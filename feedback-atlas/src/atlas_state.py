"""Corpus state, the debounce, and the single task that recomputes the layout.

The division of labour here is what keeps the event loop responsive and the
embedder safe:

  worker thread   embeddings and projection -- the blocking, CPU-bound half.
                  sentence-transformers' encode() is not safe under concurrent
                  calls, and exactly one recompute is ever in flight, so that
                  safety comes from the structure rather than from a lock nobody
                  remembers to take.
  event loop      committing results, diffing, and broadcasting. Everything that
                  touches connection state stays on one thread.

The diff deliberately happens on the loop, not in the worker: its baseline is
`sent_coords`, which only the loop may change. Computing it in the worker would
race against whatever was broadcast while the worker was running.
"""

import asyncio
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from src.config import AtlasConfig, cache_key, resolve_spec
from src.deltas import diff_layout, should_send_snapshot
from src.hub import Channel, Hub
from src.models import Opinion
from src.payloads import admin_points, participant_points
from src.projection import Layout, stabilize
from src.protocol import delta, snapshot
from src.roster import Roster
from src.store import Store
from src.textnorm import text_hash


@dataclass(frozen=True)
class RecomputeInput:
    """An immutable view of the corpus, taken on the loop and handed to the worker."""
    rev: int
    opinions: tuple[Opinion, ...]
    prev_coords: Mapping[str, tuple[float, float]]


@dataclass
class RecomputeResult:
    rev: int
    coords: dict[str, tuple[float, float]]
    drift: float = 0.0
    refit: bool = False
    encoded: int = 0
    seconds: float = 0.0
    error: str | None = None
    # Carried out of the worker so the loop can cache it for neighbour queries.
    # The matrix already exists at this point; recomputing or re-reading it per
    # query would be work we have already done.
    vector_ids: list[str] | None = None
    vectors: object | None = None


class Debouncer:
    """Trailing-edge debounce with a ceiling. Pure; the clock is injected.

    The ceiling is the part that is easy to leave out and cannot be left out. A
    plain trailing debounce resets on every request, so a continuous stream --
    a whole class typing at once, which is the busiest and most important moment
    of the session -- never fires at all.
    """

    def __init__(self, delay_s: float, *, max_delay_s: float | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.delay_s = delay_s
        self.max_delay_s = max_delay_s
        self._clock = clock
        self._last_request: float | None = None
        self._first_request: float | None = None

    def request(self, now: float | None = None) -> None:
        now = self._clock() if now is None else now
        self._last_request = now
        if self._first_request is None:
            self._first_request = now

    @property
    def pending(self) -> bool:
        return self._last_request is not None

    def due_at(self) -> float | None:
        if self._last_request is None:
            return None
        due = self._last_request + self.delay_s
        if self.max_delay_s is not None and self._first_request is not None:
            due = min(due, self._first_request + self.max_delay_s)
        return due

    # A deadline built by adding floats lands a few ulps past the number anyone
    # would write down: 0.03 + 0.4 is 0.43000000000000005, so a clock reading of
    # exactly 0.43 is "not yet". Harmless here, where a late wake-up costs
    # microseconds -- but it is the kind of comparison that is wrong in a way no
    # amount of staring reveals, so give it a tolerance far below anything that
    # could matter at a 400 ms debounce.
    _TOLERANCE_S = 1e-9

    def ready(self, now: float | None = None) -> bool:
        due = self.due_at()
        if due is None:
            return False
        return (self._clock() if now is None else now) >= due - self._TOLERANCE_S

    def clear(self) -> None:
        self._last_request = None
        self._first_request = None


class AtlasState:
    """The corpus, its layout, and what each client has been told."""

    def __init__(self, cfg: AtlasConfig, store: Store, roster: Roster, embedder):
        self.cfg = cfg
        self.store = store
        self.roster = roster
        self.embedder = embedder
        self.spec = resolve_spec(cfg.embedding_model)
        self.cache_key = cache_key(getattr(embedder, "spec", self.spec))
        self.dim = getattr(embedder, "dim", self.spec.dim)

        self.opinions: list[Opinion] = []
        self.rev = 0
        self.layout_rev = 0
        # server_coords is the truth; sent_coords is what clients were last told.
        # They are separate on purpose -- see the contract in src/deltas.py.
        self.server_coords: dict[str, tuple[float, float]] = {}
        self.sent_coords: dict[str, tuple[float, float]] = {}
        # The rev clients were last brought up to. A delta declares this as
        # from_rev, and a client whose own rev differs knows it missed one and
        # asks for a resync -- which is how gap detection stays stateless here.
        self.sent_rev = 0
        self.layout = Layout(threshold=cfg.pca_umap_threshold,
                             n_neighbors=cfg.n_neighbors)
        self.last_error: str | None = None
        # The embedding matrix from the last successful recompute, kept for
        # neighbour lookups. At 1000 x 768 float32 this is ~3 MB.
        self._vector_row: dict[str, int] = {}
        self._row_ids: list[str] = []
        self._vectors = None

    # --- startup ------------------------------------------------------------
    def load(self) -> None:
        self.opinions = self.store.all_opinions()
        self.rev = len(self.opinions)
        # layout_rev is advertised in every frame and in /healthz, so it should
        # keep counting across a restart rather than starting over. It is not used
        # for gap detection (rev is), but a counter that silently goes backwards
        # is the sort of thing that costs an hour when something else breaks.
        stored_rev = self.store.get_meta("layout_rev")
        self.layout_rev = int(stored_rev) if stored_rev is not None else 0
        stored = self.store.all_coords()
        # Anchor on what the previous process left behind, so a restart does not
        # rotate the map out from under anyone who reconnects.
        self.server_coords = {o.id: stored[o.id] for o in self.opinions if o.id in stored}
        self.sent_coords = {}
        if self.opinions:
            result = self.recompute(self.snapshot_input())
            if result.error is None:
                self.commit(result)

    def snapshot_input(self) -> RecomputeInput:
        return RecomputeInput(rev=self.rev, opinions=tuple(self.opinions),
                              prev_coords=dict(self.server_coords))

    # --- mutation (loop thread only) ----------------------------------------
    def add(self, op: Opinion) -> int:
        self.opinions.append(op)
        self.rev += 1
        return self.rev

    def commit(self, result: RecomputeResult) -> None:
        self.server_coords = result.coords
        if result.vector_ids is not None and result.vectors is not None:
            self._row_ids = list(result.vector_ids)
            self._vector_row = {oid: i for i, oid in enumerate(self._row_ids)}
            self._vectors = result.vectors
        self.layout_rev += 1
        self.store.put_coords(result.coords, self.layout_rev)
        self.store.set_meta("layout_rev", str(self.layout_rev))

    # --- the blocking half (worker thread) ----------------------------------
    def recompute(self, snap: RecomputeInput) -> RecomputeResult:
        started = time.monotonic()
        try:
            ops = snap.opinions
            if not ops:
                return RecomputeResult(rev=snap.rev, coords={})

            hashes = [text_hash(o.text) for o in ops]
            cached = self.store.get_embeddings(hashes, self.cache_key)
            missing = [(h, t) for h, t in self.store.missing_embeddings(self.cache_key)]
            encoded = 0
            if missing:
                vectors = self.embedder.encode([t for _, t in missing])
                self.store.put_embeddings(
                    self.cache_key, self.dim,
                    {h: v for (h, _), v in zip(missing, vectors, strict=True)})
                encoded = len(missing)
                cached = self.store.get_embeddings(hashes, self.cache_key)

            X = np.vstack([cached[h] for h in hashes])
            ids = [o.id for o in ops]
            raw, refit = self.layout.project(X, ids)
            coords, drift = stabilize(raw, ids, snap.prev_coords)
            placed = {oid: (float(x), float(y))
                      for oid, (x, y) in zip(ids, coords, strict=True)}
            return RecomputeResult(rev=snap.rev, coords=placed, drift=drift,
                                   refit=refit, encoded=encoded,
                                   seconds=time.monotonic() - started,
                                   vector_ids=ids, vectors=X)
        except Exception as exc:      # noqa: BLE001
            # A failed recompute must leave server_coords untouched, so the anchor
            # frame never half-updates and the next attempt retries cleanly.
            return RecomputeResult(rev=snap.rev, coords={},
                                   seconds=time.monotonic() - started,
                                   error=f"{type(exc).__name__}: {exc}")

    @property
    def vectors_ready(self) -> bool:
        """Whether a layout has been computed yet.

        Distinct from "this opinion has no neighbours". Right after startup, or
        in the seconds between a submission and the next recompute, the corpus is
        known but its vectors are not -- and answering "no such opinion" there
        would be a lie about a point the admin can see on screen.
        """
        return self._vectors is not None

    def knows(self, opinion_id: str) -> bool:
        return any(o.id == opinion_id for o in self.opinions)

    def neighbors(self, opinion_id: str, k: int = 8) -> list[dict]:
        """The k nearest opinions to this one, closest first.

        Every embedder here returns L2-normalised vectors, so the dot product is
        the cosine similarity and the distance is 1 - that. At a thousand rows
        this is one matmul against a (N, dim) matrix -- microseconds -- which is
        why there is no ANN index: it would buy nothing and cost a dependency.

        This is the question the workshop is actually about, asked one point at a
        time: given what a person wrote, what did the model write that lands
        nearest to it?
        """
        row = self._vector_row.get(opinion_id)
        if row is None or self._vectors is None:
            return []
        sims = self._vectors @ self._vectors[row]
        # argpartition rather than a full sort: we want the top k, not an order
        # over the whole corpus.
        take = min(k + 1, sims.shape[0])
        top = np.argpartition(-sims, take - 1)[:take]
        top = top[np.argsort(-sims[top])]
        out = []
        for j in top:
            j = int(j)
            if j == row:
                continue
            out.append({"id": self._row_ids[j], "distance": float(1.0 - sims[j])})
            if len(out) >= k:
                break
        return out

    # --- serialisation ------------------------------------------------------
    def participant_snapshot(self) -> dict:
        return snapshot(rev=self.rev, layout_rev=self.layout_rev,
                        points=participant_points(self.opinions, self.server_coords))

    def admin_snapshot(self) -> dict:
        return snapshot(rev=self.rev, layout_rev=self.layout_rev,
                        points=admin_points(self.opinions, self.server_coords, self.roster))

    def mark_all_sent(self) -> None:
        self.sent_coords = dict(self.server_coords)
        self.sent_rev = self.rev


class RecomputeLoop:
    """The one task that turns submissions into broadcasts."""

    def __init__(self, state: AtlasState, hub: Hub, *, on_event: Callable | None = None):
        self.state = state
        self.hub = hub
        self.debouncer = Debouncer(state.cfg.debounce_s,
                                   max_delay_s=state.cfg.max_debounce_s)
        self._wake = asyncio.Event()
        self._stop = False
        self._task: asyncio.Task | None = None
        self.on_event = on_event or (lambda *a, **k: None)
        self.in_flight = 0
        self.passes = 0

    def request(self) -> None:
        self.debouncer.request()
        self._wake.set()

    async def run(self) -> None:
        while not self._stop:
            if not self.debouncer.pending:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                if self._stop:
                    break
            due = self.debouncer.due_at()
            if due is not None:
                remaining = due - time.monotonic()
                if remaining > 0:
                    await asyncio.sleep(remaining)
            if self._stop:
                break
            self.debouncer.clear()
            await self._one_pass()

    async def _one_pass(self) -> None:
        snap = self.state.snapshot_input()
        if not snap.opinions:
            return
        self.in_flight += 1
        try:
            result = await asyncio.to_thread(self.state.recompute, snap)
        finally:
            self.in_flight -= 1
        self.passes += 1

        if result.error is not None:
            self.state.last_error = result.error
            self.on_event("recompute_failed", result)
            return

        self.state.commit(result)
        await self.broadcast(result)
        self.on_event("recomputed", result)

    async def broadcast(self, result: RecomputeResult) -> None:
        state = self.state
        added_ids, moved = diff_layout(state.sent_coords, result.coords,
                                       move_epsilon=state.cfg.move_epsilon)
        if not added_ids and not moved:
            return

        by_id = {o.id: o for o in state.opinions}
        if should_send_snapshot(len(moved), len(state.opinions)):
            # Two snapshots, one per serialiser. Never one message reused.
            await self.hub.broadcast(Channel.PARTICIPANT, state.participant_snapshot())
            await self.hub.broadcast(Channel.ADMIN, state.admin_snapshot())
            state.mark_all_sent()
            return

        fresh = [by_id[a] for a in added_ids if a in by_id]
        common = dict(from_rev=state.sent_rev, rev=result.rev,
                      layout_rev=state.layout_rev, moved=moved, drift=result.drift)
        await self.hub.broadcast(Channel.PARTICIPANT, delta(
            added=participant_points(fresh, result.coords), **common))
        await self.hub.broadcast(Channel.ADMIN, delta(
            added=admin_points(fresh, result.coords, state.roster), **common))

        for oid in added_ids:
            state.sent_coords[oid] = result.coords[oid]
        for m in moved:
            state.sent_coords[m["id"]] = (m["x"], m["y"])
        state.sent_rev = result.rev

    async def aclose(self) -> None:
        self._stop = True
        self._wake.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):   # noqa: BLE001
                pass

    def start(self) -> asyncio.Task:
        self._task = asyncio.create_task(self.run())
        return self._task
