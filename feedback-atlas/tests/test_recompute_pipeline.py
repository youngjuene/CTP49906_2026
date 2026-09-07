"""Debounce, threading, and what happens to a submit that lands mid-recompute.

Two defects. The first was visible: a burst of twenty submissions queued twenty
UMAP runs and the map fell minutes behind the room. The second was not: a submit
that arrived while a recompute was already in flight was absent from that
recompute's snapshot, got no coordinates, and -- because nothing re-armed the
debouncer -- was never picked up by another one either. It sat in the database,
correct and invisible.

The threading assertions are deliberately about thread identity rather than about
timing. "The encode must not stall the event loop" is only a hope until something
checks which thread it actually ran on.

Run:  python -m pytest feedback-atlas/tests/test_recompute_pipeline.py
  or:  python feedback-atlas/tests/test_recompute_pipeline.py
"""
import asyncio
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> feedback-atlas/

from src.atlas_state import AtlasState, Debouncer, RecomputeLoop  # noqa: E402
from src.config import AtlasConfig  # noqa: E402
from src.embedder import HashEmbedder  # noqa: E402
from src.hub import Channel, Hub  # noqa: E402
from src.models import new_opinion  # noqa: E402
from src.roster import Roster  # noqa: E402
from src.store import Store  # noqa: E402
from src.textnorm import text_hash  # noqa: E402

ROSTER = Roster.from_csv_text(
    "id,display_name,role\ntarget1,대상1,student\nwriter1,작성자1,auditor\n")


class FakeSocket:
    def __init__(self):
        self.sent = []

    async def send_json(self, message):
        self.sent.append(message)

    async def close(self, code=1000):
        pass


class SlowEmbedder(HashEmbedder):
    """Stands in for sentence-transformers, which blocks for seconds at a time."""

    def __init__(self, dim=16, delay=0.25):
        super().__init__(dim=dim)
        self.delay = delay
        self.threads: list[int] = []
        self.concurrent = 0
        self.max_concurrent = 0
        self._lock = threading.Lock()

    def encode(self, texts):
        with self._lock:
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
            self.threads.append(threading.get_ident())
        try:
            time.sleep(self.delay)
            return super().encode(texts)
        finally:
            with self._lock:
                self.concurrent -= 1


def _state(tmp_path, embedder=None, **cfg_kw):
    cfg = AtlasConfig(admin_code="x", roster_path="unused",
                      db_path=str(tmp_path / "atlas.db"),
                      debounce_s=0.05, max_debounce_s=0.30, **cfg_kw)
    store = Store(cfg.db_path)
    store.migrate()
    state = AtlasState(cfg, store, ROSTER, embedder or HashEmbedder(dim=16))
    state.load()
    return state


def _submit(state, text):
    op = new_opinion(reviewer_id="writer1", target_id="target1", text=text)
    state.store.insert_opinion(op, text_hash(op.text))
    return op, state.add(op)


# --- the debouncer, which is pure and clock-injected -------------------------

def test_a_burst_collapses_into_one_scheduled_pass():
    now = [0.0]
    d = Debouncer(0.4, max_delay_s=2.0, clock=lambda: now[0])
    for t in (0.00, 0.01, 0.02, 0.03):
        d.request(t)
    assert abs(d.due_at() - 0.43) < 1e-9, \
        "each request should push the trailing edge out"
    now[0] = 0.42
    assert d.ready() is False
    now[0] = 0.43
    assert d.ready() is True


def test_a_continuous_stream_still_fires_because_of_the_ceiling():
    """Without max_delay_s a trailing debounce never fires under a steady stream --
    which is the busiest minute of the session, when the whole class is typing."""
    now = [0.0]
    d = Debouncer(0.4, max_delay_s=2.0, clock=lambda: now[0])
    d.request(0.0)
    for t in [i * 0.1 for i in range(1, 40)]:
        now[0] = t
        d.request(t)
        if d.ready():
            break
    assert now[0] <= 2.0 + 1e-9, f"never fired; reached t={now[0]}"


def test_clearing_forgets_both_edges():
    now = [0.0]
    d = Debouncer(0.4, max_delay_s=2.0, clock=lambda: now[0])
    d.request(0.0)
    d.clear()
    assert d.pending is False and d.due_at() is None


# --- the loop ----------------------------------------------------------------

def test_twenty_submissions_in_a_burst_produce_one_or_two_passes(tmp_path):
    """The visible defect: twenty submits used to mean twenty UMAP runs."""
    async def main():
        state = _state(tmp_path)
        loop = RecomputeLoop(state, Hub())
        loop.start()
        for i in range(20):
            _submit(state, f"의견 {i}")
            loop.request()
            await asyncio.sleep(0.005)
        await asyncio.sleep(0.6)
        await loop.aclose()
        assert 1 <= loop.passes <= 3, f"{loop.passes} passes for one burst"
        assert len(state.server_coords) == 20
    asyncio.run(main())


def test_the_embedding_runs_off_the_event_loop_thread(tmp_path):
    """Asserted by thread identity, not by timing.

    "It must not stall the loop" stays a hope until something checks where the
    blocking call actually ran.
    """
    async def main():
        embedder = SlowEmbedder(delay=0.2)
        state = _state(tmp_path, embedder)
        loop = RecomputeLoop(state, Hub())
        loop.start()
        _submit(state, "느리게 계산되는 의견")
        loop.request()

        # The loop must keep answering while the encode is blocking.
        ticks = 0
        deadline = time.monotonic() + 0.6
        while time.monotonic() < deadline:
            await asyncio.sleep(0.01)
            ticks += 1
        await loop.aclose()

        assert embedder.threads, "the embedder never ran"
        assert threading.get_ident() not in embedder.threads, \
            "encode ran on the event loop thread and blocked every socket"
        assert ticks > 20, f"the loop only ticked {ticks} times while encoding"
    asyncio.run(main())


def test_only_one_recompute_is_ever_in_flight(tmp_path):
    """sentence-transformers' encode is not safe under concurrent calls. That
    safety comes from the single-task structure, so it is worth asserting that the
    structure actually holds rather than trusting it."""
    async def main():
        embedder = SlowEmbedder(delay=0.15)
        state = _state(tmp_path, embedder)
        loop = RecomputeLoop(state, Hub())
        loop.start()
        for i in range(10):
            _submit(state, f"동시성 확인 {i}")
            loop.request()
            await asyncio.sleep(0.03)
        await asyncio.sleep(0.8)
        await loop.aclose()
        assert embedder.max_concurrent == 1, \
            f"{embedder.max_concurrent} concurrent encodes"
        assert loop.in_flight == 0
    asyncio.run(main())


def test_a_submit_during_a_recompute_lands_in_the_next_one(tmp_path):
    """The invisible defect: correct in the database, absent from every map."""
    async def main():
        embedder = SlowEmbedder(delay=0.3)
        state = _state(tmp_path, embedder)
        loop = RecomputeLoop(state, Hub())
        loop.start()

        first, _ = _submit(state, "첫 번째 의견")
        loop.request()
        await asyncio.sleep(0.15)          # mid-encode
        late, _ = _submit(state, "재계산 도중에 도착한 의견")
        loop.request()

        await asyncio.sleep(1.2)
        await loop.aclose()

        assert first.id in state.server_coords
        assert late.id in state.server_coords, \
            "an opinion submitted mid-recompute never got coordinates"
    asyncio.run(main())


def test_each_pass_sends_one_message_per_channel_and_never_one_reused(tmp_path):
    """Two serialisers, two broadcasts. The participant frame must not merely
    happen to lack reviewer_id -- it must come from the other function."""
    async def main():
        state = _state(tmp_path)
        hub = Hub()
        participant, admin = FakeSocket(), FakeSocket()
        await hub.join(participant, Channel.PARTICIPANT, identity="writer1")
        await hub.join(admin, Channel.ADMIN)

        loop = RecomputeLoop(state, hub)
        loop.start()
        _submit(state, "채널마다 다른 메시지")
        loop.request()
        await asyncio.sleep(0.5)
        await loop.aclose()

        assert participant.sent and admin.sent
        p_points = participant.sent[-1].get("added") or participant.sent[-1].get("points")
        a_points = admin.sent[-1].get("added") or admin.sent[-1].get("points")
        assert p_points and a_points
        assert "reviewer_id" not in p_points[0]
        assert a_points[0]["reviewer_id"] == "writer1"
        assert a_points[0]["reviewer_name"] == "작성자1"
    asyncio.run(main())


def test_a_failing_recompute_leaves_the_previous_layout_intact(tmp_path):
    """The anchor frame must never half-update, or the next Procrustes fit is
    against a layout that never existed."""
    async def main():
        state = _state(tmp_path)
        loop = RecomputeLoop(state, Hub())
        loop.start()
        _submit(state, "정상 의견")
        loop.request()
        await asyncio.sleep(0.4)
        good_coords = dict(state.server_coords)
        good_layout_rev = state.layout_rev
        assert good_coords

        class Broken(HashEmbedder):
            def encode(self, texts):
                raise RuntimeError("model fell over")

        state.embedder = Broken(dim=16)
        _submit(state, "이 의견은 실패를 유발합니다")
        loop.request()
        await asyncio.sleep(0.4)
        await loop.aclose()

        assert state.server_coords == good_coords, "a failed pass moved the anchor"
        assert state.layout_rev == good_layout_rev
        assert state.last_error and "model fell over" in state.last_error
    asyncio.run(main())


def test_the_deltas_baseline_advances_only_by_what_was_broadcast(tmp_path):
    """sent_coords and sent_rev are what clients hold; server_coords is the truth.

    Conflating them reintroduces unbounded client divergence (see src/deltas.py).
    """
    async def main():
        state = _state(tmp_path)
        hub = Hub()
        await hub.join(FakeSocket(), Channel.PARTICIPANT, identity="writer1")
        loop = RecomputeLoop(state, hub)
        loop.start()
        for i in range(3):
            _submit(state, f"기준선 확인 {i}")
            loop.request()
            await asyncio.sleep(0.25)
        await loop.aclose()

        assert set(state.sent_coords) == set(state.server_coords)
        assert state.sent_rev == state.rev
    asyncio.run(main())


def test_reloading_the_corpus_from_disk_reproduces_the_same_layout(tmp_path):
    """A restart must not rotate the map: the previous coordinates are on disk to
    anchor the first fit against."""
    async def main():
        state = _state(tmp_path)
        loop = RecomputeLoop(state, Hub())
        loop.start()
        for i in range(5):
            _submit(state, f"재시작 확인 {i}")
        loop.request()
        await asyncio.sleep(0.5)
        await loop.aclose()
        before = dict(state.server_coords)
        state.store.close()

        restarted = _state(tmp_path)
        assert set(restarted.server_coords) == set(before)
        worst = max(abs(restarted.server_coords[k][0] - before[k][0])
                    for k in before)
        assert worst < 0.05, f"the map moved by {worst} across a restart"
    asyncio.run(main())


if __name__ == "__main__":
    import tempfile

    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    fails = 0
    for fn in fns:
        try:
            if fn.__code__.co_varnames[: fn.__code__.co_argcount] == ("tmp_path",):
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print("PASS", fn.__name__)
        except Exception as e:  # noqa: BLE001
            fails += 1
            print("FAIL", fn.__name__, "->", type(e).__name__, e)
    print(f"\n{len(fns) - fails} passed, {fails} failed")
    sys.exit(1 if fails else 0)
