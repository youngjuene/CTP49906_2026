"""FastAPI app: two websocket channels, a health probe, and a static frontend.

The roster check happens in exactly one place -- the participant `hello` frame.
There is no REST session endpoint and no way to become an admin on a participant
socket: admin is a different path, a different Channel, and a different
serialiser. A code cannot upgrade a connection in place, because there is no code
path that would.

Heartbeat, not decoration: Cloudflare quick tunnels drop an idle websocket at
roughly 100 seconds, which in this app means the quiet stretch in the middle of a
presentation. The server pings on its own timer rather than trusting clients to.
"""

import asyncio
import contextlib
import hmac
import json
import logging
import signal
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from src.atlas_state import AtlasState
from src.config import AtlasConfig, cache_key, load_config, resolve_spec
from src.embedder import build_embedder
from src.model_runtime import ModelRuntime
from src.submission_worker import SubmissionWorker
from src.submission_routes import handle_submission, hello_metadata
from src.hub import Channel, Hub
from src.mosaic_db import MosaicService, available as mosaic_available
from src.projection import warm_jit
from src.protocol import (
    MAX_FRAME_BYTES, PROTOCOL_VERSION, error, hello_ok, neighbors,
    parse_client_frame, pong,
)
from src.roster import Roster
from src.store import Store

log = logging.getLogger("atlas")
WEB_DIR = Path(__file__).resolve().parents[1] / "web"
HEARTBEAT_S = 25.0          # comfortably inside the ~100s tunnel idle timeout
MAX_HTTP_BODY_BYTES = 256 * 1024


async def _request_json(request: Request):
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_HTTP_BODY_BYTES:
            raise HTTPException(status_code=413, detail="request body is too large")
        body.extend(chunk)
    return json.loads(body)


def _valid_admin_code(supplied, expected: str) -> bool:
    if not isinstance(supplied, str):
        return False
    try:
        return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))
    except UnicodeEncodeError:
        return False


class RateLimiter:
    """One submission a second, burst of five, per connection.

    Not a security control -- the link is shared with the room. It stops one stuck
    client from queueing thirty recomputes, which would starve everyone else's
    submissions during the exact minute they are all trying to submit.
    """

    def __init__(self, per_second: float = 1.0, burst: int = 5):
        self.rate, self.burst = per_second, burst
        self._tokens: dict[str, tuple[float, float]] = {}

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        tokens, last = self._tokens.get(key, (float(self.burst), now))
        tokens = min(self.burst, tokens + (now - last) * self.rate)
        if tokens < 1.0:
            self._tokens[key] = (tokens, now)
            return False
        self._tokens[key] = (tokens - 1.0, now)
        return True

    def forget(self, key: str) -> None:
        self._tokens.pop(key, None)


class Atlas:
    """Everything the request handlers need, assembled once at startup."""

    def __init__(self, cfg: AtlasConfig):
        self.cfg = cfg
        self.hub = Hub()
        self.limiter = RateLimiter()
        self.store: Store | None = None
        self.state: AtlasState | None = None
        self.mosaic: MosaicService | None = None
        self.loop: SubmissionWorker | None = None
        self.roster: Roster | None = None
        self.viewer_changed = asyncio.Event()
        self.started_at = time.time()

    def startup(self) -> None:
        cfg = self.cfg
        # Roster before model: a mistyped path should fail in ten milliseconds,
        # not after a 1.2 GB download.
        self.roster = Roster.from_path(cfg.roster_path)
        log.info("roster: %d entries, %d targets", len(self.roster),
                 len(self.roster.targets()))

        from src.segmentation import load_policy
        policy = load_policy()
        if not policy.classroom_enabled and not cfg.fake_embedder and not cfg.allow_unreviewed_semantics:
            raise RuntimeError('semantic quality is not approved; use ATLAS_ALLOW_UNREVIEWED_SEMANTICS=1 only for development QA')
        embedder = build_embedder(cfg)
        spec = resolve_spec(cfg.embedding_model)
        log.info("embedder: %s (%s)", getattr(embedder, "model_id", spec.model_id),
                 cache_key(getattr(embedder, "spec", spec)))

        # Compile umap's numba kernels now, in the quiet minutes before anyone
        # connects. The first real fit otherwise spends 10-30s in the JIT, and
        # that lands on the first submission of the class.
        if cfg.warm_umap and warm_jit(dim=getattr(embedder, "dim", spec.dim)):
            log.info("umap JIT warmed")

        self.store = Store(cfg.db_path)
        self.store.migrate()
        self.runtime = ModelRuntime(embedder)
        self.state = AtlasState(cfg, self.store, self.roster, self.runtime)
        # Stood up before load(), because load() recomputes and commits, and
        # commit() is what refreshes the relation. Built after the store so a
        # missing database file fails on the cheap path.
        if cfg.enable_viewer and mosaic_available():
            self.mosaic = MosaicService()
            self.state.mosaic = self.mosaic
            log.info("embedding-atlas viewer: dataset relation ready")
        elif cfg.enable_viewer:
            log.warning("embedding-atlas viewer disabled: duckdb/pyarrow missing")
        self.runtime.run_sync(self.state.load)
        log.info("loaded %d opinions, layout_rev %d",
                 self.state.rev, self.state.layout_rev)
        self.loop = SubmissionWorker(self.state, self.hub, self.runtime, policy=policy)

    def reload_corpus(self) -> None:
        """SIGUSR1. Pick up opinions written straight into the database.

        scripts/seed_ai_opinions.py writes to the store rather than through a
        socket, so a running server holds a corpus that is now short of the file.
        Without this the only way to see a bulk import is a restart, which drops
        every open connection -- and PRD 4.3 puts bulk AI import in the middle of
        a session, which is exactly when that is least affordable.
        """
        if self.state is None or self.store is None:
            return
        if self.loop is not None:
            self.loop.request(reload=True)

    def reload_roster(self) -> None:
        """SIGHUP. Lets a late entry be added without dropping every open socket."""
        try:
            fresh = Roster.from_path(self.cfg.roster_path)
        except Exception as exc:            # noqa: BLE001
            log.error("roster reload refused, keeping the old one: %s", exc)
            return
        self.roster = fresh
        if self.state is not None:
            self.state.roster = fresh
            # Refresh the relation before asking open viewers to drop cached queries.
            self.state.refresh_mosaic()
            self.viewer_changed.set()
        log.info("roster reloaded: %d entries", len(fresh))


def _configure_logging() -> None:
    """Make this app's own log lines appear under `uvicorn --factory` too.

    uvicorn configures its own loggers and leaves the root alone, so without this
    every line the server writes -- which roster it loaded, which model and cache
    key, how many opinions came back, a failed recompute -- goes nowhere when the
    server is started the way the README says to start it. That is precisely the
    situation where somebody needs them: a terminal, five minutes before a class.
    """
    if log.handlers or logging.getLogger().handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s %(message)s", "%Y-%m-%d %H:%M:%S"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


def create_app(cfg: AtlasConfig | None = None, *, atlas: Atlas | None = None) -> FastAPI:
    _configure_logging()
    atlas = atlas or Atlas(cfg or load_config())

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        await asyncio.to_thread(atlas.startup)   # keeps Ctrl-C alive during a download
        task = atlas.loop.start()
        beat = asyncio.create_task(_heartbeat(atlas))
        viewer_updates = asyncio.create_task(_viewer_updates(atlas))
        # Best-effort. Unavailable when the loop is not on the main thread (a
        # TestClient, an embedded runner) and on platforms without SIGHUP, and
        # none of those is a reason to refuse to serve -- it only costs the
        # no-restart roster reload.
        with contextlib.suppress(NotImplementedError, ValueError, RuntimeError,
                                 AttributeError, OSError):
            running = asyncio.get_running_loop()
            running.add_signal_handler(signal.SIGHUP, atlas.reload_roster)
            running.add_signal_handler(signal.SIGUSR1, atlas.reload_corpus)
        try:
            yield
        finally:
            beat.cancel()
            viewer_updates.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await viewer_updates
            await atlas.loop.aclose()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            await atlas.hub.close_all()
            if atlas.mosaic is not None:
                atlas.mosaic.close()
            if atlas.store is not None:
                atlas.store.close()

    app = FastAPI(title="Feedback Atlas", lifespan=lifespan)
    app.state.atlas = atlas

    @app.get("/healthz")
    async def healthz():
        state = atlas.state
        return JSONResponse({
            "ok": state is not None,
            "protocol": PROTOCOL_VERSION,
            # The cache key names the model and its prompt convention. The access
            # code is never reported here or logged anywhere.
            "cache_key": state.cache_key if state else None,
            "opinions": len(state.opinions) if state else 0,
            "layout_rev": state.layout_rev if state else 0,
            "projector": ("umap" if state and len(state.opinions) >= atlas.cfg.pca_umap_threshold
                          else "pca"),
            "connections": atlas.hub.counts(),
            # Which front end the room can actually use, and how much the viewer's
            # relation currently holds. Reported because "the charts are empty" and
            # "the viewer never loaded" look identical from the back of a room.
            "viewer": {
                "enabled": atlas.mosaic is not None and atlas.mosaic.valid,
                "rows": atlas.mosaic.participant.rows if atlas.mosaic and atlas.mosaic.valid else 0,
            },
            "roster_count": len(atlas.roster) if atlas.roster else 0,
            "class_context": hello_metadata(atlas)['context'] if state else None,
            "limits": {"max_raw_codepoints": atlas.cfg.max_text_chars, "max_pending": 300, "automatic_units": 64},
            "model_warmed": state is not None,
            "fake_embedder": atlas.cfg.fake_embedder,
            "uptime_s": round(time.time() - atlas.started_at, 1),
            "processing": atlas.store.processing_counts() if atlas.store else {},
            "queue_diagnostics": atlas.store.queue_diagnostics() if atlas.store else {},
            "segmentation_reviewed": atlas.loop.policy.classroom_enabled if atlas.loop else False,
            "last_error": "PROCESSING_FAILED" if state and state.last_error else None,
        })

    @app.post("/api/export.csv")
    async def export_csv(request: Request):
        """The whole corpus, or a selection of it, as CSV. Admin only.

        POST rather than GET, and the code in the body rather than the query
        string, for the same reason the websocket handshake works that way: a
        query string lands in proxy logs, tunnel logs and browser history, and
        this one unlocks authorship.
        """
        try:
            body = await _request_json(request)
        except HTTPException:
            raise
        except Exception:      # noqa: BLE001
            return JSONResponse(error("MALFORMED"), status_code=400)
        if not isinstance(body, dict):
            return JSONResponse(error("MALFORMED"), status_code=400)
        if not _valid_admin_code(body.get("code"), atlas.cfg.admin_code):
            await asyncio.sleep(0.5)
            return JSONResponse(error("BAD_ACCESS_CODE"), status_code=403)

        from src.exporting import build_export
        from src.submissions import SubmissionError
        try:
            payload = build_export(atlas, body)
        except SubmissionError as exc:
            return JSONResponse(error(exc.code), status_code=409 if exc.code == 'REVISION_CONFLICT' else 400)
        return Response(
            content=payload, media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition":
                     'attachment; filename="feedback-atlas.csv"'})

    @app.post("/data/query")
    async def data_query(request: Request):
        """The SQL endpoint Embedding Atlas's viewer runs on.

        Mosaic applications do not receive data, they issue queries: every chart,
        the table, the cross-filter and the embedding view itself are SQL against
        one relation. This is `embedding_atlas.server`'s `/data/query` with the
        same three command types, so the viewer talks to it unmodified.

        Two departures, both forced by this being a classroom rather than a
        notebook on somebody's laptop:

        * **The audience is chosen before the SQL is parsed.** The admin code
          arrives in the body -- never a query string, same reason as
          /api/export.csv -- and picks which of two physically separate databases
          answers. The participant one does not contain `reviewer_id`, so the
          question cannot be asked rather than being asked and refused. See
          src/mosaic_db.py.
        * **A wrong code is a refusal, not a downgrade.** Silently answering from
          the participant relation would show an instructor a viewer that looks
          right and is quietly missing the authorship they opened it for.
        """
        if atlas.mosaic is None or not atlas.mosaic.valid:
            return JSONResponse(
                {"error": "the viewer is not enabled on this server"},
                status_code=503)
        try:
            body = await _request_json(request)
        except HTTPException:
            raise
        except Exception:      # noqa: BLE001
            return JSONResponse(error("MALFORMED"), status_code=400)

        if not isinstance(body, dict):
            return JSONResponse(error("MALFORMED"), status_code=400)
        supplied = body.get("code")
        admin = False
        if supplied not in (None, ""):
            if not _valid_admin_code(supplied, atlas.cfg.admin_code):
                await asyncio.sleep(0.5)
                return JSONResponse(error("BAD_ACCESS_CODE"), status_code=403)
            admin = True

        sql = body.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            return JSONResponse({"error": "query must carry sql"}, status_code=400)
        kind = str(body.get("type") or "arrow")

        database = atlas.mosaic.database(admin=admin)
        try:
            # Off the event loop: DuckDB blocks, and thirty phones share this loop
            # with the websocket that is delivering the map.
            payload, ctype = await asyncio.to_thread(database.query, sql, kind)
        except Exception as exc:      # noqa: BLE001
            # Mosaic reads `error` off a non-2xx body and surfaces it in the
            # viewer, so a bad query says what was wrong instead of going blank.
            return JSONResponse({"error": str(exc)}, status_code=500)
        return Response(content=payload, media_type=ctype)

    @app.websocket("/ws")
    async def participant_ws(socket: WebSocket):
        await _participant(atlas, socket)

    @app.websocket("/ws/admin")
    async def admin_ws(socket: WebSocket):
        await _admin(atlas, socket)

    if (WEB_DIR / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
    else:
        @app.get("/")
        async def placeholder():
            return HTMLResponse(
                "<title>Feedback Atlas</title>"
                "<p style='font:14px system-ui;padding:2rem'>"
                "서버는 실행 중입니다. 프론트엔드는 <code>web/</code>에 아직 없습니다."
                "</p>", status_code=200)

    return app


async def _viewer_updates(atlas: Atlas) -> None:
    """A roster edit changes viewer data without changing the opinion revision."""
    while True:
        await atlas.viewer_changed.wait()
        atlas.viewer_changed.clear()
        public = {"t": "viewer_refresh", "targets": atlas.roster.targets()}
        await atlas.hub.broadcast(Channel.PARTICIPANT, public)
        await atlas.hub.broadcast(Channel.ADMIN, {**public, "roster": [
            {"id": e.id, "display_name": e.display_name, "role": e.role}
            for e in atlas.roster.entries()]})


async def _heartbeat(atlas: Atlas) -> None:
    while True:
        await asyncio.sleep(HEARTBEAT_S)
        for channel in Channel:
            with contextlib.suppress(Exception):
                await atlas.hub.broadcast(channel, {"t": "ping"})


async def _read(socket: WebSocket):
    """-> (type, body) | error code | None on disconnect."""
    try:
        raw = await socket.receive_text()
    except (WebSocketDisconnect, RuntimeError):
        return None
    return parse_client_frame(raw, max_bytes=MAX_FRAME_BYTES)


async def _refuse(socket: WebSocket, code: str) -> None:
    with contextlib.suppress(Exception):
        await socket.send_json(error(code, fatal=True))
        await socket.close(code=4403)


async def _participant(atlas: Atlas, socket: WebSocket) -> None:
    await socket.accept()
    parsed = await _read(socket)
    if parsed is None:
        return
    if isinstance(parsed, str):
        return await _refuse(socket, parsed)
    kind, body = parsed
    if kind != "hello":
        return await _refuse(socket, "NOT_AUTHENTICATED")
    if body.get("protocol") != PROTOCOL_VERSION:
        # Loudly, so last week's cached page fails visibly instead of mis-parsing.
        return await _refuse(socket, "PROTOCOL_MISMATCH")

    entry = atlas.roster.resolve(str(body.get("id") or ""))
    if entry is None:
        return await _refuse(socket, "UNKNOWN_ID")

    state = atlas.state
    conn_id = await atlas.hub.join(socket, Channel.PARTICIPANT, identity=entry.id)
    try:
        await socket.send_json(hello_ok(
            channel="participant", rev=state.rev, layout_rev=state.layout_rev,
            targets=atlas.roster.targets(), role=entry.role,
            default_week=atlas.cfg.default_week,
            max_text_chars=atlas.cfg.max_text_chars, metadata={**hello_metadata(atlas), "submitter_key": entry.id}))
        await socket.send_json(state.participant_snapshot())
        state.mark_all_sent()
        await _serve(atlas, socket, conn_id, entry, Channel.PARTICIPANT)
    finally:
        atlas.limiter.forget(conn_id)
        await atlas.hub.leave(conn_id)


async def _admin(atlas: Atlas, socket: WebSocket) -> None:
    await socket.accept()
    parsed = await _read(socket)
    if parsed is None:
        return
    if isinstance(parsed, str):
        return await _refuse(socket, parsed)
    kind, body = parsed
    if kind != "hello" or body.get("protocol") != PROTOCOL_VERSION:
        return await _refuse(socket, "PROTOCOL_MISMATCH"
                             if kind == "hello" else "NOT_AUTHENTICATED")

    # The code arrives in the frame body, never a query string: a query string
    # lands in tunnel logs, proxy logs and browser history. compare_digest so a
    # wrong code costs the same time as a right one.
    supplied = body.get("code")
    if not _valid_admin_code(supplied, atlas.cfg.admin_code):
        await asyncio.sleep(0.5)        # flat cost on failure, no probing signal
        return await _refuse(socket, "BAD_ACCESS_CODE")

    state = atlas.state
    conn_id = await atlas.hub.join(socket, Channel.ADMIN)
    try:
        await socket.send_json(hello_ok(
            channel="admin", rev=state.rev, layout_rev=state.layout_rev,
            targets=atlas.roster.targets(),
            roster=[{"id": e.id, "display_name": e.display_name, "role": e.role}
                    for e in atlas.roster.entries()],
            default_week=atlas.cfg.default_week,
            max_text_chars=atlas.cfg.max_text_chars, metadata=hello_metadata(atlas)))
        await socket.send_json(state.admin_snapshot())
        await _serve(atlas, socket, conn_id, None, Channel.ADMIN)
    finally:
        atlas.limiter.forget(conn_id)
        await atlas.hub.leave(conn_id)


async def _serve(atlas: Atlas, socket: WebSocket, conn_id: str, entry,
                 channel: Channel) -> None:
    state = atlas.state
    while True:
        parsed = await _read(socket)
        if parsed is None:
            return
        if isinstance(parsed, str):
            with contextlib.suppress(Exception):
                await socket.send_json(error(parsed))
            continue
        kind, body = parsed

        if kind == "ping":
            await socket.send_json(pong())
        elif kind == "pong":
            continue
        elif kind == "resync":
            await socket.send_json(state.participant_snapshot()
                                   if channel is Channel.PARTICIPANT
                                   else state.admin_snapshot())
            if channel is Channel.PARTICIPANT:
                state.mark_all_sent()
        elif kind == "neighbors":
            # Admin only. Not because the distances are sensitive -- they relate
            # opinions a participant can already see -- but because it is the one
            # request whose cost scales with the corpus, and the participant
            # channel is the one with thirty sockets on it.
            if channel is not Channel.ADMIN:
                await socket.send_json(error("NOT_AUTHENTICATED"))
                continue
            target = str(body.get("id") or "")
            try:
                k = max(1, min(24, int(body.get("k", 8))))
            except (TypeError, ValueError):
                k = 8
            if not state.knows(target):
                await socket.send_json(error("UNKNOWN_OPINION"))
                continue
            await socket.send_json(neighbors(
                id=target, items=state.neighbors(target, k=k),
                ready=state.vectors_ready))
        elif await handle_submission(atlas, socket, conn_id, entry, channel, kind, body):
            continue
        else:
            await socket.send_json(error("MALFORMED"))


app = None  # uvicorn entry point is created lazily; see __main__ below


def main() -> None:
    import uvicorn
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config()
    uvicorn.run(create_app(cfg), host="0.0.0.0", port=8000, proxy_headers=True,
                forwarded_allow_ips="*")


if __name__ == "__main__":
    main()
