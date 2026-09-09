"""FastAPI app: two websocket channels, a health probe, and a static frontend.

Participant credentials are checked in the `hello` frame; the resulting signed
session is required for viewer queries and subsequent participant messages.
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
import csv
import hmac
import io
import json
import logging
import signal
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from src.atlas_state import AtlasState, RecomputeLoop
from src.access import AccessControl
from src.config import AtlasConfig, cache_key, load_config, resolve_spec
from src.embedder import build_embedder, SharedEmbedder
from src.hub import Channel, Hub
from src.models import validate_submission, _now_iso
from src.class_schedule import load_schedule, schedule_info
from src.schedule_routes import register_schedule_routes, broadcast_schedule
from src.mosaic_db import MosaicService, available as mosaic_available
from src.projection import warm_jit
from src.protocol import (
    MAX_FRAME_BYTES, PROTOCOL_VERSION, ack, error, hello_ok, neighbors,
    parse_client_frame, pong,
)
from src.roster import Roster
from src.store import Store
from src.textnorm import text_hash
from src.transport_security import TransportSecurity

log = logging.getLogger("atlas")
WEB_DIR = Path(__file__).resolve().parents[1] / "web"
HEARTBEAT_S = 25.0          # comfortably inside the ~100s tunnel idle timeout
MAX_HTTP_BODY_BYTES = 256 * 1024
HELLO_TIMEOUT_S = 10.0
MAX_CONNECTIONS = 128
MAX_CONNECTIONS_PER_IP = 64
MAX_QUERY_CONCURRENCY = 4
MAX_PENDING_QUERIES = 128


async def _request_json(request: Request):
    body = bytearray()
    try:
        async with asyncio.timeout(10):
            async for chunk in request.stream():
                if len(body) + len(chunk) > MAX_HTTP_BODY_BYTES:
                    raise HTTPException(status_code=413, detail="request body is too large")
                body.extend(chunk)
    except TimeoutError:
        raise HTTPException(status_code=408, detail="request body timed out") from None
    return json.loads(body)


def _valid_admin_code(supplied, expected: str) -> bool:
    if not isinstance(supplied, str):
        return False
    try:
        return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))
    except UnicodeEncodeError:
        return False


def _csv_cell(value):
    # CSV quoting does not prevent Excel from interpreting feedback as a formula.
    if isinstance(value, str) and (value.lstrip().startswith(("=", "+", "-", "@"))
                                   or value.startswith(("\t", "\r", "\n"))):
        return "'" + value
    return value


class RateLimiter:
    """Bounded token buckets. Callers choose identity, address, or connection keys."""

    def __init__(self, per_second: float = 1.0, burst: int = 5):
        self.rate, self.burst = per_second, burst
        self._tokens: dict[str, tuple[float, float]] = {}

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if key not in self._tokens and len(self._tokens) >= 4096:
            idle = max(60.0, self.burst / self.rate)
            self._tokens = {k: v for k, v in self._tokens.items() if now - v[1] < idle}
            if len(self._tokens) >= 4096:
                return False
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
        self.login_limiter = RateLimiter(per_second=2, burst=60)
        self.admin_limiter = RateLimiter(per_second=0.2, burst=10)
        self.query_limiter = RateLimiter(per_second=20, burst=100)
        self.message_limiter = RateLimiter(per_second=10, burst=30)
        self.connections_by_ip: dict[str, int] = {}
        self.active_queries = 0
        self.query_slots = asyncio.Semaphore(MAX_QUERY_CONCURRENCY)
        self.access: AccessControl | None = None
        self.embedder_override = None
        self.demo_submit_limiter = RateLimiter(per_second=2, burst=10)
        self.store: Store | None = None
        self.state: AtlasState | None = None
        self.mosaic: MosaicService | None = None
        self.loop: RecomputeLoop | None = None
        self.roster: Roster | None = None
        self.schedule = None
        self.schedule_marker = None
        self.roster_changed = asyncio.Event()
        self.started_at = time.time()

    def startup(self) -> None:
        cfg = self.cfg
        # Roster before model: a mistyped path should fail in ten milliseconds,
        # not after a 1.2 GB download.
        self.roster = Roster.from_path(cfg.roster_path)
        if self.access is None:
            self.access = AccessControl.from_path(cfg.access_codes_path)
        # Demo retains historical authors/targets for its example corpus, while
        # only the shared practice account has a login credential.
        if not cfg.is_demo and any(entry.id not in self.access.hashes for entry in self.roster.entries()):
            raise ValueError("Every roster ID needs a private access code; run scripts/create_access_codes.py")
        log.info("roster: %d entries, %d targets", len(self.roster),
                 len(self.roster.targets()))

        embedder = self.embedder_override or build_embedder(cfg)
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
        self.schedule = load_schedule(self.store)
        self.state = AtlasState(cfg, self.store, self.roster, embedder)
        # Stood up before load(), because load() recomputes and commits, and
        # commit() is what refreshes the relation. Built after the store so a
        # missing database file fails on the cheap path.
        if cfg.enable_viewer and mosaic_available():
            self.mosaic = MosaicService()
            self.state.mosaic = self.mosaic
            log.info("embedding-atlas viewer: dataset relation ready")
        elif cfg.enable_viewer:
            log.warning("embedding-atlas viewer disabled: duckdb/pyarrow missing")
        self.state.load()
        log.info("loaded %d opinions, layout_rev %d",
                 self.state.rev, self.state.layout_rev)
        self.loop = RecomputeLoop(self.state, self.hub)

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
        before = self.state.rev
        self.state.opinions = self.store.all_opinions()
        self.state.rev = len(self.state.opinions)
        log.info("corpus reloaded: %d -> %d opinions", before, self.state.rev)
        if self.loop is not None and self.state.rev != before:
            self.loop.request()

    def reload_roster(self) -> None:
        """SIGHUP. Lets a late entry be added without dropping every open socket."""
        try:
            fresh = Roster.from_path(self.cfg.roster_path)
            credentials = AccessControl.from_path(self.cfg.access_codes_path)
            if any(entry.id not in credentials.hashes for entry in fresh.entries()):
                raise ValueError("Every roster ID needs a private access code")
        except Exception as exc:            # noqa: BLE001
            log.error("roster reload refused, keeping the old one: %s", exc)
            return
        self.access.reload(self.cfg.access_codes_path)
        self.roster = fresh
        with contextlib.suppress(RuntimeError):
            asyncio.get_running_loop().create_task(_revoke_sessions(self))
        if self.state is not None:
            self.state.roster = fresh
            # Refresh the relation before asking open viewers to drop cached queries.
            self.state.refresh_mosaic()
        self.roster_changed.set()
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
    demo = demo_app = None
    if atlas.cfg.enable_demo:
        from src.demo import create_demo_atlas
        demo = create_demo_atlas(atlas)
        demo_app = create_app(atlas=demo)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        await asyncio.to_thread(atlas.startup)   # keeps Ctrl-C alive during a download
        demo_lifespan = contextlib.AsyncExitStack()
        if demo_app is not None:
            from src.demo import demo_accounts
            if any(atlas.access.authenticate_code(account["code"]) for account in demo_accounts()):
                if atlas.mosaic is not None:
                    atlas.mosaic.close()
                atlas.store.close()
                raise ValueError("Classroom credentials must not reuse public demo codes")
            shared = SharedEmbedder(atlas.state.embedder)
            atlas.state.embedder = shared
            demo.embedder_override = shared
            try:
                await demo_lifespan.enter_async_context(demo_app.router.lifespan_context(demo_app))
            except BaseException:
                if atlas.mosaic is not None:
                    atlas.mosaic.close()
                atlas.store.close()
                raise
        task = atlas.loop.start()
        beat = asyncio.create_task(_heartbeat(atlas))
        roster_updates = asyncio.create_task(_roster_updates(atlas))
        # Best-effort. Unavailable when the loop is not on the main thread (a
        # TestClient, an embedded runner) and on platforms without SIGHUP, and
        # none of those is a reason to refuse to serve -- it only costs the
        # no-restart roster reload.
        with contextlib.suppress(NotImplementedError, ValueError, RuntimeError,
                                 AttributeError, OSError):
            running = asyncio.get_running_loop()
            if not atlas.cfg.is_demo:
                running.add_signal_handler(signal.SIGHUP, atlas.reload_roster)
                running.add_signal_handler(signal.SIGUSR1, atlas.reload_corpus)
        try:
            yield
        finally:
            await demo_lifespan.aclose()
            beat.cancel()
            roster_updates.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await roster_updates
            await atlas.loop.aclose()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            await atlas.hub.close_all()
            if atlas.mosaic is not None:
                atlas.mosaic.close()
            if atlas.store is not None:
                atlas.store.close()

    app = FastAPI(title="Feedback Atlas", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TransportSecurity, require_https=atlas.cfg.require_https)
    app.state.atlas = atlas
    app.state.demo = demo
    register_schedule_routes(app, atlas, _request_json, _valid_admin_code, lambda: _now_iso())

    @app.get("/api/environment")
    async def environment():
        if atlas.cfg.is_demo:
            from src.demo import demo_accounts, DEMO_ADMIN_CODE, MAX_DEMO_OPINIONS
            return {"mode": "demo", "classroom_url": "/", "accounts": demo_accounts(),
                    "admin_code": DEMO_ADMIN_CODE, "opinion_limit": MAX_DEMO_OPINIONS}
        return {"mode": "classroom", "demo_url": "/demo/" if demo_app else None}

    @app.get("/healthz")
    async def healthz():
        state = atlas.state
        return JSONResponse({
            "ok": state is not None,
            "protocol": PROTOCOL_VERSION,
            # The cache key names the model and its prompt convention. The access
            # code is never reported here or logged anywhere.
            "cache_key": state.cache_key if state else None,
            "opinions": state.rev if state else 0,
            "layout_rev": state.layout_rev if state else 0,
            "projector": ("umap" if state and state.rev >= atlas.cfg.pca_umap_threshold
                          else "pca"),
            "connections": atlas.hub.counts(),
            # Which front end the room can actually use, and how much the viewer's
            # relation currently holds. Reported because "the charts are empty" and
            # "the viewer never loaded" look identical from the back of a room.
            "viewer": {
                "enabled": atlas.mosaic is not None,
                "ready": atlas.mosaic is not None and state is not None and state.mosaic_error is None,
                "rows": atlas.mosaic.participant.rows if atlas.mosaic else 0,
            },
            "uptime_s": round(time.time() - atlas.started_at, 1),
            "last_error": state.last_error if state else None,
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
            if not atlas.admin_limiter.allow(_client_ip(request)):
                return JSONResponse(error("RATE_LIMITED"), status_code=429)
            await asyncio.sleep(0.5)
            return JSONResponse(error("BAD_ACCESS_CODE"), status_code=403)

        state = atlas.state
        wanted = body.get("ids")
        if wanted is not None and (not isinstance(wanted, list)
                                   or any(not isinstance(value, str) for value in wanted)):
            return JSONResponse(error("MALFORMED"), status_code=400)
        keep = set(wanted) if isinstance(wanted, list) else None
        names = {t["id"]: t["display_name"] for t in atlas.roster.targets()}

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["id", "reviewer_id", "reviewer_name", "target_id",
                         "target_name", "text", "source", "week", "timestamp",
                         "x", "y"])
        for op in state.opinions:
            if keep is not None and op.id not in keep:
                continue
            entry = atlas.roster.resolve(op.reviewer_id)
            x, y = state.server_coords.get(op.id, (0.0, 0.0))
            writer.writerow([_csv_cell(value) for value in [op.id, op.reviewer_id,
                             entry.display_name if entry else op.reviewer_id,
                             op.target_id, names.get(op.target_id, op.target_id),
                             op.text, op.source, op.week, op.timestamp, x, y]])
        # A BOM so Excel opens Korean as UTF-8 instead of mojibake. The
        # instructor opening this is the whole audience for the feature.
        payload = ("\ufeff" + buf.getvalue()).encode("utf-8")
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
        if atlas.mosaic is None:
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
                if not atlas.admin_limiter.allow(_client_ip(request)):
                    return JSONResponse(error("RATE_LIMITED"), status_code=429)
                await asyncio.sleep(0.5)
                return JSONResponse(error("BAD_ACCESS_CODE"), status_code=403)
            admin = True

        identity = None if admin else _session_identity(atlas, _bearer(request))
        if not admin and identity is None:
            return JSONResponse(error("NOT_AUTHENTICATED"), status_code=401)
        if not atlas.query_limiter.allow("admin" if admin else identity):
            return JSONResponse(error("RATE_LIMITED"), status_code=429)

        def viewer_unavailable():
            return JSONResponse({
                "code": "VIEWER_UNAVAILABLE",
                "error": "분석 데이터를 갱신하지 못했습니다. 잠시 후 분석 보기를 다시 열어 주세요.",
            }, status_code=503)

        if atlas.state.mosaic_error is not None:
            return viewer_unavailable()

        sql = body.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            return JSONResponse({"error": "query must carry sql"}, status_code=400)
        kind = str(body.get("type") or "arrow")

        database = atlas.mosaic.database(admin=admin)
        if atlas.active_queries >= MAX_PENDING_QUERIES:
            return JSONResponse(error("RATE_LIMITED"), status_code=429)
        atlas.active_queries += 1
        try:
            # Off the event loop: DuckDB blocks, and thirty phones share this loop
            # with the websocket that is delivering the map.
            try:
                await asyncio.wait_for(atlas.query_slots.acquire(), timeout=5)
            except TimeoutError:
                return JSONResponse(error("RATE_LIMITED"), status_code=429)
            try:
                # A rebuild may fail while this request waits for capacity.
                if atlas.state.mosaic_error is not None:
                    return viewer_unavailable()
                failures_before = atlas.state.mosaic_failures
                pending = asyncio.create_task(asyncio.to_thread(database.query, sql, kind))
                try:
                    payload, ctype = await asyncio.shield(pending)
                except asyncio.CancelledError:
                    # A cancelled HTTP task must not release capacity while its
                    # SQL worker still runs. DuckDB has its own execution timeout.
                    with contextlib.suppress(Exception):
                        await pending
                    raise
                # A query already in a worker must not report a pre-failure
                # relation as current, even if a later rebuild has recovered.
                if (atlas.state.mosaic_error is not None
                        or atlas.state.mosaic_failures != failures_before):
                    return viewer_unavailable()
            finally:
                atlas.query_slots.release()
        except Exception as exc:      # noqa: BLE001
            # Mosaic reads `error` off a non-2xx body and surfaces it in the
            # viewer, so a bad query says what was wrong instead of going blank.
            return JSONResponse({"error": str(exc)}, status_code=500)
        finally:
            atlas.active_queries -= 1
        return Response(content=payload, media_type=ctype)

    @app.websocket("/ws")
    async def participant_ws(socket: WebSocket):
        await _connection(atlas, socket, _participant)

    @app.websocket("/ws/admin")
    async def admin_ws(socket: WebSocket):
        await _connection(atlas, socket, _admin)

    if demo_app is not None:
        @app.get("/demo", include_in_schema=False)
        async def demo_redirect():
            return RedirectResponse("/demo/", status_code=307)

        app.mount("/demo", demo_app, name="demo")

    @app.get("/admin", include_in_schema=False)
    async def admin_page():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/admin/", include_in_schema=False)
    async def admin_redirect():
        return RedirectResponse("../admin", status_code=307)

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


async def _roster_updates(atlas: Atlas) -> None:
    """Publish roster changes without reauthenticating or sharing author lists."""
    while True:
        await atlas.roster_changed.wait()
        atlas.roster_changed.clear()
        targets = atlas.roster.targets()
        await atlas.hub.broadcast(Channel.PARTICIPANT, {"t": "roster", "targets": targets})
        await atlas.hub.broadcast(Channel.ADMIN, {
            "t": "roster", "targets": targets,
            "roster": [{"id": e.id, "display_name": e.display_name, "role": e.role}
                       for e in atlas.roster.entries()],
        })
        if atlas.mosaic is not None:
            await atlas.hub.broadcast(Channel.PARTICIPANT, {"t": "viewer_refresh"})
            await atlas.hub.broadcast(Channel.ADMIN, {"t": "viewer_refresh"})


async def _heartbeat(atlas: Atlas) -> None:
    while True:
        await asyncio.sleep(HEARTBEAT_S)
        await _revoke_sessions(atlas)
        now = _now_iso()
        info = schedule_info(atlas, now)
        if atlas.schedule_marker != (info["revision"], info["current_week"]):
            await broadcast_schedule(atlas, now)
        for channel in Channel:
            with contextlib.suppress(Exception):
                await atlas.hub.broadcast(channel, {"t": "ping"})


async def _revoke_sessions(atlas: Atlas) -> None:
    for conn in atlas.hub.connections():
        if (conn.channel is Channel.PARTICIPANT
                and _session_identity(atlas, conn.meta.get("session_token")) != conn.identity):
            await _refuse(conn.socket, "NOT_AUTHENTICATED")
            await atlas.hub.leave(conn.id)


async def _read(socket: WebSocket):
    """-> (type, body) | error code | None on disconnect."""
    try:
        raw = await socket.receive_text()
    except (WebSocketDisconnect, RuntimeError):
        return None
    return parse_client_frame(raw, max_bytes=MAX_FRAME_BYTES)


def _client_ip(connection) -> str:
    return connection.client.host if connection.client else "unknown"


def _bearer(request: Request) -> str:
    header = request.headers.get("authorization", "")
    return header[7:] if header.lower().startswith("bearer ") else ""


def _session_identity(atlas: Atlas, token) -> str | None:
    identity = atlas.access.verify(token)
    return identity if identity and atlas.roster.resolve(identity) else None


async def _connection(atlas: Atlas, socket: WebSocket, handler) -> None:
    ip = _client_ip(socket)
    count = atlas.connections_by_ip.get(ip, 0)
    total_limit = 32 if atlas.cfg.is_demo else MAX_CONNECTIONS
    ip_limit = 32 if atlas.cfg.is_demo else MAX_CONNECTIONS_PER_IP
    if (sum(atlas.connections_by_ip.values()) >= total_limit
            or count >= ip_limit or not atlas.login_limiter.allow(ip)):
        await socket.close(code=1013)
        return
    atlas.connections_by_ip[ip] = count + 1
    try:
        await handler(atlas, socket)
    except (WebSocketDisconnect, TimeoutError):
        with contextlib.suppress(Exception):
            await socket.close(code=1008)
    finally:
        remaining = atlas.connections_by_ip[ip] - 1
        if remaining:
            atlas.connections_by_ip[ip] = remaining
        else:
            del atlas.connections_by_ip[ip]


async def _hello(socket: WebSocket):
    return await asyncio.wait_for(_read(socket), timeout=HELLO_TIMEOUT_S)


async def _refuse(socket: WebSocket, code: str) -> None:
    with contextlib.suppress(Exception):
        await socket.send_json(error(code, fatal=True))
        await socket.close(code=4403)


async def _participant(atlas: Atlas, socket: WebSocket) -> None:
    await socket.accept()
    parsed = await _hello(socket)
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

    token = body.get("session_token")
    if token:
        identity = _session_identity(atlas, token)
        entry = atlas.roster.resolve(identity) if identity else None
    else:
        identity = atlas.access.authenticate_code(body.get("code"))
        entry = atlas.roster.resolve(identity) if identity else None
        if entry:
            token = atlas.access.issue(entry.id)
    if entry is None:
        return await _refuse(socket, "BAD_ACCESS_CODE")

    state = atlas.state
    conn_id = await atlas.hub.join(socket, Channel.PARTICIPANT, identity=entry.id)
    atlas.hub.get(conn_id).meta["session_token"] = token
    try:
        welcome = hello_ok(
            channel="participant", rev=state.rev, layout_rev=state.layout_rev,
            targets=atlas.roster.targets(), role=entry.role,
            default_week=atlas.cfg.default_week,
            max_text_chars=atlas.cfg.max_text_chars)
        welcome["session_token"] = token
        welcome["submission_owner"] = atlas.access.submission_owner(
            entry.id, scope="demo" if atlas.cfg.is_demo else "classroom")
        welcome["schedule"] = schedule_info(atlas, _now_iso())
        await socket.send_json(welcome)
        await socket.send_json(state.participant_snapshot())
        state.mark_all_sent()
        await _serve(atlas, socket, conn_id, entry, Channel.PARTICIPANT, token=token)
    finally:
        atlas.message_limiter.forget(conn_id)
        if atlas.cfg.is_demo:
            atlas.limiter.forget(conn_id)
        await atlas.hub.leave(conn_id)


async def _admin(atlas: Atlas, socket: WebSocket) -> None:
    await socket.accept()
    parsed = await _hello(socket)
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
        if not atlas.admin_limiter.allow(_client_ip(socket)):
            return await _refuse(socket, "RATE_LIMITED")
        await asyncio.sleep(0.5)        # flat cost on failure, no probing signal
        return await _refuse(socket, "BAD_ACCESS_CODE")

    state = atlas.state
    conn_id = await atlas.hub.join(socket, Channel.ADMIN)
    try:
        welcome = hello_ok(
            channel="admin", rev=state.rev, layout_rev=state.layout_rev,
            targets=atlas.roster.targets(),
            roster=[{"id": e.id, "display_name": e.display_name, "role": e.role}
                    for e in atlas.roster.entries()],
            default_week=atlas.cfg.default_week,
            max_text_chars=atlas.cfg.max_text_chars)
        welcome["schedule"] = schedule_info(atlas, _now_iso())
        await socket.send_json(welcome)
        await socket.send_json(state.admin_snapshot())
        await _serve(atlas, socket, conn_id, None, Channel.ADMIN)
    finally:
        atlas.message_limiter.forget(conn_id)
        await atlas.hub.leave(conn_id)


async def _serve(atlas: Atlas, socket: WebSocket, conn_id: str, entry,
                 channel: Channel, *, token=None) -> None:
    state = atlas.state
    while True:
        parsed = await _read(socket)
        if parsed is None:
            return
        if channel is Channel.PARTICIPANT and _session_identity(atlas, token) != entry.id:
            return await _refuse(socket, "NOT_AUTHENTICATED")
        if not atlas.message_limiter.allow(conn_id):
            return await _refuse(socket, "RATE_LIMITED")
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
            nonce = body.get("nonce")
            await socket.send_json(neighbors(
                id=target, items=state.neighbors(target, k=k),
                ready=state.vectors_ready,
                nonce=nonce if isinstance(nonce, str) and len(nonce) <= 128 else None))
        elif kind == "submit":
            if channel is not Channel.PARTICIPANT or entry is None:
                await socket.send_json(error("NOT_AUTHENTICATED"))
                continue
            nonce = body.get("nonce")
            recorded = state.store.submission_receipt(entry.id, nonce)
            if recorded is not None:
                await socket.send_json(ack(nonce=nonce, id=recorded, rev=state.rev))
                continue
            received_at = _now_iso()
            submission_body = dict(body)
            if atlas.cfg.is_demo:
                submission_body["week"] = atlas.cfg.default_week
            elif atlas.cfg.auto_week:
                active_week = atlas.schedule.week_for(received_at)
                if active_week is None:
                    await socket.send_json(error("OUTSIDE_CLASS_PERIOD"))
                    continue
                submission_body["week"] = active_week
            if atlas.cfg.is_demo:
                from src.demo import MAX_DEMO_OPINIONS
                if len(state.opinions) >= MAX_DEMO_OPINIONS:
                    await socket.send_json(error("DEMO_FULL"))
                    continue
                if not atlas.demo_submit_limiter.allow("demo"):
                    await socket.send_json(error("RATE_LIMITED"))
                    continue
            if not atlas.limiter.allow(conn_id if atlas.cfg.is_demo else entry.id):
                await socket.send_json(error("RATE_LIMITED"))
                continue
            op, err = validate_submission(
                submission_body, reviewer=entry, roster=atlas.roster,
                max_chars=atlas.cfg.max_text_chars,
                default_week=atlas.cfg.default_week, now=received_at)
            if err is not None:
                # To this socket only. A refusal is not news for the room.
                await socket.send_json(error(err))
                continue
            # Durable before acknowledged: a crash between the two would otherwise
            # tell somebody their opinion landed when it did not.
            inserted, opinion_id = state.store.insert_opinion_with_receipt(
                op, text_hash(op.text), nonce)
            rev = state.add(op) if inserted else state.rev
            await socket.send_json(ack(nonce=str(body.get("nonce") or ""),
                                       id=opinion_id, rev=rev))
            if inserted:
                atlas.loop.request()
        else:
            await socket.send_json(error("MALFORMED"))


app = None  # uvicorn entry point is created lazily; see __main__ below


def main() -> None:
    raise SystemExit("Use bash scripts/serve.sh or scripts/class.sh start for verified HTTPS startup.")


if __name__ == "__main__":
    main()
