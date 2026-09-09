"""Opt-in classroom load and durability QA over a live local server.

This file is intentionally named outside pytest's default ``test_*.py`` pattern.
Run it directly when checking classroom readiness:

    .venv/bin/python -m pytest tests/classroom_load.py -s

Environment knobs:

    ATLAS_CLASSROOM_PARTICIPANTS=30
    ATLAS_CLASSROOM_SEED=1000
    ATLAS_CLASSROOM_TIMEOUT=90

It uses the fake embedder, a temporary roster/database, localhost only, explicit
timeouts, and an owned uvicorn process in a background thread. No model weights,
network services, production database, or production roster are touched.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.atlas_state import AtlasState  # noqa: E402
from src.config import load_config  # noqa: E402
from src.embedder import HashEmbedder  # noqa: E402
from src.models import Opinion  # noqa: E402
from src.protocol import PROTOCOL_VERSION  # noqa: E402
from src.roster import Roster  # noqa: E402
from src.server import create_app  # noqa: E402
from src.store import Store  # noqa: E402
from src.textnorm import text_hash  # noqa: E402
from tests.auth_helpers import code_for, write_access_codes_for  # noqa: E402


CODE = "classroom-load-code"


@dataclass
class BurstTiming:
    ack_p50_ms: float
    ack_p95_ms: float
    broadcast_max_ms: float
    broadcast_p95_ms: float
    recompute_s: float


@dataclass
class Timing:
    startup_s: float
    heartbeat_p95_ms: float
    first: BurstTiming
    second: BurstTiming


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else float(raw)


def _percentile_ms(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx] * 1000.0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _roster_csv(participants: int) -> str:
    rows = ["id,display_name,role", "target1,Target 1,student"]
    rows.extend(f"writer{i:02d},Writer {i:02d},observer" for i in range(participants))
    return "\n".join(rows) + "\n"


def _seed_database(db_path: Path, participants: int, count: int) -> None:
    store = Store(str(db_path))
    store.migrate()
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    try:
        for i in range(count):
            stamp = (start + timedelta(milliseconds=i)).isoformat(
                timespec="milliseconds"
            ).replace("+00:00", "Z")
            op = Opinion(
                id=f"seed_{i:04d}",
                reviewer_id=f"writer{i % participants:02d}",
                target_id="target1",
                text=f"Seed opinion {i:04d}: classroom baseline text for projection.",
                source="ai",
                week=(i % 4) + 1,
                timestamp=stamp,
            )
            assert store.insert_opinion(op, text_hash(op.text))
    finally:
        store.close()


class LiveServer:
    def __init__(self, *, roster: Path, db: Path):
        import uvicorn

        self.port = _free_port()
        cfg = load_config(
            {
                "ATLAS_ADMIN_CODE": CODE,
                "ATLAS_ROSTER": str(roster),
                "ATLAS_ACCESS_CODES": str(roster.parent / "access-codes.json"),
                "ATLAS_DB": str(db),
                "ATLAS_UNSAFE_FAKE_EMBEDDER": "1",
                "ATLAS_ALLOW_INSECURE_HTTP": "1",
                "ATLAS_AUTO_WEEK": "0",
            }
        )
        self.server = uvicorn.Server(
            uvicorn.Config(
                create_app(cfg),
                host="127.0.0.1",
                port=self.port,
                log_level="warning",
                access_log=False,
                lifespan="on",
            )
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def http(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def ws(self) -> str:
        return f"ws://127.0.0.1:{self.port}"

    def start(self, timeout_s: float) -> float:
        import httpx

        started = time.monotonic()
        self.thread.start()
        deadline = started + timeout_s
        last_error = None
        while time.monotonic() < deadline:
            try:
                with httpx.Client(timeout=2.0) as client:
                    body = client.get(f"{self.http}/healthz").json()
                if body.get("ok") is True:
                    return time.monotonic() - started
            except Exception as exc:  # noqa: BLE001
                last_error = exc
            time.sleep(0.1)
        raise AssertionError(f"server did not become healthy: {last_error!r}")

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)
        assert not self.thread.is_alive(), "owned uvicorn thread did not stop"


async def _json_recv(ws, timeout_s: float) -> dict[str, Any]:
    raw = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
    return json.loads(raw)


async def _recv_until(ws, wanted: set[str], timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while True:
        remaining = deadline - time.monotonic()
        assert remaining > 0, f"timed out waiting for {sorted(wanted)}"
        frame = await _json_recv(ws, remaining)
        if frame.get("t") in wanted:
            return frame


def _point_ids(frame: dict[str, Any]) -> list[str]:
    points = frame.get("points")
    if points is None:
        points = frame.get("added") or []
    return [p["id"] for p in points]


async def _connect_participant(base_ws: str, writer_id: str, timeout_s: float):
    ws = await websockets.connect(
        f"{base_ws}/ws",
        open_timeout=timeout_s,
        close_timeout=2,
        max_size=8 * 1024 * 1024,
    )
    await ws.send(json.dumps({
        "t": "hello",
        "protocol": PROTOCOL_VERSION,
        "id": writer_id,
        "code": code_for(writer_id),
    }))
    hello = await _json_recv(ws, timeout_s)
    assert hello["t"] == "hello_ok"
    snapshot = await _json_recv(ws, timeout_s)
    assert snapshot["t"] == "snapshot"
    return ws, snapshot


async def _connect_admin(base_ws: str, timeout_s: float):
    ws = await websockets.connect(
        f"{base_ws}/ws/admin",
        open_timeout=timeout_s,
        close_timeout=2,
        max_size=8 * 1024 * 1024,
    )
    await ws.send(json.dumps({"t": "hello", "protocol": PROTOCOL_VERSION, "code": CODE}))
    hello = await _json_recv(ws, timeout_s)
    assert hello["t"] == "hello_ok"
    snapshot = await _json_recv(ws, timeout_s)
    assert snapshot["t"] == "snapshot"
    return ws, snapshot


async def _submit_and_wait_ack(ws, nonce: str, text: str, timeout_s: float) -> tuple[str, float]:
    started = time.monotonic()
    await ws.send(
        json.dumps(
            {
                "t": "submit",
                "nonce": nonce,
                "target_id": "target1",
                "text": text,
                "source": "human",
                "week": 2,
            }
        )
    )
    while True:
        frame = await _json_recv(ws, timeout_s)
        if frame.get("t") == "ack" and frame.get("nonce") == nonce:
            return frame["id"], time.monotonic() - started
        assert frame.get("t") != "error", frame


async def _recv_broadcast_latency(ws, started: float, timeout_s: float):
    frame = await _recv_until(ws, {"delta", "snapshot"}, timeout_s)
    return frame, time.monotonic() - started


async def _ping_during_recompute(ws, stop: asyncio.Event, timeout_s: float) -> list[float]:
    latencies: list[float] = []
    seq = 0
    while not stop.is_set():
        started = time.monotonic()
        await ws.send(json.dumps({"t": "ping", "nonce": f"hb-{seq}"}))
        frame = await _recv_until(ws, {"pong"}, timeout_s)
        assert frame["t"] == "pong"
        latencies.append(time.monotonic() - started)
        seq += 1
        await asyncio.sleep(0.1)
    return latencies


async def _run_burst(
    *,
    participant_sockets: list[Any],
    admin_ws: Any,
    burst_name: str,
    expected_rev: int,
    timeout_s: float,
) -> tuple[BurstTiming, list[str]]:
    ack_results = await asyncio.gather(
        *[
            _submit_and_wait_ack(
                ws,
                nonce=f"{burst_name}-{i}",
                text=f"Live classroom {burst_name} opinion {i:02d}",
                timeout_s=timeout_s,
            )
            for i, ws in enumerate(participant_sockets)
        ]
    )
    acked_ids = [oid for oid, _ in ack_results]
    ack_latencies = [latency for _, latency in ack_results]
    assert len(acked_ids) == len(participant_sockets)
    assert len(set(acked_ids)) == len(participant_sockets)

    broadcast_started = time.monotonic()
    frame_results = await asyncio.gather(
        *[
            _recv_broadcast_latency(ws, broadcast_started, timeout_s)
            for ws in participant_sockets
        ]
    )
    participant_frames = [frame for frame, _ in frame_results]
    broadcast_latencies = [latency for _, latency in frame_results]

    for frame in participant_frames:
        assert frame["rev"] == expected_rev
        ids = set(_point_ids(frame))
        if frame["t"] == "delta":
            assert set(acked_ids) <= ids
            assert len(ids) == len(participant_sockets)
        else:
            assert len(ids) == expected_rev
            assert set(acked_ids) <= ids

    admin_update = await _recv_until(admin_ws, {"delta", "snapshot"}, timeout_s)
    assert admin_update["rev"] == expected_rev
    assert set(acked_ids) <= set(_point_ids(admin_update))

    timing = BurstTiming(
        ack_p50_ms=_percentile_ms(ack_latencies, 50),
        ack_p95_ms=_percentile_ms(ack_latencies, 95),
        broadcast_max_ms=max(broadcast_latencies) * 1000.0 if broadcast_latencies else 0.0,
        broadcast_p95_ms=_percentile_ms(broadcast_latencies, 95),
        recompute_s=max(broadcast_latencies) if broadcast_latencies else 0.0,
    )
    return timing, acked_ids


async def _run_classroom_scenario(
    server: LiveServer, participants: int, seed_count: int, timeout_s: float
) -> Timing:
    import httpx

    admin_ws, admin_snapshot = await _connect_admin(server.ws, timeout_s)
    heartbeat_ws = None
    participant_sockets = []
    try:
        assert admin_snapshot["rev"] == seed_count
        assert len(admin_snapshot["points"]) == seed_count
        assert len(set(_point_ids(admin_snapshot))) == seed_count

        for i in range(participants):
            ws, snapshot = await _connect_participant(server.ws, f"writer{i:02d}", timeout_s)
            assert snapshot["rev"] == seed_count
            assert len(snapshot["points"]) == seed_count
            assert len(set(_point_ids(snapshot))) == seed_count
            participant_sockets.append(ws)

        heartbeat_ws, _ = await _connect_participant(server.ws, "writer00", timeout_s)
        stop_heartbeat = asyncio.Event()
        heartbeat_task = asyncio.create_task(
            _ping_during_recompute(heartbeat_ws, stop_heartbeat, timeout_s)
        )

        first, first_ids = await _run_burst(
            participant_sockets=participant_sockets,
            admin_ws=admin_ws,
            burst_name="burst1",
            expected_rev=seed_count + participants,
            timeout_s=timeout_s,
        )
        second, second_ids = await _run_burst(
            participant_sockets=participant_sockets,
            admin_ws=admin_ws,
            burst_name="burst2",
            expected_rev=seed_count + (participants * 2),
            timeout_s=timeout_s,
        )

        stop_heartbeat.set()
        heartbeat_latencies = await heartbeat_task

        all_acked_ids = first_ids + second_ids
        expected_rev = seed_count + (participants * 2)

        deadline = time.monotonic() + timeout_s
        health = {}
        while time.monotonic() < deadline:
            with httpx.Client(timeout=2.0) as client:
                health = client.get(f"{server.http}/healthz").json()
            if health["opinions"] == expected_rev and health["layout_rev"] >= 2:
                break
            await asyncio.sleep(0.1)
        assert health["opinions"] == expected_rev, health
        assert health["last_error"] is None, health

        reconnect_ws, reconnect_snapshot = await _connect_participant(
            server.ws, "writer00", timeout_s
        )
        try:
            reconnect_ids = _point_ids(reconnect_snapshot)
            assert reconnect_snapshot["rev"] == expected_rev
            assert len(reconnect_ids) == expected_rev
            assert len(set(reconnect_ids)) == expected_rev
            assert set(all_acked_ids) <= set(reconnect_ids)
        finally:
            await reconnect_ws.close()

        with httpx.Client(timeout=5.0) as client:
            exported = client.post(f"{server.http}/api/export.csv", json={"code": CODE})
        assert exported.status_code == 200
        exported_text = exported.text
        assert exported_text.count("\n") == expected_rev + 1
        for oid in all_acked_ids:
            assert oid in exported_text

        return Timing(
            startup_s=0.0,
            heartbeat_p95_ms=_percentile_ms(heartbeat_latencies, 95),
            first=first,
            second=second,
        )
    finally:
        if heartbeat_ws is not None:
            with suppress(Exception):
                await heartbeat_ws.close()
        with suppress(Exception):
            await admin_ws.close()
        for ws in participant_sockets:
            with suppress(Exception):
                await ws.close()


def test_live_classroom_load_does_not_drop_or_duplicate_opinions(tmp_path):
    participants = _env_int("ATLAS_CLASSROOM_PARTICIPANTS", 30)
    seed_count = _env_int("ATLAS_CLASSROOM_SEED", 1000)
    timeout_s = _env_float("ATLAS_CLASSROOM_TIMEOUT", 90.0)

    assert participants > 0
    assert seed_count >= 80, "default classroom QA should exercise the UMAP path"

    roster = tmp_path / "roster.csv"
    db = tmp_path / "atlas.db"
    roster.write_text(_roster_csv(participants), encoding="utf-8")
    write_access_codes_for(
        tmp_path,
        ["target1"] + [f"writer{i:02d}" for i in range(participants)],
    )
    _seed_database(db, participants, seed_count)

    server = LiveServer(roster=roster, db=db)
    startup_s = server.start(timeout_s)
    try:
        timing = asyncio.run(
            asyncio.wait_for(
                _run_classroom_scenario(server, participants, seed_count, timeout_s),
                timeout=timeout_s,
            )
        )
        timing.startup_s = startup_s
        print(
            "CLASSROOM_LOAD "
            f"participants={participants} seed={seed_count} "
            f"startup_s={timing.startup_s:.3f} "
            f"first_ack_p50_ms={timing.first.ack_p50_ms:.1f} "
            f"first_ack_p95_ms={timing.first.ack_p95_ms:.1f} "
            f"first_broadcast_p95_ms={timing.first.broadcast_p95_ms:.1f} "
            f"first_broadcast_max_ms={timing.first.broadcast_max_ms:.1f} "
            f"first_recompute_s={timing.first.recompute_s:.3f} "
            f"second_ack_p50_ms={timing.second.ack_p50_ms:.1f} "
            f"second_ack_p95_ms={timing.second.ack_p95_ms:.1f} "
            f"second_broadcast_p95_ms={timing.second.broadcast_p95_ms:.1f} "
            f"second_broadcast_max_ms={timing.second.broadcast_max_ms:.1f} "
            f"second_recompute_s={timing.second.recompute_s:.3f} "
            f"heartbeat_p95_ms={timing.heartbeat_p95_ms:.1f}"
        )
        assert timing.first.ack_p95_ms <= 1000.0
        assert timing.second.ack_p95_ms <= 1000.0
        assert timing.first.broadcast_max_ms <= 5000.0
        assert timing.second.broadcast_max_ms <= 5000.0
        assert timing.heartbeat_p95_ms <= 1000.0
    finally:
        server.stop()


def test_threshold_transition_from_79_to_80_keeps_all_points(tmp_path):
    roster = tmp_path / "roster.csv"
    db = tmp_path / "atlas.db"
    roster.write_text(_roster_csv(1), encoding="utf-8")
    write_access_codes_for(tmp_path, ["target1", "writer00"])
    _seed_database(db, participants=1, count=79)

    cfg = load_config(
        {
            "ATLAS_ADMIN_CODE": CODE,
            "ATLAS_ROSTER": str(roster),
            "ATLAS_ACCESS_CODES": str(tmp_path / "access-codes.json"),
            "ATLAS_DB": str(db),
            "ATLAS_UNSAFE_FAKE_EMBEDDER": "1",
            "ATLAS_SKIP_WARMUP": "1",
            "ATLAS_ALLOW_INSECURE_HTTP": "1",
            "ATLAS_AUTO_WEEK": "0",
        }
    )
    store = Store(str(db))
    state = AtlasState(cfg, store, Roster.from_path(str(roster)), HashEmbedder(dim=768))
    try:
        state.load()
        assert state.rev == 79
        assert state.layout._reducer is None
        assert len(state.server_coords) == 79

        op = Opinion(
            id="threshold_0080",
            reviewer_id="writer00",
            target_id="target1",
            text="The eightieth opinion should cross into the UMAP layout path.",
            source="human",
            week=2,
            timestamp="2026-01-01T00:00:01.080Z",
        )
        assert store.insert_opinion(op, text_hash(op.text))
        state.add(op)
        result = state.recompute(state.snapshot_input())
        assert result.error is None
        state.commit(result)

        assert state.rev == 80
        assert len(state.server_coords) == 80
        assert len(set(state.server_coords)) == 80
        assert "threshold_0080" in state.server_coords
        assert state.layout._reducer is not None
    finally:
        store.close()
