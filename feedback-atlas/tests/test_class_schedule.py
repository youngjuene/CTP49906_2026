import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.class_schedule import (  # noqa: E402
    DEFAULT_STARTS,
    SCHEDULE_META_KEY,
    ScheduleConflict,
    WeekSchedule,
    load_schedule,
    schedule_info,
    update_schedule,
    validate_starts,
)
from src.schedule_routes import broadcast_schedule  # noqa: E402


class FakeStore:
    def __init__(self):
        self.meta = {}
        self.force_conflict = False

    def get_meta(self, key):
        return self.meta.get(key)

    def compare_and_set_meta(self, key, expected_raw, value):
        if self.force_conflict or self.meta.get(key) != expected_raw:
            return False
        self.meta[key] = value
        return True


@dataclass
class FakeConfig:
    default_week: int = 1
    auto_week: bool = True
    is_demo: bool = False


class FakeSocket:
    def __init__(self):
        self.messages = []

    async def send_json(self, message):
        self.messages.append(message)


class FakeHub:
    def __init__(self):
        self.messages = {"participant": [], "admin": []}

    async def join(self, socket, channel, *, identity=None):
        self.messages[channel] = socket.messages

    async def broadcast(self, channel, message):
        self.messages[channel].append(message)
        return 1


class FakeAtlas:
    def __init__(self, *, cfg=None, schedule=None):
        self.cfg = cfg or FakeConfig()
        self.schedule = schedule or WeekSchedule(DEFAULT_STARTS)
        self.hub = FakeHub()
        self.schedule_marker = None


def test_default_schedule_maps_kst_midnight_and_utc_edges():
    schedule = WeekSchedule(DEFAULT_STARTS)

    assert schedule.week_for("2026-10-14T15:00:00.000Z") == 1
    assert schedule.week_for("2026-10-21T14:59:59.999Z") == 1
    assert schedule.week_for("2026-10-21T15:00:00.000Z") == 2
    assert schedule.week_for(datetime(2026, 11, 11, 14, 59, 59, tzinfo=timezone.utc)) == 3
    assert schedule.week_for("2026-11-11T15:00:00.000Z") == 4
    assert schedule.week_for("2026-11-18T15:00:00.000Z") is None


def test_midterm_break_is_not_an_accepting_week():
    schedule = WeekSchedule(DEFAULT_STARTS)

    assert schedule.week_for("2026-10-28T14:59:59.999Z") == 2
    assert schedule.week_for("2026-10-28T15:00:00.000Z") is None
    assert schedule.week_for("2026-11-04T14:59:59.999Z") is None
    assert schedule.week_for("2026-11-04T15:00:00.000Z") == 3


@pytest.mark.parametrize(
    "starts",
    [
        ["2026-10-15", "2026-10-22", "2026-11-05"],
        ["2026-10-15", "2026-10-22", "2026-11-05", True],
        ["2026-10-15", "2026-10-22", "2026-11-05", "2026-11-12 "],
        ["2026-10-15", "2026-10-22", "2026-11-05", "2026-13-12"],
        ["2026-10-15", "2026-10-21", "2026-11-05", "2026-11-12"],
        ["2026-10-15", "2026-10-22", "2026-10-28", "2026-11-12"],
    ],
)
def test_invalid_and_overlapping_start_dates_are_refused(starts):
    with pytest.raises(ValueError):
        validate_starts(starts)


def test_schedule_loads_default_or_persisted_metadata():
    store = FakeStore()
    assert load_schedule(store).starts == DEFAULT_STARTS
    assert load_schedule(store).revision == 0

    store.meta[SCHEDULE_META_KEY] = json.dumps({
        "revision": 4,
        "starts": ["2026-10-16", "2026-10-23", "2026-11-06", "2026-11-13"],
    })
    schedule = load_schedule(store)
    assert schedule.revision == 4
    assert schedule.starts == ("2026-10-16", "2026-10-23", "2026-11-06", "2026-11-13")


def test_update_schedule_uses_optimistic_cas_and_preserves_existing_data():
    store = FakeStore()

    schedule = update_schedule(
        store,
        ["2026-10-16", "2026-10-23", "2026-11-06", "2026-11-13"],
        expected_revision=0,
    )

    assert schedule.revision == 1
    assert json.loads(store.meta[SCHEDULE_META_KEY]) == {
        "revision": 1,
        "starts": ["2026-10-16", "2026-10-23", "2026-11-06", "2026-11-13"],
    }

    with pytest.raises(ScheduleConflict):
        update_schedule(store, DEFAULT_STARTS, expected_revision=0)

    store.force_conflict = True
    with pytest.raises(ScheduleConflict):
        update_schedule(store, DEFAULT_STARTS, expected_revision=1)


def test_schedule_info_reflects_auto_week_demo_and_legacy_modes():
    atlas = FakeAtlas()
    info = schedule_info(atlas, "2026-10-28T15:00:00.000Z")
    assert info["timezone"] == "Asia/Seoul"
    assert info["server_time"] == "2026-10-28T15:00:00.000Z"
    assert info["applied_to"] == "future_submissions"
    assert info["periods"][0] == {"week": 1, "start": "2026-10-15", "end": "2026-10-21"}
    assert info["current_week"] is None
    assert info["accepting"] is False
    assert info["automatic"] is True

    legacy = schedule_info(
        FakeAtlas(cfg=FakeConfig(default_week=3, auto_week=False)),
        "2026-10-28T15:00:00.000Z",
    )
    assert legacy["current_week"] == 3
    assert legacy["accepting"] is True
    assert legacy["automatic"] is False

    demo = schedule_info(
        FakeAtlas(cfg=FakeConfig(default_week=1, auto_week=True, is_demo=True)),
        "2026-10-28T15:00:00.000Z",
    )
    assert demo["current_week"] == 1
    assert demo["accepting"] is True
    assert demo["demo"] is True
    assert demo["automatic"] is False


def test_broadcast_schedule_reaches_both_channels_and_sets_marker():
    async def main():
        atlas = FakeAtlas()
        participant = FakeSocket()
        admin = FakeSocket()
        await atlas.hub.join(participant, "participant", identity="writer1")
        await atlas.hub.join(admin, "admin")

        import src.schedule_routes as schedule_routes

        original = schedule_routes._channels
        schedule_routes._channels = lambda: ("participant", "admin")
        try:
            info = await broadcast_schedule(atlas, "2026-10-14T15:00:00.000Z")
        finally:
            schedule_routes._channels = original

        assert participant.messages == [{"t": "schedule", "schedule": info}]
        assert admin.messages == [{"t": "schedule", "schedule": info}]
        assert atlas.schedule_marker == (0, 1)

    asyncio.run(main())
