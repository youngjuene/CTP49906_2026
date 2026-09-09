"""Class-week schedule and persistence.

The schedule is metadata, not opinion data. Changing it updates future intake
classification only; stored opinions keep the week they were originally assigned.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone


SCHEDULE_META_KEY = "class_schedule_v1"
CLASS_TZ_NAME = "Asia/Seoul"
CLASS_TZ = timezone(timedelta(hours=9), CLASS_TZ_NAME)
DEFAULT_STARTS = ("2026-10-15", "2026-10-22", "2026-11-05", "2026-11-12")
PERIOD_DAYS = 7
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class ScheduleConflict(Exception):
    """Raised when an admin updates from a stale schedule revision."""


@dataclass(frozen=True)
class WeekPeriod:
    week: int
    start: date

    @property
    def start_at(self) -> datetime:
        return datetime.combine(self.start, time.min, CLASS_TZ)

    @property
    def end_at(self) -> datetime:
        return self.start_at + timedelta(days=PERIOD_DAYS)

    @property
    def inclusive_end(self) -> date:
        return self.start + timedelta(days=PERIOD_DAYS - 1)


@dataclass(frozen=True)
class WeekSchedule:
    starts: tuple[str, str, str, str]
    revision: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "starts", validate_starts(self.starts))
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise ValueError("schedule revision must be an integer")
        if self.revision < 0:
            raise ValueError("schedule revision must not be negative")

    @property
    def periods(self) -> tuple[WeekPeriod, ...]:
        return tuple(
            WeekPeriod(week=i + 1, start=date.fromisoformat(value))
            for i, value in enumerate(self.starts)
        )

    def week_for(self, value: str | datetime) -> int | None:
        when = _aware_datetime(value).astimezone(CLASS_TZ)
        for period in self.periods:
            if period.start_at <= when < period.end_at:
                return period.week
        return None


def validate_starts(values) -> tuple[str, str, str, str]:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError("schedule must contain exactly four start dates")

    starts: list[str] = []
    parsed: list[date] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, str):
            raise ValueError("schedule starts must be YYYY-MM-DD strings")
        text = value.strip()
        if text != value or not _DATE_RE.fullmatch(text):
            raise ValueError("schedule starts must be YYYY-MM-DD strings")
        try:
            parsed_date = date.fromisoformat(text)
            parsed_date + timedelta(days=PERIOD_DAYS)
        except (ValueError, OverflowError):
            raise ValueError("schedule starts must be real calendar dates") from None
        starts.append(text)
        parsed.append(parsed_date)

    for before, after in zip(parsed, parsed[1:]):
        if after < before + timedelta(days=PERIOD_DAYS):
            raise ValueError("schedule weeks must be chronological and non-overlapping")
    return tuple(starts)  # type: ignore[return-value]


def default_schedule() -> WeekSchedule:
    return WeekSchedule(DEFAULT_STARTS, revision=0)


def load_schedule(store) -> WeekSchedule:
    return _load_schedule_raw(store.get_meta(SCHEDULE_META_KEY))


def _load_schedule_raw(raw: str | None) -> WeekSchedule:
    if raw is None:
        return default_schedule()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError("stored class schedule is not valid JSON") from None
    if not isinstance(data, dict):
        raise ValueError("stored class schedule must be an object")
    return WeekSchedule(data.get("starts"), revision=data.get("revision", 0))


def update_schedule(store, starts, expected_revision: int) -> WeekSchedule:
    if isinstance(expected_revision, bool) or not isinstance(expected_revision, int):
        raise ValueError("expected_revision must be an integer")

    expected_raw = store.get_meta(SCHEDULE_META_KEY)
    current = _load_schedule_raw(expected_raw)
    if current.revision != expected_revision:
        raise ScheduleConflict("schedule revision is stale")

    updated = WeekSchedule(validate_starts(starts), revision=current.revision + 1)
    value = _dump_schedule(updated)
    if not store.compare_and_set_meta(SCHEDULE_META_KEY, expected_raw, value):
        raise ScheduleConflict("schedule revision changed")
    return updated


def schedule_info(atlas, now: str | datetime) -> dict:
    schedule = getattr(atlas, "schedule", None) or default_schedule()
    cfg = atlas.cfg
    server_time = _aware_datetime(now)
    demo = bool(getattr(cfg, "is_demo", False))
    automatic = bool(getattr(cfg, "auto_week", False))

    if demo:
        current_week = getattr(cfg, "default_week", 1)
        accepting = True
        automatic = False
    elif automatic:
        current_week = schedule.week_for(server_time)
        accepting = current_week is not None
    else:
        current_week = getattr(cfg, "default_week", 1)
        accepting = True

    return {
        "timezone": CLASS_TZ_NAME,
        "server_time": _format_utc(server_time),
        "revision": schedule.revision,
        "periods": [
            {
                "week": period.week,
                "start": period.start.isoformat(),
                "end": period.inclusive_end.isoformat(),
            }
            for period in schedule.periods
        ],
        "current_week": current_week,
        "accepting": accepting,
        "demo": demo,
        "automatic": automatic,
        "applied_to": "future_submissions",
    }


def _dump_schedule(schedule: WeekSchedule) -> str:
    return json.dumps(
        {"revision": schedule.revision, "starts": list(schedule.starts)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _aware_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        when = value
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            when = datetime.fromisoformat(text)
        except ValueError:
            raise ValueError("timestamp must be an aware ISO datetime") from None
    else:
        raise ValueError("timestamp must be an aware ISO datetime")
    if when.tzinfo is None or when.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return when.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z")
