"""Class context is metadata; changing it never retargets a stored submission."""
from dataclasses import asdict, dataclass
import json

from src.submissions import SubmissionError


@dataclass(frozen=True)
class ClassContext:
    week: int = 1
    target_id: str | None = None
    revision: int = 0
    accepting: bool = True


def current_context(store, default_week=1):
    saved = store.get_meta('class_context')
    return ClassContext(**json.loads(saved)) if saved else ClassContext(week=default_week)


def update_context(store, roster, *, expected_revision, week, target_id, accepting, default_week=1):
    if type(week) is not int or week not in (1, 2, 3, 4):
        raise SubmissionError('BAD_WEEK')
    if type(accepting) is not bool or type(expected_revision) is not int:
        raise SubmissionError('MALFORMED')
    if target_id:
        entry = roster.resolve(target_id)
        if entry is None or entry.role != 'student':
            raise SubmissionError('UNKNOWN_TARGET')
        target_id = entry.id
    else:
        target_id = None
    with store.transaction():
        old = current_context(store, default_week)
        if old.revision != expected_revision:
            raise SubmissionError('REVISION_CONFLICT')
        new = ClassContext(week, target_id, old.revision + 1, accepting)
        store._db.execute("INSERT INTO meta VALUES('class_context',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                          (json.dumps(asdict(new)),))
        return new
