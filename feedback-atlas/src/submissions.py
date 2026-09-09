"""Original submissions and immutable, contiguous feedback-unit contracts."""
from dataclasses import dataclass
from typing import Literal

Source = Literal['ai', 'human']
Status = Literal['queued', 'splitting', 'embedding', 'ready', 'failed', 'withdrawn']


class SubmissionError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class FeedbackSpan:
    start: int
    end: int


@dataclass(frozen=True)
class SplitResult:
    spans: tuple[FeedbackSpan, ...]
    algorithm_key: str


@dataclass(frozen=True)
class SubmissionRequest:
    reviewer_id: str
    target_id: str
    week: int
    source: Source
    raw_text: str
    nonce: str
    owner_capability: str
    context_revision: int
    metadata_confirmed: bool = False


@dataclass(frozen=True)
class SubmissionReceipt:
    submission_id: str
    nonce: str
    revision: int
    state: Status


def validate_partition(raw: str, spans: tuple[FeedbackSpan, ...]) -> None:
    """Offsets are Python/Unicode code points, not browser UTF-16 offsets."""
    if not isinstance(raw, str) or not raw.strip() or not spans or len(spans) > 64:
        raise SubmissionError('INVALID_SEGMENTS')
    cursor = 0
    for span in spans:
        if (type(span.start) is not int or type(span.end) is not int
                or span.start != cursor or not span.start < span.end <= len(raw)
                or not raw[span.start:span.end].strip()):
            raise SubmissionError('INVALID_SEGMENTS')
        cursor = span.end
    if cursor != len(raw):
        raise SubmissionError('INVALID_SEGMENTS')


def visible_counts(points: list[dict]) -> dict[str, int]:
    return {'units': len(points),
            'submissions': len({p.get('submission_id') or p['id'] for p in points})}
