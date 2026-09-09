"""Exact original offsets with adjacent semantic context, never output overlap."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from src.submissions import FeedbackSpan, SplitResult, validate_partition


@dataclass(frozen=True)
class SegmentationPolicy:
    algorithm: str = "adjacent-context-v1"
    threshold: float = 0.7
    context_candidates: int = 1
    max_units: int = 64
    prompt: str = "similarity"
    classroom_enabled: bool = False
    annotation_status: str = "synthetic-only-human-review-required"

    def __post_init__(self):
        if self.algorithm != "adjacent-context-v1" or not 0 <= self.threshold <= 1:
            raise ValueError("unsupported segmentation policy")
        if self.context_candidates not in (1, 2) or not 1 <= self.max_units <= 64:
            raise ValueError("invalid segmentation limits")
        if self.prompt not in ("similarity", "clustering"):
            raise ValueError("unsupported segmentation prompt")

    def fingerprint(self, runtime):
        key = runtime.similarity_key if self.prompt == "similarity" else runtime.unit_cache_key
        payload = json.dumps({**asdict(self), "model": key}, sort_keys=True, separators=(",", ":"))
        return f"{self.algorithm}:{hashlib.sha256(payload.encode()).hexdigest()}"


def load_policy(path=None):
    path = Path(path) if path else Path(__file__).resolve().parents[1] / "config/segmentation.json"
    return SegmentationPolicy(**json.loads(path.read_text(encoding="utf-8")))


def validate_spans(raw, spans):
    validate_partition(raw, spans)


# Korean final forms supply candidates even when writers omit periods.
_SENTENCE = re.compile(r"(?:[.!?。！？]+[\"'’”\)\]]*|(?:습니다|입니다|해요|어요|아요|네요|세요|죠))\s+")
_LINE = re.compile(r"\n[ \t]*\n+|\n(?=[ \t]*(?:[-*•]|\d+[.)]|#{1,6}\s))")
_HEADING = re.compile(r"^(?:#{1,6}\s+.+|[^.!?。！？\n]{1,50}:?)$")


def candidate_spans(raw):
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("empty feedback")
    boundaries = {match.end() for pattern in (_SENTENCE, _LINE) for match in pattern.finditer(raw)}
    for match in re.finditer(r"[^\n]+\n+", raw):
        line = match.group().strip()
        if _HEADING.fullmatch(line) and not _SENTENCE.search(line + " "):
            boundaries.discard(match.end())
    cuts = [0]
    for boundary in sorted(boundaries):
        if boundary < len(raw) and raw[cuts[-1]:boundary].strip() and raw[boundary:].strip():
            cuts.append(boundary)
    cuts.append(len(raw))
    return tuple(FeedbackSpan(a,b) for a,b in zip(cuts,cuts[1:]))


def boundary_scores(raw, runtime, policy):
    candidates = candidate_spans(raw)
    if len(candidates) == 1:
        return candidates, []
    width = policy.context_candidates
    pairs = []
    for i in range(1,len(candidates)):
        left = raw[candidates[max(0,i-width)].start:candidates[i-1].end]
        right = raw[candidates[i].start:candidates[min(len(candidates),i+width)-1].end]
        pairs.extend((left,right))
    encode = runtime.encode_similarity if policy.prompt == "similarity" else runtime.encode_units
    unique = list(dict.fromkeys(pairs))
    indexed = dict(zip(unique,encode(unique)))
    scores = []
    for left,right in zip(pairs[::2],pairs[1::2]):
        a,b = indexed[left],indexed[right]
        denominator = np.linalg.norm(a)*np.linalg.norm(b)
        if denominator <= 1e-12:
            raise ValueError("zero similarity vector")
        score = float(np.dot(a,b)/denominator)
        if not np.isfinite(score):
            raise ValueError("nonfinite similarity")
        scores.append(score)
    return candidates,scores


def split_feedback(raw, runtime, policy=None):
    policy = policy or load_policy()
    candidates,scores = boundary_scores(raw,runtime,policy)
    cuts = [0] + [span.end for span,score in zip(candidates,scores) if score < policy.threshold] + [len(raw)]
    if len(cuts)-1 > policy.max_units:
        raise ValueError("automatic unit limit exceeded")
    spans = tuple(FeedbackSpan(a,b) for a,b in zip(cuts,cuts[1:]))
    validate_spans(raw,spans)
    return SplitResult(spans, policy.fingerprint(runtime))
