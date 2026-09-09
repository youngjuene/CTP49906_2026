"""Structural invariants; toy vectors test decisions, never semantic quality."""
import numpy as np
import pytest


def test_units_partition_the_exact_original():
    from src.submissions import FeedbackSpan
    from src.segmentation import validate_spans
    raw = "조명이 따뜻했습니다. 색 변화도 자연스러웠습니다.\n\n음악이 큽니다. 볼륨을 낮추면 좋겠습니다."
    boundary = raw.index("음악")
    spans = (FeedbackSpan(0, boundary), FeedbackSpan(boundary, len(raw)))
    validate_spans(raw, spans)
    assert "".join(raw[s.start:s.end] for s in spans) == raw


@pytest.mark.parametrize("raw,pairs", [
    ("abcdef", [(0,4),(3,6)]), ("abcdef", [(0,2),(3,6)]),
    ("abcdef", [(1,6)]), ("abcdef", [(0,5)]), ("abcdef", [(0,0),(0,6)]),
    ("a  b", [(0,1),(1,3),(3,4)]), (" ", [(0,1)]), ("", []),
    ("abc", [(0,True),(True,3)]),
])
def test_invalid_partitions_rejected(raw,pairs):
    from src.submissions import FeedbackSpan
    from src.segmentation import validate_spans
    with pytest.raises(ValueError):
        validate_spans(raw,tuple(FeedbackSpan(a,b) for a,b in pairs))


class TopicRuntime:
    similarity_key = "toy-decisions-only"
    def encode_similarity(self,texts):
        return np.asarray([[1.,0.] if "음악" not in t and "볼륨" not in t else [0.,1.] for t in texts])


@pytest.mark.parametrize("raw", [
    "조명 👩🏽‍💻 좋았습니다. 조명 👩🏽‍💻 좋았습니다.",
    "  반복입니다. 반복입니다.\n\n반복입니다.  ",
    "조명에 대한 의견\n조명이 좋았고 색이 따뜻해서 몰입할 수 있었어요",
    "# 조명\n- 조명이 좋았습니다.\n- 색도 따뜻했습니다.\n",
    "조명이 좋아요 색도 따뜻해요 좀 더 오래 보고 싶어요",
])
def test_candidate_offsets_preserve_original_and_no_heading_or_whitespace_only(raw):
    from src.segmentation import candidate_spans,validate_spans
    spans = candidate_spans(raw)
    validate_spans(raw,spans)
    assert "".join(raw[s.start:s.end] for s in spans) == raw
    assert all(s.end>s.start and raw[s.start:s.end].strip() for s in spans)
    assert not any(raw[s.start:s.end].strip() == "# 조명" for s in spans)


def test_groups_adjacent_sentences_and_keeps_recurring_topic_in_place():
    from src.segmentation import split_feedback,SegmentationPolicy
    raw = "조명이 좋았습니다. 색이 따뜻했습니다. 음악이 큽니다. 볼륨을 낮추세요. 조명이 편안했습니다."
    result = split_feedback(raw,TopicRuntime(),SegmentationPolicy(context_candidates=1,threshold=.7))
    assert [raw[s.start:s.end].strip() for s in result.spans] == [
        "조명이 좋았습니다. 색이 따뜻했습니다.", "음악이 큽니다. 볼륨을 낮추세요.", "조명이 편안했습니다."]


def test_long_single_topic_is_not_cut_for_a_model_window():
    from src.segmentation import split_feedback,SegmentationPolicy
    raw = "조명이 좋았습니다. 색이 따뜻했습니다. " * 100
    result = split_feedback(raw,TopicRuntime(),SegmentationPolicy())
    assert len(result.spans) == 1
    assert result.spans[0].end == len(raw)


def test_unit_limit_is_error_not_truncation_or_forced_merge():
    from src.segmentation import split_feedback,SegmentationPolicy
    with pytest.raises(ValueError,match="unit limit"):
        split_feedback("조명. 음악. 조명.",TopicRuntime(),SegmentationPolicy(context_candidates=1,max_units=2))


def test_korean_unpunctuated_endings_supply_candidates():
    from src.segmentation import candidate_spans
    assert len(candidate_spans("조명이 좋았어요 음악은 너무 컸습니다 볼륨을 낮추면 좋겠어요")) == 3
