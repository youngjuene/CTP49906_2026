"""Caption display/cache regressions; these helpers do not require a GPU."""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class SplitKoreanTokenizer:
    """Actual UTF-8 boundaries: ids 2–4 encode 한, ids 6–7 encode 글."""

    eos_token_id = 8
    all_special_ids = [8]
    pieces = {2: b"\xed", 3: b"\x95", 4: b"\x9c", 5: b" ",
              6: b"\xea\xb8", 7: b"\x80", 8: b"<|im_end|>"}

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return b"".join(
            self.pieces[i] for i in ids
            if not (skip_special_tokens and i in self.all_special_ids)
        ).decode("utf-8", errors="replace")


def test_cumulative_display_keeps_korean_text_and_one_slot_per_scored_id():
    from src.classroom_display import decode_caption_tokens

    text, tokens, kinds = decode_caption_tokens(SplitKoreanTokenizer(), [2, 3, 4, 5, 6, 7, 8])
    assert text == "한 글"
    assert tokens == ["", "", "한", " ", "", "글", ""]
    assert "".join(tokens) == text
    assert kinds == ["byte_continuation", "byte_continuation", "text", "text",
                     "byte_continuation", "text", "special"]


def test_display_does_not_delete_an_actual_replacement_character():
    from src.classroom_display import decode_caption_tokens

    tokenizer = SplitKoreanTokenizer()
    tokenizer.pieces = {2: "�".encode("utf-8"), 3: b"!"}
    text, tokens, _ = decode_caption_tokens(tokenizer, [2, 3])
    assert text == "�!"
    assert tokens == ["�", "!"]


def test_selected_drop_share_uses_negative_grouped_mass():
    from src.classroom_display import selected_drop_share

    # A signed net total would report 2000%; positive mass must not cancel drops.
    assert selected_drop_share(["a", " b"], [-2, 1.9], 0.5) == 100.0
    assert selected_drop_share(["a", " b"], [-2, 3], 0.5) == 100.0
    assert selected_drop_share(["a", " b", " c"], [-2, -1, 3], 1) == pytest.approx(200 / 3)
    # Group first: a -3/+2 subword pair is only a -1 group drop.
    assert selected_drop_share(["a", "b", " c"], [-3, 2, -1], 1.5) == 0.0
    for values in ([], [0, 0], [1, 2]):
        assert selected_drop_share(["a", " b"][:len(values)], values, 0) == 0.0


def test_display_rejects_misaligned_or_nonfinite_evidence():
    from src.classroom_display import selected_drop_share

    for values in ([1], [math.nan, 0], [math.inf, 0]):
        with pytest.raises(ValueError):
            selected_drop_share(["a", " b"], values, 0)


def test_caption_cache_keys_include_content_cap_and_model(tmp_path):
    from src.classroom_display import caption_cache_key

    path = tmp_path / "clip.mp4"
    path.write_bytes(b"aaaa")
    settings = dict(clip_path=path, nframes=8, prompt="듣고 설명", max_new_tokens=8,
                    model_identity={"model": "qwen", "revision": "abc"})
    first = caption_cache_key(**settings)
    assert first == caption_cache_key(**settings)
    assert first != caption_cache_key(**(settings | {"max_new_tokens": 32}))
    assert first != caption_cache_key(**(settings | {"nframes": 4}))
    assert first != caption_cache_key(**(settings | {"prompt": "다른 질문"}))
    assert first != caption_cache_key(**(settings | {"model_identity": {"model": "qwen", "revision": "def"}}))
    path.write_bytes(b"bbbb")  # Same filename and length; content still differs.
    assert first != caption_cache_key(**settings)


def _score_korean(caption_ids, cap=32):
    import torch
    from src.teacher_forcing import teacher_forced_delta

    class Thinker:
        generation_config = SimpleNamespace(eos_token_id=[8])

        def generate(self, input_ids, **kwargs):
            return torch.cat([input_ids, torch.tensor([caption_ids])], dim=1)

        def __call__(self, input_ids, **kwargs):
            return SimpleNamespace(logits=torch.zeros(1, input_ids.shape[1], 9))

    return teacher_forced_delta(
        SimpleNamespace(thinker=Thinker()), SimpleNamespace(tokenizer=SplitKoreanTokenizer()),
        {"input_ids": torch.tensor([[0, 1]])}, ["query_text", "audio"], [],
        max_new_tokens=cap,
    )


def test_teacher_forcing_keeps_caption_intact():
    result = _score_korean([2, 3, 4, 5, 6, 7, 8])
    assert "".join(result["caption_tokens"]) == "한 글"
    assert result["caption_text"] == "한 글"
    assert len(result["caption_tokens"]) == len(result["delta"]) == 7
    assert result["generation_truncated"] is False
    assert result["generation_end_reason"] == "eos"


def test_teacher_forcing_flags_token_cap_but_not_eos_at_cap():
    capped = _score_korean([2, 3, 4], cap=3)
    assert capped["generation_truncated"] is True
    assert capped["generation_end_reason"] == "max_new_tokens"
    assert capped["generation_token_count"] == 3
    assert _score_korean([2, 3, 4, 8], cap=4)["generation_truncated"] is False


def test_grouping_accumulates_continuations_with_completed_text():
    from src.teacher_forcing import group_tokens_into_words, render_delta_strip

    # A blank byte slot before the next word must not add its score to "a".
    tokens, delta = ["a", "", " 한"], [0.1, -2, -1]
    words = group_tokens_into_words(tokens, delta)
    assert [word[0] for word in words] == ["a", " 한"]
    assert [word[1] for word in words] == [0.1, -3]
    raw = render_delta_strip(tokens, delta, word_level=False)
    assert "byte continuation" in raw
    assert "Δ=-2.00 nats" in raw


def test_whitespace_score_is_not_labeled_as_a_byte_continuation():
    from src.teacher_forcing import render_delta_strip

    html = render_delta_strip([" ", "한"], [-0.2, -0.3])
    assert "whitespace: -0.20" in html
    assert "byte continuation" not in html


@pytest.mark.parametrize("tokens,kinds,expected_text", [
    (["first", "", " second"], ["text", "special", "text"], ["first", "⟨special⟩", " second"]),
    (["first", ""], ["text", "special"], ["first", "⟨special⟩"]),
    (["first", "", "second"], ["text", "special", "text"], ["first", "⟨special⟩", "second"]),
])
def test_special_scores_do_not_color_adjacent_text(tokens, kinds, expected_text):
    from src.teacher_forcing import group_tokens_into_words, render_delta_strip

    delta = [0, -10] + ([0] if len(tokens) == 3 else [])
    groups = group_tokens_into_words(tokens, delta, token_kinds=kinds)
    assert [group[0] for group in groups] == expected_text
    assert [group[1] for group in groups] == delta
    for word_level in (True, False):
        html = render_delta_strip(tokens, delta, token_kinds=kinds, word_level=word_level,
                                  highlight_below=1, vmax=10)
        highlighted = [span for span in html.split("<span") if "outline:2px" in span]
        assert len(highlighted) == 1
        assert "⟨special⟩" in highlighted[0]
        assert "first" not in highlighted[0] and "second" not in highlighted[0]


def test_byte_scores_still_join_completed_text_across_interspersed_special():
    from src.classroom_display import decode_caption_tokens, group_tokens_into_words

    text, tokens, kinds = decode_caption_tokens(SplitKoreanTokenizer(), [2, 8, 3, 4])
    assert text == "한"
    groups = group_tokens_into_words(tokens, [-1, -10, -2, -3], token_kinds=kinds)
    assert [(group[0], group[1]) for group in groups] == [("⟨special⟩", -10), ("한", -6)]
    assert sum(len(group[2]) for group in groups) == 4


def test_special_only_caption_stays_visible_and_uncompleted_slots_stay_separate():
    from src.teacher_forcing import group_tokens_into_words, render_delta_strip

    assert "⟨special⟩" in render_delta_strip([""], [-10], token_kinds=["special"])
    groups = group_tokens_into_words(["", ""], [-2, -10],
                                    token_kinds=["byte_continuation", "special"])
    assert len(groups) == 2
    assert [group[1] for group in groups] == [-10, -2]


def test_threshold_and_drop_share_keep_special_mass_separate():
    from src.teacher_forcing import selected_drop_share, threshold_slider_params

    tokens, delta, kinds = ["first", ""], [9, -10], ["text", "special"]
    assert selected_drop_share(tokens, delta, 5, token_kinds=kinds) == 100
    params = threshold_slider_params(tokens, delta, token_kinds=kinds)
    assert params["max_value"] >= 10
    assert 5 < params["amount"] < 10


def test_misaligned_token_kind_metadata_is_rejected():
    from src.teacher_forcing import group_tokens_into_words

    with pytest.raises(ValueError, match="token_kinds"):
        group_tokens_into_words(["a", ""], [0, -1], token_kinds=["text"])


def test_baseline_full_logits_are_released_before_knockout(monkeypatch):
    import weakref
    from contextlib import nullcontext
    import torch
    from src import teacher_forcing

    class Thinker:
        previous_logits = None

        def __call__(self, input_ids, **kwargs):
            if self.previous_logits is not None:
                assert self.previous_logits() is None, "baseline full logits retained during KO"
            logits = torch.zeros(1, input_ids.shape[1], 9)
            self.previous_logits = weakref.ref(logits)
            return SimpleNamespace(logits=logits)

    monkeypatch.setattr(teacher_forcing, "block_attention", lambda *a, **kw: nullcontext())
    result = teacher_forcing.teacher_forced_delta(
        SimpleNamespace(thinker=Thinker()), SimpleNamespace(tokenizer=SplitKoreanTokenizer()),
        {"input_ids": torch.tensor([[0, 1]])}, ["query_text", "audio"],
        [("answer", "audio", 0, 1)], cached_caption_ids=torch.tensor([[2, 3, 4]]),
    )
    assert result["delta_total"] == 0
