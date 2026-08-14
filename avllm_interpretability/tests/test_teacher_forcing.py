"""CPU logic tests for F1 (teacher forcing). No GPU / no model weights.

Guards the pure logic behind the teacher-forced Δ log-likelihood: the `answer`
token type, the fail-loud guardrail, the shifted per-token log-likelihood, the
`knockout − baseline` sign convention (negative = hot color), and — the causal
core — that `answer → audio` masks exactly the (answer-query, audio-key)
attention cells in the prefill branch. The end-to-end model forward pass is only
exercisable on a GPU (molab smoke test); everything else is checked here.

Run:  python -m pytest avllm_interpretability/tests/test_teacher_forcing.py
  or:  python avllm_interpretability/tests/test_teacher_forcing.py
"""
import math
import sys
from pathlib import Path

import matplotlib
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> avllm_interpretability/

from src.attention_knockout_experiment import (  # noqa: E402
    TOKEN_TYPE_MAP, BlockAttentionHook, block_attention,
)
from src.teacher_forcing import (  # noqa: E402
    answer_distribution_summaries, build_answer_token_types, caption_logprobs, delta_logprobs,
    group_tokens_into_words, render_delta_strip, teacher_forced_delta,
    threshold_slider_params,
)

NEG = torch.finfo(torch.float32).min


def test_answer_is_a_distinct_type():
    assert TOKEN_TYPE_MAP.get("answer") == 5
    assert TOKEN_TYPE_MAP["answer"] != TOKEN_TYPE_MAP["generated"]


def test_unknown_type_fails_loud_not_silent_baseline():
    class _Layer:
        def __init__(self):
            self.self_attn = torch.nn.Identity()
        def parameters(self):
            return iter([torch.zeros(1)])
    class _Model:
        def __init__(self, n=2):
            self.device = torch.device("cpu")
            self.thinker = type("M", (), {})()
            self.thinker.model = type("M", (), {})()
            self.thinker.model.layers = [_Layer() for _ in range(n)]
    try:
        with block_attention(_Model(), [("bogus", "audio", 0, 1)], ["bogus"], 1):
            raise AssertionError("block_attention should have raised on an unknown type")
    except ValueError as e:
        assert "bogus" in str(e) or "Unknown token type" in str(e)


def test_answer_types_are_positional():
    pt = ["query_text", "audio", "audio"]
    assert build_answer_token_types(pt, 2) == pt + ["answer", "answer"]
    assert build_answer_token_types(pt, 0) == pt


def test_caption_logprobs_use_the_shifted_row():
    vocab, seq, prompt_len = 4, 4, 2
    logits = torch.zeros(1, seq, vocab)
    logits[0, 1] = torch.tensor([0., 0., 100., 0.])   # predicts pos2 -> id 2 (correct)
    logits[0, 2] = torch.tensor([0., 0., 0., 100.])   # predicts pos3 -> id 3 (correct)
    logits[0, 0] = torch.tensor([100., 0., 0., 0.])   # trap row (unused)
    logits[0, 3] = torch.tensor([0., 100., 0., 0.])   # trap row (unused)
    ids = torch.tensor([[0, 1, 2, 3]])
    lp = caption_logprobs(logits, ids, prompt_len)
    assert lp.shape[0] == 2
    assert bool((lp > -1.0).all())  # wrong (unshifted) indexing would give ~ -100
    uniform = caption_logprobs(torch.zeros(1, seq, vocab), ids, prompt_len)
    assert torch.allclose(uniform, torch.full((2,), -math.log(vocab)), atol=1e-5)
    assert caption_logprobs(logits, ids, seq).shape[0] == 0  # empty caption


def test_delta_sign_convention():
    d = delta_logprobs(torch.tensor([-4.0, -1.5]), torch.tensor([-1.0, -1.0]))
    assert torch.allclose(d, torch.tensor([-3.0, -0.5]))  # knockout - baseline
    assert bool((d < 0).all())  # belief dropped -> negative


def test_answer_distribution_summaries_are_teacher_forced_and_compact():
    logits = torch.zeros(1, 4, 5)
    logits[0, 1] = torch.tensor([0.0, 0.0, 4.0, 0.0, 0.0])
    logits[0, 2] = torch.tensor([0.0, 0.0, 0.0, 4.0, 0.0])
    ids = torch.tensor([[0, 1, 2, 3]])
    result = answer_distribution_summaries(logits, ids, prompt_len=2, top_k=2)

    assert result["measurement_kind"] == "teacher_forced_answer_distribution_dispersion"
    assert "uncalibrated proxy" in result["caveat"].lower()
    assert [row["target_token_id"] for row in result["positions"]] == [2, 3]
    assert all("logits" not in row["distribution"] for row in result["positions"])
    assert all(len(row["distribution"]["top_tokens"]) == 2 for row in result["positions"])


def test_strip_puts_belief_drop_on_the_hot_side():
    html = render_delta_strip(["a", " piano"], torch.tensor([0.01, -3.0]))
    norm = matplotlib.colors.TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=3.0)
    neg = matplotlib.colormaps["RdBu"](norm(-3.0))[:3]
    pos = matplotlib.colormaps["RdBu"](norm(0.01))[:3]
    assert neg[0] > neg[2]  # strong-negative token -> red (hot)
    assert pos[2] > pos[0]  # ~zero token -> blue side
    assert "Δ=-3.00 nats" in html


def test_word_grouping_joins_subword_pieces():
    # ' saxophone' -> [' sax', 'ophone'] must regroup into one word whose delta
    # is the SUM of its pieces (log-probs add).
    words = group_tokens_into_words(["a", " sax", "ophone", " plays"], [0.1, -2.0, -1.0, 0.2])
    assert [w[0] for w in words] == ["a", " saxophone", " plays"]
    assert abs(words[1][1] - (-3.0)) < 1e-9
    assert words[1][2] == [(" sax", -2.0), ("ophone", -1.0)]


def test_strip_word_level_display_with_token_hover():
    html = render_delta_strip(["a", " sax", "ophone"], torch.tensor([0.0, -2.0, -1.0]))
    assert "&nbsp;saxophone" in html          # one joined span, not sax + ophone
    assert ">ophone</span>" not in html
    assert "Δ=-3.00 nats" in html             # word value = sum
    assert "sax: -2.00" in html and "ophone: -1.00" in html  # per-token hover kept
    # raw per-token view still available
    raw = render_delta_strip(["a", " sax", "ophone"], torch.tensor([0.0, -2.0, -1.0]), word_level=False)
    assert ">ophone</span>" in raw


def test_strip_highlight_below_splits_emphasis_from_dimming():
    html = render_delta_strip(
        ["a", " sax", "ophone", " plays"],
        torch.tensor([0.0, -2.0, -1.0, -0.2]),
        highlight_below=0.5,
    )
    # ' saxophone' (summed Δ=-3.0) is emphasized; 'a' (0.0) and ' plays' (-0.2) dim.
    assert html.count("outline:2px solid") == 1
    assert html.count("opacity:0.3") == 2
    outlined = [s for s in html.split("<span") if "outline:2px solid" in s]
    assert "saxophone" in outlined[0] and "font-weight:700" in outlined[0]
    # strictly below -t: a word at exactly -t is dimmed, not emphasized
    at_threshold = render_delta_strip(["a"], torch.tensor([-0.5]), highlight_below=0.5)
    assert "outline" not in at_threshold and "opacity:0.3" in at_threshold
    # default (no threshold) touches neither — backward compatible
    plain = render_delta_strip(["a"], torch.tensor([-9.0]))
    assert "outline" not in plain and "opacity" not in plain


def test_threshold_params_scale_to_the_captions_own_drops():
    toks = ["a", " b", " c", " d"]
    # Worst word-level drop is 20 nats -> the range must reach it, and the step
    # must divide the range into ~100 drag steps (a fixed 0.05 would need
    # thousands of pixels to cross).
    big = threshold_slider_params(toks, torch.tensor([-1.0, -5.0, -20.0, 0.2]))
    assert big["max_value"] >= 20.0
    assert big["min_value"] == 0.0 and big["pixels_per_step"] == 3
    # The default outlines something rather than sitting above every drop.
    assert 0.0 < big["amount"] <= big["max_value"]
    words = group_tokens_into_words(toks, [-1.0, -5.0, -20.0, 0.2])
    assert any(w[1] < -big["amount"] for w in words)
    # Tiny drops: a hardcoded 0.5 default would outline nothing at all here.
    small = threshold_slider_params(toks, torch.tensor([-0.01, -0.02, -0.03, 0.0]))
    assert small["amount"] < 0.5
    assert any(
        w[1] < -small["amount"] for w in group_tokens_into_words(toks, [-0.01, -0.02, -0.03, 0.0])
    )
    # No drops at all: degrade to 0.0 instead of dividing by zero.
    flat = threshold_slider_params(toks, torch.tensor([0.0, 0.1, 0.2, 0.3]))
    assert flat["amount"] == 0.0 and flat["max_value"] > 0.0 and flat["step"] > 0.0
    # Whatever the scale, crossing the range must stay a comfortable drag: the
    # Tangle moves by floor(px / pixels_per_step) * step, so span in pixels is
    # (max/step) * pixels_per_step. Both a fixed step and a fixed precision
    # break this at one end or the other.
    for params in (big, small, flat):
        span_px = params["max_value"] / params["step"] * params["pixels_per_step"]
        assert 150 <= span_px <= 450, (params, span_px)
        assert params["step"] >= 10 ** -params["digits"], params  # step is visible at this precision


def test_answer_to_audio_masks_exactly_the_right_cells():
    types = ["query_text", "audio", "audio", "video", "answer", "answer"]
    numeric = torch.tensor([TOKEN_TYPE_MAP[t] for t in types])
    L = len(types)
    rule = torch.tensor([[TOKEN_TYPE_MAP["answer"], TOKEN_TYPE_MAP["audio"]]])
    hook = BlockAttentionHook(rule, numeric, original_input_len=L, generated_token_id=4)
    _, kw = hook(None, (), {"attention_mask": torch.zeros(1, 1, L, L)})
    out = kw["attention_mask"][0, 0]
    for q in (4, 5):            # answer queries
        for k in (1, 2):       # audio keys
            assert out[q, k] == NEG
    # everything else stays open
    assert out[4, 3] == 0 and out[0, 1] == 0 and out[2, 1] == 0


def test_answer_source_does_not_leak_to_other_query_types():
    types = ["query_text", "audio", "audio", "video", "answer", "answer"]
    numeric = torch.tensor([TOKEN_TYPE_MAP[t] for t in types])
    L = len(types)
    rule = torch.tensor([[TOKEN_TYPE_MAP["query_text"], TOKEN_TYPE_MAP["audio"]]])
    hook = BlockAttentionHook(rule, numeric, L, 4)
    _, kw = hook(None, (), {"attention_mask": torch.zeros(1, 1, L, L)})
    out = kw["attention_mask"][0, 0]
    assert out[0, 1] == NEG and out[4, 1] == 0  # only query_text rows blocked


def test_capture_requests_weights_only_on_selected_attention_modules():
    class _Attention(torch.nn.Module):
        def forward(self, hidden, output_attentions=False):
            weights = torch.ones(1, 1, 1, 1) if output_attentions else None
            return hidden, weights, None

    class _Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _Attention()

    class _Model:
        device = torch.device("cpu")

        def __init__(self):
            self.thinker = type("Thinker", (), {})()
            self.thinker.model = type("Backbone", (), {})()
            self.thinker.model.layers = [_Layer(), _Layer()]

    model = _Model()
    hidden = torch.zeros(1, 1, 2)
    with block_attention(
        model,
        [],
        ["query_text"],
        1,
        track_attention=True,
        capture_layer_range=(0, 1),
    ) as captured:
        selected = model.thinker.model.layers[0].self_attn(hidden)
        unselected = model.thinker.model.layers[1].self_attn(hidden)
        assert selected[1] is not None
        assert unselected[1] is None
    assert len(captured[0]) == 1
    assert model.thinker.model.layers[0].self_attn(hidden)[1] is None


def test_teacher_forced_delta_plumbing_with_a_fake_model():
    class _Thinker:
        def generate(self, input_ids=None, max_new_tokens=8, do_sample=None, **kw):
            assert do_sample is False, "F1 must pin greedy decoding"
            return torch.cat([input_ids, torch.tensor([[2, 3, 2]])], dim=1)
        def __call__(self, input_ids=None, **kw):
            import types as _t
            return _t.SimpleNamespace(logits=torch.zeros(1, input_ids.shape[1], 8))
    class _Full:
        thinker = _Thinker()
    class _Proc:
        class tokenizer:
            @staticmethod
            def decode(ids):
                return {2: " piano", 3: " plays"}.get(ids[0], "?")
    inputs = {"input_ids": torch.tensor([[0, 1, 2, 3, 4]]),
              "attention_mask": torch.ones(1, 5, dtype=torch.long)}
    res = teacher_forced_delta(_Full(), _Proc(), inputs,
                               prompt_token_types=["query_text"] * 5, rules=[])
    assert res["caption_ids"].tolist() == [[2, 3, 2]]
    assert res["caption_tokens"] == [" piano", " plays", " piano"]
    assert res["delta_total"] == 0.0  # rules=[] -> baseline == knockout
    assert res["delta_mean"] == 0.0
    assert res["delta"].shape[0] == 3
    assert res["baseline_distribution"]["measurement_kind"] == (
        "teacher_forced_answer_distribution_dispersion"
    )
    assert res["knockout_distribution"] == res["baseline_distribution"]


def _backgrounds(html):
    return [chunk.split(";")[0] for chunk in html.split("background:")[1:]]


def test_strip_renormalizes_per_run_without_a_shared_vmax():
    # The failure this guards: a caption whose worst token is -0.2 nats and one
    # whose worst is -9 nats both render fully saturated, so two strips shown
    # together say nothing about relative effect size.
    small = render_delta_strip([" a", " b"], [-0.2, 0.0])
    large = render_delta_strip([" a", " b"], [-9.0, 0.0])
    assert _backgrounds(small) == _backgrounds(large)


def test_shared_vmax_makes_two_strips_comparable():
    small = render_delta_strip([" a", " b"], [-0.2, 0.0], vmax=9.0)
    large = render_delta_strip([" a", " b"], [-9.0, 0.0], vmax=9.0)
    assert _backgrounds(small) != _backgrounds(large)
    # The larger drop must be the hotter end of the diverging scale.
    assert _backgrounds(large)[0] != _backgrounds(small)[0]
    # Neutral (0.0) is unchanged by the scale, so only the drop moved.
    assert _backgrounds(large)[1] == _backgrounds(small)[1]


def test_vmax_default_preserves_existing_behavior():
    assert render_delta_strip([" a"], [-1.0]) == render_delta_strip([" a"], [-1.0], vmax=None)
    assert render_delta_strip([" a"], [-1.0]) == render_delta_strip([" a"], [-1.0], vmax=1.0)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fails = 0
    for fn in fns:
        try:
            fn()
            print("PASS", fn.__name__)
        except Exception as e:  # noqa: BLE001
            fails += 1
            print("FAIL", fn.__name__, "->", type(e).__name__, e)
    print(f"\n{len(fns) - fails} passed, {fails} failed")
    sys.exit(1 if fails else 0)
