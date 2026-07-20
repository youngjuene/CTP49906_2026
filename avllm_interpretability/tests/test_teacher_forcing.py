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
    build_answer_token_types, caption_logprobs, delta_logprobs,
    render_delta_strip, teacher_forced_delta,
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


def test_strip_puts_belief_drop_on_the_hot_side():
    html = render_delta_strip(["a", " piano"], torch.tensor([0.01, -3.0]))
    norm = matplotlib.colors.TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=3.0)
    neg = matplotlib.colormaps["RdBu"](norm(-3.0))[:3]
    pos = matplotlib.colormaps["RdBu"](norm(0.01))[:3]
    assert neg[0] > neg[2]  # strong-negative token -> red (hot)
    assert pos[2] > pos[0]  # ~zero token -> blue side
    assert "Δ=-3.00 nats" in html


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
    assert res["delta"].shape[0] == 3


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
