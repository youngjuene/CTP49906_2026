"""CPU tests for the no-op knockout guard. No GPU / no model weights.

`block_attention` registers hooks only where `start <= i < end`, and the hook
only fires where a query is a source and a key is a target. A rule that matches
no layer or no token therefore runs a *baseline* while the caller believes an
intervention is applied — and the notebook renders that as "this band shows no
effect". These tests pin the four reachable ways to build such a rule.

Run:  python -m pytest avllm_interpretability/tests/test_rule_reach.py
  or:  python avllm_interpretability/tests/test_rule_reach.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> avllm_interpretability/

from src.attention_knockout_experiment import (  # noqa: E402
    block_attention, capture_vram_estimate, rule_reach,
)


def _raises(exc, substring, fn):
    """Assert `fn()` raises `exc` whose message contains `substring`.

    A local helper rather than `pytest.raises`: these test modules are also run
    directly (`python tests/test_rule_reach.py`) in environments without pytest,
    which is how the rest of this suite is written.
    """
    try:
        fn()
    except exc as e:
        assert substring in str(e), f"expected {substring!r} in {str(e)!r}"
        return
    raise AssertionError(f"expected {exc.__name__} containing {substring!r}")

# The real clip's shape, so the numbers in these tests are the ones a student
# actually meets: 1196 video, 248 audio, 32 query_text, and no image at all.
TYPES = ["query_text"] * 32 + ["video"] * 1196 + ["audio"] * 248
N_LAYERS = 36


def test_a_reaching_rule_is_ok():
    reach = rule_reach([("generated", "video", 0, 36)], TYPES, N_LAYERS)
    assert reach["ok"] and reach["reasons"] == []
    assert reach["rules"][0]["target_tokens"] == 1196
    assert reach["rules"][0]["layers"] == 36


def test_empty_layer_band_is_rejected():
    # `[12, 12)` is what dragging both range-slider handles together produces.
    reach = rule_reach([("generated", "video", 12, 12)], TYPES, N_LAYERS)
    assert not reach["ok"]
    assert "masks 0 of 36 layers" in " ".join(reach["reasons"])
    assert "exclusive" in " ".join(reach["reasons"])


def test_end_beyond_the_model_still_counts_only_real_layers():
    reach = rule_reach([("generated", "video", 30, 999)], TYPES, N_LAYERS)
    assert reach["ok"]
    assert reach["rules"][0]["layers"] == 6


def test_absent_modality_is_rejected_with_the_counts_that_prove_it():
    reach = rule_reach([("generated", "image", 0, 36)], TYPES, N_LAYERS)
    assert not reach["ok"]
    joined = " ".join(reach["reasons"])
    assert "'image' appears 0 times" in joined
    assert "video 1196" in joined  # the message shows what *is* present


def test_generated_is_live_when_generating_and_inert_in_a_forward_pass():
    rules = [("generated", "video", 0, 36)]
    assert rule_reach(rules, TYPES, N_LAYERS, context="generate")["ok"]
    forward = rule_reach(rules, TYPES, N_LAYERS, context="forward")
    assert not forward["ok"]
    assert "inert in a forward pass" in " ".join(forward["reasons"])


def test_generated_as_target_is_caught_too_not_just_as_source():
    # The notebook's prose warns only about the source half.
    forward = rule_reach([("audio", "generated", 0, 36)], TYPES, N_LAYERS, context="forward")
    assert not forward["ok"]
    assert "target 'generated'" in " ".join(forward["reasons"])


def test_answer_is_live_in_a_forward_pass():
    # `answer` is positional, so it *is* present in token_types and prefill fires.
    types = TYPES + ["answer"] * 12
    assert rule_reach([("answer", "audio", 0, 36)], types, N_LAYERS, context="forward")["ok"]


def test_unknown_type_names_the_valid_ones():
    reach = rule_reach([("text", "video", 0, 36)], TYPES, N_LAYERS)
    assert not reach["ok"]
    assert "query_text" in " ".join(reach["reasons"])


def test_malformed_rule_is_reported_not_raised():
    reach = rule_reach([("generated", "video", 0)], TYPES, N_LAYERS)
    assert not reach["ok"]
    assert "source, target, start_layer, end_layer" in " ".join(reach["reasons"])


class _Model:
    device = torch.device("cpu")

    def __init__(self, n_layers=N_LAYERS):
        self.thinker = type("Thinker", (), {})()
        self.thinker.model = type("Backbone", (), {})()
        self.thinker.model.layers = [torch.nn.Module() for _ in range(n_layers)]
        for layer in self.thinker.model.layers:
            layer.self_attn = torch.nn.Module()


def _enter(rules):
    def go():
        with block_attention(_Model(), rules, TYPES, len(TYPES)):
            pass
    return go


def test_block_attention_refuses_a_no_op_band():
    _raises(ValueError, "would mask nothing", _enter([("generated", "video", 12, 12)]))


def test_block_attention_refuses_an_absent_target():
    _raises(ValueError, "appears 0 times", _enter([("generated", "image", 0, 36)]))


def test_block_attention_allows_empty_rules():
    # Baseline runs must stay free: the guard fires only when a rule is claimed.
    with block_attention(_Model(), [], TYPES, len(TYPES)):
        pass


def test_capture_estimate_counts_heads_before_they_are_averaged():
    # The captured tensor is `[heads, q, k]`; heads are averaged only *after*
    # capture, so leaving the head factor out under-reports by 16x on this model.
    assert capture_vram_estimate(1, 100, 1, n_heads=16, dtype_bytes=2) == (
        16 * 100 * 100 * 2
    )
    assert capture_vram_estimate(2, 100, 3) == 2 * 16 * 3 * 100 * 100 * 2
    # The default (0, 2) window on the real clip against all 36 layers: the
    # ratio is what the warning is about.
    narrow = capture_vram_estimate(2, len(TYPES), 32)
    wide = capture_vram_estimate(36, len(TYPES), 32)
    assert wide == 18 * narrow
    # And the absolute number has to be big enough to justify the warning: all
    # 36 layers over a 32-token generation is tens of GB, not tens of MB.
    assert wide / 2**30 > 20


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
