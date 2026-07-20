"""CPU tests for F5a (GPU-free replay). No GPU / no model weights.

Verifies the precomputed path the notebook takes when USE_PRECOMPUTED=True:
the attention-mass reduction, the save/load round-trip (incl. the CSV copy and
the same-file guard), the fail-loud-on-missing behaviour, and the StubModel
layer count. The live model run is exercised only by generate_precompute.py on a
GPU; the notebook's *consumption* of the artifacts is fully covered here.

Run:  python -m pytest avllm_interpretability/tests/test_precompute.py
  or:  python avllm_interpretability/tests/test_precompute.py
"""
import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> avllm_interpretability/

from src.precompute import (  # noqa: E402
    DEFAULT_MODALITY_ORDER, StubModel, load_precompute, save_precompute,
    summarize_attention,
)


def test_summarize_attention_sums_mass_by_modality():
    # last-query attention row = [0.1, 0.4, 0.4, 0.1] over 4 key positions
    t = torch.zeros(1, 2, 2, 4)
    t[0, :, 1, :] = torch.tensor([0.1, 0.4, 0.4, 0.1])  # both heads, final query row
    captured = {7: [t]}
    types = ["query_text", "audio", "audio", "video"]
    layers, order, mat = summarize_attention(captured, types)
    assert layers == [7]
    assert order == DEFAULT_MODALITY_ORDER
    # query_text=0.1, audio=0.4+0.4=0.8, video=0.1, image=0, generated=0
    assert mat[0][:5] == [0.1, 0.8, 0.1, 0.0, 0.0] or (
        abs(mat[0][0] - 0.1) < 1e-6 and abs(mat[0][1] - 0.8) < 1e-6
        and abs(mat[0][2] - 0.1) < 1e-6 and mat[0][3] == 0.0 and mat[0][4] == 0.0
    )


def test_summarize_attention_none_when_empty():
    assert summarize_attention({}, ["audio"]) is None


def test_save_load_round_trip():
    with tempfile.TemporaryDirectory() as td:
        src_csv = Path(td) / "src.csv"
        src_csv.write_text("Token_Position,Token_Type,Layer_0\n3,audio,piano\n", encoding="utf-8")
        out = Path(td) / "precomputed"
        summary = ([0, 1], DEFAULT_MODALITY_ORDER, [[0.1, 0.8, 0.1, 0.0, 0.0], [0.2, 0.2, 0.5, 0.0, 0.1]])
        save_precompute(
            out, logit_csv_src=src_csv, logit_caption="a person plays the piano",
            baseline_text="baseline cap", knockout_text="knockout cap",
            knockout_rules=[("generated", "video", 0, 36)], attention_summary=summary,
            attention_token_types=["query_text", "audio"],
            meta={"clip": "02321.mp4", "n_layers": 36},
        )
        loaded = load_precompute(out)
        assert loaded["logit_caption"] == "a person plays the piano"
        assert loaded["baseline_text"] == "baseline cap"
        assert loaded["knockout_text"] == "knockout cap"
        assert loaded["knockout_rules"] == [["generated", "video", 0, 36]]
        assert loaded["attention_summary"][0] == [0, 1]
        assert loaded["attention_summary"][2][0] == [0.1, 0.8, 0.1, 0.0, 0.0]
        assert loaded["attention_token_types"] == ["query_text", "audio"]
        assert loaded["meta"]["n_layers"] == 36
        # the CSV was copied under the canonical name and is readable
        assert loaded["logit_csv"].name == "logit_lens_audio_token_analysis.csv"
        assert "piano" in loaded["logit_csv"].read_text(encoding="utf-8")


def test_save_precompute_handles_csv_already_in_place():
    # generate_precompute.py writes the CSV straight into out_dir, then calls
    # save_precompute with the same path — must not raise SameFileError.
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "precomputed"
        out.mkdir()
        csv_in_place = out / "logit_lens_audio_token_analysis.csv"
        csv_in_place.write_text("Token_Position,Token_Type,Layer_0\n3,audio,x\n", encoding="utf-8")
        save_precompute(
            out, logit_csv_src=csv_in_place, logit_caption="c", baseline_text="b",
            knockout_text="k", knockout_rules=[], attention_summary=None,
            attention_token_types=[], meta={},
        )
        assert csv_in_place.exists()
        assert load_precompute(out)["attention_summary"] is None


def test_load_precompute_fails_loud_when_missing():
    with tempfile.TemporaryDirectory() as td:
        try:
            load_precompute(Path(td) / "does_not_exist")
            raise AssertionError("expected FileNotFoundError")
        except FileNotFoundError as e:
            assert "generate_precompute" in str(e)


def test_stub_model_exposes_layer_count():
    m = StubModel(36)
    assert len(m.thinker.model.layers) == 36
    assert m.device == "cpu"


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
