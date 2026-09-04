"""CPU regression test for stale fixed-run logit-lens output."""

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import logitlens_experiment as lens  # noqa: E402


def test_empty_capture_removes_a_stale_csv():
    with tempfile.TemporaryDirectory() as td:
        output = Path(td) / "fixed.csv"
        output.write_text("plausible,old,result\n", encoding="utf-8")
        lens.logit_lens_storage.clear()
        lens.analyze_and_save_audio_logits_to_csv(
            model=None, processor=None, token_mapping=[], filename=str(output)
        )
        assert not output.exists()


def test_capture_spec_slices_only_selected_layers_and_positions():
    class _Layer(torch.nn.Module):
        def forward(self, hidden):
            return (hidden,)

    class _Thinker:
        def __init__(self):
            self.model = type("Backbone", (), {})()
            self.model.layers = [_Layer(), _Layer(), _Layer()]

    class _Model:
        def __init__(self):
            self.thinker = _Thinker()

    model = _Model()
    spec = lens.CaptureSpec(selected_layers=(1,), selected_positions=(2, 4))
    lens.register_logit_lens_hooks(model, capture_spec=spec)
    try:
        hidden = torch.arange(1 * 6 * 3, dtype=torch.float32).reshape(1, 6, 3)
        for layer in model.thinker.model.layers:
            layer(hidden)
        assert list(lens.logit_lens_storage) == [1]
        captured = lens.logit_lens_storage[1]
        assert captured.positions == (2, 4)
        assert captured.hidden_states.shape == (1, 2, 3)
        assert torch.equal(captured.hidden_states[0, 0], hidden[0, 2])
    finally:
        lens.clear_logit_lens_hooks()


def test_compact_probe_result_is_typed_and_never_serializes_full_logits():
    class _Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(4))

        def forward(self, hidden):
            return hidden @ self.weight.T

    class _Processor:
        class tokenizer:
            @staticmethod
            def decode(ids):
                return f"tok-{ids[0]}"

    model = type("Model", (), {})()
    model.thinker = type("Thinker", (), {"lm_head": _Head()})()
    lens.logit_lens_storage.clear()
    lens.logit_lens_storage[2] = lens.CapturedLayer(
        positions=(1, 3),
        hidden_states=torch.tensor([[[3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0]]]),
    )
    result = lens.build_compact_probe_result(
        model,
        _Processor(),
        ["text", "audio", "text", "audio"],
        top_k=2,
        projection_chunk_size=1,
        token_layout_fingerprint="layout-1",
    )
    payload = result.to_dict()
    assert payload["measurement_kind"] == "raw_probe_score_dispersion"
    assert [(row["layer"], row["position"]) for row in payload["summaries"]] == [
        (2, 1),
        (2, 3),
    ]
    assert all(len(row["distribution"]["top_tokens"]) == 2 for row in payload["summaries"])
    assert "logits" not in repr(payload).lower()


if __name__ == "__main__":
    fns = [value for key, value in sorted(globals().items()) if key.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
