"""Model provenance on a saved lens.

A `JacobianLens` is only meaningful in the residual basis it was fitted in, but
`d_model` agreement is the only thing a loader could otherwise check -- so a lens
fitted for a *different* model of the same width loads cleanly and then reads out
fluent nonsense. Upload is how these files travel between sessions and between
students, which makes it the path where that happens.

These fields are advisory by design: `load` never refuses, it only reports, so
callers decide what an absent or mismatched claim means.
"""

import torch

from jlens import JacobianLens


def _lens(d_model=8, layers=(0, 1)):
    return JacobianLens(
        {layer: torch.eye(d_model) for layer in layers},
        n_prompts=7,
        d_model=d_model,
    )


def test_provenance_defaults_to_absent_not_to_a_claim(tmp_path):
    lens = _lens()
    assert lens.fitted_for_model is None
    assert lens.fitted_for_revision is None


def test_provenance_round_trips_through_save_and_load(tmp_path):
    lens = _lens()
    lens.fitted_for_model = "Qwen/Qwen3.5-4B"
    lens.fitted_for_revision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    path = tmp_path / "lens.pt"
    lens.save(str(path))

    back = JacobianLens.load(str(path))
    assert back.fitted_for_model == "Qwen/Qwen3.5-4B"
    assert back.fitted_for_revision == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    assert back.source_layers == lens.source_layers
    assert back.d_model == lens.d_model
    assert back.n_prompts == lens.n_prompts


def test_a_lens_saved_before_provenance_existed_still_loads(tmp_path):
    # Absence of a claim, not a claim of compatibility.
    path = tmp_path / "legacy.pt"
    lens = _lens()
    torch.save(
        {
            "J": {layer: J.to(torch.float16) for layer, J in lens.jacobians.items()},
            "n_prompts": lens.n_prompts,
            "source_layers": lens.source_layers,
            "d_model": lens.d_model,
        },
        path,
    )
    back = JacobianLens.load(str(path))
    assert back.fitted_for_model is None
    assert back.d_model == lens.d_model


def test_scrambling_layers_keeps_the_lens_structurally_valid(tmp_path):
    # The negative control the notebook builds: same shape, same source layers,
    # each transport deliberately taken from a different layer. It has to remain a
    # usable lens, or it cannot be run as a control.
    lens = _lens(layers=(0, 1, 2, 3))
    src = lens.source_layers
    shift = max(1, len(src) // 2)
    scrambled = JacobianLens(
        {layer: lens.jacobians[src[(i + shift) % len(src)]] for i, layer in enumerate(src)},
        n_prompts=lens.n_prompts,
        d_model=lens.d_model,
    )
    assert scrambled.source_layers == lens.source_layers
    assert scrambled.d_model == lens.d_model
    assert set(scrambled.jacobians) == set(lens.jacobians)
