"""The map must not reorganise under a reader. CPU only, no weights, no network.

This file stands for the defect that would make the tool unusable in the room it
was built for: one new submission spun the whole map, and everyone lost the point
they were reading, mid-presentation.

PRD 7 recomputes coordinates over the whole corpus on every submit, and both
projectors are unstable under that. PCA mirrors when the largest loading of a
component changes as data arrives; UMAP is stochastic per fit. stabilize() fits a
rigid rotation onto the previous layout to absorb both.

The hard part is that it must absorb *noise* without absorbing *signal* -- a
layout that genuinely reorganised has to be allowed to reorganise, or the map
starts lying. Several tests below pin that boundary from each side.

Run:  python -m pytest feedback-atlas/tests/test_projection_stability.py
  or:  python feedback-atlas/tests/test_projection_stability.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> feedback-atlas/

from src.projection import (  # noqa: E402
    Layout, choose_projector, normalize_scale, project, rms_radius, stabilize,
    warm_jit,
)

SEED = 20260907


def _layout(n=30, dim=16, seed=SEED):
    X = np.random.default_rng(seed).normal(size=(n, dim))
    coords = normalize_scale(project(X))
    ids = [f"o{i}" for i in range(n)]
    return ids, coords, {i: tuple(p) for i, p in zip(ids, coords, strict=True)}


def _rotate(coords, theta, reflect=False, scale=1.0):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    out = coords @ R
    if reflect:
        out = out @ np.array([[1.0, 0.0], [0.0, -1.0]])
    return out * scale


def test_an_identical_relayout_moves_nothing():
    ids, coords, prev = _layout()
    out, drift = stabilize(coords, ids, prev)
    assert drift < 1e-9
    assert np.allclose(out, coords, atol=1e-9)


def test_a_rotated_reflected_and_rescaled_relayout_is_pulled_back_exactly():
    """The three transforms a refit is free to apply, applied at once.

    Reflection is included on purpose: a map has no chirality, so an orthogonal
    fit is allowed to mirror, and it must use that freedom to match rather than
    treat a mirrored layout as a genuine reorganisation.
    """
    ids, coords, prev = _layout()
    scrambled = _rotate(coords, theta=1.1, reflect=True, scale=3.7)
    out, drift = stabilize(scrambled, ids, prev)
    assert drift < 1e-9, f"residual {drift} -- the fit did not recover the transform"
    assert np.allclose(out, coords, atol=1e-9)


def test_the_layout_is_renormalised_so_the_map_does_not_breathe():
    """UMAP's output extent grows with n. Without renormalisation the whole map
    slowly inflates over a session with every point in the same relative place."""
    ids, coords, prev = _layout()
    out, _ = stabilize(coords * 50.0, ids, prev)
    assert abs(rms_radius(out) - rms_radius(coords)) < 1e-6


def test_adding_a_point_leaves_every_existing_point_below_the_move_threshold():
    """The submit-during-a-presentation case, which is every case.

    The bar is deliberately move_epsilon (1e-4, from src/deltas.py) and not
    "visually close". If a new opinion nudges the others by even slightly more
    than that, diff_layout reports all of them as moved, every delta becomes a
    snapshot in disguise, and the bandwidth budget the whole delta protocol was
    built for is gone. This is what forces the anchor-fitted scale in
    _similarity_fit rather than normalising the cloud as a whole.
    """
    ids, coords, prev = _layout(n=30)
    grown = np.vstack([coords, [[0.4, -0.9]]])
    out, drift = stabilize(grown, ids + ["new"], prev)
    assert drift < 1e-6
    moved = np.abs(out[:30] - coords).max()
    assert moved < 1e-4, f"existing points moved by {moved}, at/above move_epsilon"


def test_fewer_than_three_anchors_skips_rotation_instead_of_fitting_one():
    """With two anchors the rotation is exactly determined and the residual is 0
    by construction. That tells you nothing about the other 300 points, so a
    confident-looking fit here is worse than no fit."""
    ids, coords, prev = _layout()
    for k in (0, 1, 2):
        two = {i: prev[i] for i in ids[:k]}
        out, drift = stabilize(_rotate(coords, 1.1), ids, two)
        assert drift == 0.0, f"{k} anchors should not have produced a rotation"


def test_collinear_anchors_skip_rotation_from_either_side():
    """Three points on a line leave rotation about that line arbitrary, so a
    rounding error would spin the map. Degeneracy in the *previous* anchors is as
    disqualifying as in the current ones -- the fit is underdetermined either way,
    and testing only the current layout misses half of it.
    """
    rng = np.random.default_rng(1)
    line_ids = [f"c{i}" for i in range(6)]
    line = {i: (float(k) * 0.1, 0.0) for k, i in enumerate(line_ids)}

    _, drift = stabilize(rng.normal(size=(6, 2)), line_ids, line)
    assert drift == 0.0, "collinear previous anchors must not produce a rotation"

    scattered = {i: tuple(v) for i, v in zip(line_ids, rng.normal(size=(6, 2)), strict=True)}
    _, drift = stabilize(np.array([[k * 0.1, 0.0] for k in range(6)]),
                         line_ids, scattered)
    assert drift == 0.0, "collinear current anchors must not produce a rotation"


def test_a_genuine_reorganisation_is_reported_rather_than_forced_back():
    """The other side of the boundary.

    When the corpus really does restructure, the best rigid fit is a compromise.
    The transform must stay orthogonal -- no shear, no stretch, because those
    would distort real structure to flatter the residual -- and the leftover has
    to be reported as drift so the frontend can animate through it instead of
    teleporting.
    """
    ids, coords, prev = _layout(n=30)
    shuffled = np.random.default_rng(99).permutation(coords)
    out, drift = stabilize(shuffled, ids, prev)
    assert drift > 0.3, "a reorganised layout should report substantial drift"

    # Recover the transform actually applied. It must be a *similarity*:
    # M @ M.T proportional to the identity, i.e. one uniform scale factor and no
    # shear. Uniform zoom preserves every distance ratio and every angle, so it
    # cannot invent or destroy structure; anisotropic stretch could, and would let
    # the fit reduce its residual by deforming the layout instead of moving it.
    M, *_ = np.linalg.lstsq(shuffled - shuffled.mean(0), out - out.mean(0), rcond=None)
    gram = M @ M.T
    scale_sq = float(np.trace(gram)) / 2.0
    assert scale_sq > 1e-9
    assert np.allclose(gram, scale_sq * np.eye(2), atol=1e-6 * scale_sq), \
        f"fit introduced shear or an anisotropic stretch: {gram.tolist()}"


def test_reflection_parity_does_not_oscillate_across_repeated_recomputes():
    """A map that mirrors back and forth on alternate submits is unreadable, and
    an orthogonal fit is free to reflect, so parity has to be checked over a
    sequence rather than a single call."""
    ids, coords, prev = _layout(n=30)
    rng = np.random.default_rng(7)
    parities = []
    for _ in range(8):
        noisy = coords + rng.normal(scale=0.01, size=coords.shape)
        out, _ = stabilize(noisy, ids, prev)
        A = normalize_scale(noisy)
        R, *_ = np.linalg.lstsq(A - A.mean(0), out - out.mean(0), rcond=None)
        parities.append(int(np.sign(np.linalg.det(R))))
        prev = {i: tuple(p) for i, p in zip(ids, out, strict=True)}
    assert len(set(parities)) == 1, f"parity flipped across recomputes: {parities}"


def test_the_first_ever_layout_has_no_anchor_frame_and_simply_normalises():
    ids, coords, _ = _layout()
    out, drift = stabilize(coords, ids, {})
    assert drift == 0.0
    assert abs(rms_radius(out) - 1.0) < 1e-9


def test_the_first_two_submissions_of_a_session_do_not_produce_nan():
    """PCA on one point has zero variance and on two it is degenerate; the RMS
    normalisation downstream would divide by zero. These are not edge cases, they
    are the opening minute of every session."""
    X = np.random.default_rng(SEED).normal(size=(3, 16))
    assert project(X[:0]).shape == (0, 2)
    for n in (1, 2, 3):
        coords = project(X[:n])
        assert coords.shape == (n, 2)
        assert np.isfinite(coords).all(), f"n={n} produced a non-finite coordinate"
        out, _ = stabilize(coords, [f"o{i}" for i in range(n)], {})
        assert np.isfinite(out).all()


def test_a_corpus_of_identical_texts_does_not_divide_by_zero():
    """Everyone in the room types '좋아요'. Every vector is the same, the cloud has
    no extent, and the scale normalisation has nothing to divide by."""
    X = np.ones((12, 16))
    coords = project(X)
    assert np.isfinite(coords).all()
    out = normalize_scale(coords)
    assert np.isfinite(out).all()


def test_the_projector_switches_at_exactly_the_configured_threshold():
    assert choose_projector(79) == "pca"
    assert choose_projector(80) == "umap"
    assert choose_projector(80, threshold=200) == "pca"


def test_pca_runs_and_warm_jit_reports_absence_without_umap_installed():
    """The cold-start path may not depend on an optional install.

    PCA and the stabiliser are numpy-only precisely so that a failed umap install
    degrades the map rather than taking the server down, and warm_jit says so by
    returning False instead of raising.
    """
    X = np.random.default_rng(SEED).normal(size=(20, 16))
    coords = project(X, threshold=80)   # 20 < 80, so PCA regardless of umap
    assert coords.shape == (20, 2) and np.isfinite(coords).all()
    assert warm_jit(dim=8, rows=10) in (True, False)


def test_mismatched_ids_and_coordinates_are_refused_rather_than_misaligned():
    """Found by QA. ids and coords are paired positionally, so a length mismatch
    does not fail -- it attributes every point to the wrong opinion and produces a
    map that looks entirely plausible while being wrong."""
    try:
        stabilize(np.zeros((3, 2)), ["a", "b"], {"a": (0.0, 0.0)})
    except ValueError as e:
        assert "2 ids" in str(e) and "3 coordinate" in str(e)
    else:
        raise AssertionError("a length mismatch must not be silently accepted")


def test_one_non_finite_embedding_does_not_take_down_the_whole_recompute():
    """Found by QA: a NaN component made numpy's SVD raise LinAlgError straight
    out of project(), so one bad vector cost the entire class its map. The bad row
    is neutralised to the origin instead -- one misplaced point beats no map.
    """
    X = np.random.default_rng(SEED).normal(size=(12, 8))
    X[3, 2] = np.nan
    X[7, 0] = np.inf
    coords = project(X, threshold=80)
    assert coords.shape == (12, 2)
    assert np.isfinite(coords).all(), "non-finite input leaked into the layout"


@pytest.mark.needs_umap
def test_umap_is_deterministic_at_a_fixed_seed_and_stabilises_like_pca():
    """Only meaningful with umap installed; skipped otherwise by conftest."""
    X = np.random.default_rng(SEED).normal(size=(100, 16))
    a = project(X, threshold=80, seed=3)
    b = project(X, threshold=80, seed=3)
    assert np.allclose(a, b), "same seed, same data, different layout"

    ids = [f"o{i}" for i in range(100)]
    prev = {i: tuple(p) for i, p in zip(ids, normalize_scale(a), strict=True)}
    out, drift = stabilize(_rotate(a, 0.8, reflect=True, scale=2.0), ids, prev)
    assert drift < 1e-6 and np.isfinite(out).all()


def _clustered(n, dim=32, seed=11):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(3, dim))
    X = np.vstack([centres[i % 3] + rng.normal(scale=0.8, size=dim) for i in range(n)])
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def test_layout_refuses_mismatched_ids_and_rows():
    with pytest.raises(ValueError):
        Layout().project(np.zeros((3, 4)), ["a", "b"])


def test_layout_below_the_threshold_keeps_no_state_and_matches_plain_pca():
    """The cold start must behave identically with or without the incremental
    wrapper, and must not carry a stale manifold into the UMAP phase."""
    X = _clustered(20)
    ids = [f"o{i}" for i in range(20)]
    layout = Layout(threshold=80)
    coords, refit = layout.project(X, ids)
    assert refit is True
    assert np.allclose(coords, project(X, threshold=80))
    assert layout._reducer is None


@pytest.mark.needs_umap
def test_refitting_umap_every_time_would_move_every_point():
    """The measurement this whole class exists because of.

    A plain refit moves 100% of points on every recompute, at every N -- it is a
    different optimisation landing somewhere else, not a rigid motion, so the
    Procrustes stabiliser cannot take it out. Every broadcast would therefore
    exceed should_send_snapshot() and the delta protocol would be dead precisely
    above N=80, the regime it exists for. Asserted here so the justification is
    checkable rather than a claim in a docstring.
    """
    from src.deltas import MOVE_EPSILON
    n = 120
    X = _clustered(n + 1)
    ids = [f"o{i}" for i in range(n)]
    base, _ = stabilize(project(X[:n], threshold=80), ids, {})
    prev = {i: tuple(p) for i, p in zip(ids, base, strict=True)}
    grown, _ = stabilize(project(X[:n + 1], threshold=80), ids + ["new"], prev)
    moved = int((np.linalg.norm(grown[:n] - base, axis=1) > MOVE_EPSILON).sum())
    assert moved > 0.6 * n, (
        f"only {moved}/{n} moved; if a plain refit has become stable, Layout's "
        "reason for existing needs rechecking")


@pytest.mark.needs_umap
def test_a_transformed_arrival_does_not_disturb_the_points_already_placed():
    """The fix, measured the same way. Existing points are not recomputed at all,
    so they cannot move."""
    from src.deltas import MOVE_EPSILON
    X = _clustered(140)
    layout = Layout(threshold=80, refit_growth=10.0)  # no refit inside this run
    ids = [f"o{i}" for i in range(100)]
    first, refit = layout.project(X[:100], ids)
    assert refit is True

    grown_ids = ids + [f"o{i}" for i in range(100, 110)]
    second, refit = layout.project(X[:110], grown_ids)
    assert refit is False, "the manifold should have been reused, not refitted"
    moved = int((np.linalg.norm(second[:100] - first, axis=1) > MOVE_EPSILON).sum())
    assert moved == 0, f"{moved} settled points moved despite a transform-only update"


@pytest.mark.needs_umap
def test_the_manifold_is_renewed_once_the_corpus_has_outgrown_it():
    """A fit describes the corpus as it stood. Reused forever it goes stale, so
    growth past refit_growth triggers a deliberate, and increasingly rare, refit."""
    X = _clustered(200)
    layout = Layout(threshold=80, refit_growth=0.5)
    ids = [f"o{i}" for i in range(100)]
    _, first = layout.project(X[:100], ids)
    assert first is True
    _, mid = layout.project(X[:140], [f"o{i}" for i in range(140)])
    assert mid is False, "140 is within 50% growth of 100"
    _, late = layout.project(X[:160], [f"o{i}" for i in range(160)])
    assert late is True, "160 is past 50% growth of 100 and should refit"


@pytest.mark.needs_umap
def test_a_reordered_corpus_forces_a_refit_rather_than_misplacing_points():
    """The stored fit indexes rows positionally, so it is only reusable while the
    corpus still starts with the rows it was fitted on. A backfilled row or a
    clock adjustment breaks that prefix, and reusing the fit would hand stored
    coordinates to the wrong opinions -- a wrong map that looks entirely right."""
    X = _clustered(140)
    layout = Layout(threshold=80, refit_growth=10.0)
    ids = [f"o{i}" for i in range(100)]
    layout.project(X[:100], ids)
    shuffled = ["inserted"] + ids[:99]
    _, refit = layout.project(X[:100], shuffled)
    assert refit is True, "a changed prefix must invalidate the stored fit"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
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
