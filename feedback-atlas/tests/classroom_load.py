"""Opt-in live v2 delivery and PCA/UMAP threshold checks with temporary data."""
from classroom_semantic_load import run_measurement


def test_live_classroom_load_does_not_drop_or_duplicate_opinions(tmp_path):
    run_measurement(tmp_path, seed_count=1000, count=30, long=False)


def test_threshold_transition_from_79_to_80_keeps_all_points(tmp_path):
    result = run_measurement(tmp_path, seed_count=79, count=1, long=False)
    assert result['resulting_units'] == 80
