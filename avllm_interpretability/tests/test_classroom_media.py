"""Bundled classroom video selection and scientifically explicit pair identities."""
from pathlib import Path
import shutil

import pytest

from src.classroom_media import builtin_clip_choices, resolve_builtin_clip, verified_pair_key
from src.run_ledger import matched_control_ids, run_record

PROJECT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "Default clip": ("02321.mp4", False, "builtin-02321-av-pair-v1"),
    "Silent control": ("02321_silent.mp4", True, "builtin-02321-av-pair-v1"),
    "scene02": ("scene02.mp4", False, "builtin-scene02-av-pair-v1"),
    "scene02_silent": ("scene02_silent.mp4", True, "builtin-scene02-av-pair-v1"),
    "scene03": ("scene03.mp4", False, "builtin-scene03-av-pair-v1"),
    "scene03_silent": ("scene03_silent.mp4", True, "builtin-scene03-av-pair-v1"),
}


def test_dropdown_resolves_each_original_and_its_control():
    choices = builtin_clip_choices()
    assert set(choices.values()) == set(EXPECTED) | {"Upload"}
    assert set(builtin_clip_choices(include_upload=False).values()) == set(EXPECTED)
    for choice, (name, control, key) in EXPECTED.items():
        path, is_control = resolve_builtin_clip(choice, PROJECT)
        assert path == PROJECT / "assets" / name
        assert is_control is control
        assert verified_pair_key(path, PROJECT) == key


@pytest.mark.parametrize("choice", ["other", "../scene02.mp4", "Upload", ""])
def test_unknown_choice_cannot_fall_back_to_another_video(choice):
    with pytest.raises(ValueError):
        resolve_builtin_clip(choice, PROJECT)


def test_missing_or_replaced_asset_is_not_used(tmp_path):
    (tmp_path / "assets").mkdir()
    with pytest.raises(ValueError, match="찾을 수 없"):
        resolve_builtin_clip("scene02", tmp_path)
    path = tmp_path / "assets/scene02.mp4"
    path.write_bytes(b"different video")
    with pytest.raises(ValueError, match="변경"):
        resolve_builtin_clip("scene02", tmp_path)
    assert verified_pair_key(path, tmp_path) is None


def test_identical_named_upload_is_not_registered(tmp_path):
    path = tmp_path / "scene02.mp4"
    shutil.copyfile(PROJECT / "assets/scene02.mp4", path)
    assert verified_pair_key(path, PROJECT) is None


def test_each_video_matches_only_its_own_silent_control():
    import hashlib
    runs = []
    for choice in EXPECTED:
        path, control = resolve_builtin_clip(choice, PROJECT)
        runs.append(run_record(
            kind="teacher_forcing", condition="answer→audio [0,36)",
            metric_name="delta_per_token", metric_value=-0.2, metric_unit="nats/token",
            config={"clip": path.name, "clip_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "comparison_key": verified_pair_key(path, PROJECT),
                    "prompt": "영상에서 들리는 소리를 설명해 주세요", "nframes": 8},
            is_control=control,
        ))
    for index in (0, 2, 4):
        assert matched_control_ids(runs[index], runs) == [runs[index + 1]["run_id"]]
