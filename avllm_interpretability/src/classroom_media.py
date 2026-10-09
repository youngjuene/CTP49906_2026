"""Verified bundled video choices. Pure standard library; never loads a model."""
from __future__ import annotations

import hashlib
from pathlib import Path

# Preserve the original two submitted values for existing notebook records.
# Hashes describe prepared files, not the models' interpretations of them.
_CLIPS = {
    "Default clip": ("장면 1 · 원본 (10초)", "02321.mp4", False, "builtin-02321-av-pair-v1",
                     "9dcf1572e593777267eab01351022fcaf7b00f9a564da6059eb97f422266ddec"),
    "Silent control": ("장면 1 · 무음 (10초)", "02321_silent.mp4", True, "builtin-02321-av-pair-v1",
                       "88c72f00bf52f18b1a62a31d3f187990518bdb7b0b77b64b75bf2b4c5295e231"),
    "scene02": ("장면 2 · 원본 (10초)", "scene02.mp4", False, "builtin-scene02-av-pair-v1",
                "37bafc75c2a44df5d7652327d2d347c50e9db7126a0bcd1c7f664f5b64bcd2f4"),
    "scene02_silent": ("장면 2 · 무음 (10초)", "scene02_silent.mp4", True, "builtin-scene02-av-pair-v1",
                       "1f68938881c346690ef0f2e7408f02237e703717acbe1fc0dca5bf5f3be8b96f"),
    "scene03": ("장면 3 · 원본 (10초)", "scene03.mp4", False, "builtin-scene03-av-pair-v1",
                "95f2476b743e62fb1e1e39f153a24bbb324ea4d030859415afeb6f5e687d8c29"),
    "scene03_silent": ("장면 3 · 무음 (10초)", "scene03_silent.mp4", True, "builtin-scene03-av-pair-v1",
                       "67998ddadde665b567d6c938c467ed8d501a13829864baa1bd339cf7817633a0"),
}


def builtin_clip_choices(*, include_upload=True):
    choices = {entry[0]: choice for choice, entry in _CLIPS.items()}
    if include_upload:
        choices["업로드"] = "Upload"
    return choices


def verified_pair_key(clip_path, project_dir, *, clip_sha256=None):
    """Issue a pair key only for unchanged bytes at the registered asset path."""
    for _, filename, _, key, expected_hash in _CLIPS.values():
        if clip_path.name != filename:
            continue
        expected_path = Path(project_dir) / "assets" / filename
        if clip_path.resolve() != expected_path.resolve() or expected_path.is_symlink():
            return None
        try:
            actual_hash = clip_sha256 or hashlib.sha256(clip_path.read_bytes()).hexdigest()
        except OSError:
            return None
        return key if actual_hash == expected_hash else None
    return None


def resolve_builtin_clip(choice, project_dir):
    """Return (path, is_control); never substitute a default for a bad choice."""
    if choice not in _CLIPS:
        raise ValueError("지원하지 않는 영상 선택입니다. 목록에서 다시 선택하세요.")
    _, filename, is_control, _, _ = _CLIPS[choice]
    path = Path(project_dir) / "assets" / filename
    if not path.is_file():
        raise ValueError(f"수업 영상 {filename}을 찾을 수 없습니다. 새 영상이 포함된 수업 자료를 사용하세요.")
    if verified_pair_key(path, project_dir) is None:
        raise ValueError(f"수업 영상 {filename}의 파일이 변경되었습니다. 배포된 영상을 복원하거나 업로드로 선택하세요.")
    return path, is_control
