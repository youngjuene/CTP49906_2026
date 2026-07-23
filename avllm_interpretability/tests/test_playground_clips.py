"""CPU tests for path-safe, content-addressed playground clip selection."""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.playground_clips import (  # noqa: E402
    ClipInspection,
    content_id,
    extract_media_facts,
    file_content_id,
    inspect_classroom_clip,
    persist_uploaded_clip,
    register_artifact_version,
    resolve_clip_selection,
)


def test_upload_name_cannot_escape_and_bytes_define_path_and_identity():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "uploads"
        path, identity = persist_uploaded_clip(
            root, "../../outside/lesson.mp4", b"clip-a"
        )
        assert path.parent == root.resolve()
        assert path.name.startswith("upload-") and path.suffix == ".mp4"
        assert "lesson" not in path.name and ".." not in path.name
        assert identity == file_content_id(path)


def test_same_bytes_reuse_upload_even_when_original_name_changes():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        first, first_id = persist_uploaded_clip(root, "first.mp4", b"same")
        second, second_id = persist_uploaded_clip(
            root, r"C:\\fake\\second.MP4", b"same"
        )
        assert first == second
        assert first_id == second_id


def test_changed_bytes_with_same_name_get_a_new_cache_identity_and_path():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        first, first_id = persist_uploaded_clip(root, "clip.webm", b"one")
        second, second_id = persist_uploaded_clip(root, "clip.webm", b"two")
        assert first != second
        assert first_id != second_id


def test_preplanted_digest_symlink_is_replaced_not_followed():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "uploads"
        root.mkdir()
        outside = Path(td) / "outside.mp4"
        outside.write_bytes(b"same")
        digest = content_id(b"same").split(":", 1)[1]
        planted = root / f"upload-{digest}.mp4"
        planted.symlink_to(outside)

        path, _ = persist_uploaded_clip(root, "clip.mp4", b"same")

        assert path == planted and not path.is_symlink()
        assert path.read_bytes() == b"same" and outside.read_bytes() == b"same"


def test_upload_choice_without_file_fails_clearly():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        default = root / "default.mp4"
        silent = root / "silent.mp4"
        default.write_bytes(b"default")
        silent.write_bytes(b"silent")
        try:
            resolve_clip_selection(
                "Upload",
                default_path=default,
                silent_path=silent,
                upload_dir=root / "uploads",
            )
            raise AssertionError("expected missing upload to fail")
        except ValueError as exc:
            assert "Upload selected" in str(exc) and "no file" in str(exc)


def test_builtin_choices_are_content_addressed():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        default = root / "default.mp4"
        silent = root / "silent.mp4"
        default.write_bytes(b"sound")
        silent.write_bytes(b"silence")
        d = resolve_clip_selection(
            "Default",
            default_path=default,
            silent_path=silent,
            upload_dir=root / "uploads",
        )
        s = resolve_clip_selection(
            "Silent control",
            default_path=default,
            silent_path=silent,
            upload_dir=root / "uploads",
        )
        assert d.path == default.resolve() and s.path == silent.resolve()
        assert d.cache_id != s.cache_id


def test_unsupported_upload_extension_fails_before_write():
    with tempfile.TemporaryDirectory() as td:
        try:
            persist_uploaded_clip(Path(td), "payload.py", b"not a video")
            raise AssertionError("expected unsupported extension to fail")
        except ValueError as exc:
            assert "Unsupported" in str(exc)


def test_oversized_upload_fails_before_write():
    with tempfile.TemporaryDirectory() as td:
        try:
            persist_uploaded_clip(
                Path(td), "clip.mp4", b"four", max_bytes=3
            )
            raise AssertionError("expected oversized upload to fail")
        except ValueError as exc:
            assert "larger" in str(exc) and "Trim" in str(exc)
        assert not list(Path(td).iterdir())


def test_clip_preflight_checks_streams_duration_and_first_frame():
    class _Container:
        def __init__(
            self,
            stream_types,
            duration,
            frames=(object(),),
            width=640,
            height=360,
            fps=25,
        ):
            self.streams = [
                SimpleNamespace(
                    type=t,
                    width=width if t == "video" else None,
                    height=height if t == "video" else None,
                    average_rate=fps if t == "video" else None,
                )
                for t in stream_types
            ]
            self.duration = duration
            self._frames = frames

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def decode(self, *, video):
            assert video == 0
            return iter(self._frames)

    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "clip.mp4"
        clip.write_bytes(b"fixture marker")

        ok = inspect_classroom_clip(
            clip,
            _open_container=lambda _path: _Container(
                ["video", "audio"], duration=30
            ),
            _time_base=1,
        )
        assert ok.duration_seconds == 30 and ok.has_audio
        assert (ok.width, ok.height, ok.fps) == (640, 360, 25)

        for streams, duration, frames, expected in [
            (["audio"], 30, (object(),), "no video"),
            (["video"], 30, (object(),), "no audio"),
            (["video", "audio"], 121, (object(),), "120s"),
            (["video", "audio"], 30, (), "no decodable"),
        ]:
            try:
                inspect_classroom_clip(
                    clip,
                    _open_container=lambda _path, s=streams, d=duration, f=frames: _Container(s, d, f),
                    _time_base=1,
                )
                raise AssertionError("expected clip preflight to fail")
            except ValueError as exc:
                assert expected in str(exc)

        for kwargs, expected in [
            ({"duration": None}, "unknown duration"),
            ({"duration": 30, "width": 3840, "height": 2160}, "1920×1080"),
            ({"duration": 30, "fps": 120}, "60 FPS"),
            ({"duration": 100, "width": 1920, "height": 1080, "fps": 30}, "decode"),
        ]:
            try:
                inspect_classroom_clip(
                    clip,
                    _open_container=lambda _path, kw=kwargs: _Container(
                        ["video", "audio"], **kw
                    ),
                    _time_base=1,
                )
                raise AssertionError("expected resource preflight to fail")
            except ValueError as exc:
                assert expected in str(exc)


def test_media_facts_keep_filename_as_alias_not_identity():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        first = root / "first.mp4"
        second = root / "renamed.mp4"
        first.write_bytes(b"identical media bytes")
        second.write_bytes(b"identical media bytes")
        inspection = ClipInspection(
            duration_seconds=1.25,
            has_audio=True,
            width=320,
            height=180,
            fps=20.0,
            estimated_decoded_bytes=4_320_000,
            video_stream_count=1,
            audio_stream_count=1,
            audio_sample_rate_hz=16000,
            audio_channels=1,
        )
        first_facts = extract_media_facts(first, inspection=inspection)
        second_facts = extract_media_facts(second, inspection=inspection)
        assert first_facts["content_sha256"] == second_facts["content_sha256"]
        assert first_facts["local_filename_alias"] != second_facts["local_filename_alias"]
        assert first_facts["duration_ms"] == 1250
        assert first_facts["audio_sample_rate_hz"] == 16000


def test_clip_registration_integrates_with_neutral_v1_contract():
    from curriculum_common.production_manifest import EditDecisionManifest

    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "private-alias.mp4"
        clip.write_bytes(b"registered fixture")
        inspection = ClipInspection(
            duration_seconds=1.0,
            has_audio=True,
            width=320,
            height=180,
            fps=20.0,
            estimated_decoded_bytes=3_456_000,
            audio_sample_rate_hz=16000,
            audio_channels=1,
        )
        edit_manifest = EditDecisionManifest(
            editor_name_version="common-editor/1.0",
            source_assets=({"asset_id": "synthetic-source"},),
            ordering=("synthetic-source",),
            trims=({"asset_id": "synthetic-source", "in_ms": 0, "out_ms": 1000},),
            mix_levels=({"asset_id": "synthetic-source", "gain_db": 0.0},),
            accessibility_work={"captions": "fixture"},
            assistance_disclosure={"ai": False},
            export_preset_version="common-export/1.0",
        )
        artifact = register_artifact_version(
            clip,
            inspection=inspection,
            edit_manifest=edit_manifest,
            version_label="V1",
            parent_artifact_id=None,
            local_registered_at_utc="2026-07-21T00:00:00Z",
            event_index=1,
            elapsed_ms=10,
            creator_intention="fixture intention",
            intended_audience="fixture audience",
            sound_image_relation="fixture relation",
            concept_tags=("fixture",),
            cultural_aesthetic_context="fixture context",
            source_license_provenance={"status": "synthetic_fixture"},
            change_rationale="initial cut",
            processing_boundary="browser upload to Molab-hosted session",
        )
        assert artifact.content_sha256 == file_content_id(clip)
        assert artifact.media_facts["local_filename_alias"] == clip.name
        assert artifact.version_label == "V1"

if __name__ == "__main__":
    tests = [
        v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)
    ]
    failures = 0
    for test in tests:
        try:
            test()
            print("PASS", test.__name__)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print("FAIL", test.__name__, "->", type(exc).__name__, exc)
    raise SystemExit(1 if failures else 0)
