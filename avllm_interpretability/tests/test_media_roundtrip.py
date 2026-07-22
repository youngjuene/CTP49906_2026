"""Offline tests for bounded private PyAV roundtrip orchestration."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.media_roundtrip import (  # noqa: E402
    DecodedMediaFacts,
    MediaRoundtripPolicy,
    diagnose_pyav_environment,
    encode_mux_decode_private_media,
    export_private_media,
    reingest_private_media,
    reset_private_media_export,
)


class _OfflineBackend:
    """Deterministic stand-in for PyAV's encode/mux/decode boundary."""

    def __init__(self, *, fail: bool = False, payload: bytes | None = None) -> None:
        self.fail = fail
        self.payload = payload
        self.encode_calls = 0
        self.decode_calls = 0
        self.temporary_path: Path | None = None

    def encode_mux(self, path, *, video_frames, audio_frames, policy) -> None:
        self.encode_calls += 1
        self.temporary_path = Path(path)
        if self.fail:
            Path(path).write_bytes(b"partial-private-media")
            raise RuntimeError("offline encode failure")
        payload = self.payload
        if payload is None:
            payload = (
                f"offline-av|v={len(video_frames)}|a={len(audio_frames)}|"
                f"format={policy.container_format}"
            ).encode("ascii")
        Path(path).write_bytes(payload)

    def decode(self, path, *, policy) -> DecodedMediaFacts:
        self.decode_calls += 1
        payload = Path(path).read_bytes()
        assert payload.startswith(b"offline-av")
        parts = dict(
            item.split("=", 1)
            for item in payload.decode("ascii").split("|")[1:]
            if "=" in item
        )
        return DecodedMediaFacts(
            video_frame_count=int(parts["v"]),
            audio_frame_count=int(parts["a"]),
            container_format=parts["format"],
        )


class _MalformedDecodeBackend(_OfflineBackend):
    def decode(self, path, *, policy) -> DecodedMediaFacts:
        self.decode_calls += 1
        raise ValueError("malformed offline container")


def test_module_import_is_safe_without_eager_pyav_import() -> None:
    if importlib.util.find_spec("av") is None:
        assert "av" not in sys.modules


def test_offline_encode_mux_decode_is_bounded_and_locally_scoped(tmp_path) -> None:
    backend = _OfflineBackend()
    result = encode_mux_decode_private_media(
        video_frames=("v0", "v1"),
        audio_frames=("a0", "a1", "a2"),
        work_dir=tmp_path,
        policy=MediaRoundtripPolicy(
            max_video_frames=2,
            max_audio_frames=3,
            max_output_bytes=256,
        ),
        _backend=backend,
    )

    assert backend.encode_calls == backend.decode_calls == 1
    assert result.video_frame_count == result.decoded.video_frame_count == 2
    assert result.audio_frame_count == result.decoded.audio_frame_count == 3
    assert result.content_sha256.startswith("sha256:")
    assert result.byte_length == len(result.encoded_bytes)
    assert result.evidence_scope == "local_only"
    assert result.target_runtime_verified is False
    assert result.browser_compatibility.status == "UNVERIFIED"
    assert result.browser_compatibility.evidence_scope == "local_only"
    assert "no browser playback" in result.browser_compatibility.detail.lower()
    assert backend.temporary_path is not None
    assert not backend.temporary_path.exists()
    assert list(tmp_path.iterdir()) == []


def test_manual_export_and_reingest_require_the_exact_content_hash(tmp_path) -> None:
    result = encode_mux_decode_private_media(
        video_frames=("v0",),
        audio_frames=("a0",),
        work_dir=tmp_path / "roundtrip",
        _backend=_OfflineBackend(),
    )
    destination = tmp_path / "student-selected" / "variant.mp4"
    receipt = export_private_media(
        result.encoded_bytes,
        destination,
        expected_sha256=result.content_sha256,
    )
    assert receipt.path == destination.resolve()
    assert receipt.content_sha256 == result.content_sha256
    assert not (destination.parent / f".{destination.name}.tmp").exists()

    reingested = reingest_private_media(
        destination,
        expected_sha256=result.content_sha256,
        max_bytes=256,
    )
    assert reingested.payload == result.encoded_bytes
    assert reingested.content_sha256 == result.content_sha256

    destination.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        reingest_private_media(
            destination,
            expected_sha256=result.content_sha256,
            max_bytes=256,
        )

    destination.write_bytes(result.encoded_bytes)
    reset = reset_private_media_export(
        destination,
        expected_sha256=result.content_sha256,
    )
    assert reset.deleted is True
    assert not destination.exists()


def test_frame_and_output_bounds_fail_before_unbounded_state_survives(tmp_path) -> None:
    backend = _OfflineBackend()
    with pytest.raises(ValueError, match="video frame"):
        encode_mux_decode_private_media(
            video_frames=("v0", "v1"),
            audio_frames=(),
            work_dir=tmp_path,
            policy=MediaRoundtripPolicy(max_video_frames=1),
            _backend=backend,
        )
    assert backend.encode_calls == 0

    oversized = _OfflineBackend(payload=b"offline-av|v=1|a=0|format=mp4" + b"x" * 100)
    with pytest.raises(ValueError, match="output byte"):
        encode_mux_decode_private_media(
            video_frames=("v0",),
            audio_frames=(),
            work_dir=tmp_path,
            policy=MediaRoundtripPolicy(max_output_bytes=32),
            _backend=oversized,
        )
    assert list(tmp_path.iterdir()) == []


def test_encode_failure_removes_partial_private_temp_files(tmp_path) -> None:
    backend = _OfflineBackend(fail=True)
    with pytest.raises(RuntimeError, match="offline encode failure"):
        encode_mux_decode_private_media(
            video_frames=("v0",),
            audio_frames=("a0",),
            work_dir=tmp_path,
            _backend=backend,
        )
    assert backend.temporary_path is not None
    assert not backend.temporary_path.exists()
    assert list(tmp_path.iterdir()) == []


def test_malformed_decode_removes_encoded_temp_and_allows_clean_retry(tmp_path) -> None:
    malformed = _MalformedDecodeBackend()
    with pytest.raises(ValueError, match="malformed"):
        encode_mux_decode_private_media(
            video_frames=("v0",),
            audio_frames=("a0",),
            work_dir=tmp_path,
            _backend=malformed,
        )
    assert list(tmp_path.iterdir()) == []

    retry = encode_mux_decode_private_media(
        video_frames=("v0",),
        audio_frames=("a0",),
        work_dir=tmp_path,
        _backend=_OfflineBackend(),
    )
    assert retry.decoded.video_frame_count == 1
    assert list(tmp_path.iterdir()) == []


def test_offline_roundtrip_has_no_automatic_egress(tmp_path, monkeypatch) -> None:
    import socket

    socket_calls = []

    def blocked_socket(*args, **kwargs):
        socket_calls.append((args, kwargs))
        raise AssertionError("automatic network egress attempted")

    monkeypatch.setattr(socket, "socket", blocked_socket)
    encode_mux_decode_private_media(
        video_frames=("v0",),
        audio_frames=("a0",),
        work_dir=tmp_path,
        _backend=_OfflineBackend(),
    )
    assert socket_calls == []


def test_default_backend_reports_missing_pyav_only_when_called(tmp_path) -> None:
    if importlib.util.find_spec("av") is not None:
        pytest.skip("PyAV is installed in this interpreter")
    with pytest.raises(RuntimeError, match="PyAV"):
        encode_mux_decode_private_media(
            video_frames=("v0",),
            audio_frames=(),
            work_dir=tmp_path,
        )


def test_pyav_environment_diagnostic_never_claims_target_verification() -> None:
    diagnostic = diagnose_pyav_environment()
    assert diagnostic.status in {"AVAILABLE_LOCAL", "UNAVAILABLE_LOCAL"}
    assert diagnostic.evidence_scope == "local_only"
    assert diagnostic.target_runtime_verified is False


def test_real_pyav_encode_mux_decode_uses_a_tiny_offline_fixture(tmp_path) -> None:
    av = pytest.importorskip("av")

    video_frames = []
    for index in range(2):
        frame = av.VideoFrame(16, 16, "yuv420p")
        for plane in frame.planes:
            plane.update(bytes([16 + index]) * plane.buffer_size)
        frame.pts = index
        video_frames.append(frame)

    audio_frames = []
    for index in range(2):
        frame = av.AudioFrame(format="s16", layout="mono", samples=1024)
        frame.sample_rate = 8000
        frame.pts = index * 1024
        frame.planes[0].update(b"\x00\x00" * 1024)
        audio_frames.append(frame)

    result = encode_mux_decode_private_media(
        video_frames=video_frames,
        audio_frames=audio_frames,
        work_dir=tmp_path,
        policy=MediaRoundtripPolicy(
            video_rate=2,
            audio_rate=8000,
            max_video_frames=2,
            max_audio_frames=2,
            max_output_bytes=1_000_000,
        ),
    )

    assert result.encoded_bytes.startswith(b"\x00\x00")
    assert result.decoded.video_frame_count == 2
    assert result.decoded.audio_frame_count == 2
    assert result.browser_compatibility.candidate_container_codec_path is True
    assert result.browser_compatibility.status == "UNVERIFIED"
    assert list(tmp_path.iterdir()) == []
