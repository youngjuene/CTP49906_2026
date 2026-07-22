"""Bounded private media encode/mux/decode and manual export helpers.

PyAV is imported lazily only when the production backend is selected.  The
roundtrip retains encoded bytes in the caller's private session, deletes all
temporary files on success or failure, and records local evidence only.  It does
not claim browser playback, processor reingestion, or target-Molab validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import importlib
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Protocol, Sequence


_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class MediaRoundtripPolicy:
    container_format: str = "mp4"
    video_codec: str = "mpeg4"
    audio_codec: str = "aac"
    video_rate: int = 25
    audio_rate: int = 16000
    max_video_frames: int = 600
    max_audio_frames: int = 20_000
    max_output_bytes: int = 250_000_000

    def __post_init__(self) -> None:
        for name in (
            "container_format",
            "video_codec",
            "audio_codec",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty text")
        for name in (
            "video_rate",
            "audio_rate",
            "max_video_frames",
            "max_audio_frames",
            "max_output_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class DecodedMediaFacts:
    video_frame_count: int
    audio_frame_count: int
    container_format: str


@dataclass(frozen=True)
class BrowserCompatibilityDiagnostic:
    status: str
    evidence_scope: str
    candidate_container_codec_path: bool
    detail: str


@dataclass(frozen=True)
class PyAVEnvironmentDiagnostic:
    status: str
    pyav_version: str | None
    linked_ffmpeg_libraries: dict[str, str]
    evidence_scope: str = "local_only"
    target_runtime_verified: bool = False


@dataclass(frozen=True)
class MediaRoundtripResult:
    encoded_bytes: bytes
    content_sha256: str
    byte_length: int
    video_frame_count: int
    audio_frame_count: int
    decoded: DecodedMediaFacts
    container_format: str
    video_codec: str
    audio_codec: str
    browser_compatibility: BrowserCompatibilityDiagnostic
    evidence_scope: str = "local_only"
    target_runtime_verified: bool = False


@dataclass(frozen=True)
class MediaExportReceipt:
    path: Path
    content_sha256: str
    byte_length: int


@dataclass(frozen=True)
class ReingestedPrivateMedia:
    payload: bytes
    content_sha256: str
    path: Path


@dataclass(frozen=True)
class PrivateMediaResetReceipt:
    path: Path
    content_sha256: str
    deleted: bool


class _RoundtripBackend(Protocol):
    def encode_mux(
        self,
        path: Path,
        *,
        video_frames: Sequence[Any],
        audio_frames: Sequence[Any],
        policy: MediaRoundtripPolicy,
    ) -> None: ...

    def decode(
        self, path: Path, *, policy: MediaRoundtripPolicy
    ) -> DecodedMediaFacts: ...


def _content_sha256(payload: bytes) -> str:
    return "sha256:" + sha256(payload).hexdigest()


def _require_hash(value: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError("expected_sha256 must be sha256:<64 lowercase hex>")


def diagnose_pyav_environment() -> PyAVEnvironmentDiagnostic:
    """Report local PyAV availability without promoting it to target evidence."""

    try:
        av = importlib.import_module("av")
    except ImportError:
        return PyAVEnvironmentDiagnostic(
            status="UNAVAILABLE_LOCAL",
            pyav_version=None,
            linked_ffmpeg_libraries={},
        )
    versions = {
        str(name): ".".join(str(part) for part in version)
        for name, version in dict(getattr(av, "library_versions", {})).items()
    }
    return PyAVEnvironmentDiagnostic(
        status="AVAILABLE_LOCAL",
        pyav_version=str(getattr(av, "__version__", "unknown")),
        linked_ffmpeg_libraries=versions,
    )


def _bounded_frames(
    frames: Iterable[Any], *, limit: int, kind: str
) -> tuple[Any, ...]:
    bounded = []
    for index, frame in enumerate(frames):
        if index >= limit:
            raise ValueError(f"{kind} frame count exceeds the configured limit {limit}")
        bounded.append(frame)
    return tuple(bounded)


class _PyAVBackend:
    """Small PyAV adapter kept behind a lazy import boundary."""

    def __init__(self) -> None:
        try:
            import av
        except ImportError as exc:
            raise RuntimeError(
                "PyAV is required for local encode/mux/decode; install the frozen "
                "course dependency set or use a pre-rendered licensed variant."
            ) from exc
        self.av = av

    @staticmethod
    def _configure_video_stream(stream: Any, first_frame: Any) -> None:
        width = getattr(first_frame, "width", None)
        height = getattr(first_frame, "height", None)
        if width:
            stream.width = int(width)
        if height:
            stream.height = int(height)
        if hasattr(stream, "pix_fmt"):
            stream.pix_fmt = "yuv420p"

    @staticmethod
    def _configure_audio_stream(stream: Any, first_frame: Any) -> None:
        layout = getattr(getattr(first_frame, "layout", None), "name", None)
        if layout and hasattr(stream, "layout"):
            stream.layout = layout

    def encode_mux(
        self,
        path: Path,
        *,
        video_frames: Sequence[Any],
        audio_frames: Sequence[Any],
        policy: MediaRoundtripPolicy,
    ) -> None:
        with self.av.open(str(path), mode="w", format=policy.container_format) as output:
            video_stream = None
            audio_stream = None
            if video_frames:
                video_stream = output.add_stream(
                    policy.video_codec, rate=policy.video_rate
                )
                self._configure_video_stream(video_stream, video_frames[0])
            if audio_frames:
                audio_stream = output.add_stream(
                    policy.audio_codec, rate=policy.audio_rate
                )
                self._configure_audio_stream(audio_stream, audio_frames[0])

            if video_stream is not None:
                for frame in video_frames:
                    for packet in video_stream.encode(frame):
                        output.mux(packet)
                for packet in video_stream.encode():
                    output.mux(packet)
            if audio_stream is not None:
                for frame in audio_frames:
                    for packet in audio_stream.encode(frame):
                        output.mux(packet)
                for packet in audio_stream.encode():
                    output.mux(packet)

    def decode(
        self, path: Path, *, policy: MediaRoundtripPolicy
    ) -> DecodedMediaFacts:
        video_count = 0
        audio_count = 0
        with self.av.open(str(path), mode="r") as container:
            streams = [
                stream
                for stream in container.streams
                if getattr(stream, "type", None) in {"video", "audio"}
            ]
            for frame in container.decode(*streams):
                if isinstance(frame, self.av.VideoFrame):
                    video_count += 1
                elif isinstance(frame, self.av.AudioFrame):
                    audio_count += 1
        return DecodedMediaFacts(
            video_frame_count=video_count,
            audio_frame_count=audio_count,
            container_format=policy.container_format,
        )


def encode_mux_decode_private_media(
    *,
    video_frames: Iterable[Any],
    audio_frames: Iterable[Any],
    work_dir: str | Path,
    policy: MediaRoundtripPolicy | None = None,
    _backend: _RoundtripBackend | None = None,
) -> MediaRoundtripResult:
    """Encode, mux, reopen, and decode bounded frames using private temp storage."""

    selected_policy = policy or MediaRoundtripPolicy()
    if not isinstance(selected_policy, MediaRoundtripPolicy):
        raise TypeError("policy must be a MediaRoundtripPolicy")
    bounded_video = _bounded_frames(
        video_frames,
        limit=selected_policy.max_video_frames,
        kind="video",
    )
    bounded_audio = _bounded_frames(
        audio_frames,
        limit=selected_policy.max_audio_frames,
        kind="audio",
    )
    if not bounded_video and not bounded_audio:
        raise ValueError("at least one bounded audio or video frame is required")

    backend = _backend or _PyAVBackend()
    root = Path(work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    suffix = selected_policy.container_format.lstrip(".")
    with TemporaryDirectory(prefix="ctp49906-media-", dir=root) as temporary:
        encoded_path = Path(temporary) / f"roundtrip.{suffix}"
        backend.encode_mux(
            encoded_path,
            video_frames=bounded_video,
            audio_frames=bounded_audio,
            policy=selected_policy,
        )
        if not encoded_path.is_file() or encoded_path.is_symlink():
            raise RuntimeError("media backend did not create a regular encoded output")
        size = encoded_path.stat().st_size
        if size <= 0:
            raise ValueError("encoded media output is empty")
        if size > selected_policy.max_output_bytes:
            raise ValueError(
                "encoded output byte length exceeds the configured limit "
                f"{selected_policy.max_output_bytes}"
            )
        payload = encoded_path.read_bytes()
        decoded = backend.decode(encoded_path, policy=selected_policy)
        if bounded_video and decoded.video_frame_count <= 0:
            raise ValueError("roundtrip decode produced no video frames")
        if bounded_audio and decoded.audio_frame_count <= 0:
            raise ValueError("roundtrip decode produced no audio frames")

    return MediaRoundtripResult(
        encoded_bytes=payload,
        content_sha256=_content_sha256(payload),
        byte_length=len(payload),
        video_frame_count=len(bounded_video),
        audio_frame_count=len(bounded_audio),
        decoded=decoded,
        container_format=selected_policy.container_format,
        video_codec=selected_policy.video_codec,
        audio_codec=selected_policy.audio_codec,
        browser_compatibility=BrowserCompatibilityDiagnostic(
            status="UNVERIFIED",
            evidence_scope="local_only",
            candidate_container_codec_path=(
                selected_policy.container_format.lower() in {"mp4", "webm"}
            ),
            detail=(
                "Encode/mux/decode completed locally; no browser playback or "
                "target-Molab browser path was exercised."
            ),
        ),
    )


def export_private_media(
    payload: bytes | bytearray | memoryview,
    destination: str | Path,
    *,
    expected_sha256: str,
) -> MediaExportReceipt:
    """Atomically write a user-selected private export after hash validation."""

    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("private media payload must be bytes-like")
    raw = bytes(payload)
    if not raw:
        raise ValueError("private media payload cannot be empty")
    _require_hash(expected_sha256)
    actual = _content_sha256(raw)
    if actual != expected_sha256:
        raise ValueError("private media export hash does not match expected_sha256")

    path = Path(destination).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp"
    try:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
        temporary.write_bytes(raw)
        if _content_sha256(temporary.read_bytes()) != expected_sha256:
            raise ValueError("private media temporary export hash verification failed")
        temporary.replace(path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
    return MediaExportReceipt(
        path=path,
        content_sha256=expected_sha256,
        byte_length=len(raw),
    )


def reingest_private_media(
    source: str | Path,
    *,
    expected_sha256: str,
    max_bytes: int,
) -> ReingestedPrivateMedia:
    """Read a manual export back only when its exact content identity matches."""

    _require_hash(expected_sha256)
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    path = Path(source).resolve()
    if Path(source).is_symlink():
        raise ValueError("private media reingest does not follow symbolic links")
    if not path.is_file():
        raise FileNotFoundError(f"private media export not found: {path}")
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(
            f"private media reingest exceeds the configured byte limit {max_bytes}"
        )
    payload = path.read_bytes()
    actual = _content_sha256(payload)
    if actual != expected_sha256:
        raise ValueError("private media reingest hash does not match expected_sha256")
    return ReingestedPrivateMedia(
        payload=payload,
        content_sha256=actual,
        path=path,
    )


def reset_private_media_export(
    source: str | Path,
    *,
    expected_sha256: str,
) -> PrivateMediaResetReceipt:
    """Delete only the exact private export selected by the caller."""

    _require_hash(expected_sha256)
    unresolved = Path(source)
    if unresolved.is_symlink():
        raise ValueError("private media reset does not follow symbolic links")
    path = unresolved.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"private media export not found: {path}")
    actual = _content_sha256(path.read_bytes())
    if actual != expected_sha256:
        raise ValueError("private media reset hash does not match expected_sha256")
    path.unlink()
    return PrivateMediaResetReceipt(
        path=path,
        content_sha256=actual,
        deleted=not path.exists(),
    )
