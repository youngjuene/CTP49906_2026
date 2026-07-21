"""Safe, content-addressed clip selection for the classroom playgrounds.

Both marimo forms accept the same three choices: the default class clip, its
silent control, or an upload.  Uploaded names are untrusted UI input, so they
must never be joined directly to a writable directory.  This module persists
an upload under a SHA-256-derived name and exposes the same content digest as
the cache identity.  Consequently, renaming a clip cannot cause an incorrect
cache hit and re-uploading identical bytes reuses the existing file.

The helpers are deliberately model- and marimo-free so their safety properties
can be tested on CPU.
"""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePath


CLIP_CHOICES = ("Default", "Silent control", "Upload")
SUPPORTED_VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi"})
MAX_UPLOAD_BYTES = 250_000_000
MAX_VIDEO_PIXELS = 1920 * 1080
MAX_VIDEO_FPS = 60.0
MAX_ESTIMATED_DECODED_BYTES = 1_500_000_000


@dataclass(frozen=True)
class ResolvedClip:
    """A selected clip plus a stable identity for model-input caches."""

    path: Path
    cache_id: str
    choice: str


@dataclass(frozen=True)
class ClipInspection:
    """Cheap container-level facts checked before model/video preprocessing."""

    duration_seconds: float
    has_audio: bool
    width: int
    height: int
    fps: float
    estimated_decoded_bytes: int
    video_stream_count: int = 1
    audio_stream_count: int = 1
    audio_sample_rate_hz: int = 0
    audio_channels: int = 0


def content_id(contents: bytes) -> str:
    """Return a namespaced SHA-256 identity for ``contents``."""

    return f"sha256:{sha256(bytes(contents)).hexdigest()}"


def file_content_id(path) -> str:
    """Hash a file without loading it all into memory."""

    digest = sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def persist_uploaded_clip(
    upload_dir, original_name, contents, *, max_bytes=MAX_UPLOAD_BYTES
):
    """Persist uploaded bytes below ``upload_dir`` under a safe digest name.

    Returns ``(path, cache_id)``.  Only the suffix is retained from the original
    name; directory components and the stem are discarded.  Empty uploads and
    unsupported extensions fail loudly instead of silently falling back to the
    default class clip.
    """

    if contents is None or len(contents) == 0:
        raise ValueError("Upload selected but no file was provided.")
    if not isinstance(contents, (bytes, bytearray, memoryview)):
        raise TypeError("uploaded clip contents must be bytes-like")
    if len(contents) > max_bytes:
        raise ValueError(
            f"Uploaded clip is larger than the {max_bytes / 1_000_000:g} MB "
            "limit. Trim it before running the playground."
        )

    # Treat Windows separators as separators even on the Linux molab runtime.
    leaf = str(original_name or "").replace("\\", "/").rsplit("/", 1)[-1]
    suffix = PurePath(leaf).suffix.lower()
    if suffix not in SUPPORTED_VIDEO_SUFFIXES:
        allowed = ", ".join(sorted(SUPPORTED_VIDEO_SUFFIXES))
        raise ValueError(
            f"Unsupported uploaded video extension {suffix or '(none)'}; use {allowed}."
        )

    raw = bytes(contents)
    identity = content_id(raw)
    digest_hex = identity.split(":", 1)[1]
    root = Path(upload_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"upload-{digest_hex}{suffix}"

    # The generated filename contains only a fixed prefix, hex, and a validated
    # suffix.  Keep an explicit containment assertion as a future-edit guard.
    if destination.parent != root:
        raise ValueError("resolved upload path escaped the upload directory")

    # Never return a pre-planted symlink, even if its target happens to contain
    # the same bytes. This keeps the persisted path inside ``root`` in both the
    # lexical and filesystem senses.
    if destination.is_symlink():
        destination.unlink()
    if not destination.exists() or file_content_id(destination) != identity:
        temporary = root / f".{destination.name}.tmp"
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
        temporary.write_bytes(raw)
        temporary.replace(destination)
    return destination, identity


def resolve_clip_selection(
    choice,
    *,
    default_path,
    silent_path,
    upload_dir,
    upload_name=None,
    upload_contents=None,
):
    """Resolve one of ``CLIP_CHOICES`` to a path and content cache identity."""

    if choice not in CLIP_CHOICES:
        raise ValueError(
            f"Unknown clip choice {choice!r}; choose one of {CLIP_CHOICES}."
        )

    if choice == "Upload":
        path, identity = persist_uploaded_clip(upload_dir, upload_name, upload_contents)
    else:
        path = Path(default_path if choice == "Default" else silent_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{choice} clip not found: {path}")
        identity = file_content_id(path)
    return ResolvedClip(path=path, cache_id=identity, choice=choice)


def inspect_classroom_clip(
    path,
    *,
    max_duration_seconds=120.0,
    _open_container=None,
    _time_base=None,
):
    """Reject unsuitable clips before the PyAV shim decodes every frame.

    Checks container readability, a decodable first video frame, an audio
    stream (the lab scores audio positions), and a bounded duration. Private
    injection points keep the policy unit-testable without media fixtures.
    """

    clip = Path(path)
    if not clip.is_file():
        raise FileNotFoundError(f"clip not found: {clip}")
    if _open_container is None:
        import av

        _open_container = av.open
        _time_base = av.time_base
    if _time_base is None:
        _time_base = 1_000_000

    with _open_container(str(clip)) as container:
        video_streams = [stream for stream in container.streams if stream.type == "video"]
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        stream_types = [stream.type for stream in container.streams]
        if not video_streams:
            raise ValueError("Selected clip has no video stream.")
        if "audio" not in stream_types:
            raise ValueError(
                "Selected clip has no audio stream. Use the bundled silent control "
                "for a no-signal condition that still creates audio tokens."
            )
        if container.duration is None:
            raise ValueError(
                "Selected clip has unknown duration, so decoded memory cannot be bounded."
            )
        duration = float(container.duration) / float(_time_base)
        if duration > float(max_duration_seconds):
            raise ValueError(
                f"Selected clip is {duration:.1f}s long; the classroom limit is "
                f"{float(max_duration_seconds):g}s. Trim it before running."
            )

        video_stream = video_streams[0]
        audio_stream = audio_streams[0]
        width, height = int(video_stream.width), int(video_stream.height)
        try:
            fps = float(video_stream.average_rate)
        except (TypeError, ValueError, ZeroDivisionError):
            fps = 0.0
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError(
                "Selected clip lacks usable width, height, or frame-rate metadata."
            )
        if width * height > MAX_VIDEO_PIXELS:
            raise ValueError(
                f"Selected clip is {width}×{height}; the classroom limit is 1920×1080."
            )
        if fps > MAX_VIDEO_FPS:
            raise ValueError(
                f"Selected clip is {fps:g} FPS; the classroom limit is {MAX_VIDEO_FPS:g} FPS."
            )
        estimated_decoded_bytes = int(duration * fps * width * height * 3)
        if estimated_decoded_bytes > MAX_ESTIMATED_DECODED_BYTES:
            raise ValueError(
                "Selected clip would decode to approximately "
                f"{estimated_decoded_bytes / 1_000_000_000:.1f} GB before sampling; "
                f"the classroom limit is {MAX_ESTIMATED_DECODED_BYTES / 1_000_000_000:g} GB. "
                "Trim or downscale it."
            )

        try:
            next(container.decode(video=0))
        except StopIteration as exc:
            raise ValueError("Selected clip contains no decodable video frames.") from exc

    return ClipInspection(
        duration_seconds=duration,
        has_audio=True,
        width=width,
        height=height,
        fps=fps,
        estimated_decoded_bytes=estimated_decoded_bytes,
        video_stream_count=len(video_streams),
        audio_stream_count=len(audio_streams),
        audio_sample_rate_hz=int(getattr(audio_stream, "rate", 0) or 0),
        audio_channels=int(getattr(audio_stream, "channels", 0) or 0),
    )


def extract_media_facts(path, *, inspection=None, _open_container=None, _time_base=None):
    """Return stable media facts with the filename retained only as a private alias."""

    clip = Path(path)
    if inspection is None:
        inspection = inspect_classroom_clip(
            clip, _open_container=_open_container, _time_base=_time_base
        )
    return {
        "content_sha256": file_content_id(clip),
        "byte_length": clip.stat().st_size,
        "local_filename_alias": clip.name,
        "duration_ms": int(round(inspection.duration_seconds * 1000.0)),
        "has_video": inspection.video_stream_count > 0,
        "has_audio": inspection.has_audio,
        "video_stream_count": inspection.video_stream_count,
        "audio_stream_count": inspection.audio_stream_count,
        "width": inspection.width,
        "height": inspection.height,
        "fps": inspection.fps,
        "audio_sample_rate_hz": inspection.audio_sample_rate_hz,
        "audio_channels": inspection.audio_channels,
    }


def register_artifact_version(path, *, inspection=None, edit_manifest, **metadata):
    """Register immutable V1/V2 metadata through the treatment-neutral contract."""

    from curriculum_common.production_manifest import create_artifact_version

    facts = extract_media_facts(path, inspection=inspection)
    return create_artifact_version(
        content_sha256=facts["content_sha256"],
        media_facts=facts,
        edit_manifest=edit_manifest,
        **metadata,
    )
