"""Bound notebook inputs before decoding or transferring tensors to a GPU.

Pure validation imports only the standard library. PyAV, NumPy and Torch are
loaded only by the video entry points. These are conservative classroom limits,
not a prediction of peak RAM/VRAM or a guarantee against every decoder failure.
"""

from __future__ import annotations

import math
import os
from numbers import Integral
from pathlib import Path

MAX_UPLOAD_BYTES = 250 * 2**20
MAX_DURATION_S = 120.0
MAX_FPS = 60.0
MAX_PIXELS = 1920 * 1080
# A list, stack and contiguous TCHW copy can coexist. Keep their total well
# below host RAM, leaving room for model loading, audio and codec buffers.
MAX_DECODED_BYTES = 512 * 2**20
MAX_DECODED_FRAMES = 7200
MAX_AUDIO_DECODED_BYTES = 128 * 2**20
MAX_AUDIO_SAMPLE_RATE = 48000
MAX_AUDIO_CHANNELS = 2
MAX_PROMPT_CHARS = 4000
MAX_INPUT_TOKENS = 4096


def _integer(value):
    return isinstance(value, Integral) and not isinstance(value, bool)


def validate_experiment(nframes, prompt, max_new_tokens=32, capture_layers=None):
    """Raise ValueError for unsafe settings; capture_layers is [start, end).

    Only the optional *capture* range is limited to two layers. Experimental
    knockout rules have their own model-aware validation and are not limited
    by this function.
    """
    if not _integer(nframes) or not 2 <= nframes <= 16 or nframes % 2:
        raise ValueError("프레임 수는 2~16 사이의 짝수 정수여야 합니다.")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("프롬프트에 공백이 아닌 질문을 입력하세요.")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError(f"프롬프트는 {MAX_PROMPT_CHARS:,}자 이하로 줄여 주세요.")
    if not _integer(max_new_tokens) or not 8 <= max_new_tokens <= 128:
        raise ValueError("생성 상한은 8~128 사이의 정수 토큰 수여야 합니다.")
    if capture_layers is not None:
        if (
            not isinstance(capture_layers, (tuple, list))
            or len(capture_layers) != 2
            or not all(_integer(layer) for layer in capture_layers)
            or not 0 <= capture_layers[0] <= capture_layers[1]
            or capture_layers[1] - capture_layers[0] > 2
        ):
            raise ValueError("그림 캡처는 최대 2개 레이어인 (start, end) 범위로 지정하세요. end는 제외합니다.")


def validate_encoded_inputs(inputs, max_tokens=MAX_INPUT_TOKENS):
    """Return sequence length after checking the CPU processor output.

    input_ids includes the expanded multimodal positions. Call before .to(...)
    and before putting the batch in a cache; a GPU-memory check is not implied.
    """
    if not _integer(max_tokens) or max_tokens <= 0:
        raise ValueError("입력 토큰 상한은 양의 정수여야 합니다.")
    try:
        shape = inputs["input_ids"].shape
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("전처리 결과에 input_ids가 없습니다.") from exc
    if len(shape) != 2 or shape[0] != 1 or shape[1] <= 0:
        raise ValueError("한 번에 비어 있지 않은 입력 한 개만 실행할 수 있습니다.")
    length = int(shape[1])
    if length > max_tokens:
        raise ValueError(
            f"전처리 후 입력이 {length:,}토큰입니다. 상한 {max_tokens:,}에 맞게 "
            "프레임 수·영상 길이·프롬프트 길이를 줄여 주세요."
        )
    return length


def _positive_number(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} 메타데이터가 없어 안전한 크기를 확인할 수 없습니다.") from exc
    if isinstance(value, bool) or not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} 메타데이터가 유효하지 않습니다. 영상을 다시 인코딩해 주세요.")
    return number


def _dimensions(width, height):
    width = _positive_number(width, "너비")
    height = _positive_number(height, "높이")
    if not width.is_integer() or not height.is_integer():
        raise ValueError("영상 너비와 높이는 양의 정수여야 합니다.")
    # Accept a rotated 1080p clip, but not an arbitrarily wide thin image.
    if max(width, height) > 1920 or min(width, height) > 1080:
        raise ValueError("영상 해상도 상한은 1920×1080(세로 영상은 1080×1920)입니다.")
    return int(width), int(height)


def validate_audio_metadata(sample_rate, channels, duration,
                            max_decoded_bytes=MAX_AUDIO_DECODED_BYTES):
    """Return conservative float32 PCM bytes after validating audio metadata.

    Count the original channels before downmixing. The notebook must also check
    actual decoded samples before converting or appending audio frames, because
    duration metadata is only an estimate.
    """
    sample_rate = _positive_number(sample_rate, "오디오 샘플레이트")
    channels = _positive_number(channels, "오디오 채널 수")
    duration = _positive_number(duration, "오디오 길이")
    if not sample_rate.is_integer() or sample_rate > MAX_AUDIO_SAMPLE_RATE:
        raise ValueError("오디오 샘플레이트는 48,000 Hz 이하의 양의 정수여야 합니다.")
    if not channels.is_integer() or channels > MAX_AUDIO_CHANNELS:
        raise ValueError("오디오는 모노(1채널) 또는 스테레오(2채널)여야 합니다.")
    if duration > MAX_DURATION_S:
        raise ValueError("오디오 길이 상한은 120초입니다. 영상을 잘라 주세요.")
    if not _integer(max_decoded_bytes) or max_decoded_bytes <= 0:
        raise ValueError("오디오 디코딩 바이트 상한은 양의 정수여야 합니다.")
    estimated_bytes = (math.ceil(duration * sample_rate) + 1) * int(channels) * 4
    if estimated_bytes > max_decoded_bytes:
        raise ValueError(
            f"오디오 디코딩 예상량이 {estimated_bytes / 2**20:.1f} MiB입니다. "
            f"상한 {max_decoded_bytes / 2**20:.1f} MiB에 맞게 길이·샘플레이트를 줄여 주세요."
        )
    return estimated_bytes


def validate_clip_metadata(*, duration, fps, width, height, size_bytes,
                           has_video, has_audio, frame_count=None,
                           audio_sample_rate=None, audio_channels=None,
                           audio_duration=None,
                           max_decoded_bytes=MAX_DECODED_BYTES,
                           max_audio_decoded_bytes=MAX_AUDIO_DECODED_BYTES):
    """Validate metadata and return a conservative RGB-byte estimate.

    Duration/rate/dimensions must all be known, finite and positive. Unknown
    frame count is acceptable because duration × fps estimates it; the decoder
    also enforces the actual byte and frame limits independently.
    """
    if not has_video:
        raise ValueError("이 파일에는 비디오 스트림이 없습니다.")
    if not has_audio:
        raise ValueError("오디오 트랙이 없습니다. 무음 대조군도 실제 무음 오디오 트랙이 필요합니다.")
    size_bytes = _positive_number(size_bytes, "파일 크기")
    if size_bytes > MAX_UPLOAD_BYTES:
        raise ValueError("파일 크기 상한은 250 MiB입니다. 영상을 줄여 주세요.")
    duration = _positive_number(duration, "영상 길이")
    fps = _positive_number(fps, "프레임 속도")
    width, height = _dimensions(width, height)
    if duration > MAX_DURATION_S:
        raise ValueError("영상 길이 상한은 120초입니다. 영상을 잘라 주세요.")
    if fps > MAX_FPS:
        raise ValueError("영상 프레임 속도 상한은 60 fps입니다.")
    validate_audio_metadata(audio_sample_rate, audio_channels, audio_duration,
                            max_decoded_bytes=max_audio_decoded_bytes)
    if not _integer(max_decoded_bytes) or max_decoded_bytes <= 0:
        raise ValueError("디코딩 바이트 상한은 양의 정수여야 합니다.")
    # One extra frame covers a rounded-down container duration/end point.
    estimated_frames = math.ceil(duration * fps) + 1
    if frame_count is not None and frame_count != 0:
        count = _positive_number(frame_count, "프레임 수")
        if not count.is_integer() or count > MAX_DECODED_FRAMES:
            raise ValueError("영상 프레임 수가 수업 디코딩 상한을 넘습니다.")
        estimated_frames = max(estimated_frames, int(count))
    estimated_bytes = estimated_frames * width * height * 3
    if estimated_bytes > max_decoded_bytes:
        raise ValueError(
            f"RGB 디코딩 예상량이 {estimated_bytes / 2**20:.1f} MiB입니다. "
            f"상한 {max_decoded_bytes / 2**20:.1f} MiB에 맞게 영상 길이·해상도를 줄여 주세요."
        )
    return estimated_bytes


def _local_clip_path(filename):
    filename = os.fspath(filename)
    if isinstance(filename, str) and filename.startswith("file://"):
        filename = filename[len("file://"):]
    path = Path(filename)
    if not path.is_file():
        raise ValueError(f"영상 파일을 찾을 수 없습니다: {path.name}")
    # Reject oversized files before a container parser touches them.
    if not 0 < path.stat().st_size <= MAX_UPLOAD_BYTES:
        raise ValueError("영상 파일은 비어 있지 않고 250 MiB 이하여야 합니다.")
    return path


def _inspect_container(container, path, max_decoded_bytes,
                       max_audio_decoded_bytes=MAX_AUDIO_DECODED_BYTES):
    videos = container.streams.video
    stream = videos[0] if videos else None
    duration = container.duration / 1_000_000 if container.duration is not None else None
    if duration is None and stream is not None and stream.duration is not None and stream.time_base is not None:
        duration = float(stream.duration * stream.time_base)
    rate = (stream.average_rate or stream.guessed_rate or stream.base_rate) if stream is not None else None
    audio = container.streams.audio[0] if container.streams.audio else None
    # Matroska often omits per-stream duration while retaining container duration.
    # Use that known value; do not substitute the video-only duration fallback.
    audio_duration = container.duration / 1_000_000 if container.duration is not None else None
    if audio is not None and audio.duration is not None and audio.time_base is not None:
        audio_duration = float(audio.duration * audio.time_base)
    validate_clip_metadata(
        duration=duration, fps=rate,
        width=stream.width if stream is not None else None,
        height=stream.height if stream is not None else None,
        size_bytes=path.stat().st_size, has_video=bool(videos),
        has_audio=bool(container.streams.audio),
        frame_count=stream.frames if stream is not None else None,
        audio_sample_rate=audio.codec_context.sample_rate if audio is not None else None,
        audio_channels=audio.codec_context.channels if audio is not None else None,
        audio_duration=audio_duration,
        max_decoded_bytes=max_decoded_bytes,
        max_audio_decoded_bytes=max_audio_decoded_bytes,
    )
    return float(rate)


def preflight_clip(path, max_decoded_bytes=MAX_DECODED_BYTES, *,
                   max_audio_decoded_bytes=MAX_AUDIO_DECODED_BYTES):
    """Return None when allowed or an actionable Korean reason when rejected."""
    try:
        path = _local_clip_path(path)
        import av

        with av.open(str(path)) as container:
            _inspect_container(container, path, max_decoded_bytes, max_audio_decoded_bytes)
    except Exception as exc:  # a broken container becomes a form error
        return f"영상을 사용할 수 없습니다 ({type(exc).__name__}: {exc})."
    return None


def iter_budgeted_rgb_frames(frames, max_decoded_bytes=MAX_DECODED_BYTES, *,
                             start_pts=0.0, end_pts=None, pts_unit="sec"):
    """Yield RGB arrays only after bounding each frame and cumulative decode.

    The overflow check precedes RGB allocation and list/stack materialization.
    Decoder-internal buffers are outside this bound. Frames discarded by a time
    slice still count toward the limit so slicing cannot bypass the guard.
    """
    if not _integer(max_decoded_bytes) or max_decoded_bytes <= 0:
        raise ValueError("디코딩 바이트 상한은 양의 정수여야 합니다.")
    used = 0
    for count, frame in enumerate(frames, 1):
        width, height = _dimensions(frame.width, frame.height)
        used += width * height * 3
        if used > max_decoded_bytes or count > MAX_DECODED_FRAMES:
            raise ValueError("실제 RGB 디코딩 누적량이 상한을 넘었습니다. 영상 길이·해상도를 줄여 주세요.")
        if start_pts != 0 or end_pts is not None:
            timestamp = frame.time if pts_unit == "sec" else frame.pts
            if timestamp is None or not math.isfinite(float(timestamp)):
                raise ValueError("프레임 시각이 없어 요청한 영상 구간을 확인할 수 없습니다.")
            if timestamp < start_pts:
                continue
            if end_pts is not None and timestamp > end_pts:
                break
        array = frame.to_ndarray(format="rgb24")
        if array.nbytes != width * height * 3:
            raise ValueError("디코딩한 RGB 프레임 크기가 메타데이터와 다릅니다.")
        yield array


def streaming_read_video(filename, start_pts=0.0, end_pts=None, pts_unit="sec",
                         output_format="TCHW", *, max_decoded_bytes=MAX_DECODED_BYTES):
    """Bounded replacement for qwen's torchvision.io.read_video call.

    Returns (uint8 video, empty float32 audio, fps metadata), matching the old
    notebook shim. Qwen extracts audio separately. Always validates this path,
    including packaged assets and directly edited VIDEO_PATH values.
    """
    if pts_unit not in ("sec", "pts"):
        raise ValueError("pts_unit은 sec 또는 pts여야 합니다.")
    if output_format.upper() not in ("TCHW", "THWC"):
        raise ValueError("출력 형식은 TCHW 또는 THWC여야 합니다.")
    if not math.isfinite(float(start_pts)) or start_pts < 0:
        raise ValueError("시작 시각은 0 이상의 유한한 수여야 합니다.")
    if end_pts is not None and (not math.isfinite(float(end_pts)) or end_pts < start_pts):
        raise ValueError("종료 시각은 시작 시각 이상의 유한한 수여야 합니다.")
    path = _local_clip_path(filename)
    import av
    import numpy as np
    import torch

    with av.open(str(path)) as container:
        fps = _inspect_container(container, path, max_decoded_bytes)
        # Avoid an unbounded thread pool of decoder buffers on host RAM.
        container.streams.video[0].thread_count = 1
        frames = list(iter_budgeted_rgb_frames(
            container.decode(video=0), max_decoded_bytes,
            start_pts=start_pts, end_pts=end_pts, pts_unit=pts_unit,
        ))
    if not frames:
        raise ValueError("선택한 구간에서 비디오 프레임을 디코딩하지 못했습니다.")
    video = torch.from_numpy(np.stack(frames))
    del frames
    if output_format.upper() == "TCHW":
        video = video.permute(0, 3, 1, 2).contiguous()
    audio = torch.zeros((1, 0), dtype=torch.float32)
    return video, audio, {"video_fps": fps, "audio_fps": None}
