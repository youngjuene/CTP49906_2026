"""CPU regressions for classroom resource guards; no model or GPU required."""

from importlib import import_module, util
from pathlib import Path
from types import SimpleNamespace

import pytest


def safety():
    # An explicit failed assertion also documents the initial missing guard.
    assert util.find_spec("src.classroom_safety"), "classroom resource guards are missing"
    return import_module("src.classroom_safety")


@pytest.mark.parametrize("frames", [0, 1, 3, 17, 32, True, 8.0, "8"])
def test_rejects_frames_that_can_bypass_the_classroom_slider(frames):
    with pytest.raises(ValueError):
        safety().validate_experiment(frames, "소리를 설명하세요")


@pytest.mark.parametrize("prompt", [None, "", " \n\t", "가" * 4001],
                         ids=["missing", "empty", "whitespace", "too-long"])
def test_rejects_blank_and_oversized_prompts_before_processing(prompt):
    with pytest.raises(ValueError):
        safety().validate_experiment(8, prompt)


@pytest.mark.parametrize("tokens", [0, 7, 129, True, "32", 32.5])
def test_rejects_unbounded_generation_requests(tokens):
    with pytest.raises(ValueError):
        safety().validate_experiment(8, "Describe the sounds", tokens)


@pytest.mark.parametrize("layers", [(0, 3), (1, 36), (-1, 1), (3, 1), (0,), (0, 1, 2)])
def test_rejects_large_or_invalid_attention_capture_ranges(layers):
    with pytest.raises(ValueError):
        safety().validate_experiment(8, "소리", capture_layers=layers)


def test_allows_classroom_boundaries_and_exclusive_two_layer_capture():
    safety().validate_experiment(2, "가" * 4000, 8, (13, 15))
    safety().validate_experiment(16, "소리", 128, (0, 0))
    safety().validate_experiment(8, "소리")  # no limit on experimental knockout bands


def metadata(**changes):
    result = dict(duration=10.0, fps=25.0, width=640, height=360,
                  size_bytes=1056059, has_video=True, has_audio=True, frame_count=250,
                  audio_sample_rate=44100, audio_channels=2, audio_duration=10.0)
    result.update(changes)
    return result


@pytest.mark.parametrize("field", ["duration", "fps", "width", "height"])
@pytest.mark.parametrize("bad", [None, 0, -1, float("nan"), float("inf")])
def test_unknown_or_nonfinite_metadata_cannot_skip_the_decode_budget(field, bad):
    with pytest.raises(ValueError):
        safety().validate_clip_metadata(**metadata(**{field: bad}))


@pytest.mark.parametrize("changes", [
    {"has_video": False}, {"has_audio": False},
    {"size_bytes": 250 * 2**20 + 1}, {"duration": 120.1}, {"fps": 60.1},
    {"width": 3840, "height": 540},
    {"duration": 120, "fps": 60, "width": 1920, "height": 1080},
    {"frame_count": 100000},
])
def test_rejects_missing_tracks_and_individually_legal_but_huge_decode(changes):
    with pytest.raises(ValueError):
        safety().validate_clip_metadata(**metadata(**changes))


def test_default_clip_metadata_and_rotated_1080p_are_accepted():
    assert safety().validate_clip_metadata(**metadata()) == 173491200
    safety().validate_clip_metadata(**metadata(duration=1, fps=2, width=1080,
                                              height=1920, frame_count=2))


@pytest.mark.parametrize("field,bad", [
    (field, bad)
    for field in ("audio_sample_rate", "audio_channels", "audio_duration")
    for bad in (None, 0, -1, float("nan"), float("inf"))
])
def test_unknown_audio_metadata_cannot_skip_pcm_budget(field, bad):
    with pytest.raises(ValueError, match="오디오"):
        safety().validate_clip_metadata(**metadata(**{field: bad}))


@pytest.mark.parametrize("changes", [
    {"audio_sample_rate": 192000}, {"audio_channels": 8},
    {"audio_channels": 1.5}, {"audio_sample_rate": 44100.5},
    {"audio_duration": 121}, {"max_audio_decoded_bytes": 1024},
])
def test_rejects_high_rate_multichannel_or_oversized_audio(changes):
    with pytest.raises(ValueError, match="오디오"):
        safety().validate_clip_metadata(**metadata(**changes))


def test_audio_bounds_admit_120_seconds_of_48khz_stereo_float32():
    assert safety().validate_audio_metadata(48000, 2, 120) == 46080008


def test_real_pyav_preflight_applies_the_audio_budget(tiny_av_clip):
    assert "오디오" in safety().preflight_clip(tiny_av_clip, max_audio_decoded_bytes=1024)


@pytest.mark.parametrize("tiny_av_clip", [(96000, "mono"), (8000, "5.1"), (None, None)], indirect=True,
                         ids=["96khz", "six-channels", "no-audio-track"])
def test_real_pyav_preflight_rejects_unsupported_audio(tiny_av_clip):
    assert "오디오" in safety().preflight_clip(tiny_av_clip)


def test_encoded_multimodal_sequence_is_checked_before_device_transfer():
    assert safety().validate_encoded_inputs({"input_ids": SimpleNamespace(shape=(1, 4096))}) == 4096
    with pytest.raises(ValueError):
        safety().validate_encoded_inputs({"input_ids": SimpleNamespace(shape=(1, 4097))})


@pytest.mark.parametrize("shape", [(0, 5), (2, 10), (1, 0), (5,), (1, 2, 3)])
def test_encoded_validation_rejects_empty_or_unexpected_batches(shape):
    with pytest.raises(ValueError):
        safety().validate_encoded_inputs({"input_ids": SimpleNamespace(shape=shape)})


def test_encoded_validation_requires_input_ids():
    with pytest.raises(ValueError):
        safety().validate_encoded_inputs({})


def test_missing_local_clip_is_an_actionable_preflight_error(tmp_path):
    assert isinstance(safety().preflight_clip(tmp_path / "missing.mp4"), str)


def test_rgb_byte_budget_stops_before_converting_the_overflowing_frame():
    class Frame:
        width, height = 2, 2

        def to_ndarray(self, format):
            assert format == "rgb24"
            return SimpleNamespace(nbytes=12)

    class OverflowFrame(Frame):
        def to_ndarray(self, format):
            pytest.fail("overflow frame must be rejected before allocating RGB")

    frames = safety().iter_budgeted_rgb_frames([Frame(), OverflowFrame()], max_decoded_bytes=12)
    assert next(frames).nbytes == 12
    with pytest.raises(ValueError, match="디코딩"):
        next(frames)


def test_streaming_budget_checks_changed_dimensions_before_rgb_conversion():
    class OversizedFrame:
        width, height = 8000, 8000

        def to_ndarray(self, format):
            pytest.fail("oversized frame must not allocate RGB")

    with pytest.raises(ValueError):
        next(safety().iter_budgeted_rgb_frames([OversizedFrame()]))


@pytest.fixture
def tiny_av_clip(tmp_path, request):
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    path = tmp_path / "tiny.mkv"
    audio_rate, audio_layout = getattr(request, "param", (8000, "mono"))
    with av.open(str(path), "w") as container:
        video = container.add_stream("ffv1", rate=4)
        video.width, video.height, video.pix_fmt = 8, 6, "bgr0"
        audio = container.add_stream("pcm_s16le", rate=audio_rate) if audio_rate is not None else None
        if audio is not None:
            audio.layout = audio_layout
        for index in range(4):
            frame = av.VideoFrame.from_ndarray(np.full((6, 8, 3), index * 40, dtype=np.uint8), format="rgb24")
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        if audio is not None:
            channels = len(av.AudioLayout(audio_layout).channels)
            frame = av.AudioFrame.from_ndarray(np.zeros((1, audio_rate * channels), dtype=np.int16), format="s16", layout=audio_layout)
            frame.sample_rate = audio_rate
            for packet in audio.encode(frame):
                container.mux(packet)
            for packet in audio.encode():
                container.mux(packet)
    return path


def test_real_pyav_clip_preserves_torchvision_shapes_dtype_and_fps(tiny_av_clip):
    torch = pytest.importorskip("torch")
    assert safety().preflight_clip(tiny_av_clip) is None
    video, audio, info = safety().streaming_read_video(tiny_av_clip)
    assert video.shape == (4, 3, 6, 8)
    assert video.dtype == torch.uint8
    assert video[:, 0, 0, 0].tolist() == [0, 40, 80, 120]
    assert audio.shape == (1, 0)
    assert info["video_fps"] == 4.0
    cropped, _, _ = safety().streaming_read_video(tiny_av_clip, start_pts=0.25, end_pts=0.5, output_format="THWC")
    assert cropped.shape == (2, 6, 8, 3)


def test_real_pyav_clip_cannot_bypass_a_small_decode_budget(tiny_av_clip):
    assert safety().preflight_clip(tiny_av_clip, max_decoded_bytes=100) is not None
    with pytest.raises(ValueError):
        safety().streaming_read_video(tiny_av_clip, max_decoded_bytes=100)


def test_packaged_original_and_silent_controls_pass_the_same_guard():
    pytest.importorskip("av")
    for filename in ("02321.mp4", "02321_silent.mp4"):
        path = Path(__file__).resolve().parents[1] / "assets" / filename
        assert safety().preflight_clip(path) is None
