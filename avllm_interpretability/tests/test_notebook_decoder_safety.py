"""Run the notebook's setup functions without setup downloads or a model.

Video tests use Qwen's real cached backend selection and torchvision adapter.
Audio tests replace the external librosa/FFmpeg boundary with PCM readers while
retaining the notebook's actual conversion, allocation and byte accounting.
"""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from src import classroom_safety


NOTEBOOK = Path(__file__).resolve().parents[1] / "CTP49906_avllm_molab_kr.py"


def extract_setup_function(name):
    tree = ast.parse(NOTEBOOK.read_text(encoding="utf-8"))
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("cached_backend", ["decord", "torchcodec"])
@pytest.mark.parametrize("native_reader_exists", [True, False])
def test_forces_bounded_torchvision_even_with_a_cached_alternate_backend(
        monkeypatch, tmp_path, cached_backend, native_reader_exists):
    torchvision = pytest.importorskip("torchvision")
    vision = pytest.importorskip("qwen_omni_utils.v2_5.vision_process")

    def unbounded_native(*args, **kwargs):
        pytest.fail("the pre-existing native reader bypassed the classroom guard")

    # Pinned old torchvision has read_video; newer installations may not.
    if native_reader_exists:
        monkeypatch.setattr(torchvision.io, "read_video", unbounded_native, raising=False)
    else:
        monkeypatch.delattr(torchvision.io, "read_video", raising=False)
    monkeypatch.setattr(vision, "FORCE_QWENVL_VIDEO_READER", cached_backend)
    vision.get_video_reader_backend.cache_clear()
    try:
        assert vision.get_video_reader_backend() == cached_backend
        extract_setup_function("_ensure_video_reader")()
        assert vision.get_video_reader_backend() == "torchvision"
        # Exercise Qwen's real dispatch adapter, including its imported io alias.
        # The guard must reject the clip before any unbounded decoder sees it.
        backend = vision.VIDEO_READER_BACKENDS[vision.get_video_reader_backend()]
        with pytest.raises(ValueError, match="영상 파일을 찾을 수 없습니다"):
            backend({"video": str(tmp_path / "missing.mp4"), "nframes": 2})
    finally:
        vision.get_video_reader_backend.cache_clear()


@pytest.fixture
def librosa_boundary(monkeypatch):
    """Only replace the external dispatch/resampling boundary, not PCM work."""
    librosa = ModuleType("librosa")

    def native_load(*args, **kwargs):
        return "native loader", kwargs.get("sr")

    librosa.load = native_load
    librosa.core = SimpleNamespace(load=native_load, audio=SimpleNamespace(load=native_load))
    monkeypatch.setitem(sys.modules, "librosa", librosa)
    return librosa


class PcmReader:
    """The audioread interface: blocks of interleaved signed 16-bit PCM."""

    def __init__(self, frames, *, samplerate=8000, channels=1):
        self.frames = frames
        self.samplerate = samplerate
        self.channels = channels
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def read_data(self):
        yield from self.frames

    def __iter__(self):
        return self.read_data()


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_pcm_cumulative_budget_rejects_before_converting_overflow_block(
        monkeypatch, librosa_boundary, dtype):
    np = pytest.importorskip("numpy")
    first = np.array([0, 16384], dtype="<i2").tobytes()
    overflow = np.array([-16384, 32767], dtype="<i2").tobytes()
    itemsize = np.dtype(dtype).itemsize
    # One block holds two samples and fits; the second crosses the limit.
    monkeypatch.setattr(classroom_safety, "MAX_AUDIO_DECODED_BYTES", 3 * itemsize)
    frombuffer = np.frombuffer

    def convert_with_allocation_tripwire(block, *args, **kwargs):
        if block is overflow:
            pytest.fail("the overflow PCM block was converted before the byte guard")
        return frombuffer(block, *args, **kwargs)

    monkeypatch.setattr(np, "frombuffer", convert_with_allocation_tripwire)
    reader = PcmReader([first, overflow])
    extract_setup_function("_ensure_audio_decoder")()
    with pytest.raises(ValueError, match="오디오 디코딩"):
        librosa_boundary.load(reader, sr=None, mono=False, dtype=np.dtype(dtype))
    assert reader.closed  # decoder child/resources close even on rejection


def test_pcm_budget_counts_blocks_discarded_by_offset(monkeypatch, librosa_boundary):
    np = pytest.importorskip("numpy")
    block = np.array([0, 16384], dtype="<i2").tobytes()
    monkeypatch.setattr(classroom_safety, "MAX_AUDIO_DECODED_BYTES", 12)
    reader = PcmReader([block, block])
    extract_setup_function("_ensure_audio_decoder")()
    with pytest.raises(ValueError, match="오디오 디코딩"):
        librosa_boundary.load(reader, sr=None, mono=False, offset=10)
    assert reader.closed


def test_small_stereo_pcm_is_scaled_and_deinterleaved_correctly(librosa_boundary):
    np = pytest.importorskip("numpy")
    block = np.array([0, 16384, -16384, 0], dtype="<i2").tobytes()
    reader = PcmReader([block], channels=2)
    extract_setup_function("_ensure_audio_decoder")()
    decoded, rate = librosa_boundary.load(reader, sr=None, mono=False)
    assert rate == 8000
    assert decoded.dtype == np.float32
    assert decoded.tolist() == [[0.0, -0.5], [0.5, 0.0]]
    assert reader.closed


def test_previous_unbounded_audioread_shim_is_upgraded(monkeypatch, librosa_boundary):
    np = pytest.importorskip("numpy")
    # The previous notebook used this same marker before having a byte guard.
    librosa_boundary.load.__audioread_shim__ = True
    monkeypatch.setattr(classroom_safety, "MAX_AUDIO_DECODED_BYTES", 4)
    reader = PcmReader([np.array([0, 16384], dtype="<i2").tobytes()])
    extract_setup_function("_ensure_audio_decoder")()
    with pytest.raises(ValueError, match="오디오 디코딩"):
        librosa_boundary.load(reader, sr=None, mono=False)
    assert reader.closed
