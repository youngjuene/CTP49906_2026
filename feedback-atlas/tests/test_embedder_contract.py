"""The embedder's contract, and the import rule that keeps this suite offline.

The defect: the CPU test suite started downloading 1.2 GB of gated weights,
because a convenience import moved to module scope and `import src.embedder`
began pulling in sentence-transformers, which pulls in torch. Nothing failed --
CI just got slow, then flaky, then needed a network.

The first test below runs in a subprocess for that reason: once torch is in
sys.modules from any other test, an in-process check can no longer tell whether
this module was the one that imported it.

Run:  python -m pytest feedback-atlas/tests/test_embedder_contract.py
  or:  python feedback-atlas/tests/test_embedder_contract.py
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> feedback-atlas/

from src.config import AtlasConfig, EmbedderSpec  # noqa: E402
from src.embedder import HashEmbedder, _load_help, _version_tuple, build_embedder  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_importing_the_embedder_does_not_drag_in_torch():
    """The rule that keeps the suite CPU-only, offline, and fast."""
    code = (
        f"import sys; sys.path.insert(0, {str(ROOT)!r});"
        "import src.embedder;"
        "print([m for m in sys.modules if m.split('.')[0] in "
        "('torch','transformers','sentence_transformers')])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", (
        f"src.embedder imported heavy modules at module scope: {out.stdout.strip()}")


def test_the_hash_embedder_satisfies_the_whole_contract():
    """Shape, dtype, unit norm, determinism. The projection assumes all four."""
    e = HashEmbedder(dim=32)
    a = e.encode(["좋았어요", "싫었어요", "무서웠어요"])
    assert a.shape == (3, 32) and a.dtype == np.float32
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(a, e.encode(["좋았어요", "싫었어요", "무서웠어요"]))


def test_encode_preserves_batch_order():
    """A transposed or reordered batch scrambles the entire map while every
    individual vector still looks perfectly valid."""
    e = HashEmbedder(dim=16)
    texts = [f"의견 {i}" for i in range(12)]
    batch = e.encode(texts)
    for i, text in enumerate(texts):
        assert np.array_equal(batch[i], e.encode([text])[0]), f"row {i} is not {text!r}"


def test_encoding_nothing_returns_an_empty_matrix_of_the_right_width():
    """The backfill path calls this with an empty list whenever the cache is warm,
    and vstack needs the width even when there are no rows."""
    out = HashEmbedder(dim=24).encode([])
    assert out.shape == (0, 24) and out.dtype == np.float32


def test_the_fake_embedder_is_only_reachable_through_the_explicit_opt_in():
    """Never by a typo in a model name. A silently fake map is worse than no map,
    because it looks real and the class would discuss it."""
    base = dict(admin_code="x", roster_path="r.csv")
    fake = build_embedder(AtlasConfig(**base, fake_embedder=True))
    assert isinstance(fake, HashEmbedder)
    assert fake.model_id.startswith("fake:"), \
        "a fake embedder must be identifiable in /healthz and in the cache key"
    assert AtlasConfig(**base).fake_embedder is False


def test_a_gated_repository_error_names_all_three_ways_out():
    """A 401 five minutes before class needs instructions, not a stack trace.

    The third fix matters most and is the least guessable: a token belonging to a
    different account than the one that accepted the licence fails identically to
    having no token at all.
    """
    spec = EmbedderSpec(model_id="google/embeddinggemma-300m", dim=768,
                        prompt_name="Clustering")
    msg = _load_help(spec, RuntimeError(
        "401 Client Error. Access to model google/embeddinggemma-300m is restricted."))
    assert "huggingface.co/google/embeddinggemma-300m" in msg   # accept the licence
    assert "HF_TOKEN" in msg or "huggingface-cli login" in msg  # authenticate
    assert "same account" in msg                               # the non-obvious one
    assert "ATLAS_EMBEDDING_MODEL=e5-small-ko" in msg          # the ungated way out


def test_an_ordinary_load_failure_is_not_dressed_up_as_a_licence_problem():
    """Sending somebody to a licence page over a typo'd model name wastes the one
    thing they do not have."""
    spec = EmbedderSpec(model_id="some/model", dim=8, raw_input=True)
    msg = _load_help(spec, OSError("No such file or directory"))
    assert "licence" not in msg.lower() and "gated" not in msg.lower()
    assert "some/model" in msg


def test_version_floors_compare_as_numbers_not_strings():
    """'4.9' must not read as newer than '4.56.2'."""
    assert _version_tuple("4.56.2") == (4, 56, 2)
    assert _version_tuple("5.16.1") > (4, 56, 2)
    assert _version_tuple("4.9.0") < (4, 56, 2)
    assert _version_tuple("6.0.1rc1") >= (5, 0)   # a prerelease suffix must not crash


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    fails = 0
    for fn in fns:
        try:
            fn()
            print("PASS", fn.__name__)
        except Exception as e:  # noqa: BLE001
            fails += 1
            print("FAIL", fn.__name__, "->", type(e).__name__, e)
    print(f"\n{len(fns) - fails} passed, {fails} failed")
    sys.exit(1 if fails else 0)
