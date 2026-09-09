"""Text -> vectors. The only module that touches model weights.

`sentence_transformers` is imported inside __init__, never at module scope, and
tests/test_embedder_contract.py asserts that in a subprocess. That single rule is
what keeps the test suite CPU-only and offline: importing this module must not
pull in torch, and must not be able to start a 1.2 GB download.

The other thing this module exists to prevent is silent. EmbeddingGemma is used
with a *clustering* prompt, which is the whole reason it was chosen -- opinions are
grouped against each other, not retrieved against a query. sentence-transformers
below 5.0 accepts `prompt_name=` and ignores it: no exception, no warning, and a
perfectly plausible map built in the wrong vector space. A version pin alone
cannot catch that, because the pin can be edited and the failure is invisible. So
the prompt is checked against the loaded model's own prompt table, after load.
"""

from collections.abc import Sequence

import numpy as np
import threading

from src.config import AtlasConfig, EmbedderSpec, apply_prefix, encode_kwargs, resolve_spec

# Floors that matter, with the reason each one is a floor.
_MIN_TRANSFORMERS = (4, 56, 2)      # below this, EmbeddingGemma's head is unknown
_MIN_SENTENCE_TRANSFORMERS = (5, 0)  # below this, prompt_name is silently ignored


class SharedEmbedder:
    """Serialize inference when classroom and demo share one loaded model."""

    def __init__(self, embedder):
        self._embedder = embedder
        self._lock = threading.Lock()
        self.dim = embedder.dim
        self.spec = embedder.spec
        self.model_id = embedder.model_id

    def encode(self, texts):
        with self._lock:
            return self._embedder.encode(texts)


def _version_tuple(raw: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in raw.split(".")[:3]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _require(package: str, minimum: tuple[int, ...], why: str) -> None:
    from importlib.metadata import PackageNotFoundError, version
    try:
        found = version(package)
    except PackageNotFoundError:
        raise RuntimeError(f"{package} is not installed; needed to embed text") from None
    if _version_tuple(found) < minimum:
        raise RuntimeError(
            f"{package} {found} is older than "
            f"{'.'.join(str(p) for p in minimum)}: {why}")


class HashEmbedder:
    """A deterministic, weightless stand-in. Development and tests only.

    Reachable only through ATLAS_UNSAFE_FAKE_EMBEDDER=1, never by mistyping a
    model name -- a model name typo raises. The distinction matters: a silently
    fake map is worse than no map, because it looks exactly like a real one and
    the class would discuss it.
    """

    def __init__(self, dim: int = 768, model_id: str = "hash-test"):
        self.dim = dim
        self.model_id = model_id
        self.spec = EmbedderSpec(model_id=model_id, dim=dim, raw_input=True)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        import hashlib
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = hashlib.sha256(text.encode("utf-8")).digest()[:8]
            rng = np.random.default_rng(int.from_bytes(seed, "big"))
            v = rng.normal(size=self.dim)
            out[i] = v / (np.linalg.norm(v) + 1e-12)
        return out


class SentenceTransformerEmbedder:
    def __init__(self, spec: EmbedderSpec, device: str = "cpu"):
        _require("transformers", _MIN_TRANSFORMERS,
                 "EmbeddingGemma's model head is unknown to older versions and the "
                 "weights load and then fail with KeyError: 'gemma3_text'")
        _require("sentence-transformers", _MIN_SENTENCE_TRANSFORMERS,
                 "older versions accept prompt_name= and silently ignore it, which "
                 "produces a valid-looking map in the wrong vector space")

        from sentence_transformers import SentenceTransformer

        self.spec = spec
        self.dim = spec.dim
        self.model_id = spec.model_id
        try:
            self._model = SentenceTransformer(spec.model_id, device=device)
        except Exception as exc:            # noqa: BLE001 -- re-raised with guidance
            raise RuntimeError(_load_help(spec, exc)) from exc

        if spec.prompt_name:
            prompts = getattr(self._model, "prompts", None) or {}
            if spec.prompt_name not in prompts:
                # The check the version pin cannot make for us.
                raise RuntimeError(
                    f"{spec.model_id} was loaded but does not expose a "
                    f"{spec.prompt_name!r} prompt (it has: {sorted(prompts) or 'none'}). "
                    "Embedding would silently fall back to no prompt at all, so the "
                    "map would be built in a different vector space than intended.")

        reported = getattr(self._model, "get_sentence_embedding_dimension", lambda: None)()
        if reported is not None and int(reported) != spec.dim:
            raise RuntimeError(
                f"{spec.model_id} returns {reported}-dimensional vectors but the "
                f"registry declares {spec.dim}; fix src/config.py before the "
                "mismatch reaches the embedding cache")

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        prepared = apply_prefix(self.spec, list(texts))
        vectors = self._model.encode(
            prepared,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            **encode_kwargs(self.spec),
        )
        out = np.asarray(vectors, dtype=np.float32)
        if out.shape != (len(texts), self.dim):
            # A transposed or truncated batch would scramble the entire map while
            # every individual vector still looked fine.
            raise RuntimeError(
                f"expected {(len(texts), self.dim)} embeddings, got {out.shape}")
        return out


def _load_help(spec: EmbedderSpec, exc: Exception) -> str:
    """Turn a 401/403 into instructions somebody can act on five minutes before class."""
    text = f"{type(exc).__name__}: {exc}"
    gated = any(k in text.lower() for k in
                ("gated", "401", "403", "unauthorized", "awaiting a review",
                 "access to model", "restricted"))
    if not gated:
        return f"could not load {spec.model_id} -- {text}"
    import os
    shadowing = (
        "\n  NOTE: HF_TOKEN is set in this environment, and it takes precedence "
        "over\n        whatever `huggingface-cli login` stored. An invalid or "
        "revoked HF_TOKEN\n        therefore shadows a perfectly good stored "
        "login, and the hub reports\n        that as a gated-repo error -- which "
        "sends you to the licence page when\n        the licence was never the "
        "problem. Try: env -u HF_TOKEN <command>\n"
        if os.environ.get("HF_TOKEN") else "")
    return (
        f"could not load {spec.model_id}: it is a gated repository.\n"
        f"  1. Accept the licence at https://huggingface.co/{spec.model_id}\n"
        "  2. Log in as that same account: `huggingface-cli login` (or set HF_TOKEN).\n"
        "     A token from a different account than the one that accepted the "
        "licence fails exactly like no token at all.\n"
        "  3. Or run ungated: ATLAS_EMBEDDING_MODEL=e5-small-ko\n"
        f"{shadowing}"
        f"original error -- {text}")


def build_embedder(cfg: AtlasConfig):
    if cfg.fake_embedder:
        spec = resolve_spec(cfg.embedding_model)
        return HashEmbedder(dim=spec.dim, model_id=f"fake:{spec.model_id}")
    return SentenceTransformerEmbedder(resolve_spec(cfg.embedding_model), device=cfg.device)
