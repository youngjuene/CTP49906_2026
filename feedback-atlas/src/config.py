"""Model registry and server configuration.

PRD 8 requires the embedding model be replaceable if Korean quality disappoints.
That requirement is the whole reason this file exists: the *prompt convention* is
data, not code. Each model wants its input prepared differently and gets it wrong
silently rather than loudly when you don't --

  embeddinggemma-300m  wants prompt_name="Clustering", which sentence-transformers
                       expands to "task: clustering | query: ". This is the one
                       model here with a prompt trained for grouping texts against
                       each other rather than retrieving documents for a query,
                       which is exactly what this app does.
  multilingual-e5-*    wants a literal "query: " prefix on every text. Omitting it
                       does not raise; it measurably degrades the embedding.
  KURE-v1              wants neither.

Hardcoding either convention would make a model swap a silent quality regression,
so a spec declares one or the other and never both.
"""

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.deltas import MOVE_EPSILON
from src.models import WEEKS


@dataclass(frozen=True)
class EmbedderSpec:
    """How to prepare text for one model, and what comes back."""

    model_id: str
    dim: int
    # Exactly one of the three must be chosen. prompt_name is looked up in the
    # model's own prompt table by sentence-transformers; query_prefix is pasted
    # on by us; raw_input says this model wants the text untouched.
    #
    # raw_input has to be stated rather than inferred from "neither of the other
    # two is set", because that is also what a forgotten convention looks like,
    # and a forgotten convention is invisible -- no exception, no warning, just a
    # worse map.
    prompt_name: str | None = None
    query_prefix: str = ""
    raw_input: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        chosen = [bool(self.prompt_name), bool(self.query_prefix), self.raw_input]
        if sum(chosen) != 1:
            raise ValueError(
                f"{self.model_id}: choose exactly one of prompt_name, query_prefix "
                "or raw_input -- a model preparing its input two ways, or no way "
                "at all, is a silent quality bug rather than a crash"
            )


MODEL_REGISTRY: dict[str, EmbedderSpec] = {
    "embeddinggemma": EmbedderSpec(
        model_id="google/embeddinggemma-300m",
        dim=768,
        prompt_name="Clustering",
        note="Default. Gated: needs an accepted Gemma licence and a matching HF token.",
    ),
    "kure": EmbedderSpec(
        model_id="nlpai-lab/KURE-v1",
        dim=1024,
        raw_input=True,
        note="MIT, ungated, near the top of the Korean retrieval leaderboard.",
    ),
    "e5-small-ko": EmbedderSpec(
        model_id="dragonkue/multilingual-e5-small-ko-v2",
        dim=384,
        query_prefix="query: ",
        note="Apache-2.0, ungated, 384-dim. The fast fallback when the Gemma gate blocks a clone.",
    ),
}

DEFAULT_MODEL = "embeddinggemma"


def resolve_spec(name: str) -> EmbedderSpec:
    """Look up a spec by registry key or by full model id."""
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    for spec in MODEL_REGISTRY.values():
        if spec.model_id == name:
            return spec
    raise KeyError(
        f"unknown embedding model {name!r}; known: "
        + ", ".join(sorted(MODEL_REGISTRY)) + ". "
        "An unknown name is a hard error on purpose -- falling back to a default "
        "would hand you a map built by a model you did not choose."
    )


def encode_kwargs(spec: EmbedderSpec) -> dict:
    """Extra kwargs for SentenceTransformer.encode for this spec."""
    return {"prompt_name": spec.prompt_name} if spec.prompt_name else {}


def apply_prefix(spec: EmbedderSpec, texts: Sequence[str]) -> list[str]:
    """Paste on the literal prefix this spec wants, if any.

    Idempotent: a text that already carries the prefix is left alone, so a
    re-encode of cached text cannot end up with "query: query: ...".
    """
    if not spec.query_prefix:
        return list(texts)
    prefix = spec.query_prefix
    return [t if t.startswith(prefix) else prefix + t for t in texts]


def cache_key(spec: EmbedderSpec) -> str:
    """Identity of the vector space these embeddings live in.

    Not just the model id: the same weights under a different prompt convention
    produce a different, incomparable space. Storing this per row is what makes a
    model swap a cache miss that re-embeds, instead of a silent mix of 384- and
    768-dimensional vectors that only fails later at vstack.

    Deliberately excludes library versions. A transformers patch bump would
    invalidate every cached vector in the class for no benefit; src/store.py
    records the resolved versions in `meta` and warns instead.
    """
    convention = spec.prompt_name or spec.query_prefix.strip() or "raw"
    return f"{spec.model_id}#{convention}#{spec.dim}"


@dataclass(frozen=True)
class AtlasConfig:
    admin_code: str
    roster_path: str
    db_path: str = "atlas.db"
    embedding_model: str = DEFAULT_MODEL
    device: str = "cpu"
    default_week: int = 1
    auto_week: bool = True
    # Below this many opinions, project with PCA. UMAP's spectral init is
    # unusable on tiny inputs and its layout is dominated by initialisation
    # randomness up to ~50. Configurable because crossing it *reorganises the
    # map once* -- see README: cross it between sessions, not mid-presentation.
    pca_umap_threshold: int = 80
    n_neighbors: int = 15
    # How many neighbours ride on each point in the payload. This is
    # embedding-atlas's `neighbors` column, which its viewer reads directly rather
    # than asking for; there is no request to answer, so the number is a bandwidth
    # decision. Eight fills the panel and costs roughly 200 bytes a point. The
    # graph itself is still built at n_neighbors, because that is what UMAP was
    # fitted on and shrinking it would change the layout.
    neighbors_k: int = 8
    # Trailing-edge debounce, with a cap. Without the cap a continuous stream --
    # a whole class typing at once -- never fires at all.
    debounce_s: float = 0.4
    max_debounce_s: float = 2.0
    max_text_chars: int = 1000
    move_epsilon: float = MOVE_EPSILON
    fake_embedder: bool = False
    warm_umap: bool = True
    # Whether to stand up the DuckDB relation Embedding Atlas's viewer queries.
    # On by default and switchable off, because the viewer is the heavier of the
    # two front ends in every sense -- it needs duckdb and pyarrow on the server, a
    # vendored bundle on the client, and WebGPU in the browser -- and a room where
    # any of those is missing should still get its map.
    enable_viewer: bool = True
    access_codes_path: str = ".run/access-codes.json"
    require_https: bool = True
    enable_demo: bool = False
    demo_db_path: str = ".run/demo/atlas.db"
    is_demo: bool = False


def _int_env(env: Mapping[str, str], name: str, default: int) -> int:
    """int() with an error that names the variable.

    Bare int() raises "invalid literal for int() with base 10: 'two'", which does
    not say which setting is wrong -- and this runs at startup, in a terminal, five
    minutes before a class.
    """
    raw = env.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a whole number") from None


def load_config(env: Mapping[str, str] | None = None) -> AtlasConfig:
    """Build config from the environment. Raises rather than defaulting secrets."""
    env = os.environ if env is None else env

    admin_code = (env.get("ATLAS_ADMIN_CODE") or "").strip()
    if not admin_code:
        raise ValueError(
            "ATLAS_ADMIN_CODE is not set. There is no default: a fallback admin "
            "code is a published admin code, and admin mode is the only surface "
            "that shows who wrote what (PRD 5.6)."
        )

    roster_path = (env.get("ATLAS_ROSTER") or "").strip()
    if not roster_path:
        raise ValueError(
            "ATLAS_ROSTER is not set. Registration is strict: an unlisted id "
            "cannot get in, so there is nothing sensible to do without a roster."
        )

    model = env.get("ATLAS_EMBEDDING_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    resolve_spec(model)  # fail here, not at model-load time

    default_week = _int_env(env, "ATLAS_DEFAULT_WEEK", 1)
    if default_week not in WEEKS:
        # Not clamped. An out-of-range default silently stamps every submission
        # that omits a week with a value no week checkbox can select, so those
        # opinions are on the map and invisible -- the worst of both.
        raise ValueError(
            f"ATLAS_DEFAULT_WEEK={default_week} is not one of {WEEKS}; every "
            "submission that omits a week would be filed under a week the filter "
            "cannot show")

    threshold = _int_env(env, "ATLAS_PCA_UMAP_THRESHOLD", 80)
    if threshold < 3:
        raise ValueError(
            f"ATLAS_PCA_UMAP_THRESHOLD={threshold} is below 3; UMAP cannot lay out "
            "fewer than three points and the cold-start path exists for that reason")

    return AtlasConfig(
        admin_code=admin_code,
        roster_path=roster_path,
        db_path=env.get("ATLAS_DB", "atlas.db"),
        embedding_model=model,
        device=env.get("ATLAS_DEVICE", "cpu"),
        default_week=default_week,
        auto_week=env.get("ATLAS_AUTO_WEEK", "1") != "0",
        pca_umap_threshold=threshold,
        fake_embedder=env.get("ATLAS_UNSAFE_FAKE_EMBEDDER", "") == "1",
        warm_umap=env.get("ATLAS_SKIP_WARMUP", "") != "1",
        enable_viewer=env.get("ATLAS_DISABLE_VIEWER", "") != "1",
        access_codes_path=env.get("ATLAS_ACCESS_CODES", ".run/access-codes.json"),
        require_https=env.get("ATLAS_ALLOW_INSECURE_HTTP", "") != "1",
        enable_demo=env.get("ATLAS_ENABLE_DEMO", "") == "1",
        demo_db_path=env.get("ATLAS_DEMO_DB", ".run/demo/atlas.db"),
    )
