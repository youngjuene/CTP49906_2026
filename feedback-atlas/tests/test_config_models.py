"""The model registry, and the secret with no default. CPU only, no weights.

Two defects. The first: a swap from EmbeddingGemma to an e5 model kept
EmbeddingGemma's prompt_name and dropped e5's "query: " prefix, so the server ran
happily and produced a map in a subtly worse vector space -- no exception, no
warning, and nothing on screen to suggest anything had changed.

The second is the one that would matter more: an admin access code with a
fallback value. A default admin code is a published admin code, and admin mode is
the only surface in this system that shows who wrote what.

Run:  python -m pytest feedback-atlas/tests/test_config_models.py
  or:  python feedback-atlas/tests/test_config_models.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> feedback-atlas/

from src.config import (  # noqa: E402
    DEFAULT_MODEL, MODEL_REGISTRY, AtlasConfig, EmbedderSpec, apply_prefix,
    cache_key, encode_kwargs, load_config, resolve_spec,
)

ENV = {"ATLAS_ADMIN_CODE": "swordfish", "ATLAS_ROSTER": "roster.csv"}


def test_every_registered_model_declares_exactly_one_input_convention():
    """prompt_name, query_prefix, or raw_input -- one of the three, never two and
    never none. 'None' is what a forgotten convention looks like, and a forgotten
    convention is invisible at runtime."""
    for name, spec in MODEL_REGISTRY.items():
        chosen = [bool(spec.prompt_name), bool(spec.query_prefix), spec.raw_input]
        assert sum(chosen) == 1, f"{name} declares {sum(chosen)} conventions"


def test_a_spec_with_two_conventions_or_none_refuses_to_exist():
    for kwargs in ({}, {"prompt_name": "C", "query_prefix": "query: "},
                   {"prompt_name": "C", "raw_input": True},
                   {"query_prefix": "q: ", "raw_input": True}):
        try:
            EmbedderSpec(model_id="m/x", dim=8, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{kwargs} should not have produced a spec")


def test_embeddinggemma_uses_its_clustering_prompt_and_no_manual_prefix():
    """The reason it is the default: a prompt trained for grouping texts against
    each other rather than retrieving documents for a query, which is what this
    app actually does."""
    spec = MODEL_REGISTRY["embeddinggemma"]
    assert spec.model_id == "google/embeddinggemma-300m"
    assert spec.prompt_name == "Clustering"
    assert spec.query_prefix == "" and spec.raw_input is False
    assert encode_kwargs(spec) == {"prompt_name": "Clustering"}
    assert apply_prefix(spec, ["좋았어요"]) == ["좋았어요"]


def test_e5_gets_its_prefix_and_no_prompt_name():
    """e5 does not raise without the prefix, it just gets quietly worse -- which is
    exactly the failure mode this registry exists to prevent."""
    spec = MODEL_REGISTRY["e5-small-ko"]
    assert spec.query_prefix == "query: " and spec.prompt_name is None
    assert encode_kwargs(spec) == {}
    assert apply_prefix(spec, ["좋았어요"]) == ["query: 좋았어요"]


def test_applying_a_prefix_twice_does_not_double_it():
    """Cached text is re-encoded on a model swap; a second pass must be a no-op."""
    spec = MODEL_REGISTRY["e5-small-ko"]
    once = apply_prefix(spec, ["좋았어요"])
    assert apply_prefix(spec, once) == once


def test_kure_declares_raw_input_and_is_left_untouched():
    spec = MODEL_REGISTRY["kure"]
    assert spec.raw_input is True
    assert apply_prefix(spec, ["좋았어요"]) == ["좋았어요"]
    assert encode_kwargs(spec) == {}


def test_each_model_declares_its_dimension_so_a_mismatch_is_caught_early():
    """A wrong dim surfaces at the cached-vector read with the model name in the
    message, rather than deep inside numpy at vstack time."""
    assert MODEL_REGISTRY["embeddinggemma"].dim == 768
    assert MODEL_REGISTRY["kure"].dim == 1024
    assert MODEL_REGISTRY["e5-small-ko"].dim == 384


def test_the_cache_key_separates_models_conventions_and_dimensions():
    keys = {cache_key(s) for s in MODEL_REGISTRY.values()}
    assert len(keys) == len(MODEL_REGISTRY)
    same_weights_different_prompt = {
        cache_key(EmbedderSpec(model_id="m/x", dim=768, prompt_name="Clustering")),
        cache_key(EmbedderSpec(model_id="m/x", dim=768, query_prefix="query: ")),
        cache_key(EmbedderSpec(model_id="m/x", dim=768, raw_input=True)),
    }
    assert len(same_weights_different_prompt) == 3


def test_the_cache_key_ignores_library_versions():
    """Deliberate: including them would make a transformers patch bump discard
    every cached vector in the class for no benefit. src/store.py records the
    resolved versions in `meta` and warns instead."""
    key = cache_key(MODEL_REGISTRY["embeddinggemma"])
    assert "transformers" not in key and "sentence" not in key
    assert key.startswith("google/embeddinggemma-300m#")


def test_a_model_can_be_named_by_registry_key_or_by_full_id():
    assert resolve_spec("embeddinggemma") is resolve_spec("google/embeddinggemma-300m")


def test_an_unknown_model_name_is_a_hard_error_that_lists_the_known_ones():
    """Never a silent fallback: falling back would hand the class a map built by
    a model nobody chose."""
    try:
        resolve_spec("gpt-embeddings-9")
    except KeyError as e:
        assert "embeddinggemma" in str(e) and "kure" in str(e)
    else:
        raise AssertionError("an unknown model must not resolve")


def test_load_config_refuses_to_run_without_an_admin_code():
    """The defect this file's second half is named for."""
    try:
        load_config({"ATLAS_ROSTER": "roster.csv"})
    except ValueError as e:
        assert "ATLAS_ADMIN_CODE" in str(e)
    else:
        raise AssertionError("a missing admin code must not fall back to a default")

    for blank in ("", "   ", "\t"):
        try:
            load_config({"ATLAS_ADMIN_CODE": blank, "ATLAS_ROSTER": "r.csv"})
        except ValueError:
            pass
        else:
            raise AssertionError(f"admin code {blank!r} must not be accepted")


def test_load_config_refuses_to_run_without_a_roster():
    """With strict registration and no roster, nobody can get in at all."""
    try:
        load_config({"ATLAS_ADMIN_CODE": "x"})
    except ValueError as e:
        assert "ATLAS_ROSTER" in str(e)
    else:
        raise AssertionError("a missing roster path must not be tolerated")


def test_an_unknown_model_in_the_environment_fails_at_startup_not_at_model_load():
    """Ten milliseconds of failure beats failing after a gigabyte of download."""
    try:
        load_config({**ENV, "ATLAS_EMBEDDING_MODEL": "nope"})
    except KeyError:
        pass
    else:
        raise AssertionError("an unknown model name must fail at config load")


def test_a_default_config_uses_the_documented_defaults():
    cfg = load_config(ENV)
    assert isinstance(cfg, AtlasConfig)
    assert cfg.embedding_model == DEFAULT_MODEL == "embeddinggemma"
    assert cfg.admin_code == "swordfish"
    assert cfg.pca_umap_threshold == 80
    assert cfg.debounce_s == 0.4 and cfg.max_debounce_s == 2.0
    assert cfg.fake_embedder is False


def test_the_fake_embedder_requires_an_explicit_opt_in():
    """It must never be reachable by a typo in a model name: a silently fake map
    is worse than no map."""
    assert load_config({**ENV, "ATLAS_UNSAFE_FAKE_EMBEDDER": "true"}).fake_embedder is False
    assert load_config({**ENV, "ATLAS_UNSAFE_FAKE_EMBEDDER": "1"}).fake_embedder is True


def test_an_out_of_range_default_week_is_refused_rather_than_clamped():
    """Found by QA, not by a user, which is the only reason it is cheap.

    ATLAS_DEFAULT_WEEK=7 was accepted and stamped on every submission that
    omitted a week. Those opinions land on the map and no week checkbox can ever
    show them -- present in the data, invisible on screen, and nothing anywhere
    reports a problem.
    """
    for bad in ("0", "5", "7", "-1"):
        try:
            load_config({**ENV, "ATLAS_DEFAULT_WEEK": bad})
        except ValueError as e:
            assert "ATLAS_DEFAULT_WEEK" in str(e)
        else:
            raise AssertionError(f"ATLAS_DEFAULT_WEEK={bad} must not be accepted")
    assert load_config({**ENV, "ATLAS_DEFAULT_WEEK": "3"}).default_week == 3


def test_a_non_numeric_setting_names_itself_in_the_error():
    """Bare int() says "invalid literal for int() with base 10: 'two'", which does
    not say which setting is wrong. This runs at startup, in a terminal, minutes
    before a class."""
    for var in ("ATLAS_DEFAULT_WEEK", "ATLAS_PCA_UMAP_THRESHOLD"):
        try:
            load_config({**ENV, var: "two"})
        except ValueError as e:
            assert var in str(e) and "two" in str(e), str(e)
        else:
            raise AssertionError(f"{var}='two' must not be accepted")


def test_a_threshold_below_three_is_refused():
    """UMAP cannot lay out fewer than three points; the cold-start path exists
    for exactly that reason, and a threshold under it would bypass it."""
    try:
        load_config({**ENV, "ATLAS_PCA_UMAP_THRESHOLD": "1"})
    except ValueError as e:
        assert "ATLAS_PCA_UMAP_THRESHOLD" in str(e)
    else:
        raise AssertionError("a threshold below 3 must not be accepted")


def test_an_empty_optional_setting_falls_back_to_its_default():
    """An unset variable exported as "" is the normal shape of a half-filled .env
    file and must not be a startup failure."""
    cfg = load_config({**ENV, "ATLAS_DEFAULT_WEEK": "", "ATLAS_PCA_UMAP_THRESHOLD": "  "})
    assert cfg.default_week == 1 and cfg.pca_umap_threshold == 80


def test_the_pca_umap_threshold_is_configurable():
    """Crossing it reorganises the map once, so an instructor needs to be able to
    cross it deliberately between sessions rather than mid-presentation."""
    assert load_config({**ENV, "ATLAS_PCA_UMAP_THRESHOLD": "200"}).pca_umap_threshold == 200


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
