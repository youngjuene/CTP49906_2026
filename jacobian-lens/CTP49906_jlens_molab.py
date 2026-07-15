# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "huggingface-hub",
#     # jlens is not on PyPI; install it from the repository subdirectory.
#     # If youngjuene/CTP49906_2026 is private, connect GitHub in molab, or
#     # swap in the upstream: "jlens @ git+https://github.com/anthropics/jacobian-lens"
#     "jlens @ git+https://github.com/youngjuene/CTP49906_2026.git#subdirectory=jacobian-lens",
# ]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Jacobian lens — molab demo

    Inspect a released Jacobian lens on Qwen3.5-4B: compare it with the vanilla
    logit lens, then render the interactive layer × position slice. Pick a
    different example from the dropdown and the slice recomputes reactively.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Running in molab

    - **GPU:** click the notebook-specs button in the header and attach a GPU.
      This notebook uses `cuda:0` (molab exposes a single GPU) — no device pinning.
    - **Dependencies:** the setup cell below clones this repo, imports `jlens`
      from source, and pip-installs `transformers` into the kernel. Torch is left
      untouched so molab's GPU-matched build (Blackwell needs a cu128 wheel) is
      preserved. First run pulls a few GB of model weights.
    - The **local 100-prompt fit** (Section "Local lens") lives on your workstation,
      so it reports *not available* here unless you upload the artifact.
    """)
    return


@app.cell
def _(mo):
    import importlib.metadata
    import importlib.util
    import subprocess
    import sys
    from pathlib import Path

    def _ver_tuple(v):
        out = []
        for part in v.split(".")[:3]:
            digits = "".join(ch for ch in part if ch.isdigit())
            out.append(int(digits) if digits else 0)
        return tuple(out)

    def _ensure_packages(specs):
        # specs: (import_name, dist_name, min_version_or_None, pip_spec).
        # molab does not install the `# /// script` block into the running
        # kernel, so pip-install anything missing (or too old) at runtime.
        # torch is intentionally NEVER touched: molab's base image ships a build
        # matched to its GPU (Blackwell / sm_120 needs a cu128 wheel), and
        # pinning torch here would replace it with an unrunnable one.
        to_install = []
        for import_name, dist_name, min_version, pip_spec in specs:
            if importlib.util.find_spec(import_name) is None:
                to_install.append(pip_spec)
                continue
            if min_version is not None:
                try:
                    have = importlib.metadata.version(dist_name)
                except importlib.metadata.PackageNotFoundError:
                    to_install.append(pip_spec)
                    continue
                if _ver_tuple(have) < _ver_tuple(min_version):
                    to_install.append(pip_spec)
        if to_install:
            with mo.status.spinner(title=f"Installing {', '.join(to_install)}…"):
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", *to_install], check=True
                )

    _ensure_packages([
        # jlens needs transformers>=5.5, but Qwen3.5-4B's architecture
        # (`qwen3_5`) only became natively supported around 5.13, so floor at the
        # locally-validated version to guarantee the model itself loads.
        ("transformers", "transformers", "5.13", "transformers>=5.13,<6"),
        ("huggingface_hub", "huggingface_hub", None, "huggingface_hub"),
        ("numpy", "numpy", None, "numpy"),
    ])

    # jlens is not on PyPI; clone the repo and import it from source. This avoids
    # pip's resolver rebuilding/replacing torch on Blackwell. The package lives
    # in the `jacobian-lens/` subdirectory of the repo.
    REPO_DIR = Path("CTP49906_2026").resolve()
    if not REPO_DIR.exists():
        with mo.status.spinner(title="Cloning CTP49906_2026 (jlens source + assets)…"):
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/youngjuene/CTP49906_2026.git", str(REPO_DIR)],
                check=True,
            )
    JLENS_DIR = REPO_DIR / "jacobian-lens"
    assert JLENS_DIR.is_dir(), f"expected jlens dir not found: {JLENS_DIR}"
    if str(JLENS_DIR) not in sys.path:
        sys.path.insert(0, str(JLENS_DIR))
    print("jlens dir:", JLENS_DIR)
    return (JLENS_DIR,)


@app.cell
def _(JLENS_DIR):
    import os
    from pathlib import Path

    _ = JLENS_DIR  # ensure the clone / sys.path / deps cell ran first
    import jlens
    import torch

    jlens.configure_logging()
    os.environ.setdefault("HF_HOME", "/tmp/hf-cache")

    assert torch.cuda.is_available(), (
        "No GPU visible. In molab, attach a GPU via the notebook-specs button in the header."
    )
    device = torch.device("cuda:0")
    OUTPUT_ROOT = Path("artifacts/jlens-molab")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    _free, _total = torch.cuda.mem_get_info(0)
    print(f"torch={torch.__version__}, CUDA={torch.version.cuda}, GPU={torch.cuda.get_device_name(0)}")
    print(f"VRAM free/total GiB={_free / 2**30:.1f}/{_total / 2**30:.1f}")
    return OUTPUT_ROOT, device, jlens, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Configure the released model and lens

    The released lens is fitted for this exact model architecture; a lens for
    another model must be fitted separately.
    """)
    return


@app.cell
def _(OUTPUT_ROOT):
    MODEL_NAME = "Qwen/Qwen3.5-4B"
    LENS_REPO = "neuronpedia/jacobian-lens"
    LENS_REVISION = "qwen-n1000"
    LENS_FILE = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"
    LOCAL_FITTED_LENS = OUTPUT_ROOT / "jacobian_lens.pt"
    print({"model": MODEL_NAME, "lens_file": LENS_FILE})
    return LENS_FILE, LENS_REPO, LENS_REVISION, LOCAL_FITTED_LENS, MODEL_NAME


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Load the model
    """)
    return


@app.cell
def _(MODEL_NAME, device, jlens, mo, torch):
    import transformers

    with mo.status.spinner(title="Loading Qwen3.5-4B (first run downloads several GB)…"):
        torch.cuda.reset_peak_memory_stats(device)
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.bfloat16
        ).to(device)
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
        model = jlens.from_hf(hf_model, tokenizer)

    print(f"model-load peak GiB={torch.cuda.max_memory_allocated(device) / 2**30:.2f}")
    model
    return model, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Load the published Jacobian lens

    A pre-fitted artifact for the model above — nothing is trained here.
    """)
    return


@app.cell
def _(LENS_FILE, LENS_REPO, LENS_REVISION, jlens, mo):
    with mo.status.spinner(title="Downloading the published Jacobian lens (first run only)…"):
        lens = jlens.JacobianLens.from_pretrained(
            LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION
        )
    lens
    return (lens,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Compare J-lens and vanilla logit lens

    Both read out the next token from an intermediate layer. The logit lens
    decodes it directly; the J-lens estimates the remaining computation first.
    """)
    return


@app.cell
def _(lens, mo, model, tokenizer):
    prompt_compare = "Fact: The currency used in the country shaped like a boot is"
    # Pick four representative layers FROM the lens's fitted set: the J-lens path
    # requires layers ⊆ source_layers, so deriving them from model.n_layers
    # fractions would raise if the released lens skipped any of them.
    _src = lens.source_layers
    layers = sorted({_src[len(_src) // 4], _src[len(_src) // 2], _src[len(_src) * 3 // 4], _src[-1]})
    jlens_logits, model_logits, _ = lens.apply(model, prompt_compare, layers=layers, positions=[-2])
    logit_lens_out, _, _ = lens.apply(
        model, prompt_compare, layers=layers, positions=[-2], use_jacobian=False
    )

    def _top5(logits):
        return "`" + ", ".join(repr(tokenizer.decode([t])) for t in logits.topk(5).indices) + "`"

    _rows = "\n".join(
        f"| L{l} | {_top5(logit_lens_out[l][0])} | {_top5(jlens_logits[l][0])} |" for l in layers
    )
    mo.md(
        f"**Prompt:** {prompt_compare!r}\n\n"
        "| layer | logit-lens top-5 | J-lens top-5 |\n|---|---|---|\n"
        + _rows
        + f"\n\n**model (final) top-5:** {_top5(model_logits[0])}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Interactive slice

    The slice tracks candidate next tokens across layers and positions. Change
    the example below to recompute it.
    """)
    return


@app.cell
def _(mo):
    from jlens.examples import EXAMPLES

    example_choice = mo.ui.dropdown(
        options={e.section: e.slug for e in EXAMPLES},
        value=next(e.section for e in EXAMPLES if e.slug == "multihop"),
        label="Example prompt",
    )
    example_choice
    return EXAMPLES, example_choice


@app.cell
def _(EXAMPLES, JLENS_DIR, example_choice, lens, mo, model, tokenizer):
    import gzip
    import json

    from jlens.examples import resolve_prompt
    from jlens.vis import build_page, compute_slice

    _example = next(e for e in EXAMPLES if e.slug == example_choice.value)
    _prompt = resolve_prompt(_example, tokenizer)

    # The English-gloss file ships in the repo (assets/), not the installed
    # package, so it is optional; the slice renders fine without it.
    _gloss = None
    _gloss_path = JLENS_DIR / "assets" / "qwen_gloss.json.gz"
    if _gloss_path.exists():
        _gloss = {int(k): v for k, v in json.load(gzip.open(_gloss_path)).items()}

    with mo.status.spinner(title=f"Computing slice for “{_example.section}”…"):
        _slice = compute_slice(model, lens, _prompt, layer_stride=2, mask_display=True)
        _page, _, _ = build_page(
            _slice,
            _prompt,
            title=_example.section,
            description=_example.description,
            alt_token=_gloss,
        )
    mo.iframe(_page, height="660px")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bundled examples
    """)
    return


@app.cell
def _(EXAMPLES, mo):
    mo.md(
        "Available example slugs:\n\n"
        + "\n".join(f"- `{_e.slug}` — {_e.section}" for _e in EXAMPLES)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Local 100-prompt lens

    The long fitting job on your workstation writes `jacobian_lens.pt`. Upload it
    to the path below to verify it here; otherwise this reports *not available*.
    """)
    return


@app.cell
def _(LOCAL_FITTED_LENS, jlens, mo, model):
    if LOCAL_FITTED_LENS.exists():
        _fitted = jlens.JacobianLens.load(str(LOCAL_FITTED_LENS))
        _msg = mo.md(
            f"✅ Loaded local lens: **{_fitted.n_prompts} prompts**, "
            f"d_model matches model: **{_fitted.d_model == model.d_model}**."
        )
    else:
        _msg = mo.md(
            f"ℹ️ No local fitted lens at `{LOCAL_FITTED_LENS}`. "
            "Sections 1–5 use the released lens."
        )
    _msg
    return


if __name__ == "__main__":
    app.run()
