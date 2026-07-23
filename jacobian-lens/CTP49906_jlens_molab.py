# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     # Only marimo is declared here. jlens (imported from the cloned source),
#     # transformers, huggingface-hub, and numpy are installed at runtime by the
#     # setup cell, which never touches torch so molab's GPU-matched build (Blackwell
#     # needs a cu128 wheel) is preserved. Don't add jlens or torch here: jlens
#     # depends on an unpinned torch transitively and would risk replacing that build.
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
    # Jacobian lens interpretability lab

    **The question:** when an intermediate residual stream decodes to a token,
    is that token evidence about the computation the model will eventually
    perform, or an artifact of reading the activation in the wrong basis?

    The guided demo compares a direct residual readout with a readout transported
    through a corpus-average Jacobian. The formulas and their limits appear beside
    the first measurement rather than being treated as hidden ground truth.

    **Learning route:** prepare the course reference model and lens, compare both
    readouts in a guided demo, test a falsifiable claim in the research playground,
    and finish by transferring the idea to a different architecture. The optional
    appendix lets you fit a smaller lens after the required class path.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare the experiment

    ### 1.1 Before you run

    - **GPU:** click the notebook-specs button in the header and attach a GPU.
      This notebook uses `cuda:0` (molab exposes a single GPU) — no device pinning.
    - **Dependencies:** the setup cell below clones this repo, imports `jlens`
      from source, and pip-installs `transformers` into the kernel. Torch is left
      untouched so molab's GPU-matched build (Blackwell needs a cu128 wheel) is
      preserved. First run pulls a few GB of model weights.
    - The guided demo and playground use the **course reference lens**, fitted on
      1,000 WikiText prompts. The advanced appendix runs only when clicked; you
      can instead upload a previously fitted lens file.
    """)
    return


@app.cell(hide_code=True)
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
        # specs: (import_name, dist_name, min_version, max_exclusive, pip_spec).
        # molab does not install the `# /// script` block into the running
        # kernel, so pip-install anything missing (or too old) at runtime.
        # torch is intentionally NEVER touched: molab's base image ships a build
        # matched to its GPU (Blackwell / sm_120 needs a cu128 wheel), and
        # pinning torch here would replace it with an unrunnable one.
        to_install = []
        for import_name, dist_name, min_version, max_version, pip_spec in specs:
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
                    continue
            if max_version is not None:
                try:
                    have = importlib.metadata.version(dist_name)
                except importlib.metadata.PackageNotFoundError:
                    to_install.append(pip_spec)
                    continue
                if _ver_tuple(have) >= _ver_tuple(max_version):
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
        ("transformers", "transformers", "5.13", "6", "transformers>=5.13,<6"),
        ("huggingface_hub", "huggingface_hub", None, None, "huggingface_hub"),
        ("numpy", "numpy", None, None, "numpy"),
    ])

    # jlens is not on PyPI; clone the repo and import it from source. This avoids
    # pip's resolver rebuilding/replacing torch on Blackwell. The package lives
    # in the `jacobian-lens/` subdirectory of the repo.
    # If the clone already exists, hard-sync it to REPO_REF so pushed fixes
    # reach molab (a kernel restart is still needed to re-import modules).
    # Use "main" while iterating; distribute an immutable course tag so later
    # repository changes cannot alter the class run. FETCH_HEAD supports branches
    # and tags.
    REPO_REF = "main"
    _local_jlens = Path(__file__).resolve().parent
    if (_local_jlens / "jlens").is_dir():
        # Prefer the exact checked-out source when opened locally or from a
        # course-release checkout; do not shadow it with a nested clone.
        JLENS_DIR = _local_jlens
        print(f"using checked-out jlens source: {JLENS_DIR}")
    else:
        REPO_DIR = Path("CTP49906_2026").resolve()
        if REPO_REF != "main":
            print(f"Notebook source pinned to {REPO_REF!r}.")
        if REPO_DIR.exists():
            _sync_title = f"Updating CTP49906_2026 to {REPO_REF}…"
        else:
            _sync_title = f"Cloning CTP49906_2026 @ {REPO_REF}…"
        with mo.status.spinner(title=_sync_title):
            if not REPO_DIR.exists():
                subprocess.run(
                    ["git", "clone", "--depth", "1", "--branch", REPO_REF,
                     "https://github.com/youngjuene/CTP49906_2026.git", str(REPO_DIR)],
                    check=True,
                )
            subprocess.run(
                ["git", "-C", str(REPO_DIR), "fetch", "--depth", "1", "origin", REPO_REF],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(REPO_DIR), "reset", "--hard", "FETCH_HEAD"], check=True
            )
        JLENS_DIR = REPO_DIR / "jacobian-lens"
    assert JLENS_DIR.is_dir(), f"expected jlens dir not found: {JLENS_DIR}"
    if str(JLENS_DIR) not in sys.path:
        sys.path.insert(0, str(JLENS_DIR))
    print("jlens dir:", JLENS_DIR)
    return (JLENS_DIR,)


@app.cell(hide_code=True)
def _(JLENS_DIR):
    import os
    from pathlib import Path as _Path

    _ = JLENS_DIR  # ensure the clone / sys.path / deps cell ran first
    import torch

    import jlens

    jlens.configure_logging()
    os.environ.setdefault("HF_HOME", "/tmp/hf-cache")

    assert torch.cuda.is_available(), (
        "No GPU visible. In molab, attach a GPU via the notebook-specs button in the header."
    )
    device = torch.device("cuda:0")
    OUTPUT_ROOT = _Path("artifacts/jlens-molab")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    _free, _total = torch.cuda.mem_get_info(0)
    print(f"torch={torch.__version__}, CUDA={torch.version.cuda}, GPU={torch.cuda.get_device_name(0)}")
    print(f"VRAM free/total GiB={_free / 2**30:.1f}/{_total / 2**30:.1f}")
    return OUTPUT_ROOT, device, jlens, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.2 Set the course model and reference lens

    The course reference lens is fitted for this exact model architecture; a lens for
    another model must be fitted separately.
    """)
    return


@app.cell
def _(OUTPUT_ROOT):
    MODEL_NAME = "Qwen/Qwen3.5-4B"
    # Immutable Hugging Face revisions validated for this class notebook.
    MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    LENS_REPO = "neuronpedia/jacobian-lens"
    LENS_REVISION = "16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a"
    LENS_FILE = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"
    LOCAL_FITTED_LENS = OUTPUT_ROOT / "jacobian_lens.pt"
    print({
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION[:12],
        "lens_file": LENS_FILE,
        "lens_revision": LENS_REVISION[:12],
    })
    return (
        LENS_FILE,
        LENS_REPO,
        LENS_REVISION,
        LOCAL_FITTED_LENS,
        MODEL_NAME,
        MODEL_REVISION,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.3 Load the course model
    """)
    return


@app.cell
def _(MODEL_NAME, MODEL_REVISION, device, jlens, mo, torch):
    import transformers

    with mo.status.spinner(title="Loading Qwen3.5-4B (first run downloads several GB)…"):
        torch.cuda.reset_peak_memory_stats(device)
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, revision=MODEL_REVISION, dtype=torch.bfloat16
        ).to(device)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            MODEL_NAME, revision=MODEL_REVISION
        )
        model = jlens.from_hf(hf_model, tokenizer)

    print(f"model-load peak GiB={torch.cuda.max_memory_allocated(device) / 2**30:.2f}")
    mo.md(
        f"**Course model ready:** `{MODEL_NAME}` · {model.n_layers} layers · "
        f"residual width {model.d_model}."
    )
    return model, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.4 Load the course reference lens

    This instructor-provided lens is already fitted for the model above; nothing
    is trained in the required class path.
    """)
    return


@app.cell
def _(LENS_FILE, LENS_REPO, LENS_REVISION, jlens, mo):
    with mo.status.spinner(title="Downloading the course reference lens (first run only)…"):
        lens = jlens.JacobianLens.from_pretrained(
            LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION
        )
    mo.md(
        f"**Course reference lens ready:** {len(lens.source_layers)} fitted source "
        f"layers · `n_prompts={lens.n_prompts}`."
    )
    return (lens,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Guided demo

    ### 2.1 Compare J-lens and vanilla logit lens

    For residual $h_{l,p}$ at layer $l$ and position $p$:

    - **Vanilla:** $h_{l,p} \rightarrow \mathrm{final\ norm} \rightarrow U$
    - **J-lens:** $h_{l,p} \rightarrow \bar{J}_l h_{l,p}
      \rightarrow \mathrm{final\ norm} \rightarrow U$

    Here $\bar{J}_l$ is an **average linear transport**, fitted over generic
    WikiText prompts, source positions, and current/future target positions. It
    is not the local Jacobian for this prompt, a causal intervention, or a
    decoder of a hidden sentence. Both methods ask what a residual is disposed
    to make the model say under different readout assumptions.

    **Pause before running:** the prompt ends in `the`. Predict which method will
    put the final model's preferred continuation nearer rank 1 at early, middle,
    and late layers. What result would count against your prediction?
    """)
    return


@app.cell
def _(tokenizer, torch):
    def compare_readouts(jacobian_logits, vanilla_logits, final_logits, layers, top_k):
        """Compare each lens with the model's final distribution at one position.

        The reference token/distribution is the model's own final-layer readout,
        so these are *self-consistency* metrics, not factual-correctness scores.
        """
        _reference = final_logits[0].float()
        _reference_logp = torch.log_softmax(_reference, dim=-1)
        _reference_top = _reference.topk(top_k).indices.tolist()
        _target_id = int(_reference.argmax())

        def _rank(_logits):
            _x = _logits.float()
            return int((_x > _x[_target_id]).sum().item()) + 1

        def _js(_logits):
            _logp = torch.log_softmax(_logits.float(), dim=-1)
            _logm = torch.logaddexp(_logp, _reference_logp) - 0.6931471805599453
            return float(
                0.5
                * (
                    (_logp.exp() * (_logp - _logm)).sum()
                    + (_reference_logp.exp() * (_reference_logp - _logm)).sum()
                )
            )

        def _top(_logits):
            return ", ".join(
                repr(tokenizer.decode([int(_t)]))
                for _t in _logits.topk(top_k).indices
            )

        def _overlap(_logits):
            _ids = set(int(_t) for _t in _logits.topk(top_k).indices)
            return len(_ids.intersection(_reference_top))

        _rows = []
        for _layer in layers:
            _vanilla = vanilla_logits[_layer][0]
            _jacobian = jacobian_logits[_layer][0]
            _vanilla_rank = _rank(_vanilla)
            _jacobian_rank = _rank(_jacobian)
            _rows.append({
                "Layer": _layer,
                "Vanilla target rank": _vanilla_rank,
                "J-lens target rank": _jacobian_rank,
                "Rank gain (V−J)": _vanilla_rank - _jacobian_rank,
                f"Vanilla top-{top_k} overlap": _overlap(_vanilla),
                f"J-lens top-{top_k} overlap": _overlap(_jacobian),
                "Vanilla JS": round(_js(_vanilla), 4),
                "J-lens JS": round(_js(_jacobian), 4),
                "Vanilla candidates": _top(_vanilla),
                "J-lens candidates": _top(_jacobian),
            })
        _target = tokenizer.decode([_target_id], clean_up_tokenization_spaces=False)
        _reference_tokens = [
            tokenizer.decode([int(_t)], clean_up_tokenization_spaces=False)
            for _t in _reference_top
        ]
        return _rows, _target, _reference_tokens

    return (compare_readouts,)


@app.cell
def _(compare_readouts, lens, mo, model, tokenizer):
    prompt_compare = "Fact: The currency used in the country shaped like a boot is the"
    # Pick four representative layers FROM the lens's fitted set: the J-lens path
    # requires layers ⊆ source_layers, so deriving them from model.n_layers
    # fractions would raise if the course reference lens skipped any of them.
    _src = lens.source_layers
    demo_layers = sorted(
        {_src[len(_src) // 4], _src[len(_src) // 2], _src[len(_src) * 3 // 4], _src[-1]}
    )
    # `-1` is the final prompt token (` the`), so its residual predicts the unseen
    # continuation after the complete prompt.
    _position = -1
    jlens_logits, model_logits, _input_ids = lens.apply(
        model, prompt_compare, layers=demo_layers, positions=[_position]
    )
    logit_lens_out, _, _ = lens.apply(
        model,
        prompt_compare,
        layers=demo_layers,
        positions=[_position],
        use_jacobian=False,
    )

    _metric_rows, _target, _final_top = compare_readouts(
        jlens_logits, logit_lens_out, model_logits, demo_layers, 5
    )
    _source_token = tokenizer.decode(
        [int(_input_ids[0, _position])], clean_up_tokenization_spaces=False
    )
    mo.vstack([
        mo.md(
            f"**Prompt:** `{prompt_compare}`  \n"
            f"**Probe:** position `{_position}` = token `{_source_token!r}`; its residual "
            "predicts the next token after the complete prompt.  \n"
            f"**Final model target:** `{_target!r}` · **final top-5:** "
            + ", ".join(repr(_t) for _t in _final_top)
        ),
        mo.ui.table(_metric_rows, selection=None, pagination=False),
        mo.callout(
            mo.md(
                "**How to read this:** lower target rank and lower Jensen–Shannon "
                "divergence mean closer agreement with the model's final-layer "
                "distribution. Positive `Rank gain (V−J)` favors J-lens. This is "
                "fidelity to the model's own output distribution — **not evidence "
                "that the output is true, safe, or causally explained.**"
            ),
            kind="neutral",
        ),
    ])
    return (demo_layers,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.2 Inspect a layer × position slice

    The slice tracks candidate tokens across residual positions and layers.
    Submit one bundled example at a time; this prevents an expensive recompute
    while you are still changing controls.

    `Word-like display` filters which candidates are *shown* (special tokens and
    punctuation disappear), but ranks are still calculated against the full
    vocabulary. Toggle it off whenever the polished view looks too coherent —
    selection can change the story you tell about the same activations.
    """)
    return


@app.cell
def _(mo):
    from jlens.examples import EXAMPLES

    _example_template = (
        "**Bundled prompt** {example}\n\n"
        "**Layer stride** {layer_stride} &nbsp; **Last prompt positions shown** {last_n_tokens}\n\n"
        "**Word-like display** {mask_display}"
    )
    example_controls = mo.md(_example_template).batch(
        example=mo.ui.dropdown(
            options={e.section: e.slug for e in EXAMPLES},
            value=next(e.section for e in EXAMPLES if e.slug == "multihop"),
        ),
        layer_stride=mo.ui.slider(1, 8, step=1, value=2, show_value=True),
        last_n_tokens=mo.ui.slider(
            8, 128, step=8, value=64, show_value=True, include_input=True
        ),
        mask_display=mo.ui.checkbox(value=True),
    ).form(submit_button_label="▶ Build guided slice", bordered=True)
    example_controls
    return EXAMPLES, example_controls


@app.cell(hide_code=True)
def _(EXAMPLES, JLENS_DIR, example_controls, lens, mo, model, tokenizer):
    import gzip
    import json

    from jlens import vis
    from jlens.examples import resolve_prompt
    from jlens.vis import build_page, compute_slice

    _cfg = example_controls.value
    mo.stop(
        _cfg is None,
        mo.callout(mo.md("Choose the guided-slice settings and press **▶**."), kind="info"),
    )

    # The embed page inlines d3. If the runtime blocks Python's socket, fetch the
    # same SRI-pinned file with curl; the verified template is then memoized.
    try:
        vis._template("embed")
    except RuntimeError:
        import base64 as _b64
        import hashlib as _hashlib
        import subprocess as _sp

        _d3 = _sp.run(
            ["curl", "--fail", "--silent", "--show-error", "-L", vis._D3_URL],
            check=True,
            capture_output=True,
        ).stdout
        _sri = "sha384-" + _b64.b64encode(_hashlib.sha384(_d3).digest()).decode()
        if _sri != vis._D3_SRI:
            raise RuntimeError(f"d3 integrity check failed: {_sri}") from None
        vis._TEMPLATE_FOR_MODE["embed"] = vis.PAGE_TEMPLATE.replace(
            "__D3__", f"<script>\n{_d3.decode()}\n</script>"
        )

    _example = next(e for e in EXAMPLES if e.slug == _cfg["example"])
    _prompt = resolve_prompt(_example, tokenizer)

    # The English-gloss file ships in the repo (assets/), not the installed
    # package, so it is optional; the slice renders fine without it.
    _gloss = None
    _gloss_path = JLENS_DIR / "assets" / "qwen_gloss.json.gz"
    if _gloss_path.exists():
        _gloss = {int(k): v for k, v in json.load(gzip.open(_gloss_path)).items()}

    with mo.status.spinner(title=f"Computing slice for “{_example.section}”…"):
        _slice = compute_slice(
            model,
            lens,
            _prompt,
            layer_stride=int(_cfg["layer_stride"]),
            last_n_tokens=int(_cfg["last_n_tokens"]),
            max_tracked=_example.n_tracked if _example.n_tracked is not None else 128,
            mask_display=bool(_cfg["mask_display"]),
        )
        _page, _, _ = build_page(
            _slice,
            _prompt,
            title=_example.section,
            description=_example.description,
            alt_token=_gloss,
        )

    # molab renders cell output inside a locked-down iframe that won't run this
    # page's inlined scripts, so the inline view often comes up blank. The page
    # is fully self-contained (d3 inlined), so always offer it as a download —
    # opening the file in a browser tab gives the real interactive slice. The
    # inline iframe stays below for runtimes (local marimo, Jupyter) that do
    # render it.
    _download = mo.download(
        _page.encode(),
        filename=f"slice_{_example.slug}.html",
        mimetype="text/html",
        label="⬇ Download this slice, then open it in a new browser tab",
    )
    mo.vstack([
        mo.md(
            f"**{_example.section}** — interactive slice. If the view below is "
            "blank, use the download button above (molab blocks scripted iframes)."
        ),
        _download,
        mo.iframe(_page, height="660px"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Research playground

    Now choose the prompt, offset from the end, fitted layers, comparison depth,
    and display settings. Every submit reports numeric **agreement with the
    model's final distribution**: target rank, top-$k$ overlap, and Jensen–Shannon
    divergence. Agreement is a fidelity check, not a truth score.

    **Investigation routes — change one variable at a time**

    1. **Try to falsify the headline.** Find a prompt/layer where vanilla has a
       lower target rank or JS divergence than J-lens. A counterexample is a
       successful result.
    2. **Move the probe.** Keep the prompt fixed and compare offsets 1, 2, and 4.
       Does a story about “what the model knows” survive a one-token move?
    3. **Shift the distribution.** Compare factual prose with code, Korean,
       poetry, or deliberately broken text. Where should a transport averaged
       over WikiText fail, and why?
    4. **Expose selection.** Build the same slice with word-like filtering on
       and off. Can the two views invite different narratives from identical
       ranks?
    5. **Compare estimators.** After Appendix A, repeat a prompt with the course
       reference lens and your 100-prompt fit. Which differences look like
       architecture, and which look like estimation noise?

    Before ▶, write a prediction that could be wrong. Afterward ask: *what
    alternate mechanism could produce the same table, and what next run would
    distinguish it?*

    ### 3.1 Choose a lens
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    lens_upload = mo.ui.file(
        filetypes=[".pt"],
        kind="button",
        max_size=2_000_000_000,
        label="⬆ Upload a jacobian_lens.pt (used by 'Uploaded lens file' below)",
    )
    lens_upload
    return (lens_upload,)


@app.cell(hide_code=True)
def _(mo):
    pg_source = mo.ui.dropdown(
        options={
            "Course reference lens": "reference",
            "Student-fitted lens": "fitted",
            "Uploaded lens file": "uploaded",
        },
        value="Course reference lens",
        label="Lens source (loaded once, then reused across submits)",
    )
    pg_source
    return (pg_source,)


@app.cell(hide_code=True)
def _(
    LOCAL_FITTED_LENS,
    OUTPUT_ROOT,
    fitted_lens_version,
    jlens,
    lens,
    lens_upload,
    mo,
    model,
    pg_source,
):
    # Reload only when the source, upload, or student-fitted file changes. Prompt
    # edits and repeated submits reuse the already-loaded lens.
    _ = fitted_lens_version
    _load_error = None
    active_lens = None
    active_lens_label = ""
    try:
        if pg_source.value == "uploaded":
            if not lens_upload.value:
                _load_error = "Select an uploaded `.pt` file, or choose another lens source."
            else:
                _p = OUTPUT_ROOT / "uploaded_lens.pt"
                _p.write_bytes(lens_upload.value[0].contents)
                with mo.status.spinner(title="Loading uploaded lens (once)…"):
                    active_lens = jlens.JacobianLens.load(str(_p))
                active_lens_label = f"uploaded · {lens_upload.value[0].name}"
        elif pg_source.value == "fitted":
            if not LOCAL_FITTED_LENS.exists():
                _load_error = (
                    "No student-fitted lens exists in this session. Run Appendix A, "
                    "then return here."
                )
            else:
                with mo.status.spinner(title="Loading the student-fitted lens (once)…"):
                    active_lens = jlens.JacobianLens.load(str(LOCAL_FITTED_LENS))
                active_lens_label = "student-fitted lens"
        else:
            active_lens = lens
            active_lens_label = "course reference lens · 1,000 prompts"
    except Exception as _e:
        _load_error = f"{type(_e).__name__}: {_e}"

    if active_lens is not None:
        if active_lens.d_model != lens.d_model:
            _load_error = (
                f"Lens d_model={active_lens.d_model} does not match this model/lens "
                f"configuration ({lens.d_model})."
            )
        elif not active_lens.source_layers:
            _load_error = "The selected lens has no fitted source layers."
        elif max(active_lens.source_layers) >= model.n_layers:
            _load_error = (
                f"Lens source layer {max(active_lens.source_layers)} is out of range "
                f"for this {model.n_layers}-layer model."
            )

    mo.stop(
        _load_error is not None,
        mo.callout(mo.md(f"**Lens unavailable or incompatible** — {_load_error}"), kind="danger"),
    )
    return active_lens, active_lens_label


@app.cell(hide_code=True)
def _(active_lens, active_lens_label, demo_layers, mo):
    _default_layers = [
        _layer for _layer in demo_layers if _layer in active_lens.source_layers
    ]
    if not _default_layers:
        _default_layers = [active_lens.source_layers[-1]]

    def _validate(_value):
        if not _value or not _value["prompt"].strip():
            return "Enter a non-empty prompt."
        if not _value["hypothesis"].strip():
            return "Write a falsifiable prediction before running."
        if not _value["layers"]:
            return "Select at least one fitted layer."
        return None

    _template = (
        "### 3.2 State a prediction and choose variables\n\n"
        f"**Active lens:** {active_lens_label} (`n_prompts={active_lens.n_prompts}`)\n\n"
        "*The initial controls reproduce the guided comparison; after that, change one variable at a time.*\n\n"
        "**Prediction before ▶** — name a layer/position trend that could be wrong:\n\n"
        "{hypothesis}\n\n"
        "**Prompt** {prompt}\n\n"
        "Probe **{position_from_end} token(s) from the end** "
        "(1 = final prompt token, whose residual predicts the unseen continuation).\n\n"
        "**Fitted layers** {layers}\n\n"
        "Compare top **{top_k}** candidates.\n\n"
        "---\n\n"
        "**Also build a downloadable slice** {make_slice}\n\n"
        "Slice layer stride {slice_stride} · last positions {slice_window} · "
        "word-like display {mask_display}"
    )
    playground_controls = mo.md(_template).batch(
        hypothesis=mo.ui.text_area(
            placeholder="e.g. J-lens will beat vanilla before the midpoint; a code prompt will shrink that gain.",
            rows=2,
            full_width=True,
        ),
        prompt=mo.ui.text_area(
            value="Fact: The currency used in the country shaped like a boot is the",
            rows=3,
            full_width=True,
        ),
        position_from_end=mo.ui.slider(
            1, 32, step=1, value=1, show_value=True, include_input=True
        ),
        layers=mo.ui.multiselect(
            options={f"Layer {_layer}": _layer for _layer in active_lens.source_layers},
            value=[f"Layer {_layer}" for _layer in _default_layers],
        ),
        top_k=mo.ui.slider(1, 10, step=1, value=5, show_value=True),
        make_slice=mo.ui.checkbox(value=False),
        slice_stride=mo.ui.slider(1, 8, step=1, value=2, show_value=True),
        slice_window=mo.ui.slider(
            8, 128, step=8, value=64, show_value=True, include_input=True
        ),
        mask_display=mo.ui.checkbox(value=True),
    ).form(
        submit_button_label="▶ Test the prediction",
        bordered=True,
        validate=_validate,
    )
    playground_controls
    return (playground_controls,)


@app.cell
def _(
    active_lens,
    active_lens_label,
    compare_readouts,
    mo,
    model,
    playground_controls,
    tokenizer,
):
    _cfg = playground_controls.value
    mo.stop(
        _cfg is None,
        mo.callout(mo.md("Set the playground controls and press **▶**."), kind="info"),
    )

    # Keep the playground renderer independent from the guided-slice renderer.
    from jlens.vis import build_page as _build_page
    from jlens.vis import compute_slice as _compute_slice

    _prompt = _cfg["prompt"].strip()
    _input_ids = model.encode(_prompt, max_length=512)
    _seq_len = int(_input_ids.shape[1])
    _offset = int(_cfg["position_from_end"])
    mo.stop(
        _offset > _seq_len,
        mo.callout(
            mo.md(
                f"This prompt has only **{_seq_len} tokens** after tokenization; "
                f"offset {_offset} is out of range."
            ),
            kind="danger",
        ),
    )
    _position = -_offset
    _absolute_position = _seq_len - _offset

    _layers = sorted(int(_layer) for _layer in _cfg["layers"])

    with mo.status.spinner(title="Comparing readouts at the selected position…"):
        _jl, _ml, _ = active_lens.apply(
            model, _prompt, layers=_layers, positions=[_position]
        )
        _ll, _, _ = active_lens.apply(
            model, _prompt, layers=_layers, positions=[_position], use_jacobian=False
        )

    _top_k = int(_cfg["top_k"])
    _rows, _target, _final_top = compare_readouts(
        _jl, _ll, _ml, _layers, _top_k
    )
    _source_token = tokenizer.decode(
        [int(_input_ids[0, _absolute_position])],
        clean_up_tokenization_spaces=False,
    )
    _observed_next = (
        tokenizer.decode(
            [int(_input_ids[0, _absolute_position + 1])],
            clean_up_tokenization_spaces=False,
        )
        if _absolute_position + 1 < _seq_len
        else None
    )
    _table = mo.vstack([
        mo.md("### 3.3 Result and verdict"),
        mo.md(
            f"**Prediction recorded before run:** {_cfg['hypothesis']}  \n"
            f"**Lens:** {active_lens_label} · `n_prompts={active_lens.n_prompts}`  \n"
            f"**Probe:** absolute position `{_absolute_position}` / offset `{_position}` "
            f"= `{_source_token!r}`  \n"
            + (
                f"**Observed next context token:** `{_observed_next!r}`  \n"
                if _observed_next is not None
                else "**Observed next context token:** _not supplied; this is a continuation probe_  \n"
            )
            + f"**Final-model target:** `{_target!r}` · **final top-{_top_k}:** "
            + ", ".join(repr(_t) for _t in _final_top)
        ),
        mo.ui.table(_rows, selection=None, pagination=False),
    ])

    # Offer the matching interactive slice as a download. A visualization failure
    # must not hide the numeric comparison above.
    if not _cfg["make_slice"]:
        _viz = mo.md("_Slice skipped for this submit._")
    else:
        try:
            with mo.status.spinner(title="Building the interactive slice…"):
                _slice = _compute_slice(
                    model,
                    active_lens,
                    _prompt,
                    layer_stride=int(_cfg["slice_stride"]),
                    last_n_tokens=int(_cfg["slice_window"]),
                    max_tracked=64,
                    mask_display=bool(_cfg["mask_display"]),
                )
                _page, _, _ = _build_page(
                    _slice,
                    _prompt,
                    title="Playground slice",
                    description=(
                        f"{active_lens_label} · probe offset {_position} · {_prompt!r}"
                    ),
                )
            _viz = mo.download(
                _page.encode(),
                filename="playground_slice.html",
                mimetype="text/html",
                label="⬇ Download the interactive slice for this prompt",
            )
        except Exception as _e:
            _viz = mo.callout(
                mo.md(
                    f"**Slice unavailable:** `{type(_e).__name__}: {_e}`. "
                    "The comparison table above is still valid."
                ),
                kind="warn",
            )

    _reflection = mo.callout(
        mo.md(
            "**Verdict before the next run:** Did the result support, refute, or "
            "fail to test your prediction? Name one competing explanation. Then "
            "change exactly one variable or design a control whose pass *and* "
            "fail outcomes would both teach you something."
        ),
        kind="neutral",
    )
    mo.vstack([_table, _viz, _reflection])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Synthesis challenge — transfer the estimator across architectures

    The course reference transport belongs to a **text decoder** and was fitted on
    a generic text corpus. It cannot simply be inserted into Qwen2.5-Omni: the
    model, residual basis, width, layer structure, and modality positions differ.
    An audio-position J-lens therefore forces new estimator and evidence choices.

    In groups, sketch a defensible experiment:

    1. What prompt distribution should define $\bar{J}_l$ — text only,
       synchronized audio/video, or deliberately conflicting modalities?
    2. Which source and future target positions should the Jacobian average?
    3. What held-out metric separates **faithful transport** from a lens that
       merely emits plausible vocabulary?
    4. Design a negative control. What would both its pass and fail outcomes let
       you conclude—and what would remain unresolved?
    5. What observation would make you abandon the claim that the same global
       workspace principle transfers across architectures or modalities?

    **Design deliverable:** draw the proposed information path, state one
    falsifiable prediction, name a rival explanation, and choose the next
    measurement that would distinguish them. A creative proposal changes the
    estimator or its evidence, not just a slider; a critical proposal states what
    would prove it wrong.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Appendix A. Fit a student lens from scratch (optional)

    Fit $J_l$ over 100 WikiText prompts and save a student lens file. This is a
    long GPU job (~15–20 minutes for the 4B model) and pulls `datasets` on demand,
    so it runs **only when you click**. When it finishes, validate the file below,
    then return to the research playground and select **Student-fitted lens**.
    """)
    return


@app.cell
def _(mo):
    run_fit = mo.ui.run_button(label="Fit a 100-prompt lens (~15–20 min on GPU)")
    run_fit
    return (run_fit,)


@app.cell
def _(LOCAL_FITTED_LENS, OUTPUT_ROOT, jlens, mo, model, run_fit):
    if run_fit.value:
        import subprocess as _sp
        import sys as _sys

        # load_wikitext_prompts streams WikiText via `datasets`; install it on
        # demand (only when fitting) so ordinary runs stay lean and torch is
        # never touched.
        with mo.status.spinner(title="Installing datasets…"):
            _sp.run([_sys.executable, "-m", "pip", "install", "datasets"], check=True)
        from jlens.examples import load_wikitext_prompts

        with mo.status.spinner(title="Fitting a 100-prompt Jacobian lens (~15–20 min)…"):
            _prompts = load_wikitext_prompts(n_prompts=100)
            _fitted = jlens.fit(
                model,
                _prompts,
                dim_batch=32,
                max_seq_len=128,
                checkpoint_path=str(OUTPUT_ROOT / "ckpt.pt"),
            )
            _fitted.save(str(LOCAL_FITTED_LENS))
        _out = mo.md(
            f"✅ Fitted **{_fitted.n_prompts} prompts**, saved to "
            f"`{LOCAL_FITTED_LENS}`. Use the validation below before comparing it."
        )
    else:
        _out = mo.md(
            "_Idle — click the button above to fit. Nothing runs until you do._"
        )
    _stat = LOCAL_FITTED_LENS.stat() if LOCAL_FITTED_LENS.exists() else None
    fitted_lens_version = (
        (_stat.st_mtime_ns, _stat.st_size) if _stat is not None else None
    )
    _out
    return (fitted_lens_version,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A.1 Validate and export the student-fitted lens

    Loading and matching `d_model`/layer bounds proves only **structural
    compatibility**. It does not prove lens quality, model identity, corpus fit,
    or fidelity. Held-out playground prompts provide the behavioral check.
    """)
    return


@app.cell
def _(LOCAL_FITTED_LENS, fitted_lens_version, jlens, mo, model):
    _ = fitted_lens_version
    if LOCAL_FITTED_LENS.exists():
        _fitted = jlens.JacobianLens.load(str(LOCAL_FITTED_LENS))
        _shape_ok = (
            _fitted.d_model == model.d_model
            and bool(_fitted.source_layers)
            and max(_fitted.source_layers) < model.n_layers
        )
        _msg = mo.md(
            f"✅ Loaded student-fitted lens: **{_fitted.n_prompts} prompts**, "
            f"structurally compatible: **{_shape_ok}**. "
            "This is a file/shape check, not a quality verdict."
        )
    else:
        _msg = mo.md(
            f"ℹ️ No student-fitted lens at `{LOCAL_FITTED_LENS}`. "
            "The guided demo and playground default to the course reference lens."
        )
    _msg
    return


@app.cell
def _(LOCAL_FITTED_LENS, fitted_lens_version, mo):
    _ = fitted_lens_version
    if LOCAL_FITTED_LENS.exists():
        _mb = LOCAL_FITTED_LENS.stat().st_size / 2**20
        _dl = mo.download(
            data=lambda: LOCAL_FITTED_LENS.read_bytes(),
            filename="jacobian_lens.pt",
            mimetype="application/octet-stream",
            label=f"⬇ Download your fitted lens ({_mb:.0f} MB)",
        )
    else:
        _dl = mo.md("_No fitted lens on disk yet._")
    _dl
    return


if __name__ == "__main__":
    app.run()
