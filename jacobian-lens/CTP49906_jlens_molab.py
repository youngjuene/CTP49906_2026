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
      1,000 WikiText prompts. The advanced appendix runs only when clicked.
    - **Nothing in this notebook requires a file round-trip.** Slices render in
      place, and a lens you fit in Appendix A is selectable in §3.1 straight
      away — it stays on this session's disk. The download and upload buttons
      exist only to carry a fit *out of* this session and back into a later one;
      skip them and the whole path still works.
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
def _():
    # Model / revision pairs this notebook has actually been validated against.
    # The reference lens is fitted for one specific residual basis, so changing
    # the model without a matching lens does not produce a worse readout — it
    # produces a meaningless one. Editing MODEL_NAME to something not in this
    # table used to start a multi-gigabyte download and then fail on the pinned
    # revision hash; now it stops here, before the download.
    VALIDATED_MODELS = {
        # model id: (immutable HF revision, lens file fitted for it)
        "Qwen/Qwen3.5-4B": (
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt",
        ),
    }

    MODEL_NAME = "Qwen/Qwen3.5-4B"
    assert MODEL_NAME in VALIDATED_MODELS, (
        f"{MODEL_NAME!r} has no validated revision or reference lens in this "
        f"notebook. Known: {sorted(VALIDATED_MODELS)}. Pointing this at another "
        "model needs a lens fitted for that model too — see Appendix A; a lens "
        "from a different model will load happily and read out nonsense."
    )
    MODEL_REVISION, LENS_FILE = VALIDATED_MODELS[MODEL_NAME]
    LENS_REPO = "neuronpedia/jacobian-lens"
    LENS_REVISION = "16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a"
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
def _(mo, model, tokenizer, torch):
    # `mask_display=True` needs a vocab-wide mask built by decoding every token id
    # in a Python loop, memoised per kernel inside jlens.vis. Built lazily it lands
    # on whichever slice a student submits first — and route 4 asks them to toggle
    # exactly that checkbox, so the first slice of a session is dearer than every
    # later one for a reason that has nothing to do with the setting being
    # compared. Pay it here instead, where it is labelled as a one-off and cannot
    # be mistaken for the cost of a lens.
    from jlens.vis import _meaningful_token_mask

    # The cache key is (tokenizer, vocab_size), and `vocab_size` inside
    # `compute_slice` is the *unembedding* width read off real logits — not
    # `len(tokenizer)`, which is smaller (the unembedding is padded). Warming with
    # the wrong number would build the mask twice and cache neither usefully, so
    # take the width from one cheap matmul on a zero residual.
    with torch.no_grad():
        _probe = model.unembed(
            torch.zeros(
                1, model.d_model,
                device=next(model.layers[0].parameters()).device,
                dtype=next(model.layers[0].parameters()).dtype,
            )
        )
    VOCAB_SIZE = int(_probe.shape[-1])

    with mo.status.spinner(
        title=f"Indexing {VOCAB_SIZE:,} tokens for word-like display (one time)…"
    ):
        _mask = _meaningful_token_mask(tokenizer, VOCAB_SIZE, _probe.device)
    mo.md(
        f"**Vocabulary indexed:** {int(_mask.sum()):,} of {VOCAB_SIZE:,} tokens are "
        "word-like. Slices now cost the same whether it is your first or your tenth.\n\n"
        f"Note the width: ranks below are against **{VOCAB_SIZE:,}** columns, so "
        f"chance is rank ≈ **{VOCAB_SIZE // 2:,}** — not `len(tokenizer)` = "
        f"{len(tokenizer):,}, which is a different number."
    )
    return


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

    **How to read the grid.** One row per prompt position, one column per
    rendered layer, and each cell holds that readout's top displayed candidate.
    Shading answers *where does the readout already agree with the model?* —
    dark where the cell's own top-1 is the final layer's top-1 for that
    position, mid where the final answer is merely somewhere in the cell's
    top-K, pale where it is absent. The `★` column is the model's final layer
    (identity transport), so it is the reference the other columns are shaded
    against, not a lens readout. **Hover any cell** for its full candidate list
    with full-vocabulary ranks.

    Everything under the grid — which tokens to track, which position to read
    exactly — is computed from the slice that is already in memory, so moving
    those controls costs nothing and never refits or re-runs the model.

    `Word-like display` filters which candidates are *shown* (special tokens and
    punctuation disappear), but ranks are still calculated against the full
    vocabulary. Toggle it off whenever the polished view looks too coherent —
    selection can change the story you tell about the same activations.
    """)
    return


@app.cell(hide_code=True)
def _(JLENS_DIR):
    import gzip as _gzip
    import json as _json

    # The English-gloss file ships in the repo (assets/), not the installed
    # package, so it is optional; slices render fine without it. Loaded in its own
    # cell so *both* the guided slice and the playground slice can use it — the
    # playground one was silently dropping it, which is exactly where investigation
    # route 3 sends students (Korean and other non-English prompts).
    _gloss_path = JLENS_DIR / "assets" / "qwen_gloss.json.gz"
    token_gloss = (
        {int(k): v for k, v in _json.load(_gzip.open(_gloss_path)).items()}
        if _gloss_path.exists()
        else None
    )
    return (token_gloss,)


@app.cell(hide_code=True)
def _(JLENS_DIR, token_gloss):
    from jlens import vis

    _ = JLENS_DIR  # the clone has to be on sys.path before jlens.vis imports

    def standalone_page(slice_data, prompt, title, description):
        """The d3 page for one slice, as bytes, built only when clicked.

        The notebook's own view is script-free and renders in place; this
        builds the d3 instrument as a file for a reader who wants it in a
        browser tab of its own. `mo.download` resolves a callable lazily, so
        nothing in here — including the d3 fetch — runs unless the button is
        pressed. Shared by §2.2 and §3 so the two paths cannot drift apart.
        """
        try:
            vis._template("embed")
        except RuntimeError:
            # The embed page inlines d3. If the runtime blocks Python's socket,
            # fetch the same SRI-pinned file with curl; either way it is
            # verified before it is inlined, and the template is then memoized.
            import base64
            import hashlib
            import subprocess

            body = subprocess.run(
                ["curl", "--fail", "--silent", "--show-error", "-L", vis._D3_URL],
                check=True,
                capture_output=True,
            ).stdout
            sri = "sha384-" + base64.b64encode(hashlib.sha384(body).digest()).decode()
            if sri != vis._D3_SRI:
                raise RuntimeError(f"d3 integrity check failed: {sri}") from None
            vis._TEMPLATE_FOR_MODE["embed"] = vis.PAGE_TEMPLATE.replace(
                "__D3__", f"<script>\n{body.decode()}\n</script>"
            )
        page, _, _ = vis.build_page(
            slice_data,
            prompt,
            title=title,
            description=description,
            alt_token=token_gloss,
        )
        return page.encode()

    return (standalone_page,)


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
def _(EXAMPLES, example_controls, lens, mo, model, tokenizer):
    from jlens.examples import resolve_prompt
    from jlens.vis import compute_slice

    _cfg = example_controls.value
    mo.stop(
        _cfg is None,
        mo.callout(mo.md("Choose the guided-slice settings and press **▶**."), kind="info"),
    )

    _example = next(e for e in EXAMPLES if e.slug == _cfg["example"])
    guided_prompt = resolve_prompt(_example, tokenizer)
    guided_title = _example.section

    # `compute_slice` truncates at max_seq_len=512 and says nothing. The longest
    # bundled example is several times that, so the window lands on positions
    # nowhere near the decision point the example exists to show — labelled with
    # absolute indices that look entirely reasonable.
    _MAX_SEQ = 512
    _full_len = len(tokenizer(guided_prompt).input_ids)
    _win = int(_cfg["last_n_tokens"])
    _trunc = (
        mo.callout(
            mo.md(
                f"**“{_example.section}” is truncated for this slice.** It tokenizes "
                f"to **{_full_len}** tokens; only the first **{_MAX_SEQ}** are used, "
                f"and the view then shows the last **{_win}** of *those* — positions "
                f"{_MAX_SEQ - _win}–{_MAX_SEQ - 1}, not the end of the text. If the "
                "part of this example you care about sits later, it is not on screen."
            ),
            kind="warn",
        )
        if _full_len > _MAX_SEQ
        else None
    )

    with mo.status.spinner(title=f"Computing slice for “{_example.section}”…"):
        guided_slice = compute_slice(
            model,
            lens,
            guided_prompt,
            layer_stride=int(_cfg["layer_stride"]),
            last_n_tokens=int(_cfg["last_n_tokens"]),
            max_tracked=_example.n_tracked if _example.n_tracked is not None else 128,
            mask_display=bool(_cfg["mask_display"]),
        )

    # Name the lens wherever this slice is shown. The guided slice always uses
    # the *course reference* lens — it sits above §3.1, so it is not affected by
    # the lens picker there. Without saying so, a student who has just fitted
    # their own lens will scroll up, see this picture, and file it under their
    # fit: a plausible-but-wrong attribution manufactured by the notebook itself.
    guided_description = (
        f"{_example.description}  —  course reference lens "
        f"(n_prompts={lens.n_prompts}); the lens picker in §3.1 does not "
        f"affect this slice. Layer stride {int(_cfg['layer_stride'])}, "
        f"last {int(_cfg['last_n_tokens'])} positions, word-like display "
        f"{'on' if _cfg['mask_display'] else 'off'}."
    )
    guided_filename = (
        f"slice_{_example.slug}"
        f"_{int(_cfg['layer_stride'])}stride"
        f"_{int(_cfg['last_n_tokens'])}win"
        f"_{'masked' if _cfg['mask_display'] else 'raw'}.html"
    )
    # Only the slice is computed here. Rendering it lives downstream so that the
    # view controls cannot trigger a recompute: this cell is the 10–30 s one.
    # The settings and the lens name travel with the picture instead of being
    # printed here, since it is the picture people screenshot and quote.
    mo.vstack([_c for _c in (
        _trunc,
        mo.md(
            f"**{_example.section}** — slice ready: "
            f"{guided_slice.seq_len} positions × {len(guided_slice.layers)} "
            f"layers, {len(guided_slice.tracked_token_ids)} tracked tokens."
        ),
    ) if _c is not None])
    return guided_description, guided_filename, guided_prompt, guided_slice, guided_title


@app.cell(hide_code=True)
def _(guided_slice, mo, token_gloss):
    from jlens.vis import token_label

    # Re-minted whenever a new slice is computed, which is correct here: the
    # options *are* this slice's tracked tokens and this slice's positions, and
    # a stale selection would silently point at another prompt's vocabulary.
    _tracked = list(guided_slice.tracked_token_ids)
    _token_options = {}
    for _tid in _tracked:
        _label = repr(token_label(guided_slice, _tid, token_gloss))
        # Two ids can decode to the same string; without the id, the duplicate
        # key would silently drop one of them from the picker.
        if _label in _token_options:
            _label = f"{_label} · id {_tid}"
        _token_options[_label] = int(_tid)

    # Default to the model's own answer for the last position: the final layer
    # (identity transport) top-1 there. `compute_slice` tracks by top-K
    # frequency, so it is usually but not always tracked — fall back rather
    # than open on an empty rank map.
    _answer = int(guided_slice.top_ids[-1, -1, 0])
    _default_id = _answer if _answer in _tracked else (_tracked[0] if _tracked else None)
    _default = [_k for _k, _v in _token_options.items() if _v == _default_id][:1]

    _positions = {
        f"{guided_slice.ctx_offset + _i}  "
        f"{guided_slice.context_token_strs[guided_slice.ctx_offset + _i]!r}": _i
        for _i in range(guided_slice.seq_len)
    }

    guided_view = mo.md(
        "**Track these tokens through the stack** {tokens} *(rank map below; "
        "the default is the model's own answer at the last position)*\n\n"
        "**Read the exact readout at position** {position}"
    ).batch(
        tokens=mo.ui.multiselect(
            options=_token_options, value=_default, max_selections=4
        ),
        position=mo.ui.dropdown(
            options=_positions, value=list(_positions)[-1]
        ),
    )
    guided_view
    return (guided_view,)


@app.cell(hide_code=True)
def _(
    guided_description,
    guided_filename,
    guided_prompt,
    guided_slice,
    guided_title,
    guided_view,
    mo,
    standalone_page,
    token_gloss,
):
    from functools import partial as _partial

    from jlens.vis import (
        position_top_k_rows,
        slice_grid_html,
        token_rank_grid_html,
    )

    # No `mo.iframe` here. marimo sanitizes HTML output and drops `<script>`,
    # which is why displaying the d3 page at all means nesting a document — and
    # in molab that nested document never ran the page's inlined scripts, so it
    # came up as a 660 px blank rectangle carrying a multi-megabyte srcdoc,
    # shipped as the *default* view of the notebook's only real instrument.
    # These fragments need no script: plain tables, inline colour, native
    # `title` tooltips. Nothing to strip and nothing to block, so they render
    # in molab, in local marimo, and in Jupyter alike.
    _sel = guided_view.value
    _tokens = [int(_t) for _t in (_sel["tokens"] or [])]
    _position = int(_sel["position"])
    _absolute = guided_slice.ctx_offset + _position

    mo.vstack([
        mo.Html(
            slice_grid_html(
                guided_slice, alt_token=token_gloss, caption=guided_description
            )
        ),
        mo.md(
            "#### Rank of the tracked token(s)\n\n"
            "Full-vocabulary rank in every cell, shaded on a log scale — the "
            "same quantity the interactive page charts. Dark is rank 0; a token "
            "that turns dark early and stays dark was a candidate long before "
            "the final layer."
        ),
        mo.Html(token_rank_grid_html(guided_slice, _tokens, alt_token=token_gloss)),
        mo.md(f"#### Exact readout at prompt position {_absolute}"),
        mo.ui.table(
            position_top_k_rows(
                guided_slice,
                _position,
                top_k=5,
                token_ids=_tokens,
                alt_token=token_gloss,
            ),
            selection=None,
            pagination=False,
        ),
        mo.accordion({
            "Optional — the same slice as a standalone d3 page (a file to open "
            "in its own browser tab; nothing above needs it)": mo.download(
                # Lazy: `mo.download` resolves a callable only when clicked, so
                # neither the page nor its d3 fetch costs anything unless asked
                # for. Nothing in the notebook's own path depends on it.
                data=_partial(
                    standalone_page,
                    guided_slice,
                    guided_prompt,
                    guided_title,
                    guided_description,
                ),
                filename=guided_filename,
                mimetype="text/html",
                label=f"⬇ Build and download {guided_filename}",
            )
        }),
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
       reference lens and your own fit. Fit **25** prompts and **100** and compare
       all three: which differences look like architecture, and which look like
       estimation noise? The form below keeps your prompt and prediction when you
       switch lens, so this is one variable moving.
    6. **Run the negative control.** Select **Scrambled-layer control** in §3.1 —
       the reference lens with each layer's transport swapped for another
       layer's. It costs nothing and refits nothing. If a deliberately mismatched
       transport still emits fluent, on-topic candidates, then fluency is not
       evidence that a readout is faithful, and any claim you make from the
       candidate lists alone is unsupported. If instead it collapses to junk,
       that is also a result — say which of the two you got before reading the
       rank columns.

    Before ▶, write a prediction that could be wrong. Afterward ask: *what
    alternate mechanism could produce the same table, and what next run would
    distinguish it?*

    ### 3.1 Choose a lens
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # Both controls live inside a form. Ungated, changing the dropdown or picking
    # a file *immediately* ran `JacobianLens.load()` — an 812 MB `torch.load` on
    # one click, with no way to change your mind. Nothing here touches the disk
    # until ▶.
    lens_form = mo.md(
        "**Lens source** {source}\n\n"
        "For **Student-fitted lens** — which of your fits: {fit_size}. A lens "
        "fitted in Appendix A during *this* session appears here with no file "
        "handling at all.\n\n"
        "Only for **Uploaded lens file** — a `jacobian_lens.pt` from a *previous* "
        "session; not needed for anything you fit today:\n\n"
        "{upload}\n\n"
        "The **scrambled control** is the reference lens with each layer's transport "
        "swapped for a different layer's. It is a *negative control*: if a mismatched "
        "transport still emits fluent, on-topic words, then fluency is not evidence "
        "that the readout is faithful. Costs no GPU and nothing is refitted."
    ).batch(
        source=mo.ui.dropdown(
            options={
                "Course reference lens": "reference",
                "Scrambled-layer control": "scrambled",
                "Student-fitted lens": "fitted",
                "Uploaded lens file": "uploaded",
            },
            value="Course reference lens",
        ),
        # Options are the constant fit sizes, NOT the file list: keeping this
        # cell's only ref as `mo` is what stops the form being re-minted (and its
        # value reset to None) the moment a fit completes. Sizes that do not exist
        # yet fall back with a message.
        fit_size=mo.ui.dropdown(
            options={"newest fit": 0, "25 prompts": 25, "50 prompts": 50,
                     "100 prompts": 100},
            value="newest fit",
        ),
        upload=mo.ui.file(filetypes=[".pt"], kind="button", max_size=2_000_000_000),
    ).form(
        submit_button_label="Load this lens",
        bordered=True,
    )
    lens_form
    return (lens_form,)


@app.cell(hide_code=True)
def _(
    MODEL_NAME,
    OUTPUT_ROOT,
    fitted_lens_files,
    jlens,
    lens,
    lens_form,
    mo,
    model,
):
    # Loads only on ▶. Prompt edits and repeated playground submits reuse the
    # already-loaded lens.
    _sel = lens_form.value
    _choice = _sel["source"] if _sel else "reference"
    _load_error = None
    _load_warning = None
    active_lens = None
    active_lens_label = ""
    try:
        if _choice == "uploaded":
            if not _sel or not _sel["upload"]:
                _load_warning = (
                    "**Uploaded lens file** is selected but no file was chosen — "
                    "falling back to the course reference lens."
                )
            else:
                _p = OUTPUT_ROOT / "uploaded_lens.pt"
                _p.write_bytes(_sel["upload"][0].contents)
                with mo.status.spinner(title="Loading uploaded lens (once)…"):
                    active_lens = jlens.JacobianLens.load(str(_p))
                active_lens_label = f"uploaded · {_sel['upload'][0].name}"
                # Shape compatibility is not identity. Any lens fitted for any
                # other model of the same residual width loads cleanly and then
                # produces confident, fluent, meaningless readouts — and upload is
                # the only way to carry a fit across sessions, so it is the path
                # students will actually use.
                _meta_model = getattr(active_lens, "fitted_for_model", None)
                if _meta_model is None:
                    _load_warning = (
                        f"`{_sel['upload'][0].name}` records no model identity, so "
                        "there is no way to check it was fitted for "
                        f"`{MODEL_NAME}`. Shape compatibility is not identity — "
                        "treat any agreement it reports as unverified."
                    )
                elif _meta_model != MODEL_NAME:
                    _load_error = (
                        f"that lens was fitted for `{_meta_model}`, not `{MODEL_NAME}`. "
                        "The residual bases are unrelated; the readout would be fluent "
                        "and meaningless."
                    )
                    active_lens = None
        elif _choice == "scrambled":
            # The negative control. `JacobianLens.__init__` takes a plain
            # {layer: J} dict, so shifting each layer's transport to a different
            # layer's costs no GPU and refits nothing.
            _src = lens.source_layers
            _shift = max(1, len(_src) // 2)
            active_lens = jlens.JacobianLens(
                {
                    _l: lens.jacobians[_src[(_i + _shift) % len(_src)]]
                    for _i, _l in enumerate(_src)
                },
                n_prompts=lens.n_prompts,
                d_model=lens.d_model,
            )
            active_lens_label = (
                f"scrambled-layer CONTROL · each J_l replaced by J of the layer "
                f"{_shift} positions away"
            )
        elif _choice == "fitted":
            if not fitted_lens_files:
                # Not `mo.stop`: cancelling here killed the form, the results
                # and the slice view all at once, and the recovery action was 250
                # lines further down.
                _load_warning = (
                    "No student-fitted lens exists in this session yet — run "
                    "**Appendix A** at the bottom of the notebook, then come back "
                    "and press ▶ again. Using the course reference lens meanwhile."
                )
            else:
                _want = int((_sel or {}).get("fit_size") or 0)
                if _want:
                    _match = [_f for _f in fitted_lens_files
                              if _f.name == f"jacobian_lens_n{_want}.pt"]
                    if not _match:
                        _load_warning = (
                            f"No {_want}-prompt fit on disk "
                            f"(have: {', '.join(_f.name for _f in fitted_lens_files)}). "
                            "Using your newest fit instead."
                        )
                _pick = _match[0] if _want and _match else fitted_lens_files[-1]
                with mo.status.spinner(title=f"Loading {_pick.name} (once)…"):
                    active_lens = jlens.JacobianLens.load(str(_pick))
                active_lens_label = f"student-fitted · {_pick.name}"
    except Exception as _e:
        _load_error = f"{type(_e).__name__}: {_e}"

    if active_lens is None and _load_error is None:
        active_lens = lens
        active_lens_label = "course reference lens · 1,000 prompts"

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
    mo.vstack([
        mo.callout(mo.md(_load_warning), kind="warn") if _load_warning else mo.md(""),
        mo.md(f"**Active lens:** {active_lens_label} · `n_prompts={active_lens.n_prompts}`"),
    ])
    return active_lens, active_lens_label


@app.cell(hide_code=True)
def _(demo_layers, lens, mo, model):
    # Options come from the *reference* lens, not the active one. marimo mints a
    # fresh id for any UI element whose defining cell re-runs, so depending on
    # `active_lens` here meant that switching lens source — the whole point of
    # investigation route 5 — wiped the prompt, the hypothesis and the result you
    # were about to compare against, all at once. The layer sets are checked
    # against the active lens in the result cell instead, where a mismatch can be
    # explained rather than silently erasing the form.
    # `demo_layers` is built in §2.1 by indexing into `lens.source_layers`, so it
    # is a subset by construction. Filtering it again was load-bearing when this
    # cell selected against `active_lens`; after the switch to `lens` it only
    # implied the two sets could disagree.
    _default_layers = list(demo_layers)

    def _validate(_value):
        # `.get` throughout: a batch's value is a partial dict until the frontend
        # has pushed state for every child, so indexing directly raises KeyError
        # on the first render rather than validating.
        if not _value or not (_value.get("prompt") or "").strip():
            return "Enter a non-empty prompt."
        if not (_value.get("hypothesis") or "").strip():
            return "Write a falsifiable prediction before running."
        if not _value.get("layers"):
            return "Select at least one fitted layer."
        return None

    _template = (
        "### 3.2 State a prediction and choose variables\n\n"
        "*The initial controls reproduce the guided comparison; after that, change one "
        "variable at a time. This form keeps its contents when you switch lens source "
        "above — that is what makes route 5 runnable.*\n\n"
        "*Cost: the numeric comparison is ~2 s. Ticking the slice adds ~10–30 s — "
        "the same for your first slice as your tenth, since the vocabulary was "
        "indexed once in §1.4. Appendix A is 5–20 min depending on the size you "
        "pick.*\n\n"
        "**Prediction before ▶** — name a layer/position trend that could be wrong:\n\n"
        "{hypothesis}\n\n"
        "**Prompt** {prompt}\n\n"
        f"Probe **{{position_from_end}} token(s) from the end** "
        "(1 = final prompt token, whose residual predicts the unseen continuation).\n\n"
        f"**Fitted layers** {{layers}} — this model has **{model.n_layers}** layers and "
        f"the reference lens fits **{len(lens.source_layers)}** of them "
        f"(`{lens.source_layers[0]}`–`{lens.source_layers[-1]}`). Layer *l* is counted "
        "from the embedding, so depth fraction = *l* / "
        f"{model.n_layers}.\n\n"
        "Compare top **{top_k}** candidates.\n\n"
        "---\n\n"
        "**Also build the layer × position slice view** {make_slice} *(rendered "
        "below the result table, in the notebook)*\n\n"
        "Slice layer stride {slice_stride} · last positions {slice_window} · "
        "word-like display {mask_display} *(these three do nothing unless the box "
        "above is ticked)*"
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
            options={
                f"Layer {_layer} ({_layer / model.n_layers:.0%} depth)": _layer
                for _layer in lens.source_layers
            },
            value=[
                f"Layer {_layer} ({_layer / model.n_layers:.0%} depth)"
                for _layer in _default_layers
            ],
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
    # The form's `validate=` runs only in the submit-button handler; marimo's
    # Ctrl/Cmd+Enter shortcut sets the value directly and skips it. This gate is
    # the notebook's best pedagogical mechanism — it is what makes it structurally
    # impossible to run without a prediction on record — so it needs a backstop
    # rather than relying on which key the student pressed.
    mo.stop(
        not (_cfg.get("hypothesis") or "").strip(),
        mo.callout(
            mo.md(
                "**Write a falsifiable prediction first.** (Ctrl/Cmd+Enter skips "
                "the form's own check; the run is held here instead.)"
            ),
            kind="warn",
        ),
    )

    from jlens.vis import compute_slice as _compute_slice

    _MAX_SEQ = 512
    _prompt = _cfg["prompt"].strip()
    _input_ids = model.encode(_prompt, max_length=_MAX_SEQ)
    _seq_len = int(_input_ids.shape[1])
    _offset = int(_cfg["position_from_end"])

    # Truncation is silent inside `model.encode` (tokenizer truncation=True), and
    # the positions it leaves behind still print as perfectly reasonable absolute
    # indices. A long prompt would quietly make "N tokens from the end" mean "from
    # token 512" instead.
    _full_len = len(tokenizer(_prompt).input_ids)
    _truncated = _full_len > _seq_len
    _trunc_note = (
        mo.callout(
            mo.md(
                f"**This prompt was truncated** — it tokenizes to **{_full_len}** "
                f"tokens and only the first **{_seq_len}** are used. Your probe "
                f"offset counts back from token {_seq_len}, *not* from the end of "
                "the text you typed. Shorten the prompt if the part you care about "
                "is past the cut."
            ),
            kind="warn",
        )
        if _truncated
        else None
    )

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
    # The form's options come from the reference lens so it survives a lens
    # switch; the active lens may fit a different set, so reconcile here where
    # the mismatch can be explained instead of erasing the form.
    _missing = [_l for _l in _layers if _l not in active_lens.source_layers]
    _layers = [_l for _l in _layers if _l in active_lens.source_layers]
    mo.stop(
        not _layers,
        mo.callout(
            mo.md(
                f"None of the selected layers are fitted by **{active_lens_label}** "
                f"(it fits {active_lens.source_layers[0]}–{active_lens.source_layers[-1]}). "
                "Pick layers this lens actually has."
            ),
            kind="danger",
        ),
    )
    _layer_note = (
        mo.callout(
            mo.md(
                f"Dropped layer(s) {_missing} — not fitted by **{active_lens_label}**. "
                f"Comparing the remaining {len(_layers)}."
            ),
            kind="warn",
        )
        if _missing
        else None
    )

    with mo.status.spinner(title="Comparing readouts at the selected position…"):
        # Two calls, one per readout: each runs its own forward pass over the same
        # activations and only the transport differs. Left as-is deliberately —
        # collapsing them needs a change in `lens.apply`, and at ~2 s a submit the
        # duplicated pass is not what is costing the student anything.
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
    # The same reference `compare_readouts` ranks against — tracked in the slice
    # below so the rank map opens on the token this table is about.
    _target_id = int(_ml[0].float().argmax())
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

    # Compute the slice here; render it downstream, so that moving the view
    # controls never re-runs this cell. A visualization failure must not hide
    # the numeric comparison above, so a failed slice is a note, not a stop.
    playground_slice = None
    playground_prompt = _prompt
    playground_target_id = _target_id
    playground_title = f"Slice · {active_lens_label}"
    playground_description = (
        f"{active_lens_label} · probe offset {_position} · "
        f"stride {int(_cfg['slice_stride'])} · "
        f"last {int(_cfg['slice_window'])} positions · "
        f"word-like display {'on' if _cfg['mask_display'] else 'off'} · "
        f"{_prompt!r}"
    )
    # Settings in the filename: two downloads at different strides used to
    # arrive as `playground_slice.html` and `playground_slice (1).html`, with
    # nothing on either page saying which was which.
    playground_filename = (
        f"slice_{_cfg['slice_stride']}stride"
        f"_{_cfg['slice_window']}win"
        f"_off{_offset}"
        f"_{'masked' if _cfg['mask_display'] else 'raw'}"
        f"_{active_lens_label.split(' ')[0].lower()}.html"
    )
    if not _cfg["make_slice"]:
        _viz = mo.md("_Slice view skipped for this submit._")
    else:
        try:
            with mo.status.spinner(title="Computing the slice…"):
                playground_slice = _compute_slice(
                    model,
                    active_lens,
                    _prompt,
                    layer_stride=int(_cfg["slice_stride"]),
                    last_n_tokens=int(_cfg["slice_window"]),
                    max_tracked=64,
                    # Track the model's own target whatever else scores highly,
                    # so the rank map below opens on the token this result table
                    # is about instead of on an arbitrary frequent one.
                    pinned_token_ids={_target_id},
                    mask_display=bool(_cfg["mask_display"]),
                )
            _viz = None
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
            "fail outcomes would both teach you something.\n\n"
            "One caution when you sweep offsets or swap prompts: the reference "
            "here is `argmax` of the **final layer for that prompt at that "
            "position**, so changing either changes the target token. Two runs' "
            "ranks are then ranks *of different tokens* — comparable only if you "
            "say so explicitly."
        ),
        kind="neutral",
    )
    # `is not None`, not truthiness: `_viz` is None on a successful slice (the
    # view itself is rendered two cells down), and several of these entries are
    # marimo objects whose truthiness is either meaningless or noisy.
    mo.vstack([
        _n for _n in (_trunc_note, _layer_note, _table, _viz, _reflection)
        if _n is not None
    ])
    return (
        playground_description,
        playground_filename,
        playground_prompt,
        playground_slice,
        playground_target_id,
        playground_title,
    )


@app.cell(hide_code=True)
def _(mo, playground_slice, playground_target_id, token_gloss):
    from jlens.vis import token_label as _token_label

    mo.stop(
        playground_slice is None,
        mo.md(
            "_Tick **Also build the layer × position slice view** in the form "
            "above and press ▶ to get the grid here._"
        ),
    )

    _tracked = list(playground_slice.tracked_token_ids)
    _options = {}
    for _tid in _tracked:
        _label = repr(_token_label(playground_slice, _tid, token_gloss))
        if _label in _options:
            _label = f"{_label} · id {_tid}"
        _options[_label] = int(_tid)

    # The target is pinned at compute time, so it is always tracked: the rank
    # map opens on the same token the result table above ranks.
    _default = [_k for _k, _v in _options.items() if _v == int(playground_target_id)][:1]
    _positions = {
        f"{playground_slice.ctx_offset + _i}  "
        f"{playground_slice.context_token_strs[playground_slice.ctx_offset + _i]!r}": _i
        for _i in range(playground_slice.seq_len)
    }

    playground_view = mo.md(
        "**Track these tokens through the stack** {tokens} *(the model's own "
        "target for this run is selected by default)*\n\n"
        "**Read the exact readout at position** {position}"
    ).batch(
        tokens=mo.ui.multiselect(options=_options, value=_default, max_selections=4),
        position=mo.ui.dropdown(options=_positions, value=list(_positions)[-1]),
    )
    playground_view
    return (playground_view,)


@app.cell(hide_code=True)
def _(
    mo,
    playground_description,
    playground_filename,
    playground_prompt,
    playground_slice,
    playground_title,
    playground_view,
    standalone_page,
    token_gloss,
):
    from functools import partial as _partial

    # Underscore aliases: marimo forbids two cells defining the same name, and
    # §2.2's view cell already imports these under their real names.
    from jlens.vis import position_top_k_rows as _position_rows
    from jlens.vis import slice_grid_html as _grid_html
    from jlens.vis import token_rank_grid_html as _rank_html

    _sel = playground_view.value
    _tokens = [int(_t) for _t in (_sel["tokens"] or [])]
    _position = int(_sel["position"])

    mo.vstack([
        mo.Html(
            _grid_html(
                playground_slice,
                alt_token=token_gloss,
                # The settings and the lens ride with the picture. Two submits
                # at different strides are otherwise two identical-looking
                # grids, and the one on screen is whichever ran last.
                caption=playground_description,
            )
        ),
        mo.md("#### Rank of the tracked token(s)"),
        mo.Html(_rank_html(playground_slice, _tokens, alt_token=token_gloss)),
        mo.md(
            "#### Exact readout at prompt position "
            f"{playground_slice.ctx_offset + _position}"
        ),
        mo.ui.table(
            _position_rows(
                playground_slice,
                _position,
                top_k=5,
                token_ids=_tokens,
                alt_token=token_gloss,
            ),
            selection=None,
            pagination=False,
        ),
        mo.accordion({
            "Optional — the same slice as a standalone d3 page (a file to open "
            "in its own browser tab; nothing above needs it)": mo.download(
                data=_partial(
                    standalone_page,
                    playground_slice,
                    playground_prompt,
                    playground_title,
                    playground_description,
                ),
                filename=playground_filename,
                mimetype="text/html",
                label=f"⬇ Build and download {playground_filename}",
            )
        }),
    ])
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

    Fit $J_l$ over WikiText prompts and save a student lens file. This is a GPU job
    that pulls `datasets` on demand, so it runs **only when you click**.

    **Pick the prompt count deliberately.** Estimator quality saturates quickly, so
    a **25-prompt** fit takes 4–5 minutes — a real experiment inside one class
    period — while 100 takes 15–20. Each count is saved under its own filename, so
    fitting 25 and then 100 leaves you with *two* lenses to compare. That is what
    makes investigation route 5 answerable: which differences between lenses are
    architecture, and which are just estimation noise?
    """)
    return


@app.cell
def _(mo):
    fit_controls = mo.md(
        "Fit over {n_prompts} WikiText prompts.\n\n"
        "*25 ≈ 4–5 min · 50 ≈ 8–10 min · 100 ≈ 15–20 min. Each is saved separately.*"
    ).batch(
        n_prompts=mo.ui.dropdown(
            options={"25 prompts (~5 min)": 25,
                     "50 prompts (~10 min)": 50,
                     "100 prompts (~20 min)": 100},
            value="25 prompts (~5 min)",
        ),
    ).form(submit_button_label="▶ Fit this lens", bordered=True)
    fit_controls
    return (fit_controls,)


@app.cell
def _(MODEL_NAME, MODEL_REVISION, OUTPUT_ROOT, fit_controls, jlens, mo, model):
    from functools import partial as _partial
    from pathlib import Path as _Path

    _cfg = fit_controls.value
    # A form keeps its submitted value indefinitely (unlike `run_button`, which
    # resets itself). This cell refs the model and the config cells, so any
    # upstream re-run — editing MODEL_NAME, a kernel reconnect, Run-all — would
    # re-enter the body with no click and silently start another 20-minute fit.
    # Guard on the *artifact*, not the trigger: an existing file means the work is
    # already done.
    _n = int(_cfg["n_prompts"]) if _cfg else 0
    _dest = OUTPUT_ROOT / f"jacobian_lens_n{_n}.pt" if _cfg else None
    if _cfg and _dest.exists():
        _mb = _dest.stat().st_size / 2**20
        _out = mo.vstack([
            mo.md(
                f"`{_dest.name}` already exists ({_mb:.0f} MB) — not refitting. "
                "Delete the file, or choose a different size, to fit again."
            ),
            mo.download(
                data=_partial(_Path.read_bytes, _dest),
                filename=_dest.name,
                mimetype="application/octet-stream",
                label=f"⬇ Download {_dest.name} ({_mb:.0f} MB)",
            ),
        ])
    elif _cfg:
        import subprocess as _sp
        import sys as _sys

        # load_wikitext_prompts streams WikiText via `datasets`; install it on
        # demand (only when fitting) so ordinary runs stay lean and torch is
        # never touched.
        with mo.status.spinner(title="Installing datasets…"):
            _sp.run([_sys.executable, "-m", "pip", "install", "datasets"], check=True)
        from jlens.examples import load_wikitext_prompts

        with mo.status.spinner(title=f"Fitting a {_n}-prompt Jacobian lens…"):
            _prompts = load_wikitext_prompts(n_prompts=_n)
            _fitted = jlens.fit(
                model,
                _prompts,
                dim_batch=32,
                max_seq_len=128,
                checkpoint_path=str(OUTPUT_ROOT / f"ckpt_n{_n}.pt"),
            )
            # Stamp the model identity into the file. A lens is only meaningful
            # for the residual basis it was fitted in, but `d_model` agreement is
            # all a loader can otherwise check — and upload is how these files
            # travel between sessions and between students.
            _fitted.fitted_for_model = MODEL_NAME
            _fitted.fitted_for_revision = MODEL_REVISION
            _fitted.save(str(_dest))
        _mb = _dest.stat().st_size / 2**20
        _out = mo.vstack([
            mo.md(
                f"✅ Fitted **{_fitted.n_prompts} prompts** → `{_dest.name}` "
                f"({_mb:.0f} MB). Select **Student-fitted lens** in §3.1 and press "
                "▶ to use it."
            ),
            mo.callout(
                mo.md(
                    "**Nothing further is needed to use it here** — it is on this "
                    "session's disk and §3.1 can load it now. Download it only if "
                    "you want it *after* this session: the disk does not survive a "
                    "restart, and re-fitting costs the same minutes again."
                ),
                kind="neutral",
            ),
            mo.download(
                # Lazy, not eager bytes. marimo materializes an eager payload into
                # a virtual file *and* a shared-memory segment at render time; a
                # ~400 MB lens would be copied twice for a button nobody may click.
                # A callable is resolved only when the download is requested.
                data=_partial(_Path.read_bytes, _dest),
                filename=_dest.name,
                mimetype="application/octet-stream",
                label=f"⬇ Download {_dest.name} ({_mb:.0f} MB)",
            ),
        ])
    else:
        _out = mo.md("_Idle — choose a size and press ▶. Nothing runs until you do._")

    # Every fitted lens on disk, newest last. The §3.1 loader reads this, so a
    # fresh fit becomes selectable without the notebook tracking mtimes by hand.
    fitted_lens_files = sorted(
        OUTPUT_ROOT.glob("jacobian_lens_n*.pt"), key=lambda _p: _p.stat().st_mtime
    )
    _out
    return (fitted_lens_files,)


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
def _(MODEL_NAME, fitted_lens_files, mo, model, torch):
    if not fitted_lens_files:
        _msg = mo.md(
            "ℹ️ No student-fitted lens on disk yet. The guided demo and playground "
            "use the course reference lens."
        )
    else:
        _rows = []
        for _p in fitted_lens_files:
            # Read the metadata, not the lens. Every field this table shows is a
            # plain scalar in the checkpoint dict, but `JacobianLens.load` calls
            # `J.float()` on 31 2560x2560 matrices — turning a ~390 MB fp16 file
            # into ~780 MB of fp32 resident, per row, just to print four numbers.
            # `mmap=True` keeps the tensors on disk; we never touch them.
            _ck = torch.load(str(_p), map_location="cpu", weights_only=True, mmap=True)
            _layers = _ck["source_layers"]
            _rows.append({
                "File": _p.name,
                "Prompts": _ck["n_prompts"],
                "Fitted layers": len(_layers),
                "Structurally compatible": (
                    _ck["d_model"] == model.d_model
                    and bool(_layers)
                    and max(_layers) < model.n_layers
                ),
                "Fitted for": _ck.get("fitted_for_model") or "— not recorded —",
                "Size (MB)": round(_p.stat().st_size / 2**20),
            })
            del _ck
        _msg = mo.vstack([
            mo.ui.table(_rows, selection=None, pagination=False),
            mo.callout(
                mo.md(
                    "**Structural compatibility is not quality.** Matching `d_model` "
                    "and layer bounds says a file will load, nothing more — a lens "
                    f"fitted for a different model of the same width passes every "
                    f"check in this table. The `Fitted for` column is the only "
                    f"identity claim, and it should read `{MODEL_NAME}`. The real "
                    "check is behavioral: run the same held-out prompt through two "
                    "of these in §3 and see whether the difference looks like "
                    "estimation noise or like something systematic."
                ),
                kind="neutral",
            ),
        ], gap=0.4)
    _msg
    return


@app.cell
def _(fitted_lens_files, mo):
    from functools import partial as _partial
    from pathlib import Path as _P

    # Kept as a backstop; the fit cell already offers the download inline, next to
    # the result, which is where it is actually needed.
    #
    # `partial(_P.read_bytes, _p)`, NOT `lambda _f=_p: _f.read_bytes()`. Both are
    # lazy; only one is safe.
    #
    # marimo rewrites underscore-prefixed names to per-cell mangled ones. Written
    # as a lambda default over the comprehension variable, this line raised
    # `NameError: name '_cell_NCOB_p' is not defined. Did you mean
    # '_cell_pHFh_p'?` -- the default resolved against a *different* cell's
    # prefix. Reproduced in a fresh interpreter, and only when `_p` also appeared
    # as a lambda parameter in the fit cell above; adding or removing an unrelated
    # underscore name in either cell made it come and go. So the exact trigger is
    # incidental, which is the argument for not writing the construct at all:
    # `partial` binds the value with no default and no closure, so it cannot
    # depend on what other cells happen to name their locals.
    #
    # It mattered because this branch is empty until a student has fitted a lens
    # -- so the first time it could ever fire is in a classroom, on the download
    # button they need right after Appendix A.
    if fitted_lens_files:
        _dl = mo.vstack([
            mo.download(
                data=_partial(_P.read_bytes, _p),
                filename=_p.name,
                mimetype="application/octet-stream",
                label=f"⬇ {_p.name} ({_p.stat().st_size / 2**20:.0f} MB)",
            )
            for _p in fitted_lens_files
        ], gap=0.3)
    else:
        _dl = mo.md("_No fitted lens on disk yet._")
    _dl
    return


if __name__ == "__main__":
    app.run()
