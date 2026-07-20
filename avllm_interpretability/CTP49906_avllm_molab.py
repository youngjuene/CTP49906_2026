# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "torch==2.6.0",
#     "torchvision==0.21.0",
#     "transformers==4.52.0",
#     "accelerate==1.14.0",
#     "qwen-omni-utils==0.0.9",
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
    # AVLLM interpretability — molab demo

    **The question:** when Qwen2.5-Omni-3B captions a video, is it really using what it
    *sees* and what it *hears* — or narrating from language priors? A caption like
    *"a person is playing the piano"* looks equally correct either way, so the output
    alone can't tell you. This notebook opens the model up with two tools:

    1. **Logit Lens** — decode the model's intermediate predictions at audio-token
       positions across thinker layers: watch *when* (and whether) audio content
       becomes legible inside the network.
    2. **Attention Knockout** — surgically cut one information pathway
       (source→target attention) and re-run: see *causally* what breaks when a
       modality is taken away.

    Read the fixed run top-to-bottom once for the mechanics — then the real work is
    the two interactive sections at the bottom (🎛️ and 🎯), where **you** pick the
    clip, the prompt, and the intervention. The habit to practice there: before every
    ▶, write down a falsifiable hypothesis — *"if I block X, the result should change
    like Y"* — and then check whether the model agrees.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Running in molab

    - **GPU:** attach one via the notebook-specs button in the header; this notebook
      uses `cuda:0`. The two 3B models load comfortably in molab's VRAM.
    - **Dependencies:** the setup cell pip-installs them into the kernel (molab does
      not honor the `# /// script` block automatically) and restores
      `torchvision.io.read_video` with a small PyAV shim, since molab's bundled
      torchvision no longer ships a video decoder.
    - The experiment code (`src/`) and the sample clip are cloned from
      `youngjuene/CTP49906_2026` by the setup cell below.
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
        ("transformers", "transformers", "4.52.0", "transformers==4.52.0"),
        ("accelerate", "accelerate", "1.14.0", "accelerate==1.14.0"),
        ("qwen_omni_utils", "qwen-omni-utils", None, "qwen-omni-utils==0.0.9"),
        ("av", "av", None, "av"),  # PyAV — backs the video-decode shim below
    ])

    def _ensure_video_reader():
        # molab ships its own recent torch/torchvision and ignores the
        # `# /// script` pins above. torchvision >= 0.23 dropped the built-in
        # video decoder, so `torchvision.io.read_video` no longer exists and
        # qwen-omni-utils' default torchvision backend dies with
        # `AttributeError: module 'torchvision.io' has no attribute 'read_video'`.
        # PyAV is already installed (qwen uses it to read the audio track), so
        # restore read_video on top of PyAV — no version-fragile CUDA wheels
        # (torchcodec/decord) and no reliance on system codecs.
        import torchvision

        if hasattr(torchvision.io, "read_video"):
            return  # normal torchvision (e.g. the pinned 0.21.0) — nothing to do
        import av
        import numpy as np
        import torch

        def _read_video_pyav(
            filename, start_pts=0.0, end_pts=None, pts_unit="sec", output_format="TCHW"
        ):
            # Minimal torchvision.io.read_video replacement covering the single
            # call qwen makes: it only reads `video.size(0)` and `info["video_fps"]`.
            if isinstance(filename, str) and filename.startswith("file://"):
                filename = filename[len("file://") :]
            container = av.open(filename)
            try:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                rate = stream.average_rate or stream.guessed_rate or stream.base_rate
                video_fps = float(rate) if rate else 30.0
                frames = []
                for frame in container.decode(video=0):
                    ts = frame.time
                    if pts_unit == "sec" and ts is not None:
                        if ts < start_pts:
                            continue
                        if end_pts is not None and ts > end_pts:
                            break
                    frames.append(frame.to_ndarray(format="rgb24"))  # (H, W, C) uint8
            finally:
                container.close()
            if frames:
                video = torch.from_numpy(np.stack(frames))  # (T, H, W, C)
            else:
                video = torch.zeros((0, 0, 0, 3), dtype=torch.uint8)
            if output_format.upper() == "TCHW":
                video = video.permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)
            # qwen extracts audio separately (process_audio_info), so an empty
            # placeholder here is fine; it only unpacks and discards this value.
            audio = torch.zeros((1, 0), dtype=torch.float32)
            return video, audio, {"video_fps": video_fps, "audio_fps": None}

        torchvision.io.read_video = _read_video_pyav
        print("patched torchvision.io.read_video (PyAV shim) for molab compatibility")

    _ensure_video_reader()

    # The experiment code (src/) and sample video live under the
    # `avllm_interpretability/` subdirectory of this repo. If the clone already
    # exists, hard-sync it to REPO_REF so pushed fixes reach molab (a kernel
    # restart is still needed to re-import updated modules).
    #
    # REPO_REF selects which branch or tag to sync: "main" for normal class use;
    # a feature branch to smoke-test unmerged work; a release tag (risk R7 in
    # the PRD) to pin the semester so September pushes can't change what
    # students execute mid-course. Works for branches and tags alike (fetch +
    # FETCH_HEAD, not origin/<branch>).
    REPO_REF = "main"
    REPO_DIR = Path("CTP49906_2026").resolve()
    if REPO_REF != "main":
        print(f"⚠️ REPO_REF={REPO_REF!r} — this notebook is pinned to a non-main ref.")
    if REPO_DIR.exists():
        with mo.status.spinner(title=f"Updating CTP49906_2026 to latest {REPO_REF}…"):
            subprocess.run(
                ["git", "-C", str(REPO_DIR), "fetch", "--depth", "1", "origin", REPO_REF],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(REPO_DIR), "reset", "--hard", "FETCH_HEAD"], check=True
            )
    else:
        with mo.status.spinner(title=f"Cloning CTP49906_2026 @ {REPO_REF} (src + sample video)…"):
            subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", REPO_REF,
                 "https://github.com/youngjuene/CTP49906_2026.git", str(REPO_DIR)],
                check=True,
            )
    PROJECT_DIR = REPO_DIR / "avllm_interpretability"
    assert PROJECT_DIR.is_dir(), f"expected code dir not found: {PROJECT_DIR}"
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    print("project dir:", PROJECT_DIR)
    return (PROJECT_DIR,)


@app.cell
def _(PROJECT_DIR):
    # F5a — GPU-free replay. Flip to True to render every non-interactive
    # fixed-run plot from committed artifacts (no GPU, no 8 GB download): a
    # break-glass mode for when molab's GPU is unavailable. Default False = live
    # model. The interactive playground / teacher-forcing sections still need a
    # GPU and fail loudly if submitted in this mode. Generate the artifacts on a
    # GPU with:
    #   python avllm_interpretability/scripts/generate_precompute.py
    USE_PRECOMPUTED = False
    PRECOMPUTED_DIR = PROJECT_DIR / "precomputed"
    if USE_PRECOMPUTED:
        print(f"USE_PRECOMPUTED=True — replaying the fixed run from {PRECOMPUTED_DIR} (no GPU)")
    return PRECOMPUTED_DIR, USE_PRECOMPUTED


@app.cell
def _(USE_PRECOMPUTED):
    import torch

    if USE_PRECOMPUTED:
        DEVICE = torch.device("cpu")
        print(f"torch={torch.__version__}, USE_PRECOMPUTED=True → CPU (no GPU required)")
    else:
        assert torch.cuda.is_available(), (
            "No GPU visible. In molab, attach a GPU via the notebook-specs button in the header. "
            "(Or set USE_PRECOMPUTED=True in the cell above to replay the fixed run from committed artifacts.)"
        )
        DEVICE = torch.device("cuda:0")
        _free, _total = torch.cuda.mem_get_info(0)
        print(f"torch={torch.__version__}, GPU={torch.cuda.get_device_name(0)}, VRAM={_total / 2**30:.0f} GiB")
    return DEVICE, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parameters

    Edit these to point at your own video or change the intervention. On molab's
    large GPU you can safely raise `NFRAMES`.

    The parameters are split across three cells *by re-run cost* (marimo re-runs
    every cell downstream of an edit): the model id — editing it reloads the
    models; the paths; and the **knobs** — prompts, rules, frames. Tweak the knob
    cell freely: it re-runs the experiments against the already-loaded models
    (seconds), never the model loads themselves.
    """)
    return


@app.cell
def _():
    # Own cell on purpose: nothing but a genuine model change should ever
    # invalidate the loader cells below.
    MODEL_PATH = "Qwen/Qwen2.5-Omni-3B"
    return (MODEL_PATH,)


@app.cell
def _(PROJECT_DIR):
    VIDEO_PATH = PROJECT_DIR / "assets" / "02321.mp4"

    RESULTS_DIR = PROJECT_DIR / "notebook_results"
    RESULTS_DIR.mkdir(exist_ok=True)
    LOGIT_CSV_PATH = RESULTS_DIR / "logit_lens_audio_token_analysis.csv"

    assert VIDEO_PATH.is_file(), f"video not found: {VIDEO_PATH}"
    print("video:", VIDEO_PATH)
    return LOGIT_CSV_PATH, VIDEO_PATH


@app.cell
def _():
    # The knobs — cheap to tweak: re-runs the experiments, not the model loads.
    NFRAMES = 8
    LOGIT_PROMPT = "Describe what you hear in the video"
    ATTENTION_PROMPT = "Describe what you see and hear in the video"
    KNOCKOUT_RULES = [("generated", "video", 0, 36)]  # block generated→video, all 36 thinker layers
    MAX_NEW_TOKENS = 32
    ATTENTION_CAPTURE_LAYERS = (0, 2)
    return (
        ATTENTION_CAPTURE_LAYERS,
        ATTENTION_PROMPT,
        KNOCKOUT_RULES,
        LOGIT_PROMPT,
        MAX_NEW_TOKENS,
        NFRAMES,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preview the video (frames + embedded audio go to Qwen)
    """)
    return


@app.cell
def _(VIDEO_PATH, mo):
    mo.video(src=VIDEO_PATH.read_bytes(), width=640)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model and input helpers
    """)
    return


@app.cell
def _(DEVICE, MODEL_PATH, PROJECT_DIR):
    import csv
    from collections import Counter

    import matplotlib.pyplot as plt
    import numpy as np
    from qwen_omni_utils import process_mm_info
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    _ = PROJECT_DIR  # ensure the clone / sys.path cell ran first
    from src.attention_knockout_experiment import block_attention
    from src.attention_knockout_experiment import (
        create_token_type_mapping as create_attention_token_mapping,
    )
    from src.logitlens_experiment import (
        analyze_and_save_audio_logits_to_csv,
        clear_logit_lens_hooks,
        create_token_type_mapping,
        register_logit_lens_hooks,
    )

    def load_model_and_processor(attn_implementation):
        _model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            MODEL_PATH, torch_dtype="auto", attn_implementation=attn_implementation
        )
        # Free the talker + (float32) token2wav BEFORE moving to GPU so they never
        # occupy VRAM — this experiment only needs the thinker.
        _model.disable_talker()
        _model = _model.to(DEVICE)
        _model.eval()
        _proc = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
        return _model, _proc

    # video_path/nframes are arguments, not closures: this cell must depend only
    # on the model constants, or a knob tweak would cascade into the loaders.
    def prepare_video_inputs(model, processor, prompt, token_mapping_fn, video_path, nframes):
        _conv = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "video", "video": str(video_path), "nframes": nframes},
        ]}]
        _text = processor.apply_chat_template(_conv, add_generation_prompt=True, tokenize=False)
        _audios, _images, _videos = process_mm_info(_conv, use_audio_in_video=True)
        _inputs = processor(
            text=_text, audio=_audios, images=_images, videos=_videos,
            return_tensors="pt", padding=True, use_audio_in_video=True,
        )
        _inputs = {k: v.to(model.device) for k, v in _inputs.items()}
        _types = token_mapping_fn(_inputs["input_ids"], model.config.thinker_config)
        print("token counts:", Counter(_types))
        return _inputs, _types

    return (
        Counter,
        analyze_and_save_audio_logits_to_csv,
        block_attention,
        clear_logit_lens_hooks,
        create_attention_token_mapping,
        create_token_type_mapping,
        csv,
        load_model_and_processor,
        np,
        plt,
        prepare_video_inputs,
        register_logit_lens_hooks,
    )


@app.cell
def _(USE_PRECOMPUTED, load_model_and_processor, mo):
    # Dedicated loader cell: depends only on the model constants, so tweaking
    # prompts/rules/frames re-runs the experiments against this already-loaded
    # instance instead of reloading ~7 GB of weights.
    if USE_PRECOMPUTED:
        logit_model, logit_processor = None, None
    else:
        with mo.status.spinner(title="Loading Qwen2.5-Omni-3B (SDPA, first run downloads ~8 GB)…"):
            logit_model, logit_processor = load_model_and_processor("sdpa")
    return logit_model, logit_processor


@app.cell
def _(PRECOMPUTED_DIR, USE_PRECOMPUTED, load_model_and_processor, mo):
    # Dedicated loader cell for the eager model (knockout hooks + both
    # playgrounds). In precomputed mode a layer-count stub stands in so the
    # playground forms can render; it cannot compute, and the forms fail loudly
    # if submitted.
    if USE_PRECOMPUTED:
        from src.precompute import StubModel as _StubModel
        from src.precompute import load_precompute as _load_pre

        attention_model = _StubModel(_load_pre(PRECOMPUTED_DIR)["meta"].get("n_layers", 36))
        attention_processor = None
    else:
        with mo.status.spinner(title="Loading Qwen2.5-Omni-3B (eager attention)…"):
            attention_model, attention_processor = load_model_and_processor("eager")
    return attention_model, attention_processor


@app.cell
def _(attention_model):
    # Submit-to-submit caches for the two playground forms, keyed on
    # (clip name, clip bytes, nframes, prompt): "encode" holds prepared inputs +
    # token types, "caption" holds greedy caption ids for teacher forcing — so a
    # layer-band sweep re-encodes and re-captions nothing after the first ▶.
    # Depending on attention_model flushes them whenever the model is reloaded.
    _ = attention_model
    playground_caches = {"encode": {}, "caption": {}}

    def cache_put(cache, key, value, keep=4):
        cache[key] = value
        while len(cache) > keep:  # bound GPU-resident entries; FIFO eviction
            cache.pop(next(iter(cache)))
        return value

    return cache_put, playground_caches


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Logit Lens

    A multimodal forward pass; the CSV analysis focuses on `audio` token positions.
    """)
    return


@app.cell
def _(
    LOGIT_CSV_PATH,
    LOGIT_PROMPT,
    MAX_NEW_TOKENS,
    NFRAMES,
    PRECOMPUTED_DIR,
    USE_PRECOMPUTED,
    VIDEO_PATH,
    analyze_and_save_audio_logits_to_csv,
    clear_logit_lens_hooks,
    create_token_type_mapping,
    logit_model,
    logit_processor,
    mo,
    prepare_video_inputs,
    register_logit_lens_hooks,
    torch,
):
    if USE_PRECOMPUTED:
        from src.precompute import load_precompute as _load_pre

        _pre = _load_pre(PRECOMPUTED_DIR)
        logit_csv_written = _pre["logit_csv"]
        _logit_out = mo.vstack([
            mo.callout(mo.md("**Replayed from cache** — precomputed, no GPU."), kind="neutral"),
            mo.md(f"**Generated caption:**\n\n> {_pre['logit_caption']}"),
        ])
    else:
        logit_inputs, logit_token_types = prepare_video_inputs(
            logit_model, logit_processor, LOGIT_PROMPT, create_token_type_mapping,
            VIDEO_PATH, NFRAMES,
        )

        register_logit_lens_hooks(logit_model)
        try:
            with mo.status.spinner(title="Forward pass + decoding per-layer predictions…"):
                with torch.no_grad():
                    _ = logit_model.thinker(**logit_inputs, output_hidden_states=True)
                analyze_and_save_audio_logits_to_csv(
                    logit_model, logit_processor, logit_token_types, filename=str(LOGIT_CSV_PATH)
                )
        finally:
            clear_logit_lens_hooks()
        logit_csv_written = LOGIT_CSV_PATH

        with mo.status.spinner(title="Generating the caption…"):
            with torch.no_grad():
                # Generate from the thinker directly: the omni wrapper's generate()
                # defaults to audio output and errors because we freed the talker
                # (transformers >=5 dropped the has-talker fallback). The thinker is a
                # plain causal LM and yields the same text, version-agnostically.
                # do_sample=False pins greedy decoding explicitly: the shipped
                # generation_config is an empty stub that happens to resolve to
                # greedy today; an upstream change must not silently flip it.
                _ids = logit_model.thinker.generate(
                    **logit_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
                )
        _logit_caption = logit_processor.batch_decode(
            _ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        _logit_out = mo.md(f"**Generated caption:**\n\n> {_logit_caption}")
    _logit_out
    return (logit_csv_written,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Logit-lens diversity by layer

    Left: how many distinct decoded predictions appear across audio-token positions
    at each layer. Right: how dominant the most common prediction is.
    """)
    return


@app.cell
def _(Counter, USE_PRECOMPUTED, csv, logit_csv_written, mo, np, plt):
    with open(logit_csv_written, newline="", encoding="utf-8") as _fh:
        _all = list(csv.reader(_fh))
    _header, _data = _all[0], _all[1:]
    _layer_names = _header[2:]
    _preds = list(zip(*(r[2:] for r in _data)))
    _unique = [len(set(p)) for p in _preds]
    _dominant = [Counter(p).most_common(1)[0][1] / len(p) for p in _preds]

    _x = np.arange(len(_layer_names))
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
    _axes[0].bar(_x, _unique, color="#4C78A8")
    _axes[0].set(title="Logit-lens diversity by layer", xlabel="Thinker layer", ylabel="Unique predictions")
    _axes[1].plot(_x, _dominant, marker="o", color="#F58518")
    _axes[1].set(title="Most-common prediction share", xlabel="Thinker layer", ylabel="Share", ylim=(0, 1))
    for _ax in _axes:
        _ax.grid(axis="y", alpha=0.25)
    if USE_PRECOMPUTED:
        _div_out = mo.vstack([
            mo.callout(mo.md("**Replayed from cache** — precomputed, no GPU."), kind="neutral"),
            _fig,
        ])
    else:
        _div_out = _fig
    _div_out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Attention Knockout

    `KNOCKOUT_RULES` are `(source_type, target_type, start_layer, end_layer)` tuples.
    The default blocks generated tokens from attending to video tokens in layers 0–35.
    """)
    return


@app.cell
def _(Counter, KNOCKOUT_RULES, attention_token_types, mo, plt):
    # Visual primer: THIS run's actual token sequence, and what the default rule
    # cuts. Rebuilt from attention_token_types every run, so it is never a stale
    # abstraction — the striping is the real audio/video interleaving.
    _TYPE_COLORS = {
        "query_text": "#9E9E9E", "audio": "#4C78A8", "video": "#F58518",
        "image": "#B279A2", "generated": "#54A24B", "answer": "#E45756",
    }
    _pt = list(attention_token_types)
    _n = max(1, len(_pt))

    # Contiguous runs -> (type, start, width)
    _runs = []
    for _i, _t in enumerate(_pt):
        if _runs and _runs[-1][0] == _t:
            _runs[-1][2] += 1
        else:
            _runs.append([_t, _i, 1])

    _fig, _ax = plt.subplots(figsize=(14, 2.8), constrained_layout=True)
    for _t, _s, _w in _runs:
        _ax.barh(0, _w, left=_s, height=0.55, color=_TYPE_COLORS.get(_t, "#cccccc"))
    # The reply, appended after the prompt: `generated` while decoding,
    # `answer` once teacher-forced back in.
    _gap, _rw = _n * 0.015, max(8.0, _n * 0.08)
    _r0 = _n + _gap
    _ax.barh(0, _rw, left=_r0, height=0.55, color="#54A24B", hatch="//", edgecolor="white")
    _ax.text(_r0 + _rw / 2, 0, "reply\n(`generated` / `answer`)",
             ha="center", va="center", fontsize=8)

    def _center_of(type_name):
        if type_name in ("generated", "answer"):
            return _r0 + _rw / 2
        _cands = [(_w, _s + _w / 2) for _t, _s, _w in _runs if _t == type_name]
        return max(_cands)[1] if _cands else None

    _src, _tgt, _a, _b = (KNOCKOUT_RULES[0] if KNOCKOUT_RULES
                          else ("generated", "video", 0, 36))
    _sx, _tx = _center_of(_src), _center_of(_tgt)
    if _sx is not None and _tx is not None:
        _ax.annotate("", xy=(_tx, 0.33), xytext=(_sx, 0.33),
                     arrowprops=dict(arrowstyle="-|>", color="#E45756", lw=2,
                                     connectionstyle="arc3,rad=-0.22"))
        _ax.text((_sx + _tx) / 2, 0.92,
                 f"✂  {_src} → {_tgt}   ·   layers [{_a}, {_b})",
                 ha="center", va="center", fontsize=10, color="#E45756")

    from matplotlib.patches import Patch as _Patch
    _counts = Counter(_pt)
    _legend = [
        _Patch(color=_TYPE_COLORS[_t], label=f"{_t} ({_counts[_t]})")
        for _t in ("query_text", "audio", "video", "image") if _counts.get(_t)
    ]
    _ax.legend(handles=_legend, loc="lower left", ncols=len(_legend),
               bbox_to_anchor=(0, -0.42), frameon=False, fontsize=9)
    _ax.set(xlim=(-_n * 0.01, _r0 + _rw + _n * 0.01), ylim=(-0.75, 1.15),
            title="How to read a knockout rule — this run's token sequence, one position per pixel-width")
    _ax.axis("off")

    mo.vstack([
        _fig,
        mo.md(
            "Each colored sliver is one token **position** — note the audio/video "
            "striping: the two modalities are *interleaved* in time, not separate "
            "blocks. A rule `(source, target, start, end)` means: **in layers "
            "`[start, end)` (end exclusive), forbid `source` tokens from attending "
            "to `target` tokens.** The arrow points *from the token doing the "
            "looking to the token being read* — and because the model is causally "
            "masked (a token sees only itself and earlier positions), the "
            "meaningful direction is almost always a **later** token reading an "
            "**earlier** one. That is why the reply (`generated` while decoding, "
            "`answer` when teacher-forced) is the natural *source*, and why "
            "`video → generated` would cut almost nothing."
        ),
    ])
    return


@app.cell
def _(
    ATTENTION_CAPTURE_LAYERS,
    ATTENTION_PROMPT,
    KNOCKOUT_RULES,
    MAX_NEW_TOKENS,
    NFRAMES,
    PRECOMPUTED_DIR,
    USE_PRECOMPUTED,
    VIDEO_PATH,
    attention_model,
    attention_processor,
    block_attention,
    create_attention_token_mapping,
    mo,
    prepare_video_inputs,
    torch,
):
    if USE_PRECOMPUTED:
        from src.precompute import load_precompute as _load_pre

        _pre = _load_pre(PRECOMPUTED_DIR)
        baseline_text = _pre["baseline_text"]
        knockout_text = _pre["knockout_text"]
        attention_summary = _pre["attention_summary"]
        attention_token_types = _pre["attention_token_types"]
        attention_inputs = None
        attention_baseline_ids = None
        _ko_rules = _pre["knockout_rules"]
        _ko_banner = mo.callout(mo.md("**Replayed from cache** — precomputed, no GPU."), kind="neutral")
    else:
        from src.precompute import summarize_attention as _summarize_attention

        attention_inputs, attention_token_types = prepare_video_inputs(
            attention_model, attention_processor, ATTENTION_PROMPT, create_attention_token_mapping,
            VIDEO_PATH, NFRAMES,
        )

        with mo.status.spinner(title="Baseline generation…"):
            with torch.no_grad():
                # Thinker-direct generation (see the logit cell): avoids the omni
                # wrapper's talker requirement. Knockout hooks live on the thinker's
                # layers, so they still fire below. Greedy (do_sample=False) so the
                # baseline caption — reused as C by the teacher-forced cell below —
                # is deterministic.
                _base = attention_model.thinker.generate(
                    **attention_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                    return_dict_in_generate=True,
                )
        attention_baseline_ids = _base.sequences
        baseline_text = attention_processor.batch_decode(
            _base.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        with block_attention(
            attention_model, KNOCKOUT_RULES, attention_token_types, len(attention_token_types),
            track_attention=True, capture_layer_range=ATTENTION_CAPTURE_LAYERS,
        ) as _cap:
            with mo.status.spinner(title="Knockout generation…"):
                with torch.no_grad():
                    _ko = attention_model.thinker.generate(
                        **attention_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                        output_attentions=True, return_dict_in_generate=True,
                    )
            _captured = {layer: list(v) for layer, v in _cap.items()}
        knockout_text = attention_processor.batch_decode(
            _ko.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        # Reduce to the plot-ready matrix now, so the heatmap cell consumes the
        # same shape whether live or replayed (raw tensors are never committed).
        attention_summary = _summarize_attention(_captured, attention_token_types)
        _ko_rules = KNOCKOUT_RULES
        _ko_banner = None

    _ko_cmp = mo.hstack(
        [
            mo.vstack([mo.md("**Baseline**"), mo.md(baseline_text)]),
            mo.vstack([mo.md(f"**Knockout** `{_ko_rules}`"), mo.md(knockout_text)]),
        ],
        widths="equal",
    )
    _ko_display = mo.vstack([_ko_banner, _ko_cmp]) if _ko_banner is not None else _ko_cmp
    _ko_display
    return (
        attention_baseline_ids,
        attention_inputs,
        attention_summary,
        attention_token_types,
        knockout_text,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Captured attention by key modality

    A **descriptive** summary (not causal importance): for each captured layer we
    average heads and sum the final query's attention over each token group. Read it
    alongside the baseline-vs-knockout text above.
    """)
    return


@app.cell
def _(attention_summary, mo, np, plt):
    # `attention_summary` is `(layers, modalities, matrix)` — computed live from
    # captured tensors, or loaded from the committed matrix in USE_PRECOMPUTED
    # mode. Same shape either way, so the plot below is unchanged.
    if attention_summary is None:
        _out = mo.md("> No attention tensors were returned by this build; the text comparison above is the result.")
    else:
        _layers, _mods, _mat = attention_summary
        _mat = np.asarray(_mat, dtype=float)
        _fig, _ax = plt.subplots(figsize=(8, max(3, len(_layers) * 0.6)), constrained_layout=True)
        _im = _ax.imshow(_mat, aspect="auto", cmap="magma")
        _ax.set(
            title="Captured final-query attention mass by key modality",
            xlabel="Key modality", ylabel="Thinker layer",
            xticks=np.arange(len(_mods)), xticklabels=_mods,
            yticks=np.arange(len(_layers)), yticklabels=_layers,
        )
        for _ri in range(_mat.shape[0]):
            for _ci in range(_mat.shape[1]):
                _ax.text(_ci, _ri, f"{_mat[_ri, _ci]:.2f}", ha="center", va="center", color="white", fontsize=9)
        _fig.colorbar(_im, ax=_ax, label="Attention mass")
        _out = _fig
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Teacher-forced Δ log-likelihood (fixed parameters)

    The string diff above is **visceral but binary** — you can't see a *small*
    effect, and it depends on how generation happens to continue. This cell asks
    the same question as a **measurement**: it feeds the baseline caption back in
    tagged `answer` and scores, per token, **how much less the model believes what
    it said** when the answer is cut off from the same target modality as
    `KNOCKOUT_RULES` (same clip, same prompt, same layers — only the source becomes
    `answer`, because the caption is now *input*, not generation).

    **Δ = knockout − baseline** per caption token; *negative = believed less = hot
    color*. The 🎯 playground section below runs the same measurement on your own
    clip, prompt, and layer band.
    """)
    return


@app.cell
def _(
    KNOCKOUT_RULES,
    USE_PRECOMPUTED,
    attention_baseline_ids,
    attention_inputs,
    attention_model,
    attention_processor,
    attention_token_types,
    mo,
):
    if USE_PRECOMPUTED:
        fixed_tf_result = None
        _fixed_out = mo.callout(
            mo.md(
                "**Teacher forcing needs the live model** — this cell is skipped while "
                "`USE_PRECOMPUTED=True`. (Cached replay of this measurement lands with F5b.)"
            ),
            kind="warn",
        )
    else:
        from src.teacher_forcing import render_delta_strip as _fixed_strip
        from src.teacher_forcing import teacher_forced_delta as _fixed_tfd

        # Mirror the params-cell intervention with `answer` as the source: the
        # caption is input now, so `answer → target` is the measurable counterpart
        # of the generation-time `generated → target` diff above.
        _fixed_rules = [("answer", _t, _a, _b) for (_s, _t, _a, _b) in KNOCKOUT_RULES]
        _fixed_prompt_len = attention_inputs["input_ids"].shape[1]
        _fixed_c_ids = attention_baseline_ids[:, _fixed_prompt_len:]

        fixed_tf_result = None
        try:
            with mo.status.spinner(title="Teacher-forced scoring (2 forward passes)…"):
                fixed_tf_result = _fixed_tfd(
                    attention_model,
                    attention_processor,
                    attention_inputs,
                    attention_token_types,
                    _fixed_rules,
                    cached_caption_ids=_fixed_c_ids,
                )
        except Exception as _e:  # noqa: BLE001 — surface any failure in-notebook
            _fixed_out = mo.callout(
                mo.md(f"**Teacher-forced scoring failed** — `{type(_e).__name__}: {_e}`"),
                kind="danger",
            )

        if fixed_tf_result is not None:
            _fixed_delta = [float(x) for x in fixed_tf_result["delta"].detach().cpu().float().tolist()]
            _fixed_total = fixed_tf_result["delta_total"]
            _fixed_rule_txt = " + ".join(f"`answer→{_r[1]}` [{_r[2]},{_r[3]})" for _r in _fixed_rules)
            _fixed_out = mo.vstack([
                mo.md(f"**Knockout** {_fixed_rule_txt} &nbsp;·&nbsp; baseline caption teacher-forced as `answer`"),
                mo.hstack([
                    mo.stat(
                        value=f"{_fixed_total:+.2f}",
                        label="Σ Δ log-lik (nats)",
                        caption="knockout − baseline · negative = believed less",
                        direction="decrease" if _fixed_total < 0 else "increase",
                        bordered=True,
                    ),
                    mo.stat(
                        value=str(len(_fixed_delta)),
                        label="Caption tokens scored",
                        caption="greedy baseline, teacher-forced",
                        bordered=True,
                    ),
                ], widths="equal", gap=1),
                mo.md("###### Per-token Δ log-likelihood (hover a word for its tokens' nats)"),
                mo.Html(
                    "<div style='line-height:2.1;font-family:monospace;font-size:15px'>"
                    + _fixed_strip(fixed_tf_result["caption_tokens"], _fixed_delta)
                    + "</div>"
                ),
            ])
    _fixed_out
    return (fixed_tf_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wrap-up
    """)
    return


@app.cell
def _(knockout_text, logit_csv_written, mo):
    _ = knockout_text  # depend on the knockout run
    _ok = logit_csv_written.is_file() and logit_csv_written.stat().st_size > 0
    mo.md(
        f"### Done — the fixed run ends here; the exploration starts below\n\n"
        f"- Logit-lens CSV written: **{_ok}** — `{logit_csv_written}`\n"
        f"- Baseline vs knockout compared, and the caption scored under teacher forcing, above.\n\n"
        "Two interactive sections follow, each with suggested missions in its intro:\n\n"
        "- **🎛️ Diversity scoreboard** — how do the *audio positions* respond to your "
        "prompt, clip, and knockout choices?\n"
        "- **🎯 Teacher forcing** — how much less does the model *believe its own caption* "
        "when you cut a pathway — and on **your** clip, which words lose the belief?\n\n"
        "Form the hypothesis first, then press ▶. (You can also edit `KNOCKOUT_RULES`, "
        "`NFRAMES`, or `VIDEO_PATH` in the parameters cell to change the fixed run itself.)"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🎛️ Interactive: logit-lens diversity scoreboard

    Everything above ran once with the fixed parameters. This section turns the
    **logit-lens diversity** measurement into a live playground: pick a clip, the
    number of frames, the prompt, and (optionally) an attention knockout to apply
    **during** the forward pass, then submit to score every thinker layer by how
    many *distinct* tokens it decodes across the audio-token positions.

    Nothing runs until you press submit (the controls are wrapped in a form), and
    the eager model from the knockout experiment is reused — so runs are quick and
    need no extra VRAM.

    The score is measured at **audio** token positions, so knockouts with an `audio`
    source reshape it most directly (a `generated` source does nothing in a forward
    pass). Build one rule with the dropdowns, or enter several in the advanced field.

    ### What to try (predict the trend *before* you press ▶)

    - **Steer the prompt, touch nothing else.** Run the same clip with *"describe
      what you **hear**"* and then *"describe what you **see**"* — no knockout.
      Hypothesis first: should what the prompt asks for change what the *audio
      positions* decode, before any pathway is cut? Whatever you find is a finding.
    - **Hunt the fusion band.** Knock out `audio → video` over `[0, 12)`, then
      `[12, 24)`, then `[24, 36)`. Which band moves the Δ trend the most — early,
      middle, or late? What would each answer imply about *where* the visual stream
      touches the audio representations?
    - **Starve the stream.** Stack `audio,video,0,36 ; audio,query_text,0,36` in the
      advanced field. Is the effect of cutting both neighbors the sum of cutting
      each alone — and what would it mean if it isn't?

    A flat Δ is a result too — record it. And be careful *reading* a big one: a
    diversity shift tells you the representations changed, not by itself *why*
    (that question gets its own week). Log every run in the lab worksheet
    (`avllm_interpretability/WORKSHEET.md`): hypothesis **before** ▶, result,
    verdict. The full experiment catalog is in the repo's
    `avllm_interpretability/README.md`.
    """)
    return


@app.cell
def _(KNOCKOUT_RULES, LOGIT_PROMPT, NFRAMES, attention_model, mo):
    _n_layers = len(attention_model.thinker.model.layers)
    _modalities = ["audio", "video", "query_text", "image", "generated"]
    # Scoreboard-appropriate defaults: the source must be a modality that is
    # actually PRESENT in the prompt, so `audio` (the positions being scored) —
    # not the params cell's `generated`, which is inert in a forward pass. The
    # target follows the params rule; the window spans every layer ([0, N)).
    _def_source = "audio"
    _def_target = KNOCKOUT_RULES[0][1] if KNOCKOUT_RULES else "video"

    _hint = (
        f"Source/target ∈ `audio · video · query_text · image · generated` — but "
        f"`generated` is **inert** here (there are no generated tokens during a forward "
        f"pass). Layer `end` is exclusive; this thinker has **{_n_layers}** layers, so "
        f"`[0, {_n_layers})` spans all of them."
    )
    _template = (
        "**Video** — upload `mp4 / mov / mkv / webm`, or leave empty to reuse the default clip:\n\n"
        "{video}\n\n"
        "**Frames sampled from the clip** {nframes}\n\n"
        "**Prompt** {prompt}\n\n"
        "---\n\n"
        "**Apply attention knockout during the pass** {ko_enable}\n\n"
        "Single rule — block {ko_source} → {ko_target} across thinker layers {ko_layers}\n\n"
        "Advanced — several rules as `source,target,start,end` separated by `;` "
        "(overrides the single rule above when filled):\n\n"
        "{ko_rules_text}\n\n"
        + _hint + "\n\n"
        "**Also run a no-knockout baseline to compare against** {compare}"
    )

    scoreboard_controls = mo.md(_template).batch(
        video=mo.ui.file(
            filetypes=[".mp4", ".mov", ".mkv", ".webm", ".avi"],
            multiple=False,
            kind="area",
        ),
        nframes=mo.ui.slider(
            2, 32, step=2, value=NFRAMES, show_value=True, include_input=True
        ),
        prompt=mo.ui.text(value=LOGIT_PROMPT, full_width=True),
        ko_enable=mo.ui.checkbox(value=bool(KNOCKOUT_RULES)),
        ko_source=mo.ui.dropdown(_modalities, value=_def_source),
        ko_target=mo.ui.dropdown(_modalities, value=_def_target),
        ko_layers=mo.ui.range_slider(
            0, _n_layers, step=1, value=[0, _n_layers], show_value=True
        ),
        ko_rules_text=mo.ui.text(
            placeholder="e.g.  audio,video,0,36 ; audio,image,0,36", full_width=True
        ),
        compare=mo.ui.checkbox(value=True),
    ).form(
        submit_button_label="▶ Run logit-lens diversity",
        bordered=True,
    )
    scoreboard_controls
    return (scoreboard_controls,)


@app.cell
def _(
    Counter,
    LOGIT_CSV_PATH,
    LOGIT_PROMPT,
    VIDEO_PATH,
    analyze_and_save_audio_logits_to_csv,
    attention_model,
    attention_processor,
    block_attention,
    cache_put,
    clear_logit_lens_hooks,
    create_attention_token_mapping,
    csv,
    scoreboard_controls,
    mo,
    np,
    playground_caches,
    plt,
    register_logit_lens_hooks,
    torch,
):
    from contextlib import nullcontext as _nullcontext

    from qwen_omni_utils import process_mm_info as _process_mm_info

    _p = scoreboard_controls.value
    mo.stop(
        _p is None,
        mo.callout(
            mo.md("Set the parameters above and press **▶ Run logit-lens diversity**."),
            kind="info",
        ),
    )

    _results_dir = LOGIT_CSV_PATH.parent

    # Resolve the video: an uploaded clip wins, otherwise reuse the default sample.
    _uploads = _p["video"]
    if _uploads and _uploads[0].contents:
        _video_path = _results_dir / f"uploaded_{_uploads[0].name}"
        _video_path.write_bytes(_uploads[0].contents)
    else:
        _video_path = VIDEO_PATH
    _nframes = int(_p["nframes"])
    _prompt = _p["prompt"].strip() or LOGIT_PROMPT

    # Build the knockout rules. The advanced text field (several `src,tgt,start,end`
    # rules separated by `;`) overrides the single-rule builder when it is filled.
    _modalities = ["audio", "video", "query_text", "image", "generated"]
    _n_layers = len(attention_model.thinker.model.layers)

    def _parse_rules(text):
        _out = []
        for _seg in text.split(";"):
            _seg = _seg.strip()
            if not _seg:
                continue
            _f = [c.strip() for c in _seg.split(",")]
            if len(_f) != 4:
                return [], f"`{_seg}` needs 4 fields: `source,target,start,end`"
            _s, _t, _a, _b = _f
            if _s not in _modalities:
                return [], f"unknown source `{_s}` — use {' / '.join(_modalities)}"
            if _t not in _modalities:
                return [], f"unknown target `{_t}` — use {' / '.join(_modalities)}"
            try:
                _a, _b = int(_a), int(_b)
            except ValueError:
                return [], f"start/end must be integers in `{_seg}`"
            if not (0 <= _a < _b <= _n_layers):
                return [], f"need 0 ≤ start < end ≤ {_n_layers} in `{_seg}`"
            _out.append((_s, _t, _a, _b))
        if not _out:
            return [], "no rules parsed — try `audio,video,0,36`"
        return _out, None

    _rules_err = None
    if not _p["ko_enable"]:
        _rules = []
    elif _p["ko_rules_text"].strip():
        _rules, _rules_err = _parse_rules(_p["ko_rules_text"])
    else:
        _lo, _hi = _p["ko_layers"]
        _rules = [(_p["ko_source"], _p["ko_target"], int(_lo), int(_hi))]
    mo.stop(
        _rules_err is not None,
        mo.callout(mo.md(f"**Invalid knockout rules** — {_rules_err}"), kind="danger"),
    )
    _compare = bool(_p["compare"])

    def _prep(video_path, nframes, prompt):
        # Encoding (video decode + feature extraction) dominates a submit when
        # only the rule/layer band changed — cache it across ▶ presses.
        _key = (video_path.name, video_path.stat().st_size, nframes, prompt)
        if _key in playground_caches["encode"]:
            return playground_caches["encode"][_key]
        _conv = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "video", "video": str(video_path), "nframes": nframes},
        ]}]
        _text = attention_processor.apply_chat_template(
            _conv, add_generation_prompt=True, tokenize=False
        )
        _audios, _images, _videos = _process_mm_info(_conv, use_audio_in_video=True)
        _inp = attention_processor(
            text=_text, audio=_audios, images=_images, videos=_videos,
            return_tensors="pt", padding=True, use_audio_in_video=True,
        )
        _inp = {k: v.to(attention_model.device) for k, v in _inp.items()}
        _types = create_attention_token_mapping(
            _inp["input_ids"], attention_model.config.thinker_config
        )
        return cache_put(playground_caches["encode"], _key, (_inp, _types))

    def _diversity(csv_path):
        # Reproduce the "diversity by layer" logic: per layer, count distinct decoded
        # tokens across the audio-token rows, and the most-common prediction's share.
        with open(csv_path, newline="", encoding="utf-8") as _fh:
            _data = list(csv.reader(_fh))[1:]  # drop the header row
        if not _data:
            return [], [], 0
        _cols = list(zip(*(_r[2:] for _r in _data)))  # one tuple per thinker layer
        _uniq = [len(set(_c)) for _c in _cols]
        _dom = [Counter(_c).most_common(1)[0][1] / len(_c) for _c in _cols]
        return _uniq, _dom, len(_data)

    def _run_pass(rules, tag, inp, types):
        _csv_path = _results_dir / f"interactive_logit_lens_{tag}.csv"
        if _csv_path.exists():
            _csv_path.unlink()  # no stale results if this run has no audio tokens
        register_logit_lens_hooks(attention_model)
        try:
            _ctx = (
                block_attention(
                    attention_model, rules, types, len(types), track_attention=False
                )
                if rules else _nullcontext()
            )
            with _ctx:
                with torch.no_grad():
                    attention_model.thinker(**inp, output_hidden_states=True)
            analyze_and_save_audio_logits_to_csv(
                attention_model, attention_processor, types, filename=str(_csv_path)
            )
        finally:
            clear_logit_lens_hooks()
        if not _csv_path.exists():
            return [], [], 0
        return _diversity(_csv_path)

    _scoreboard = None
    try:
        with mo.status.spinner(
            title=f"Logit-lens forward pass · {_nframes} frames · {_video_path.name}…"
        ):
            _inp, _types = _prep(_video_path, _nframes, _prompt)  # encode the clip once
            if _rules:
                _ko_u, _ko_d, _n_audio = _run_pass(_rules, "knockout", _inp, _types)
                _bl_u, _bl_d = (None, None)
                if _compare:
                    _bl_u, _bl_d, _ = _run_pass([], "baseline", _inp, _types)
            else:
                _bl_u, _bl_d, _n_audio = _run_pass([], "baseline", _inp, _types)
                _ko_u, _ko_d = (None, None)
    except Exception as _e:  # noqa: BLE001 — surface any run failure in-notebook
        _scoreboard = mo.callout(
            mo.md(f"**Run failed** — `{type(_e).__name__}: {_e}`"), kind="danger"
        )

    if _scoreboard is None:
        _primary_u = _ko_u if _ko_u else _bl_u
        _primary_d = _ko_d if _ko_d else _bl_d
        _both = bool(_ko_u) and bool(_bl_u)

    if _scoreboard is not None:
        pass
    elif not _primary_u:
        _scoreboard = mo.callout(
            mo.md(
                f"**No audio tokens** were produced for `{_video_path.name}` with this "
                "prompt, so there are no audio-position predictions to score. Try a clip "
                "that carries an audio track."
            ),
            kind="warn",
        )
    else:
        _n_l = len(_primary_u)
        _order = sorted(range(_n_l), key=lambda k: _primary_u[k], reverse=True)

        _rows = []
        for _rank, _i in enumerate(_order, 1):
            _row = {"Rank": _rank, "Layer": _i, "Unique preds": _primary_u[_i]}
            if _both:
                _row["Baseline"] = _bl_u[_i]
                _row["Δ vs base"] = _ko_u[_i] - _bl_u[_i]
            _row["Dominant share"] = round(_primary_d[_i], 3)
            _rows.append(_row)
        _table = mo.ui.table(_rows, selection=None, pagination=True, page_size=12)

        _peak = _order[0]
        _stats = [
            mo.stat(
                value=f"Layer {_peak}",
                label="Peak diversity",
                caption=f"{_primary_u[_peak]} unique predictions",
                bordered=True,
            ),
            mo.stat(
                value=f"{sum(_primary_u) / _n_l:.1f}",
                label="Mean unique / layer",
                caption=f"across {_n_l} thinker layers",
                bordered=True,
            ),
            mo.stat(
                value=str(_n_audio),
                label="Audio tokens scored",
                caption="positions decoded per layer",
                bordered=True,
            ),
        ]
        if _both:
            _mean_delta = sum(_ko_u[k] - _bl_u[k] for k in range(_n_l)) / _n_l
            _less = sum(1 for k in range(_n_l) if _ko_u[k] < _bl_u[k])
            _stats.append(
                mo.stat(
                    value=f"{_mean_delta:+.1f}",
                    label="Mean Δ from knockout",
                    caption=f"{_less}/{_n_l} layers less diverse",
                    direction="decrease" if _mean_delta < 0 else "increase",
                    bordered=True,
                )
            )

        _x = np.arange(_n_l)
        _fig, _axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
        if _both:
            _axes[0].bar(_x, _ko_u, color="#4C78A8", label="knockout")
            _axes[0].plot(_x, _bl_u, color="#F58518", marker="o", ms=3, lw=1.5, label="baseline")
            _axes[0].legend()
            _axes[0].set(title="Unique predictions by layer",
                         xlabel="Thinker layer", ylabel="Unique predictions")
            _delta = [_ko_u[k] - _bl_u[k] for k in range(_n_l)]
            _axes[1].bar(_x, _delta, color=["#E45756" if d < 0 else "#54A24B" for d in _delta])
            _axes[1].axhline(0, color="black", lw=0.8)
            _axes[1].set(title="Δ diversity (knockout − baseline)",
                         xlabel="Thinker layer", ylabel="Δ unique predictions")
        else:
            _axes[0].bar(_x, _primary_u, color="#4C78A8")
            _axes[0].set(title="Logit-lens diversity by layer",
                         xlabel="Thinker layer", ylabel="Unique predictions")
            _axes[1].plot(_x, _primary_d, marker="o", color="#F58518")
            _axes[1].set(title="Most-common prediction share",
                         xlabel="Thinker layer", ylabel="Share", ylim=(0, 1))
        for _ax in _axes:
            _ax.grid(axis="y", alpha=0.25)

        _rule_txt = (
            " + ".join(f"`{r[0]}→{r[1]}` [{r[2]},{r[3]})" for r in _rules)
            if _rules else "_none (baseline only)_"
        )
        _children = []
        _gen = [r for r in _rules if r[0] == "generated"]
        if _gen:
            _children.append(mo.callout(
                mo.md(
                    f"**Heads-up:** {len(_gen)} rule(s) use `generated` as the source, which "
                    "is **inert** in this forward-pass scoreboard — there are no generated "
                    "tokens during prefill, so those rules block nothing (expect a flat Δ). "
                    "Use `audio`, `video`, or `query_text` as the source to reshape the score."
                ),
                kind="warn",
            ))
        _children += [
            mo.md(
                f"**Video** `{_video_path.name}` &nbsp;·&nbsp; **Frames** {_nframes} "
                f"&nbsp;·&nbsp; **Prompt** _{_prompt}_ &nbsp;·&nbsp; **Knockout** {_rule_txt}"
            ),
            mo.hstack(_stats, widths="equal", gap=1),
            _fig,
            mo.md("###### Layers ranked by decoded-prediction diversity (higher = more distinct audio-token predictions)"),
            _table,
        ]
        _scoreboard = mo.vstack(_children)
    _scoreboard
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🎯 Interactive: teacher-forced Δ log-likelihood

    The diversity scoreboard above runs one forward pass over the **prompt**, so —
    exactly like `generated` — an **`answer`** source is inert there (there are no
    answer tokens to block). This section closes that gap. It generates the caption
    once, feeds it back in tagged **`answer`**, and measures **how much less the
    model believes what it said** when the answer is forbidden from attending to a
    modality.

    The metric is **Δ log-likelihood, `knockout − baseline`** — *negative* means the
    model believed its own caption **less** after the knockout, i.e. that pathway was
    holding the caption up. Unlike the free-generation string diff above it is
    **continuous** (you can see a *small* effect) and **deterministic** (greedy
    caption, forward-only scoring). Nothing runs until you press ▶.

    ### What to try (predict the sign and the size *before* you press ▶)

    - **Sight vs. sound, in nats.** Same clip, same prompt: run `answer → audio`,
      then `answer → video`. Which Δ is more negative — and does that agree with
      which knockout changed the *free-generated* caption more in the knockout cell above?
      If the binary diff and the continuous measurement disagree, which do you
      believe, and why?
    - **Where does the caption's audio grounding live?** Keep `answer → audio` and
      narrow the layer band: `[0, 12)`, `[12, 24)`, `[24, 36)`. Which band costs the
      caption the most belief? Compare with the fusion band you found in 🎛️ — do the
      layers that *build* the audio representation match the layers the *answer*
      reads it from?
    - **Bring your own clip — make the modalities disagree.** The most interesting
      trends come from clips where sound and sight tell different stories (narration
      over unrelated footage, a music video with off-screen audio, dubbed speech).
      Upload one, caption it, and knock out `answer → audio` vs `answer → video`:
      which sense was the caption actually standing on? Hover the colored strip —
      *which words* lose the belief?

    > **The control that keeps you honest.** Upload `assets/02321_silent.mp4` (the
    > same frames, but the audio track is digital silence) and run `answer → audio`:
    > the audio tokens exist but carry no signal, so Δ should be ≈ 0. Compare against
    > the default clip (real soundtrack), same prompt and layers — a real audio
    > dependency shows up as a clearly larger negative Δ. Pair *every* interesting
    > effect you find above with a control like this: a control that *can* fail is
    > the whole point. Log each run in `avllm_interpretability/WORKSHEET.md` —
    > hypothesis **before** ▶, result, verdict; its last block is the exact
    > question your final project is graded on.
    """)
    return


@app.cell
def _(LOGIT_PROMPT, NFRAMES, attention_model, mo):
    _n_layers = len(attention_model.thinker.model.layers)
    _tf_targets = ["audio", "video", "query_text", "image"]
    _tf_template = (
        "**Video** — upload `mp4 / mov / mkv / webm`, or leave empty to reuse the default clip:\n\n"
        "{video}\n\n"
        "**Frames sampled from the clip** {nframes}\n\n"
        "**Prompt** {prompt}\n\n"
        "---\n\n"
        "Forbid the **answer** from attending to {target} across thinker layers {layers}\n\n"
        f"(`answer` is the model's own caption, teacher-forced back in; this thinker has "
        f"**{_n_layers}** layers, `end` exclusive.)"
    )
    teacher_forcing_controls = mo.md(_tf_template).batch(
        video=mo.ui.file(
            filetypes=[".mp4", ".mov", ".mkv", ".webm", ".avi"], multiple=False, kind="area"
        ),
        nframes=mo.ui.slider(2, 32, step=2, value=NFRAMES, show_value=True, include_input=True),
        prompt=mo.ui.text(value=LOGIT_PROMPT, full_width=True),
        target=mo.ui.dropdown(_tf_targets, value="audio"),
        layers=mo.ui.range_slider(0, _n_layers, step=1, value=[0, _n_layers], show_value=True),
    ).form(submit_button_label="▶ Run teacher-forced Δ log-lik", bordered=True)
    teacher_forcing_controls
    return (teacher_forcing_controls,)


@app.cell
def _(
    LOGIT_PROMPT,
    VIDEO_PATH,
    attention_model,
    attention_processor,
    cache_put,
    create_attention_token_mapping,
    mo,
    np,
    playground_caches,
    teacher_forcing_controls,
):
    from pathlib import Path as _Path

    from qwen_omni_utils import process_mm_info as _tf_mm_info

    from src.teacher_forcing import render_delta_strip as _render_strip
    from src.teacher_forcing import teacher_forced_delta as _tfd

    _tp = teacher_forcing_controls.value
    mo.stop(
        _tp is None,
        mo.callout(
            mo.md("Set the parameters above and press **▶ Run teacher-forced Δ log-lik**."),
            kind="info",
        ),
    )

    # Resolve the clip: an uploaded file wins, else reuse the sample.
    _tf_uploads = _tp["video"]
    if _tf_uploads and _tf_uploads[0].contents:
        _tf_video = _Path(VIDEO_PATH).parent / "notebook_results" / f"tf_upload_{_tf_uploads[0].name}"
        _tf_video.parent.mkdir(exist_ok=True)
        _tf_video.write_bytes(_tf_uploads[0].contents)
    else:
        _tf_video = _Path(VIDEO_PATH)
    _tf_nframes = int(_tp["nframes"])
    _tf_prompt = _tp["prompt"].strip() or LOGIT_PROMPT
    _tf_lo, _tf_hi = int(_tp["layers"][0]), int(_tp["layers"][1])
    _tf_rules = [("answer", _tp["target"], _tf_lo, _tf_hi)]

    def _tf_prep(video_path, nframes, prompt):
        # Shared encode cache with the 🎛️ section: a layer-band or target sweep
        # on the same clip/prompt re-encodes nothing after the first ▶.
        _key = (video_path.name, video_path.stat().st_size, nframes, prompt)
        if _key in playground_caches["encode"]:
            return playground_caches["encode"][_key]
        _conv = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "video", "video": str(video_path), "nframes": nframes},
        ]}]
        _text = attention_processor.apply_chat_template(
            _conv, add_generation_prompt=True, tokenize=False
        )
        _audios, _images, _videos = _tf_mm_info(_conv, use_audio_in_video=True)
        _inp = attention_processor(
            text=_text, audio=_audios, images=_images, videos=_videos,
            return_tensors="pt", padding=True, use_audio_in_video=True,
        )
        _inp = {k: v.to(attention_model.device) for k, v in _inp.items()}
        _types = create_attention_token_mapping(
            _inp["input_ids"], attention_model.config.thinker_config
        )
        return cache_put(playground_caches["encode"], _key, (_inp, _types))

    _tf_out = None
    try:
        # Caption cache (the F1 spec's "cached keyed on (clip, prompt, nframes)"):
        # the greedy caption depends only on the encoded inputs, so a rule/layer
        # sweep reuses C instead of regenerating it every submit.
        _tf_cap_key = (_tf_video.name, _tf_video.stat().st_size, _tf_nframes, _tf_prompt)
        _tf_cached_c = playground_caches["caption"].get(_tf_cap_key)
        with mo.status.spinner(
            title=f"Teacher forcing · {_tf_nframes} frames · {_tf_video.name}"
            + (" · caption cached…" if _tf_cached_c is not None else "…")
        ):
            _tf_inp, _tf_types = _tf_prep(_tf_video, _tf_nframes, _tf_prompt)
            _tf_res = _tfd(
                attention_model, attention_processor, _tf_inp, _tf_types, _tf_rules,
                cached_caption_ids=_tf_cached_c,
            )
            cache_put(playground_caches["caption"], _tf_cap_key, _tf_res["caption_ids"])
    except Exception as _e:  # noqa: BLE001 — surface any failure in-notebook
        _tf_out = mo.callout(
            mo.md(f"**Run failed** — `{type(_e).__name__}: {_e}`"), kind="danger"
        )

    if _tf_out is None:
        _tf_delta = [float(x) for x in _tf_res["delta"].detach().cpu().float().tolist()]
        _tf_total = _tf_res["delta_total"]
        _tf_toks = _tf_res["caption_tokens"]
        _tf_worst = int(np.argmin(_tf_delta)) if _tf_delta else 0
        _tf_rule_txt = f"`answer→{_tp['target']}` [{_tf_lo},{_tf_hi})"
        _tf_stats = [
            mo.stat(
                value=f"{_tf_total:+.2f}",
                label="Σ Δ log-lik (nats)",
                caption="knockout − baseline · negative = believed less",
                direction="decrease" if _tf_total < 0 else "increase",
                bordered=True,
            ),
            mo.stat(
                value=(_tf_toks[_tf_worst].strip() or "·") if _tf_toks else "—",
                label="Most affected token",
                caption=(f"Δ = {_tf_delta[_tf_worst]:+.2f} nats" if _tf_delta else ""),
                bordered=True,
            ),
            mo.stat(
                value=str(len(_tf_toks)),
                label="Caption tokens scored",
                caption="teacher-forced, greedy",
                bordered=True,
            ),
        ]
        _tf_rows = [
            {"pos": _i, "token": _t, "Δ log-lik": round(_d, 3)}
            for _i, (_t, _d) in enumerate(zip(_tf_toks, _tf_delta))
        ]
        _tf_out = mo.vstack([
            mo.md(
                f"**Video** `{_tf_video.name}` &nbsp;·&nbsp; **Frames** {_tf_nframes} "
                f"&nbsp;·&nbsp; **Prompt** _{_tf_prompt}_ &nbsp;·&nbsp; **Knockout** {_tf_rule_txt}"
            ),
            mo.hstack(_tf_stats, widths="equal", gap=1),
            mo.md("###### Per-token Δ log-likelihood (hot = believed less after the knockout; hover a word for its tokens' nats)"),
            mo.Html(f"<div style='line-height:2.1;font-family:monospace;font-size:15px'>{_render_strip(_tf_toks, _tf_delta)}</div>"),
            mo.ui.table(_tf_rows, selection=None, pagination=True, page_size=16),
        ])
    _tf_out
    return


if __name__ == "__main__":
    app.run()
