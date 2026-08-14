# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "torch==2.6.0",
#     "torchvision==0.21.0",
#     "transformers==4.52.0",
#     "accelerate==1.14.0",
#     "qwen-omni-utils==0.0.9",
#     "wigglystuff==0.5.21",
#     "anywidget>=0.9.2",
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

    Two interpretability experiments on one video with **Qwen2.5-Omni-3B**:

    1. **Logit Lens** — decode the model's intermediate predictions at audio-token
       positions across thinker layers.
    2. **Attention Knockout** — compare a baseline response with one generated after
       blocking a chosen source→target attention path.
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
        # qwen-omni-utils 0.0.9 imports these at module scope but declares
        # neither: `audioread` (used for real, via audioread.ffdec) is missing
        # from its metadata entirely, and `librosa` is declared but absent on any
        # kernel where its dependency set was not resolved. Both are listed here
        # so the common case is fixed deterministically rather than by the
        # import-repair loop below.
        ("audioread", "audioread", None, "audioread"),
        ("librosa", "librosa", None, "librosa"),
        # anywidget-based classroom widgets (caption diff, Δ threshold).
        # 0.5.15+ needs Python >= 3.11 — molab qualifies.
        ("wigglystuff", "wigglystuff", "0.5.21", "wigglystuff==0.5.21"),
        # Listed explicitly even though wigglystuff pulls it in: `src/probe_grid.py`
        # is a first-party anywidget and should not depend on another package's
        # dependency graph to be importable.
        ("anywidget", "anywidget", "0.9.2", "anywidget>=0.9.2"),
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

    def _ensure_importable(module, attempts=4):
        # `find_spec` above only proves a package is on disk, not that it
        # imports. qwen-omni-utils 0.0.9 imports `audioread`, `numpy`, `torch`,
        # `torchvision` and `torchcodec` at module scope while declaring none of
        # them, so a kernel whose dependency set was resolved differently has the
        # package present and unimportable — and the failure surfaces as a
        # ModuleNotFoundError deep inside a later cell, long after setup claimed
        # success.
        #
        # So import it here and install whatever it actually asks for. Bounded,
        # and every install is printed: this repairs a broken environment, it
        # does not paper over one.
        for _ in range(attempts):
            try:
                importlib.import_module(module)
                return
            except ModuleNotFoundError as _exc:
                _missing = (_exc.name or "").split(".")[0]
                if not _missing or _missing == module:
                    raise
                if _missing.startswith("torch"):
                    # Never reinstall torch/torchvision: molab ships a build
                    # matched to its GPU, and replacing it yields an unrunnable
                    # one. Fail with the reason instead.
                    raise ModuleNotFoundError(
                        f"{module} needs {_missing!r}, which is missing. Refusing "
                        "to pip-install it: molab's torch build is GPU-matched and "
                        "reinstalling would replace it. Attach a GPU runtime that "
                        "provides it."
                    ) from _exc
                print(f"{module} needs {_missing!r} (undeclared) — installing it")
                with mo.status.spinner(title=f"Installing {_missing} for {module}…"):
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", _missing], check=True
                    )
                importlib.invalidate_caches()
        importlib.import_module(module)  # last try; let it raise if still broken

    _ensure_importable("qwen_omni_utils")

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
    # F5a — GPU-free replay. Flip to True to render every non-interactive W7-W9
    # plot from committed artifacts (no GPU, no 8 GB download): a break-glass mode
    # for when molab's GPU is unavailable. Default False = live model. The
    # interactive playground / teacher-forcing sections still need a GPU and fail
    # loudly if submitted in this mode. Generate the artifacts on a GPU with:
    #   python avllm_interpretability/scripts/generate_precompute.py
    USE_PRECOMPUTED = False
    PRECOMPUTED_DIR = PROJECT_DIR / "precomputed"
    if USE_PRECOMPUTED:
        print(f"USE_PRECOMPUTED=True — replaying W7-W9 from {PRECOMPUTED_DIR} (no GPU)")
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
            "(Or set USE_PRECOMPUTED=True in the cell above to replay W7-W9 from committed artifacts.)"
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

    The knobs, and what each one actually moves:

    | Knob | Moves |
    |---|---|
    | `NFRAMES` | how many **video** frames are sampled. Audio-token count is fixed by the clip's duration, so this does *not* change the logit-lens scoreboard's row count. |
    | `LOGIT_PROMPT` / `ATTENTION_PROMPT` | the instruction; changes the `query_text` positions and can change the caption. |
    | `KNOCKOUT_RULES` | `(source, target, start_layer, end_layer)`, `end` **exclusive**. A rule that matches no layer or no token is refused rather than silently returning a baseline. |
    | `MAX_NEW_TOKENS` | caption length — and therefore Σ Δ log-lik, which is why the per-token mean is the number to compare across runs. |
    | `ATTENTION_CAPTURE_LAYERS` | which layers the attention heatmap below covers, `(0, 2)` by default. Widening it is a real experiment with a real cost: each captured layer holds a `seq × seq` tensor **per decode step**, so `(0, 36)` is the usual way to lose both loaded models on a 24 GB GPU. |
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

    # The silent-clip control ships in the repo next to the default clip. It is
    # the only control in this lab that can fail, so its absence is a hard error
    # rather than something the forms below discover at submit time.
    SILENT_VIDEO_PATH = PROJECT_DIR / "assets" / "02321_silent.mp4"

    assert VIDEO_PATH.is_file(), f"video not found: {VIDEO_PATH}"
    assert SILENT_VIDEO_PATH.is_file(), f"silent control not found: {SILENT_VIDEO_PATH}"
    print("video:", VIDEO_PATH)
    print("silent control:", SILENT_VIDEO_PATH)
    return LOGIT_CSV_PATH, RESULTS_DIR, SILENT_VIDEO_PATH, VIDEO_PATH


@app.cell
def _():
    # The knobs — cheap to tweak: re-runs the experiments, not the model loads.
    NFRAMES = 8
    LOGIT_PROMPT = "Describe what you hear in the video"
    ATTENTION_PROMPT = "Describe what you see and hear in the video"
    KNOCKOUT_RULES = [("generated", "video", 0, 36)]  # block generated→video, all 36 thinker layers
    MAX_NEW_TOKENS = 32
    ATTENTION_CAPTURE_LAYERS = (0, 2)  # heatmap rows; widening this costs VRAM per decode step

    # `end` is exclusive, so `(…, 12, 12)` masks nothing and would come back
    # labelled "no effect". The layer *count* is deliberately not checked here:
    # that would make this cell depend on the loaded model and undo the split
    # that keeps knob edits cheap. `block_attention` does the model-aware check.
    assert all(
        len(_r) == 4 and 0 <= _r[2] < _r[3] for _r in KNOCKOUT_RULES
    ), f"each rule must be (source, target, start, end) with 0 <= start < end: {KNOCKOUT_RULES}"
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
        # Spell out every modality including the zeros. `Counter` omits absent
        # keys, so `image` — which is always 0 for a video clip — simply did not
        # appear, and an absence reads as an oversight rather than as a fact.
        _counts = Counter(_types)
        print("token counts:", ", ".join(
            f"{_m}={_counts.get(_m, 0)}"
            for _m in ("query_text", "audio", "video", "image")
        ))
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
def _(attention_model, attention_processor):
    # The logit lens shares the eager model rather than loading a second SDPA
    # copy. *Measured on an RTX 3090:* each copy is 8.88 GiB (this "3B" Omni model
    # is 4.70 B parameters once the talker is freed), so two copies plus eager
    # attention over the 1,476-token multimodal prompt peaked at 22.08 GiB and
    # died with `CUDA out of memory` on a 24 GB card -- the exact floor the README
    # advertises. One shared model peaks at 13.70 GiB and completes.
    #
    # Nothing is lost: the logit lens hooks each layer's *output* and projects it
    # through `lm_head`. It never reads attention weights, so it has no reason to
    # prefer SDPA -- that choice only ever bought speed on one forward pass, at
    # the cost of the notebook not running at all.
    logit_model, logit_processor = attention_model, attention_processor
    return logit_model, logit_processor


@app.cell
def _(PRECOMPUTED_DIR, USE_PRECOMPUTED, load_model_and_processor, mo):
    # Dedicated loader cell for the eager model (knockout hooks + both
    # playgrounds). In precomputed mode a layer-count stub stands in so the
    # playground forms can render; it cannot compute, and the forms fail loudly
    # if submitted.
    if USE_PRECOMPUTED:
        from src.precompute import StubModel as _StubModel
        from src.precompute import load_precompute as _stub_pre

        attention_model = _StubModel(_stub_pre(PRECOMPUTED_DIR)["meta"].get("n_layers", 36))
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


@app.cell
def _(RESULTS_DIR, mo):
    # The run ledger's state. Its only references are `mo` and `RESULTS_DIR`, both
    # computed once and never touched again — deliberately. marimo mints a fresh
    # `SetFunctor` every time a state cell re-runs, and every cell holding the old
    # setter re-runs with it; since the three GPU cells below call `set_runs`, a
    # dependency here on anything reactive would re-fire all three at once.
    #
    # Seeded from the JSONL rather than from `[]`: the log is the durability
    # promise this section makes to a student whose molab kernel dies mid-lab,
    # and a promise nothing reads back is not a promise.
    from src.run_ledger import load_log as _load_log
    #
    # Not an anywidget: a widget the GPU cells push to would have to be *named* by
    # them, which makes them referring cells, and any synced trait changing from
    # the browser re-runs every referring cell (marimo has no per-trait
    # subscription). One click on a verdict chip would re-run a 60-second
    # generation. An HTML string rendered with `mo.Html` has no such edge —
    # the same shape `render_delta_strip` already uses.
    get_runs, set_runs = mo.state(_load_log(RESULTS_DIR / "lab_log.jsonl"))
    return get_runs, set_runs


@app.cell
def _(RESULTS_DIR):
    # Writers only. Deliberately does NOT reference `get_runs`: the three GPU
    # cells import `run_record`/`append_run` from here, so if this cell also
    # referred to the state it would re-run on every append — and take all three
    # generations with it.
    from src.run_ledger import append_run, apply_verdict, run_record

    LEDGER_LOG = RESULTS_DIR / "lab_log.jsonl"
    return LEDGER_LOG, append_run, apply_verdict, run_record


@app.cell
def _(RESULTS_DIR, SILENT_VIDEO_PATH, VIDEO_PATH):
    import hashlib as _hashlib

    # Upload limits. The PyAV shim above decodes *every* frame into a Python list
    # before stacking, so a two-minute 4K clip materializes tens of gigabytes and
    # takes the kernel — and both loaded models — with it. "Bring your own clip"
    # is the point of the last session, and the natural student clip is a 1080p60
    # phone video, so these are checked before a single frame is decoded.
    MAX_UPLOAD_BYTES = 250 * 2**20
    MAX_DURATION_S = 120.0
    MAX_PIXELS = 1920 * 1080
    MAX_FPS = 60.0

    def preflight_clip(path):
        """`None` if the clip is safe to decode, else a sentence explaining why not."""
        import av

        try:
            with av.open(str(path)) as _c:
                if not _c.streams.video:
                    return "this file has no video stream."
                _v = _c.streams.video[0]
                _dur = float(_c.duration / 1_000_000) if _c.duration else None
                _rate = _v.average_rate or _v.guessed_rate
                _fps = float(_rate) if _rate else None
                _px = int(_v.width or 0) * int(_v.height or 0)
                if _dur is not None and _dur > MAX_DURATION_S:
                    return (
                        f"it is {_dur:.0f}s long; the limit here is "
                        f"{MAX_DURATION_S:.0f}s. Trim it and try again."
                    )
                if _px > MAX_PIXELS:
                    return (
                        f"it is {_v.width}×{_v.height}; the limit is 1920×1080. "
                        "Every frame is decoded into memory uncompressed."
                    )
                if _fps is not None and _fps > MAX_FPS:
                    return f"it is {_fps:.0f} fps; the limit is {MAX_FPS:.0f}."
                if not _c.streams.audio:
                    return (
                        "this file has no audio track. Both playgrounds measure at "
                        "`audio` token positions, so there would be nothing to score. "
                        "(If you *want* an audio-free control, use the silent clip: "
                        "it has a real, digitally silent audio track.)"
                    )
        except Exception as _e:  # noqa: BLE001 — a broken container is a message, not a crash
            return f"it could not be opened ({type(_e).__name__}: {_e})."
        return None

    def resolve_clip(choice, uploads):
        """Turn a clip choice into `(path, is_control, error)`.

        The silent control lives in the repo, so in molab it exists only on the
        kernel side — there is nothing on the student's machine for a browser file
        picker to select. Choosing it by name and resolving server-side is what
        makes the one control in this lab that can fail actually reachable.
        """
        if choice == "Default clip":
            return VIDEO_PATH, False, None
        if choice == "Silent control":
            return SILENT_VIDEO_PATH, True, None
        if not uploads or not uploads[0].contents:
            # Deliberately an error, not a fallback: silently substituting the
            # default clip is how a "silent control" run became a duplicate of
            # the experiment it was supposed to falsify.
            return None, False, "**Upload** is selected but no file was chosen."
        _blob = uploads[0].contents
        if len(_blob) > MAX_UPLOAD_BYTES:
            return None, False, (
                f"that file is {len(_blob) / 2**20:.0f} MB; the limit is "
                f"{MAX_UPLOAD_BYTES // 2**20} MB."
            )
        # Hash into the filename so two different clips that happen to share a
        # name and byte size cannot collide in the encode cache below. The
        # student-supplied name is reduced to its basename and stripped of
        # anything but word characters, dots and dashes: it arrives from a
        # browser and is interpolated straight into a path, so `../` in it would
        # write outside RESULTS_DIR.
        _digest = _hashlib.sha256(_blob).hexdigest()[:12]
        _safe = "".join(
            _c for _c in Path(uploads[0].name).name if _c.isalnum() or _c in "._-"
        )[:80] or "clip.mp4"
        _dest = RESULTS_DIR / f"upload_{_digest}_{_safe}"
        if not _dest.exists():
            _dest.write_bytes(_blob)
        _why = preflight_clip(_dest)
        if _why is not None:
            return None, False, f"`{uploads[0].name}` was rejected: {_why}"
        return _dest, False, None

    return resolve_clip,


@app.cell
def _(get_runs, mo):
    # The reader. This one *does* reference the state, so it re-runs on every
    # append — which is what keeps the views and the worksheet export fresh.
    from src.run_ledger import build_worksheet_md as worksheet_md
    from src.run_ledger import ledger_counts as _counts
    from src.run_ledger import render_ledger_html as _render_ledger

    def ledger_view(highlight=()):
        """The ledger, rendered wherever a result lands.

        Called from cheap display cells only. A cell calling this depends on this
        cell and therefore re-renders on every append — the point for a view, and
        exactly why no GPU cell and no form cell may call it.
        """
        _runs = get_runs()
        _c = _counts(_runs)
        _head = mo.md(
            f"###### Run ledger — {_c['n']} run(s) · "
            f"**{_c['n_uncontrolled_claims']}** without a control · "
            f"{_c['n_unresolved']} without a verdict"
        )
        return mo.vstack([_head, mo.Html(_render_ledger(_runs, highlight_ids=highlight))], gap=0.3)

    return ledger_view, worksheet_md


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
        from src.precompute import validate_precompute_meta as _validate_pre

        _pre = _load_pre(PRECOMPUTED_DIR)
        # Replay mode keeps LOGIT_PROMPT / NFRAMES / VIDEO_PATH as dependencies of
        # this cell, so editing them re-runs a cell that ignores them. Naming the
        # pinned values — and refusing to relabel a cached run as if it came from
        # the current knobs — is what stops every knob quietly lying for a session.
        _validate_pre(
            _pre["meta"], clip=VIDEO_PATH.name, nframes=NFRAMES, logit_prompt=LOGIT_PROMPT
        )
        logit_csv_written = _pre["logit_csv"]
        _logit_out = mo.vstack([
            mo.callout(
                mo.md(
                    "**Replayed from cache** — precomputed, no GPU. These artifacts were "
                    f"produced with clip `{_pre['meta'].get('clip')}`, "
                    f"`nframes={_pre['meta'].get('nframes')}`, prompt "
                    f"_{_pre['meta'].get('logit_prompt')}_. Editing the knobs above "
                    "cannot change them."
                ),
                kind="neutral",
            ),
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
                # Keep the return value. `build_compact_probe_result` already ran
                # inside this call and carries top-5, entropy and the top1-top2
                # margin for every (layer, position); the CSV adapter keeps only
                # `top_tokens[0]["token_text"]`. Throwing the rest away here is
                # what left the probe's degeneracy invisible.
                _probe_result = analyze_and_save_audio_logits_to_csv(
                    logit_model, logit_processor, logit_token_types, filename=str(LOGIT_CSV_PATH)
                )
        finally:
            clear_logit_lens_hooks()
        # `analyze_and_save_audio_logits_to_csv` returns None on every failure path
        # (no capture, no audio tokens, unwritable file) and deletes the previous
        # CSV first, so an unconditional assignment here pointed downstream cells
        # at a file that may not exist.
        logit_csv_written = LOGIT_CSV_PATH if _probe_result is not None else None

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
    # Twenty lines below, the scrubber handles the identical case with a callout.
    # A bare open() here meant a clip with no audio track ended the notebook in a
    # raw traceback instead of a sentence naming the cause.
    mo.stop(
        logit_csv_written is None or not logit_csv_written.is_file(),
        mo.callout(
            mo.md(
                "**No logit-lens CSV** — the run above wrote no audio-token rows. "
                "The most likely cause is a clip with no audio track: the probe is "
                "measured at `audio` positions only."
            ),
            kind="warn",
        ),
    )
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
    ## 🎞️ Interactive: the probe surface, all 36 layers at once

    The diversity bars above aggregate away *what* the model is predicting. This
    grid shows every layer × every audio position at once: **y = thinker layer,
    x = audio position**, coloured by what kind of token the probe decodes to —
    **content**, **junk** (punctuation, whitespace, symbols), or undecodable. A
    thin ring marks the cells that already equal that position's own last-layer
    token.

    **Drag anywhere on the grid** to move the active layer; the chip strip below
    it updates with no round trip to Python. **Click a column** to pin that
    position and read its whole 36-layer trajectory as tokens.

    > **The thing to notice.** An earlier version of this section invited you to
    > "watch early-layer noise crystallize into the final prediction". Look at
    > what the final layers actually decode to before believing that. The probe is
    > uncalibrated at audio positions — that degeneracy *is* this week's result,
    > and the junk/content split is what makes it visible. The junk rule is
    > printed inside the widget so you can argue with it.

    Pure re-render of the CSV written above: no GPU, and it works in
    `USE_PRECOMPUTED` replay mode.
    """)
    return


@app.cell(hide_code=True)
def _(logit_csv_written, mo):
    from src.probe_grid import ProbeGrid as _ProbeGrid
    from src.probe_grid import build_probe_grid_pack as _build_pack
    from src.probe_grid import probe_grid_layer_summary as _layer_summary

    _pack = (
        _build_pack(logit_csv_written)
        if logit_csv_written is not None and logit_csv_written.is_file()
        else None
    )
    mo.stop(
        _pack is None,
        mo.callout(
            mo.md("**Nothing to show** — the logit-lens run above wrote no audio-token rows."),
            kind="warn",
        ),
    )

    # Constructed and displayed in one cell, and nothing anywhere reads its
    # `.value`: every trait is py→js, so no interaction with this widget can make
    # marimo re-run anything. That is what lets it sit upstream of the GPU cells.
    probe_grid = mo.ui.anywidget(_ProbeGrid(**_pack))
    probe_summary = _layer_summary(_pack)
    probe_grid
    return probe_summary,


@app.cell(hide_code=True)
def _(mo, probe_summary):
    # The numbers the widget states, restated from the same pure function that
    # feeds it — so the claim in the prose can never drift from the data.
    _worst = max(probe_summary, key=lambda _r: _r["junk"])
    _rows = [
        {
            "Layer": _r["name"].replace("Layer_", ""),
            "Unique tokens": _r["unique"],
            "Junk": _r["junk"],
            "Undecodable": _r["undecodable"],
            "Matches final layer": _r["matches_final"],
            "Modal token": repr(_r["modal_token"]),
        }
        for _r in probe_summary
    ]
    mo.vstack([
        mo.md(
            f"###### Per-layer census — the junkiest layer is **{_worst['name']}** "
            f"with **{_worst['junk']}** junk cells and only **{_worst['unique']}** "
            "distinct tokens across every audio position"
        ),
        mo.ui.table(_rows, selection=None, pagination=True, page_size=12),
    ], gap=0.4)
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
        from src.precompute import validate_precompute_meta as _validate_pre

        _pre = _load_pre(PRECOMPUTED_DIR)
        # This cell takes four knobs it cannot honour in replay mode. All four are
        # pinned in the artifact's meta, so check them rather than let each one
        # quietly lie for the whole session — the heatmap caption below reads
        # ATTENTION_CAPTURE_LAYERS to say which layers are on screen, and would
        # otherwise describe the current knob rather than the cached data.
        _validate_pre(
            _pre["meta"],
            attention_prompt=ATTENTION_PROMPT,
            knockout_rules=[list(_r) for _r in KNOCKOUT_RULES],
            max_new_tokens=MAX_NEW_TOKENS,
            attention_capture_layers=list(ATTENTION_CAPTURE_LAYERS),
        )
        baseline_text = _pre["baseline_text"]
        knockout_text = _pre["knockout_text"]
        attention_summary = _pre["knockout_attention_summary"]
        baseline_attention_summary = _pre["baseline_attention_summary"]
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

        # Capture during the baseline too. The heatmap below used to show *only*
        # the knockout run while calling itself "captured attention", so the
        # blocked modality read as ~0 — which is the intervention, not a finding.
        # This costs nothing extra: the baseline generation already happens, and
        # an empty rule list means no knockout hooks are registered.
        with block_attention(
            attention_model, [], attention_token_types, len(attention_token_types),
            track_attention=True, capture_layer_range=ATTENTION_CAPTURE_LAYERS,
        ) as _base_cap:
            with mo.status.spinner(title="Baseline generation (+ attention capture)…"):
                with torch.no_grad():
                    # Thinker-direct generation (see the logit cell): avoids the omni
                    # wrapper's talker requirement. Knockout hooks live on the thinker's
                    # layers, so they still fire below. Greedy (do_sample=False) so the
                    # baseline caption — reused as C by the teacher-forced cell below —
                    # is deterministic.
                    # No `output_attentions=True` here. The capture is done by
                    # per-module hooks (`force_attention_output_hook` +
                    # `attention_hook_fn`), which copy only the selected layers to
                    # CPU. Setting the flag at the `generate` level additionally
                    # makes Transformers retain *every* layer's attention for
                    # *every* decode step in the return object — precisely the
                    # 24 GB-classroom OOM that hook exists to avoid.
                    _base = attention_model.thinker.generate(
                        **attention_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                        return_dict_in_generate=True,
                    )
            _base_captured = {layer: list(v) for layer, v in _base_cap.items()}
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
                    # Same as the baseline: the per-module capture hooks do the
                    # work, so the model-level flag is redundant here and costs
                    # every layer x every step of retained attention.
                    _ko = attention_model.thinker.generate(
                        **attention_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                        return_dict_in_generate=True,
                    )
            _captured = {layer: list(v) for layer, v in _cap.items()}
        knockout_text = attention_processor.batch_decode(
            _ko.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        # Reduce to the plot-ready matrix now, so the heatmap cell consumes the
        # same shape whether live or replayed (raw tensors are never committed).
        # `decode_only=True` matches `scripts/generate_precompute.py`, which
        # produced the committed matrices: it drops the multi-query prefill
        # snapshot, whose row has zero `generated` key mass. Without it the live
        # numbers would be the same *shape* but a different *quantity* than the
        # replayed ones (~1/MAX_NEW_TOKENS of the mass shifted off `generated`).
        attention_summary = _summarize_attention(
            _captured, attention_token_types, decode_only=True
        )
        baseline_attention_summary = _summarize_attention(
            _base_captured, attention_token_types, decode_only=True
        )
        _ko_rules = KNOCKOUT_RULES
        _ko_banner = None

    from wigglystuff import TextCompare as _TextCompare

    _ko_cmp = mo.vstack([
        mo.md(
            f"**Baseline** (left) vs **knockout** `{_ko_rules}` (right) — shared "
            "phrases highlight on hover; **unhighlighted text is where the "
            "knockout changed the caption**. This is one fixed layer band; the "
            "🎚️ section below sweeps the band interactively."
        ),
        mo.ui.anywidget(_TextCompare(
            text_a=baseline_text, text_b=knockout_text, min_match_words=2
        )),
    ])
    _ko_display = mo.vstack([_ko_banner, _ko_cmp]) if _ko_banner is not None else _ko_cmp
    _ko_display
    return (
        attention_baseline_ids,
        attention_inputs,
        attention_summary,
        attention_token_types,
        baseline_attention_summary,
        knockout_text,
    )


@app.cell(hide_code=True)
def _(ATTENTION_CAPTURE_LAYERS, mo):
    mo.md(f"""
    ## Captured attention by key modality — baseline **and** knockout

    A **descriptive** summary (not causal importance): for each captured layer we
    average heads and sum the final query's attention over each token group.

    Read the two panels together — they share one color scale, so a cell that
    looks darker really is smaller. The **knockout** panel's blocked column is
    near-zero *by construction*: that is the intervention working, not a finding
    about the model. Only the **baseline** panel says anything about where this
    model's attention normally goes, and only the **Δ** panel shows where the mask
    pushed the mass instead. Reading the knockout panel alone is how "the model
    ignores the video" gets concluded from a cell that was told to ignore the video.

    Only layers `{ATTENTION_CAPTURE_LAYERS[0]}`–`{ATTENTION_CAPTURE_LAYERS[1] - 1}`
    are shown: that is `ATTENTION_CAPTURE_LAYERS` in the parameters cell. Widening
    it is a legitimate experiment with a real cost — see the knob table above.
    """)
    return


@app.cell
def _(attention_summary, baseline_attention_summary, mo, np, plt):
    # Each summary is `(layers, modalities, matrix)` — computed live from captured
    # tensors, or loaded from the committed matrices in USE_PRECOMPUTED mode. Same
    # shape either way, so this plot is unchanged between the two.
    if attention_summary is None:
        _out = mo.md("> No attention tensors were returned by this build; the text comparison above is the result.")
    else:
        _layers, _mods, _ko_mat = attention_summary
        _ko_mat = np.asarray(_ko_mat, dtype=float)
        # Panels are `(title, matrix, cmap, vmin, vmax)`. The two mass panels
        # SHARE a scale: they exist to be read against each other, and letting
        # each autoscale to its own max is the same defect as a per-run color
        # scale on the Δ strip — two pictures that look alike while describing
        # different numbers. The Δ panel keeps its own symmetric diverging scale,
        # because it is a different quantity with a meaningful zero.
        _mass_hi = max(1e-9, float(_ko_mat.max()))
        _panels = [("Knockout run", _ko_mat, "magma", 0.0, _mass_hi)]
        if baseline_attention_summary is not None:
            _bl_mat = np.asarray(baseline_attention_summary[2], dtype=float)
            if _bl_mat.shape == _ko_mat.shape:
                _delta = _ko_mat - _bl_mat
                _lim = max(1e-9, float(np.abs(_delta).max()))
                _mass_hi = max(1e-9, float(max(_bl_mat.max(), _ko_mat.max())))
                _panels = [
                    ("Baseline (no knockout)", _bl_mat, "magma", 0.0, _mass_hi),
                    ("Knockout run", _ko_mat, "magma", 0.0, _mass_hi),
                    ("Δ = knockout − baseline", _delta, "RdBu_r", -_lim, _lim),
                ]

        _fig, _axes = plt.subplots(
            1, len(_panels),
            figsize=(4.6 * len(_panels), max(3, len(_layers) * 0.6)),
            constrained_layout=True,
        )
        _axes = np.atleast_1d(_axes)
        for _ax, (_title, _mat, _cmap, _vmin, _vmax) in zip(_axes, _panels):
            _diverging = _vmin < 0
            _im = _ax.imshow(_mat, aspect="auto", cmap=_cmap, vmin=_vmin, vmax=_vmax)
            _ax.set(
                title=_title, xlabel="Key modality", ylabel="Thinker layer",
                xticks=np.arange(len(_mods)), xticklabels=_mods,
                yticks=np.arange(len(_layers)), yticklabels=_layers,
            )
            # Annotate against the cell's own background: magma runs dark->bright
            # with value, so white only reads on the low end; the diverging Δ panel
            # is lightest in the middle, where black always reads. Contrast is
            # judged against the *drawn* scale, which is now shared across the two
            # mass panels.
            for _ri in range(_mat.shape[0]):
                for _ci in range(_mat.shape[1]):
                    _v = float(_mat[_ri, _ci])
                    if _diverging:
                        _color = "#111"
                        _label = f"{_v:+.2f}"
                    else:
                        _color = "white" if _v < 0.6 * _vmax else "#111"
                        _label = f"{_v:.2f}"
                    _ax.text(_ci, _ri, _label, ha="center", va="center",
                             color=_color, fontsize=9)
            _fig.colorbar(_im, ax=_ax, label="Attention mass")
        _out = _fig
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🎚️ Interactive: which layer band carries the pathway?

    The comparison above is a **single** knockout — one modality, one fixed layer
    band (all 36 layers). Blocking everywhere tells you *that* a pathway matters,
    not *where* it is used. This section sweeps the band: pick a target modality
    and a layer window, regenerate, and see how the caption drifts from the
    baseline.

    Try `[0, 12)` vs `[12, 24)` vs `[24, 36)` against the same target.

    **Reading a null.** A band whose caption is unchanged shows *no effect under
    this measurement* — which is consistent with redundancy, an indirect route
    that this rule does not cut, or a metric too coarse to see the change. It is
    **not** evidence that the pathway isn't there. The string diff is binary; the
    teacher-forced Δ below is the continuous version of the same question, and it
    is the one that can show a *small* effect.

    Each ▶ runs one greedy generation on the already-encoded clip (a few
    seconds); the baseline is reused, never regenerated. Both captions are shown
    **answer-only** — the shared prompt is stripped so the diff is about the
    model's words, not the instruction. Every ▶ is appended to the **run ledger**
    below, so the previous band stays on screen to compare against.
    """)
    return


@app.cell
def _(Counter, attention_token_types, mo):
    # The modality census, shown next to the controls rather than folded into a
    # dropdown label. `Counter` omits absent keys, so `image` — offered in every
    # picker — used to be invisible rather than visibly zero, and a rule targeting
    # it masked nothing while reporting "no effect".
    _census = Counter(attention_token_types)
    _rows = [
        {
            "Modality": _m,
            "Tokens in this input": _census.get(_m, 0),
            "Usable as a target here": "yes" if _census.get(_m, 0) else "no — none present",
        }
        for _m in ("video", "audio", "query_text", "image")
    ]
    mo.vstack([
        mo.md("###### What is actually in this encoded input"),
        mo.ui.table(_rows, selection=None, pagination=False),
        mo.md(
            "`generated` is not listed: those positions do not exist until the model "
            "decodes. That makes a `generated` source **live during generation** "
            "(this section) and **inert in a forward pass** (the 🎛️ scoreboard below)."
        ),
    ], gap=0.4)
    return


@app.cell
def _(KNOCKOUT_RULES, attention_model, mo):
    _band_layers = len(attention_model.thinker.model.layers)
    _band_targets = ["video", "audio", "image", "query_text"]
    _band_default = KNOCKOUT_RULES[0][1] if KNOCKOUT_RULES else "video"

    def _band_validate(_v):
        if not _v:
            return None
        if not (_v.get("prediction") or "").strip():
            return "Write a prediction before running — one that could turn out wrong."
        # `.get` with a default throughout: a batch's value is a partial dict
        # until the frontend has pushed state for every child, so indexing
        # directly raises KeyError on the first render instead of validating.
        _lo, _hi = _v.get("layers") or (0, 1)
        if int(_hi) <= int(_lo):
            return (
                f"[{int(_lo)}, {int(_hi)}) masks 0 layers — `end` is exclusive. "
                "This would run a baseline and report it as 'no effect'."
            )
        return None

    band_controls = mo.md(
        "**Prediction before ▶** — if this band carries the pathway, what should "
        "happen to the caption, and what would count against you?\n\n"
        "{prediction}\n\n"
        "Forbid **generated** tokens from attending to {target} "
        "across thinker layers {layers}\n\n"
        "{null_band} — treat this run as my **null band**: a band I predict will "
        "do nothing. That is the control for this family, and it is what moves "
        "the ledger's *claims with no control* count off red.\n\n"
        f"(`end` is exclusive; this thinker has **{_band_layers}** layers. The clip, "
        "prompt, and frame count stay as set in the parameters cell.)"
    ).batch(
        prediction=mo.ui.text_area(
            placeholder="e.g. blocking video in [0,12) will change the caption more "
                        "than [24,36), because the description is assembled early.",
            rows=2,
            full_width=True,
        ),
        target=mo.ui.dropdown(
            _band_targets,
            value=_band_default if _band_default in _band_targets else "video",
        ),
        layers=mo.ui.range_slider(
            0, _band_layers, step=1, value=[0, _band_layers // 3], show_value=True
        ),
        null_band=mo.ui.checkbox(value=False),
    ).form(
        submit_button_label="▶ Regenerate with this band",
        bordered=True,
        # Refuse the empty band *before* the GPU runs. Dragging both handles onto
        # the same layer is what a student does to ask "is it exactly layer 12?",
        # and the answer used to be a confident, bordered "no effect" tile.
        validate=_band_validate,
    )
    band_controls
    return (band_controls,)


@app.cell
def _(
    ATTENTION_PROMPT,
    LEDGER_LOG,
    MAX_NEW_TOKENS,
    NFRAMES,
    USE_PRECOMPUTED,
    VIDEO_PATH,
    append_run,
    attention_baseline_ids,
    attention_inputs,
    attention_model,
    attention_processor,
    attention_token_types,
    band_controls,
    block_attention,
    mo,
    run_record,
    set_runs,
    torch,
):
    import difflib as _difflib

    from wigglystuff import TextCompare as _BandCompare

    _bp = band_controls.value
    mo.stop(
        _bp is None,
        mo.callout(
            mo.md("Pick a target and a layer band, then press **▶ Regenerate with this band**."),
            kind="info",
        ),
    )
    # The form's `validate=` runs only in the submit-button handler; marimo's
    # Ctrl/Cmd+Enter shortcut sets the value directly and skips it. The empty-band
    # check has a backstop (`rule_reach` raises inside `block_attention`), but the
    # prediction gate has none — and it is the one that makes hypothesis-before-▶
    # structurally unavoidable rather than merely suggested.
    mo.stop(
        not (_bp.get("prediction") or "").strip(),
        mo.callout(
            mo.md(
                "**Write a prediction first** — one that could turn out wrong. "
                "(Ctrl/Cmd+Enter skips the form's own check; the run is held here "
                "instead.)"
            ),
            kind="warn",
        ),
    )
    mo.stop(
        USE_PRECOMPUTED or attention_inputs is None,
        mo.callout(
            mo.md(
                "**This sweep needs the live model** — it regenerates a caption per "
                "band, so it is skipped while `USE_PRECOMPUTED=True`."
            ),
            kind="warn",
        ),
    )

    _lo, _hi = int(_bp["layers"][0]), int(_bp["layers"][1])
    _band_rules = [("generated", _bp["target"], _lo, _hi)]
    _plen = attention_inputs["input_ids"].shape[1]
    # Answer-only text: slicing off the shared prompt keeps the diff focused on
    # the generated words (the prompt would otherwise dominate as one big match).
    _base_ans = attention_processor.batch_decode(
        attention_baseline_ids[:, _plen:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    _band_out = None
    try:
        with mo.status.spinner(
            title=f"Knockout generation · generated→{_bp['target']} [{_lo},{_hi})…"
        ):
            with block_attention(
                attention_model, _band_rules, attention_token_types,
                len(attention_token_types), track_attention=False,
            ):
                with torch.no_grad():
                    _band_ids = attention_model.thinker.generate(
                        **attention_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                    )
        _band_ans = attention_processor.batch_decode(
            _band_ids[:, _plen:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    except Exception as _e:  # noqa: BLE001 — surface any run failure in-notebook
        _band_out = mo.callout(
            mo.md(f"**Run failed** — `{type(_e).__name__}: {_e}`"), kind="danger"
        )

    if _band_out is None:
        _ratio = _difflib.SequenceMatcher(
            None, _base_ans.split(), _band_ans.split()
        ).ratio()
        _unchanged = _band_ans.strip() == _base_ans.strip()
        _band_out = mo.vstack([
            mo.md(
                f"**Knockout** `generated→{_bp['target']}` **[{_lo}, {_hi})** "
                f"&nbsp;·&nbsp; {_hi - _lo} of "
                f"{len(attention_model.thinker.model.layers)} layers blocked"
            ),
            mo.hstack([
                mo.stat(
                    value=f"{_ratio:.0%}",
                    label="Caption similarity to baseline",
                    caption="word-level; 100% = this band changed nothing",
                    # No `direction=`: an unchanged caption used to get the green
                    # up-arrow and a moved one the red down-arrow, so a three-band
                    # sweep read as two failures and one success rather than as a
                    # localization. Neither outcome is the good one here.
                    bordered=True,
                ),
                mo.stat(
                    value="unchanged" if _unchanged else "changed",
                    label="Effect of this band",
                    caption=(
                        "no effect under this measurement — consistent with "
                        "redundancy or an indirect route, not proof of absence"
                        if _unchanged else "blocking here moved the caption"
                    ),
                    bordered=True,
                ),
            ], widths="equal", gap=1),
            mo.md(
                "**Baseline** (left) vs **this band** (right) — shared phrases "
                "highlight on hover; **unhighlighted text is what this band changed**."
            ),
            mo.ui.anywidget(_BandCompare(
                text_a=_base_ans, text_b=_band_ans, min_match_words=2
            )),
        ])
        # Record it. `set_runs` is a SetFunctor, not the State object, so a cell
        # that only *sets* never re-runs itself — this append cannot re-trigger
        # the generation above. `run_record(...)` is bound as a default argument
        # so it evaluates here, in the GPU cell, leaving `prev` as the only lazy
        # input to the lambda.
        try:
            set_runs(
                lambda _prev, _r=run_record(
                    kind="band_sweep",
                    condition=f"generated→{_bp['target']} [{_lo},{_hi})",
                    metric_name="caption_similarity",
                    metric_value=round(_ratio, 4),
                    metric_unit="ratio",
                    # Every input that can move the number belongs in `config`:
                    # `run_id` digests config *and* metric, so an input left out
                    # produces a second row with a different number, an identical
                    # `condition`, and an empty `changed` column — two runs that
                    # look controlled and are not.
                    config={
                        "clip": VIDEO_PATH.name, "nframes": NFRAMES,
                        "prompt": ATTENTION_PROMPT,
                        "target": _bp["target"], "start": _lo, "end": _hi,
                        "max_new_tokens": MAX_NEW_TOKENS,
                    },
                    prediction=_bp.get("prediction", ""),
                    is_control=bool(_bp.get("null_band")),
                ): append_run(_prev, _r, log_path=LEDGER_LOG)
            )
        except Exception as _le:  # noqa: BLE001 — a ledger bug must never eat a run
            print("ledger append failed:", type(_le).__name__, _le)
    _band_out
    return


@app.cell
def _(ledger_view):
    ledger_view()
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
        w9_tf_result = None
        _w9_out = mo.callout(
            mo.md(
                "**Teacher forcing needs the live model** — this cell is skipped while "
                "`USE_PRECOMPUTED=True`. (Cached replay of this measurement lands with F5b.)"
            ),
            kind="warn",
        )
    else:
        from src.teacher_forcing import teacher_forced_delta as _w9_tfd

        # Mirror the params-cell intervention with `answer` as the source: the
        # caption is input now, so `answer → target` is the measurable counterpart
        # of the generation-time `generated → target` diff above.
        _w9_rules = [("answer", _t, _a, _b) for (_s, _t, _a, _b) in KNOCKOUT_RULES]
        _w9_prompt_len = attention_inputs["input_ids"].shape[1]
        _w9_c_ids = attention_baseline_ids[:, _w9_prompt_len:]

        w9_tf_result = None
        try:
            with mo.status.spinner(title="Teacher-forced scoring (2 forward passes)…"):
                w9_tf_result = _w9_tfd(
                    attention_model,
                    attention_processor,
                    attention_inputs,
                    attention_token_types,
                    _w9_rules,
                    cached_caption_ids=_w9_c_ids,
                )
        except Exception as _e:  # noqa: BLE001 — surface any failure in-notebook
            _w9_out = mo.callout(
                mo.md(f"**Teacher-forced scoring failed** — `{type(_e).__name__}: {_e}`"),
                kind="danger",
            )

        if w9_tf_result is not None:
            _w9_delta = [float(x) for x in w9_tf_result["delta"].detach().cpu().float().tolist()]
            _w9_total = w9_tf_result["delta_total"]
            _w9_rule_txt = " + ".join(f"`answer→{_r[1]}` [{_r[2]},{_r[3]})" for _r in _w9_rules)
            _w9_out = mo.vstack([
                mo.md(f"**Knockout** {_w9_rule_txt} &nbsp;·&nbsp; baseline caption teacher-forced as `answer`"),
                mo.hstack([
                    mo.stat(
                        value=f"{_w9_total:+.2f}",
                        label="Σ Δ log-lik (nats)",
                        caption="knockout − baseline · negative = believed less",
                        direction="decrease" if _w9_total < 0 else "increase",
                        bordered=True,
                    ),
                    mo.stat(
                        value=f"{w9_tf_result['delta_mean']:+.3f}",
                        label="Δ / token (nats)",
                        caption=(
                            "length-normalized — **this** is the number to compare "
                            "across runs; Σ scales with caption length"
                        ),
                        direction="decrease" if w9_tf_result["delta_mean"] < 0 else "increase",
                        bordered=True,
                    ),
                    mo.stat(
                        value=str(len(_w9_delta)),
                        label="Caption tokens scored",
                        caption="greedy baseline, teacher-forced",
                        bordered=True,
                    ),
                ], widths="equal", gap=1),
            ])
    _w9_out
    return (w9_tf_result,)


@app.cell
def _(mo, w9_tf_result):
    # Skipped quietly in USE_PRECOMPUTED mode / after a scoring failure.
    mo.stop(w9_tf_result is None)
    from src.teacher_forcing import threshold_slider_params as _w9_params
    from wigglystuff import TangleSlider as _W9Tangle

    # Bounds/step/default derived from this caption's own drops, so the drag
    # spans the range in ~300 px and starts with a meaningful set outlined.
    w9_threshold = mo.ui.anywidget(_W9Tangle(
        suffix=" nats",
        **_w9_params(w9_tf_result["caption_tokens"], w9_tf_result["delta"]),
    ))
    mo.md(
        "###### Per-token Δ log-likelihood (hover a word for its tokens' nats)\n\n"
        f"Show only the words that lost more than {w9_threshold} — "
        "**drag the underlined number sideways** (or click it and type). Words "
        "past the threshold turn **bold with an outline**; the rest fade out, so "
        "the strip visibly re-sorts as you drag. Nothing here touches the model."
    )
    return (w9_threshold,)


@app.cell
def _(mo, w9_threshold, w9_tf_result):
    from src.teacher_forcing import group_tokens_into_words as _w9_group
    from src.teacher_forcing import render_delta_strip as _w9_strip

    _delta = [float(_x) for _x in w9_tf_result["delta"].detach().cpu().float().tolist()]
    _th = abs(float(w9_threshold.value.get("amount", 0.0)))
    _words = _w9_group(w9_tf_result["caption_tokens"], _delta)
    _hit = [_w for _w in _words if _w[1] < -_th]
    _share = (
        100.0 * sum(_w[1] for _w in _hit) / w9_tf_result["delta_total"]
        if w9_tf_result["delta_total"]
        else 0.0
    )
    mo.vstack([
        mo.Html(
            "<div style='line-height:2.1;font-family:monospace;font-size:15px'>"
            + _w9_strip(w9_tf_result["caption_tokens"], _delta, highlight_below=_th)
            + "</div>"
        ),
        mo.md(
            f"**{len(_hit)}/{len(_words)}** words drop more than −{_th:.2f} nats — together "
            f"Δ = {sum(_w[1] for _w in _hit):+.2f} nats, **{_share:.0f}%** of the total "
            f"{w9_tf_result['delta_total']:+.2f}."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wrap-up
    """)
    return


@app.cell
def _(knockout_text, logit_csv_written, mo):
    _ = knockout_text  # depend on the knockout run
    _ok = (
        logit_csv_written is not None
        and logit_csv_written.is_file()
        and logit_csv_written.stat().st_size > 0
    )
    mo.md(
        f"### Fixed run complete\n\n"
        + (
            f"- Logit-lens CSV written: **True** — `{logit_csv_written}`\n"
            if _ok
            else "- Logit-lens CSV: **not written** — the run produced no "
                 "audio-token rows (most likely a clip with no audio track).\n"
        )
        + "- Baseline vs knockout compared, both attention panels shown.\n\n"
        "**Everything above is the worked example — one setting, run for you.** The "
        "two playgrounds below are where you run your own: each ▶ takes a prediction "
        "first and appends its result to the run ledger, so the previous run stays on "
        "screen to compare against. Start with the 🎯 section's silent-clip control."
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

    **What moves what.** The score is measured at **audio** token positions, and
    the number of those is fixed by the clip's *duration*. So `Frames` changes the
    **video** token count and will not move "Audio tokens scored" at all — if you
    sweep it expecting that number to change, nothing is broken. Both counts are
    reported below so you can see which one you moved.

    This is a single forward pass over the prompt, so **`generated` and `answer`
    are inert here on either side of a rule** — there are no such positions until
    the model decodes. Rules that reach nothing are refused rather than run.
    Use `audio`, `video` or `query_text`. Build one rule with the dropdowns, or
    enter several in the advanced field.
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
        f"`generated` is **inert** here on *either* side of a rule (there are no "
        f"generated positions during a forward pass), and `image` is 0 tokens for a "
        f"video clip. Rules that reach nothing are refused, not run. Layer `end` is "
        f"exclusive; this thinker has **{_n_layers}** layers, so `[0, {_n_layers})` "
        f"spans all of them."
    )
    _template = (
        "**Prediction before ▶** — which layers should lose diversity, and why?\n\n"
        "{prediction}\n\n"
        "**Clip** {clip} &nbsp; (the silent control is in the repo — pick it by name; "
        "there is nothing to upload)\n\n"
        "Only if you chose **Upload** — `mp4 / mov / mkv / webm`, ≤ 250 MB, ≤ 120 s, ≤ 1080p:\n\n"
        "{video}\n\n"
        "**Frames sampled from the clip** {nframes} &nbsp; *(moves the **video** token "
        "count; audio positions are fixed by duration)*\n\n"
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

    def _ko_validate(_v):
        if not _v:
            return None
        if not (_v.get("prediction") or "").strip():
            return "Write a prediction before running — one that could turn out wrong."
        if _v.get("clip") == "Upload" and not _v.get("video"):
            return "You chose Upload but did not select a file."
        # `validate` receives the *frontend* value, not the converted Python one
        # (`form._validate` calls `self.validate(value.value)`), and a dropdown
        # arrives as `list[str]` — `['generated']`, not `'generated'`. Comparing
        # the raw value to a string silently never matches, which would let the
        # inert-rule refusal this section promises fail open: the student pays a
        # full clip encode and then gets a raw ValueError from the backstop.
        def _one(_x):
            return _x[0] if isinstance(_x, list) and _x else _x

        # `.get` with defaults: the batch value is partial until the frontend has
        # pushed state for every child.
        _lo, _hi = _v.get("ko_layers") or (0, 1)
        if _v.get("ko_enable") and not (_v.get("ko_rules_text") or "").strip():
            if int(_hi) <= int(_lo):
                return (
                    f"[{int(_lo)}, {int(_hi)}) masks 0 layers — `end` is exclusive."
                )
            if _one(_v.get("ko_source")) == "generated" or _one(_v.get("ko_target")) == "generated":
                return (
                    "`generated` is inert in a forward pass on either side of a rule. "
                    "Use audio / video / query_text, or this run is just a baseline."
                )
        return None

    ko_controls = mo.md(_template).batch(
        prediction=mo.ui.text_area(
            placeholder="e.g. blocking audio→video will cut diversity most in the "
                        "middle layers, where the two streams are being fused.",
            rows=2,
            full_width=True,
        ),
        clip=mo.ui.radio(
            ["Default clip", "Silent control", "Upload"],
            value="Default clip",
            inline=True,
        ),
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
        validate=_ko_validate,
    )
    ko_controls
    return (ko_controls,)


@app.cell
def _(
    Counter,
    LEDGER_LOG,
    LOGIT_CSV_PATH,
    LOGIT_PROMPT,
    USE_PRECOMPUTED,
    analyze_and_save_audio_logits_to_csv,
    append_run,
    attention_model,
    attention_processor,
    block_attention,
    cache_put,
    clear_logit_lens_hooks,
    create_attention_token_mapping,
    csv,
    ko_controls,
    mo,
    np,
    playground_caches,
    plt,
    register_logit_lens_hooks,
    resolve_clip,
    run_record,
    set_runs,
    torch,
):
    from contextlib import nullcontext as _nullcontext

    from qwen_omni_utils import process_mm_info as _process_mm_info

    _p = ko_controls.value
    mo.stop(
        _p is None,
        mo.callout(
            mo.md("Set the parameters above and press **▶ Run logit-lens diversity**."),
            kind="info",
        ),
    )
    # Backstop for the prediction gate: `validate=` is skipped by marimo's
    # Ctrl/Cmd+Enter shortcut. See the band-sweep cell above.
    mo.stop(
        not (_p.get("prediction") or "").strip(),
        mo.callout(
            mo.md("**Write a prediction first** — one that could turn out wrong."),
            kind="warn",
        ),
    )
    # The band cell already guards replay mode; these two did not, and submitting
    # either called `apply_chat_template` on a `None` processor — an AttributeError
    # instead of the honest sentence the band cell knows how to print.
    mo.stop(
        USE_PRECOMPUTED,
        mo.callout(
            mo.md(
                "**This playground needs the live model** — it runs a fresh forward "
                "pass per submit, so it is skipped while `USE_PRECOMPUTED=True`."
            ),
            kind="warn",
        ),
    )

    _results_dir = LOGIT_CSV_PATH.parent

    _video_path, _is_control, _clip_err = resolve_clip(_p["clip"], _p["video"])
    mo.stop(
        _clip_err is not None,
        mo.callout(mo.md(f"**Clip unavailable** — {_clip_err}"), kind="danger"),
    )
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
                    attention_model, rules, types, len(types), track_attention=False,
                    # One forward pass over the prompt: only the prefill branch
                    # fires, so `generated`/`answer` rules would mask nothing.
                    # Saying so here is what turns a silent baseline into a
                    # refusal the student can read.
                    context="forward",
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
                caption="fixed by clip duration — Frames does not move this",
                bordered=True,
            ),
            mo.stat(
                value=str(Counter(_types).get("video", 0)),
                label="Video tokens encoded",
                caption=f"this is what Frames={_nframes} moves",
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
        _children = [
            mo.md(
                f"**Clip** `{_video_path.name}`"
                + (" _(silent control)_" if _is_control else "")
                + f" &nbsp;·&nbsp; **Frames** {_nframes} "
                f"&nbsp;·&nbsp; **Prompt** _{_prompt}_ &nbsp;·&nbsp; **Knockout** {_rule_txt}"
            ),
            mo.hstack(_stats, widths="equal", gap=1),
            _fig,
            mo.md("###### Layers ranked by decoded-prediction diversity (higher = more distinct audio-token predictions)"),
            _table,
        ]
        _scoreboard = mo.vstack(_children)

        # Record it. `_mean_delta` only exists when a baseline was run alongside
        # the knockout; without one there is no effect size to log, only a
        # description — so the metric reported changes with it, rather than
        # stacking two different quantities in one column.
        try:
            if _both:
                _metric = ("mean_delta_diversity", round(_mean_delta, 3), "unique preds")
            else:
                _metric = (
                    "mean_unique_per_layer", round(sum(_primary_u) / _n_l, 3), "unique preds"
                )
            set_runs(
                lambda _prev, _r=run_record(
                    kind="diversity",
                    condition=(
                        " + ".join(f"{r[0]}→{r[1]} [{r[2]},{r[3]})" for r in _rules)
                        if _rules else "baseline (no knockout)"
                    ),
                    metric_name=_metric[0],
                    metric_value=_metric[1],
                    metric_unit=_metric[2],
                    config={
                        "clip": _video_path.name, "nframes": _nframes, "prompt": _prompt,
                        "rules": [list(r) for r in _rules], "compare": _compare,
                    },
                    prediction=_p.get("prediction", ""),
                    is_control=_is_control,
                    extra={"audio_tokens": _n_audio, "peak_layer": _peak},
                ): append_run(_prev, _r, log_path=LEDGER_LOG)
            )
        except Exception as _le:  # noqa: BLE001 — a ledger bug must never eat a run
            print("ledger append failed:", type(_le).__name__, _le)
    _scoreboard
    return


@app.cell
def _(ledger_view):
    ledger_view()
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
    holding the caption up. Unlike the W9 free-generation string diff it is
    **continuous** (you can see a *small* effect) and **deterministic** (greedy
    caption, forward-only scoring). Nothing runs until you press ▶.

    > **Run the control first.** Set **Clip** to `Silent control` and run
    > `answer → audio`. It is the same frames with a digitally silent audio track:
    > the audio tokens exist but carry no signal, so **Δ / token should be ≈ 0**.
    > Then switch to `Default clip` with the same prompt and layers. A real audio
    > dependency shows up as a clearly larger negative Δ / token.
    >
    > Compare **Δ / token**, not Σ. The two clips produce different captions of
    > different lengths, and Σ scales with length — comparing totals would be
    > meaningless. The acceptance criterion is `|Δ_silent| / token ≪ |Δ_audio| / token`.
    >
    > A control that *can* fail is the whole point, and both runs land in the
    > ledger so you can put the two numbers side by side.
    """)
    return


@app.cell
def _(LOGIT_PROMPT, NFRAMES, attention_model, mo):
    _n_layers = len(attention_model.thinker.model.layers)
    _tf_targets = ["audio", "video", "query_text", "image"]
    _tf_template = (
        "**Prediction before ▶** — name the Δ / token you expect and what would "
        "refute you:\n\n"
        "{prediction}\n\n"
        "**Clip** {clip} &nbsp; (`Silent control` is the falsifiable one — it is in "
        "the repo, so pick it by name rather than uploading it)\n\n"
        "Only if you chose **Upload** — `mp4 / mov / mkv / webm`, ≤ 250 MB, ≤ 120 s, ≤ 1080p:\n\n"
        "{video}\n\n"
        "**Frames sampled from the clip** {nframes}\n\n"
        "**Prompt** {prompt}\n\n"
        "---\n\n"
        "Forbid the **answer** from attending to {target} across thinker layers {layers}\n\n"
        f"(`answer` is the model's own caption, teacher-forced back in; this thinker has "
        f"**{_n_layers}** layers, `end` exclusive.)"
    )

    def _tf_validate(_v):
        if not _v:
            return None
        if not (_v.get("prediction") or "").strip():
            return "Write a prediction before running — one that could turn out wrong."
        if _v.get("clip") == "Upload" and not _v.get("video"):
            return "You chose Upload but did not select a file."
        # `.get` with a default: the batch value is partial on first render.
        _lo, _hi = _v.get("layers") or (0, 1)
        if int(_hi) <= int(_lo):
            return f"[{int(_lo)}, {int(_hi)}) masks 0 layers — `end` is exclusive."
        return None

    tf_controls = mo.md(_tf_template).batch(
        prediction=mo.ui.text_area(
            placeholder="e.g. on the silent clip Δ/token will be within ±0.02 nats; "
                        "on the real clip it will be at least 5x more negative.",
            rows=2,
            full_width=True,
        ),
        clip=mo.ui.radio(
            ["Default clip", "Silent control", "Upload"],
            value="Default clip",
            inline=True,
        ),
        video=mo.ui.file(
            filetypes=[".mp4", ".mov", ".mkv", ".webm", ".avi"], multiple=False, kind="area"
        ),
        nframes=mo.ui.slider(2, 32, step=2, value=NFRAMES, show_value=True, include_input=True),
        prompt=mo.ui.text(value=LOGIT_PROMPT, full_width=True),
        target=mo.ui.dropdown(_tf_targets, value="audio"),
        layers=mo.ui.range_slider(0, _n_layers, step=1, value=[0, _n_layers], show_value=True),
    ).form(
        submit_button_label="▶ Run teacher-forced Δ log-lik",
        bordered=True,
        validate=_tf_validate,
    )
    tf_controls
    return (tf_controls,)


@app.cell
def _(
    LEDGER_LOG,
    LOGIT_PROMPT,
    MAX_NEW_TOKENS,
    USE_PRECOMPUTED,
    append_run,
    attention_model,
    attention_processor,
    cache_put,
    create_attention_token_mapping,
    mo,
    np,
    playground_caches,
    resolve_clip,
    run_record,
    set_runs,
    tf_controls,
):
    from qwen_omni_utils import process_mm_info as _tf_mm_info

    from src.teacher_forcing import teacher_forced_delta as _tfd

    _tp = tf_controls.value
    mo.stop(
        _tp is None,
        mo.callout(
            mo.md("Set the parameters above and press **▶ Run teacher-forced Δ log-lik**."),
            kind="info",
        ),
    )
    # Backstop for the prediction gate: `validate=` is skipped by marimo's
    # Ctrl/Cmd+Enter shortcut. See the band-sweep cell above.
    mo.stop(
        not (_tp.get("prediction") or "").strip(),
        mo.callout(
            mo.md("**Write a prediction first** — one that could turn out wrong."),
            kind="warn",
        ),
    )
    mo.stop(
        USE_PRECOMPUTED,
        mo.callout(
            mo.md(
                "**This measurement needs the live model** — it generates a caption "
                "and scores two forward passes, so it is skipped while "
                "`USE_PRECOMPUTED=True`."
            ),
            kind="warn",
        ),
    )

    _tf_video, _tf_is_control, _tf_clip_err = resolve_clip(_tp["clip"], _tp["video"])
    mo.stop(
        _tf_clip_err is not None,
        mo.callout(mo.md(f"**Clip unavailable** — {_tf_clip_err}"), kind="danger"),
    )
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

    tf_result = None
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
                # Without this the playground scored 32-token captions (the
                # function's own default) while the fixed cell above used
                # MAX_NEW_TOKENS — two different caption lengths, one Σ column.
                max_new_tokens=MAX_NEW_TOKENS,
                cached_caption_ids=_tf_cached_c,
            )
            cache_put(playground_caches["caption"], _tf_cap_key, _tf_res["caption_ids"])
            tf_result = _tf_res
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
        _tf_mean = _tf_res["delta_mean"]
        _tf_stats = [
            mo.stat(
                value=f"{_tf_mean:+.3f}",
                label="Δ / token (nats)",
                caption="**compare this across runs** — Σ scales with caption length",
                direction="decrease" if _tf_mean < 0 else "increase",
                bordered=True,
            ),
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
        _tf_out = mo.vstack([
            mo.md(
                f"**Clip** `{_tf_video.name}`"
                + (" _(silent control)_" if _tf_is_control else "")
                + f" &nbsp;·&nbsp; **Frames** {_tf_nframes} "
                f"&nbsp;·&nbsp; **Prompt** _{_tf_prompt}_ &nbsp;·&nbsp; **Knockout** {_tf_rule_txt}"
            ),
            mo.hstack(_tf_stats, widths="equal", gap=1),
        ])
        try:
            set_runs(
                lambda _prev, _r=run_record(
                    kind="teacher_forcing",
                    condition=f"answer→{_tp['target']} [{_tf_lo},{_tf_hi})",
                    # Δ/token, not Σ: the control and the experiment score
                    # different captions of different lengths, so the total is not
                    # the comparable quantity — logging Σ as the headline would
                    # rebuild the exact confusion this section exists to remove.
                    metric_name="delta_per_token",
                    metric_value=round(_tf_mean, 4),
                    metric_unit="nats/token",
                    config={
                        "clip": _tf_video.name, "nframes": _tf_nframes,
                        "prompt": _tf_prompt, "target": _tp["target"],
                        "start": _tf_lo, "end": _tf_hi,
                        "max_new_tokens": MAX_NEW_TOKENS,
                    },
                    prediction=_tp.get("prediction", ""),
                    is_control=_tf_is_control,
                    extra={"delta_total": round(_tf_total, 4), "n_tokens": len(_tf_toks)},
                ): append_run(_prev, _r, log_path=LEDGER_LOG)
            )
        except Exception as _le:  # noqa: BLE001 — a ledger bug must never eat a run
            print("ledger append failed:", type(_le).__name__, _le)
    _tf_out
    return (tf_result,)


@app.cell
def _(mo, tf_result):
    # No output until the form above has produced a result (and skipped after a
    # failed run) — mirrors the W9 threshold cells.
    mo.stop(tf_result is None)
    from src.teacher_forcing import threshold_slider_params as _tf_params
    from wigglystuff import TangleSlider as _TfTangle

    tf_threshold = mo.ui.anywidget(_TfTangle(
        suffix=" nats",
        **_tf_params(tf_result["caption_tokens"], tf_result["delta"]),
    ))
    mo.md(
        "###### Per-token Δ log-likelihood (hot = believed less after the knockout; "
        "hover a word for its tokens' nats)\n\n"
        f"Show only the words that lost more than {tf_threshold} — "
        "**drag the underlined number sideways** (or click it and type). Only this "
        "strip re-renders, never the model."
    )
    return (tf_threshold,)


@app.cell
def _(mo, tf_result, tf_threshold):
    from src.teacher_forcing import group_tokens_into_words as _tf_group
    from src.teacher_forcing import render_delta_strip as _tf_strip

    _delta = [float(_x) for _x in tf_result["delta"].detach().cpu().float().tolist()]
    _toks = tf_result["caption_tokens"]
    _th = abs(float(tf_threshold.value.get("amount", 0.0)))
    _words = _tf_group(_toks, _delta)
    _hit = [_w for _w in _words if _w[1] < -_th]
    _share = (
        100.0 * sum(_w[1] for _w in _hit) / tf_result["delta_total"]
        if tf_result["delta_total"]
        else 0.0
    )
    _rows = [
        {"pos": _i, "token": _t, "Δ log-lik": round(_d, 3)}
        for _i, (_t, _d) in enumerate(zip(_toks, _delta))
    ]
    mo.vstack([
        mo.Html(
            "<div style='line-height:2.1;font-family:monospace;font-size:15px'>"
            + _tf_strip(_toks, _delta, highlight_below=_th)
            + "</div>"
        ),
        mo.md(
            f"**{len(_hit)}/{len(_words)}** words drop more than −{_th:.2f} nats — together "
            f"Δ = {sum(_w[1] for _w in _hit):+.2f} nats, **{_share:.0f}%** of the total "
            f"{tf_result['delta_total']:+.2f}."
        ),
        mo.ui.table(_rows, selection=None, pagination=True, page_size=16),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 📓 Lab ledger — what you actually ran

    Every ▶ in this notebook is recorded below with the prediction you wrote
    beforehand, the setting you changed since your last run of the same kind, and
    the measurement it produced. Metrics keep their **name and unit**: a caption
    similarity and a Δ/token are not the same quantity and are never stacked in
    one column.

    Two counts are worth watching. **Runs without a control** is the number of
    claims you could not currently defend — pair each experiment with the silent
    clip, or a layer band you expect to do nothing, to bring it to zero. **Runs
    without a verdict** is the number you have not yet said *supported*,
    *refuted*, or *untested* about.

    The ledger survives a form reset, and every run **and verdict** is appended to
    `notebook_results/lab_log.jsonl` as you go, so it survives a molab kernel
    restart too — the table reloads from that file when the notebook starts.
    """)
    return


@app.cell
def _(mo):
    # Refs = {mo} only, deliberately. A form rebuilt on every append would clear
    # itself the moment the run it is describing is recorded.
    verdict_form = mo.md(
        "**Resolve a run** — copy its `id` from the ledger above.\n\n"
        "Run {run_id} was {verdict}. A rival explanation that would produce the "
        "same number: {rival}"
    ).batch(
        run_id=mo.ui.text(placeholder="e.g. 3f9a1c02"),
        verdict=mo.ui.dropdown(["supported", "refuted", "untested"], value="untested"),
        rival=mo.ui.text(placeholder="e.g. the caption just got shorter", full_width=True),
    ).form(submit_button_label="Record verdict", bordered=True)
    verdict_form
    return (verdict_form,)


@app.cell
def _(LEDGER_LOG, apply_verdict, mo, set_runs, verdict_form):
    # Sets but never reads: referencing `get_runs` here would re-run this cell on
    # every append and re-apply the last verdict.
    _v = verdict_form.value
    mo.stop(_v is None or not (_v.get("run_id") or "").strip())
    set_runs(
        lambda _prev, _id=_v["run_id"].strip(), _k=_v.get("verdict", "untested"),
        _r=_v.get("rival", ""): apply_verdict(
            _prev, _id, _k, rival=_r, log_path=LEDGER_LOG
        )
    )
    # Deliberately *not* `kind="success"`. This cell cannot read `get_runs`
    # (that would make it re-run on every append), so it cannot know whether the
    # id matched — and `apply_verdict` is a no-op on an unknown one. Claiming
    # success here would be a confident, wrong report in the one notebook whose
    # whole subject is that those are the enemy.
    mo.callout(
        mo.md(
            f"Sent **{_v.get('verdict', 'untested')}** to run "
            f"`{_v['run_id'].strip()}`. An id that matches nothing is ignored — "
            "check that the **verdict** column below actually changed."
        ),
        kind="info",
    )
    return


@app.cell
def _(ledger_view):
    ledger_view()
    return


@app.cell
def _(RESULTS_DIR, get_runs, mo, worksheet_md):
    # References `get_runs`, which is exactly what keeps the download current: the
    # cell re-runs on every append, so the file offered is never a stale snapshot.
    _runs = get_runs()
    _md = worksheet_md(_runs)
    mo.vstack([
        mo.md("###### Export for `WORKSHEET.md`"),
        mo.download(
            _md.encode(),
            filename="lab_log.md",
            mimetype="text/markdown",
            label=f"⬇ Download your {len(_runs)} run(s) as a worksheet table",
        ),
        mo.accordion({
            "Show the markdown (select and copy)": mo.md(f"```markdown\n{_md}\n```")
        }),
        mo.md(
            f"_Also being appended live to `{RESULTS_DIR / 'lab_log.jsonl'}`._"
        ),
    ], gap=0.4)
    return


if __name__ == "__main__":
    app.run()
