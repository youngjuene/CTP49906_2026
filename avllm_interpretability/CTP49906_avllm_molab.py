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
    # Counterpoint Lens studio

    **Studio question:** when a multimodal model captions a video, what evidence
    suggests that sound, image, prompt, or learned language patterns shaped its
    answer? A plausible caption alone cannot settle that question.

    You will make and register a first audiovisual cut, observe a short saved
    reference replay, change one condition at a time, compare three readings,
    revise your explanation and artifact, and finish with a bounded architecture
    proposal.

    **Required** activities form the shortest complete route. **Choice** activities
    let you follow one evidence question. **Advanced** controls stay collapsed and
    are never prerequisites. Before every run, commit a prediction that could be
    wrong; after every run, separate observation, interpretation, limitation, and
    next test.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare your project

    ### 1.1 Orient — Make, test, revise

    **Required · about 10 minutes.** Follow the eight stages in order. Begin with
    the saved course replay; it is deterministic, uses the checked-in reference
    results, and does not allocate a GPU or download model weights. Choose the live
    model only when your instructor has prepared the runtime.

    **Evidence legend**

    - **Observed:** a displayed value, text difference, or validated record.
    - **Inferred:** a bounded interpretation that may still have rival explanations.
    - **Not measured:** a question this activity cannot answer.

    **Data boundary.** Teaching mode is the default and fail-safe mode. Uploads and
    session records are processed inside the hosted Molab session/container, not
    only on your device. Nothing in this notebook automatically sends student media,
    process records, or audience readings to an instructor or outside service.
    Private downloads happen only when you press a download control.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    def _validate_execution_mode(_value):
        if not _value:
            return "Choose a replay or live route."
        return None

    execution_mode_form = mo.ui.dropdown(
        ["Saved course replay", "Live model"],
        value="Saved course replay",
        label="Execution route",
    ).form(
        submit_button_label="Apply execution route",
        validate=_validate_execution_mode,
        bordered=True,
    )
    execution_mode_form
    return (execution_mode_form,)


@app.cell
def _(execution_mode_form):
    USE_PRECOMPUTED = execution_mode_form.value != "Live model"
    return (USE_PRECOMPUTED,)


@app.cell(hide_code=True)
def _(USE_PRECOMPUTED, mo):
    import importlib.metadata
    import importlib.util
    import subprocess
    import sys
    from pathlib import Path

    def _ensure_packages(specs):
        # specs: (import_name, dist_name, exact_version_or_None, pip_spec).
        # molab does not install the `# /// script` block into the running
        # kernel, so enforce the versions used in the rehearsal at runtime.
        to_install = []
        for import_name, dist_name, exact_version, pip_spec in specs:
            if importlib.util.find_spec(import_name) is None:
                to_install.append(pip_spec)
                continue
            if exact_version is not None:
                try:
                    have = importlib.metadata.version(dist_name)
                except importlib.metadata.PackageNotFoundError:
                    to_install.append(pip_spec)
                    continue
                if have != exact_version:
                    to_install.append(pip_spec)
        if to_install:
            with mo.status.spinner(title=f"Installing {', '.join(to_install)}…"):
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", *to_install], check=True
                )

    if not USE_PRECOMPUTED:
        _ensure_packages([
            ("transformers", "transformers", "4.52.0", "transformers==4.52.0"),
            ("accelerate", "accelerate", "1.14.0", "accelerate==1.14.0"),
            ("qwen_omni_utils", "qwen-omni-utils", "0.0.9", "qwen-omni-utils==0.0.9"),
            ("av", "av", None, "av"),
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

    if not USE_PRECOMPUTED:
        _ensure_video_reader()

    # The experiment code (src/) and sample video live under the
    # `avllm_interpretability/` subdirectory of this repo. If the clone already
    # exists, hard-sync it to REPO_REF so pushed fixes reach molab (a kernel
    # restart is still needed to re-import updated modules).
    #
    # REPO_REF selects the source version. Use "main" while iterating; distribute
    # an immutable course tag so later repository changes cannot alter the class
    # run. Fetching through FETCH_HEAD supports both branches and tags.
    REPO_REF = "bb12df8686c0179bf95f0cc90b90f2319ad4040c"
    _local_project = Path(__file__).resolve().parent
    if (_local_project / "src").is_dir() and (_local_project / "assets").is_dir():
        # Local development / a notebook opened from a checked-out release:
        # use that exact working tree rather than cloning a second, stale copy.
        PROJECT_DIR = _local_project
        print(f"using checked-out project: {PROJECT_DIR}")
    else:
        REPO_DIR = Path("CTP49906_2026").resolve()
        if REPO_REF != "main":
            print(f"⚠️ REPO_REF={REPO_REF!r} — notebook source pinned to that ref.")
        if REPO_DIR.exists():
            _sync_title = f"Updating CTP49906_2026 to {REPO_REF}…"
        else:
            _sync_title = f"Cloning CTP49906_2026 @ {REPO_REF} (src + sample video)…"
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
        PROJECT_DIR = REPO_DIR / "avllm_interpretability"
    assert PROJECT_DIR.is_dir(), f"expected code dir not found: {PROJECT_DIR}"
    for _source_root in (PROJECT_DIR, PROJECT_DIR.parent):
        if str(_source_root) not in sys.path:
            sys.path.insert(0, str(_source_root))
    print("project dir:", PROJECT_DIR)
    return (PROJECT_DIR,)


@app.cell
def _(PROJECT_DIR, USE_PRECOMPUTED):
    PRECOMPUTED_DIR = PROJECT_DIR / "precomputed"
    if USE_PRECOMPUTED:
        print(f"USE_PRECOMPUTED=True — replaying saved course results from {PRECOMPUTED_DIR} (no GPU)")
    return (PRECOMPUTED_DIR,)


@app.cell
def _(PROJECT_DIR, mo):
    from hashlib import sha256
    import json
    from uuid import uuid4

    from curriculum_common.audience_packets import (
        AudiencePacket,
        AudienceReading,
        validate_audience_exchange,
    )
    from curriculum_common.portfolio_export import (
        build_private_portfolio,
        serialize_private_portfolio,
    )
    from curriculum_common.production_manifest import (
        ArtifactVersion,
        EditDecisionManifest,
    )
    from curriculum_common.session_records import (
        load_jsonl,
        new_session,
        reduce_command,
        teaching_mode,
        validate_process_log,
    )
    from src.playground_clips import register_artifact_version

    COURSE_RELEASE_ID = "counterpoint-lens-classroom-2026-07-22"
    classroom_mode = teaching_mode(
        "Teaching is the default; no approved Research configuration is loaded"
    )
    _session_pseudonym = f"studio-{uuid4().hex[:12]}"
    _initial_log = new_session(
        _session_pseudonym,
        decision=classroom_mode,
        course_release_id=COURSE_RELEASE_ID,
    )
    get_classroom_log, set_classroom_log = mo.state(_initial_log)
    get_artifact_versions, set_artifact_versions = mo.state(tuple())
    get_audience_exchange, set_audience_exchange = mo.state((None, tuple()))

    PRIVATE_UPLOAD_DIR = PROJECT_DIR / "notebook_results" / "uploads"
    PRIVATE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return (
        ArtifactVersion,
        AudiencePacket,
        AudienceReading,
        COURSE_RELEASE_ID,
        EditDecisionManifest,
        PRIVATE_UPLOAD_DIR,
        build_private_portfolio,
        classroom_mode,
        get_artifact_versions,
        get_audience_exchange,
        get_classroom_log,
        json,
        load_jsonl,
        new_session,
        reduce_command,
        register_artifact_version,
        serialize_private_portfolio,
        set_artifact_versions,
        set_audience_exchange,
        set_classroom_log,
        sha256,
        teaching_mode,
        uuid4,
        validate_audience_exchange,
        validate_process_log,
    )


@app.cell(hide_code=True)
def _(USE_PRECOMPUTED, classroom_mode, mo):
    _route = "Saved course replay" if USE_PRECOMPUTED else "Live model"
    mo.hstack(
        [
            mo.stat(
                value=classroom_mode.mode.value,
                label="Operating mode",
                caption="no research destination or collection action",
                bordered=True,
            ),
            mo.stat(
                value=_route,
                label="Result provenance",
                caption="replay and live results are always labeled",
                bordered=True,
            ),
            mo.stat(
                value="Manual only",
                label="Student-data egress",
                caption="private download or permission-authorized exchange",
                bordered=True,
            ),
        ],
        widths="equal",
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(USE_PRECOMPUTED):
    if USE_PRECOMPUTED:
        DEVICE = "cpu"
        torch = None
        print("Saved course replay → CPU (no GPU allocation or model download)")
    else:
        import torch

        assert torch.cuda.is_available(), (
            "No GPU visible. In molab, attach a GPU via the notebook-specs button in the header. "
            "Choose Saved course replay to continue without a GPU."
        )
        DEVICE = torch.device("cuda:0")
        _free, _total = torch.cuda.mem_get_info(0)
        print(f"torch={torch.__version__}, GPU={torch.cuda.get_device_name(0)}, VRAM={_total / 2**30:.0f} GiB")
    return DEVICE, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.2 Compose — Plan your first cut

    **Required.** Complete the planning card before registering V1. Name the
    audience and artistic intention, the sound–image relationship you want to try,
    relevant cultural or aesthetic references, source and license provenance, and
    accessibility work. Record the common editor/export route and any human or AI
    assistance so your later revision has a trustworthy starting point.

    The saved course clip remains the shared reference for the guided demonstration;
    your planning card belongs to your own project and is not sent anywhere.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    def _validate_planning_card(_value):
        if not _value:
            return "Complete the planning card."
        _required = (
            "creator_intention",
            "intended_audience",
            "sound_image_relation",
            "cultural_context",
            "concept_tags",
            "source_license",
            "accessibility_plan",
            "editor_name_version",
            "source_assets",
            "edit_order",
            "mix_levels",
            "assistance",
        )
        if any(not str(_value[_field]).strip() for _field in _required):
            return "Every required planning field needs a short response."
        return None

    planning_form = mo.md(r"""
    **Artistic intention** {creator_intention}

    **Intended audience** {intended_audience}

    **Planned sound–image relationship** {sound_image_relation}

    **Cultural or aesthetic context** {cultural_context}

    **Concept tags** — comma separated {concept_tags}

    **Sources and licenses** {source_license}

    **Accessibility plan** — captions/transcript and visual-context support {accessibility_plan}

    **Editor and version** {editor_name_version}

    **Source assets and order** {source_assets} {edit_order}

    **Mix/level decisions** {mix_levels}

    **Human/AI assistance disclosure** {assistance}
    """).batch(
        creator_intention=mo.ui.text_area(rows=2, full_width=True),
        intended_audience=mo.ui.text(full_width=True),
        sound_image_relation=mo.ui.text_area(rows=2, full_width=True),
        cultural_context=mo.ui.text_area(rows=2, full_width=True),
        concept_tags=mo.ui.text(full_width=True),
        source_license=mo.ui.text_area(rows=2, full_width=True),
        accessibility_plan=mo.ui.text_area(rows=2, full_width=True),
        editor_name_version=mo.ui.text(value="Common no-cost editor route", full_width=True),
        source_assets=mo.ui.text_area(rows=2, full_width=True),
        edit_order=mo.ui.text_area(rows=2, full_width=True),
        mix_levels=mo.ui.text_area(rows=2, full_width=True),
        assistance=mo.ui.text_area(rows=2, full_width=True),
    ).form(
        submit_button_label="Commit planning card",
        validate=_validate_planning_card,
        bordered=True,
    )
    planning_form
    return (planning_form,)


@app.cell
def _():
    # Own cell on purpose: nothing but a genuine model change should ever
    # invalidate the loader cells below.
    MODEL_PATH = "Qwen/Qwen2.5-Omni-3B"
    MODEL_REVISION = "f75b40e3da2003cdd6e1829b1f420ca70797c34e"
    return MODEL_PATH, MODEL_REVISION


@app.cell
def _(PROJECT_DIR):
    VIDEO_PATH = PROJECT_DIR / "assets" / "02321.mp4"
    SILENT_VIDEO_PATH = PROJECT_DIR / "assets" / "02321_silent.mp4"

    RESULTS_DIR = PROJECT_DIR / "notebook_results"
    RESULTS_DIR.mkdir(exist_ok=True)
    LOGIT_CSV_PATH = RESULTS_DIR / "logit_lens_audio_token_analysis.csv"

    assert VIDEO_PATH.is_file(), f"video not found: {VIDEO_PATH}"
    assert SILENT_VIDEO_PATH.is_file(), f"silent control not found: {SILENT_VIDEO_PATH}"
    print("video:", VIDEO_PATH)
    return LOGIT_CSV_PATH, SILENT_VIDEO_PATH, VIDEO_PATH


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
    ### 1.3 Register — First cut (V1)

    **Required.** Register an immutable first cut from the saved course clip for
    practice or from your own upload. The record keeps a content hash, a private
    filename alias, media facts, your committed planning card, and the common edit
    manifest. Registering a later V2 will link to this V1 instead of overwriting it.

    A browser upload crosses into the hosted Molab session/container for processing.
    It is not automatically transmitted to an instructor, model endpoint, or other
    destination. The private portfolio download near the end is user initiated.
    """)
    return


@app.cell
def _(VIDEO_PATH, mo):
    mo.video(src=VIDEO_PATH.read_bytes(), width=640)
    return


@app.cell(hide_code=True)
def _(mo):
    def _validate_v1_registration(_value):
        if not _value or not _value["registration_key"].strip():
            return "Give this registration a stable key."
        if not _value["change_rationale"].strip():
            return "State what makes this your first committed cut."
        if _value["clip_source"] == "Upload":
            _files = _value["video"]
            if not _files or not _files[0].contents:
                return "Choose an upload or use the saved course clip for practice."
        return None

    v1_registration_form = mo.md(r"""
    **Clip source** {clip_source}

    **Optional upload** — MP4/MOV/MKV/WEBM/AVI, up to 250 MB {video}

    **Stable registration key** {registration_key}

    **First-cut rationale** {change_rationale}
    """).batch(
        clip_source=mo.ui.dropdown(
            ["Saved course clip", "Upload"], value="Saved course clip"
        ),
        video=mo.ui.file(
            filetypes=[".mp4", ".mov", ".mkv", ".webm", ".avi"],
            multiple=False,
            kind="area",
            max_size=250_000_000,
        ),
        registration_key=mo.ui.text(value="first-cut-v1", full_width=True),
        change_rationale=mo.ui.text_area(rows=2, full_width=True),
    ).form(
        submit_button_label="Register immutable V1",
        validate=_validate_v1_registration,
        bordered=True,
    )
    v1_registration_form
    return (v1_registration_form,)


@app.cell(hide_code=True)
def _(
    EditDecisionManifest,
    PRIVATE_UPLOAD_DIR,
    SILENT_VIDEO_PATH,
    VIDEO_PATH,
    get_artifact_versions,
    mo,
    planning_form,
    register_artifact_version,
    set_artifact_versions,
    v1_registration_form,
):
    from datetime import datetime as _datetime
    from datetime import timezone as _timezone

    _registration = v1_registration_form.value
    _plan = planning_form.value
    _current = get_artifact_versions()
    _existing = next(
        (_artifact for _artifact in _current if _artifact.version_label == "V1"),
        None,
    )
    if _existing is not None:
        _v1_card = mo.callout(
            mo.md(
                f"**V1 committed** · `{_existing.artifact_id}`  \n"
                f"The original record remains immutable; continue to Observe."
            ),
            kind="success",
        )
    elif _registration is None or _plan is None:
        _v1_card = mo.callout(
            mo.md("Commit the planning card and registration form to create V1."),
            kind="info",
        )
    else:
        _source_path = VIDEO_PATH
        if _registration["clip_source"] == "Upload":
            from src.playground_clips import resolve_clip_selection as _resolve_v1_clip

            _upload = _registration["video"][0]
            _resolved = _resolve_v1_clip(
                "Upload",
                default_path=VIDEO_PATH,
                silent_path=SILENT_VIDEO_PATH,
                upload_dir=PRIVATE_UPLOAD_DIR,
                upload_name=_upload.name,
                upload_contents=_upload.contents,
            )
            _source_path = _resolved.path
        _tags = tuple(
            _tag.strip() for _tag in _plan["concept_tags"].split(",") if _tag.strip()
        )
        _ordering = tuple(
            _item.strip() for _item in _plan["edit_order"].split(",") if _item.strip()
        ) or ("recorded in planning card",)
        _edit_manifest = EditDecisionManifest(
            editor_name_version=_plan["editor_name_version"].strip(),
            source_assets=({"source_note": _plan["source_assets"].strip()},),
            ordering=_ordering,
            trims=(),
            mix_levels=({"decision": _plan["mix_levels"].strip()},),
            accessibility_work={"plan": _plan["accessibility_plan"].strip()},
            assistance_disclosure={"disclosure": _plan["assistance"].strip()},
            export_preset_version="common-classroom-export/1.0.0",
        )
        try:
            _artifact = register_artifact_version(
                _source_path,
                edit_manifest=_edit_manifest,
                version_label="V1",
                parent_artifact_id=None,
                local_registered_at_utc=_datetime.now(_timezone.utc).isoformat(),
                event_index=len(_current),
                elapsed_ms=0,
                creator_intention=_plan["creator_intention"].strip(),
                intended_audience=_plan["intended_audience"].strip(),
                sound_image_relation=_plan["sound_image_relation"].strip(),
                concept_tags=_tags or ("unclassified",),
                cultural_aesthetic_context=_plan["cultural_context"].strip(),
                source_license_provenance={"notes": _plan["source_license"].strip()},
                change_rationale=_registration["change_rationale"].strip(),
                processing_boundary=(
                    "hosted Molab session/container; no automatic student-data egress"
                ),
            )
        except Exception as _error:  # noqa: BLE001 — registration errors belong in the UI
            _v1_card = mo.callout(
                mo.md(f"**V1 registration needs attention** — `{type(_error).__name__}: {_error}`"),
                kind="danger",
            )
        else:
            set_artifact_versions(_current + (_artifact,))
            _v1_card = mo.callout(
                mo.md(
                    f"**V1 committed** · `{_artifact.artifact_id}`  \n"
                    "The planning snapshot, media facts, and first-cut rationale are now immutable."
                ),
                kind="success",
            )
    _v1_card
    return


@app.cell(hide_code=True)
def _(get_artifact_versions, mo):
    _ready = any(
        _artifact.version_label == "V1" for _artifact in get_artifact_versions()
    )
    _status = "Complete" if _ready else "Waiting for V1"
    mo.callout(
        mo.md(
            f"**Checkpoint · Prepare your project — {_status}**  \n"
            "Next: replay the shared reference and write one interpretation checkpoint after each measure."
        ),
        kind="success" if _ready else "info",
    )
    return


@app.cell(hide_code=True)
def _(USE_PRECOMPUTED, mo):
    _engine = "saved replay ready" if USE_PRECOMPUTED else "live model requested"
    mo.md(f"**Reference engine:** {_engine}.")
    return


@app.cell(hide_code=True)
def _(DEVICE, MODEL_PATH, MODEL_REVISION, PROJECT_DIR, USE_PRECOMPUTED):
    import csv
    from collections import Counter

    import matplotlib.pyplot as plt
    import numpy as np
    _ = PROJECT_DIR  # ensure the clone / sys.path cell ran first
    from src.playground_clips import (
        CLIP_CHOICES,
        inspect_classroom_clip,
        resolve_clip_selection,
    )

    if USE_PRECOMPUTED:
        Qwen2_5OmniForConditionalGeneration = None
        Qwen2_5OmniProcessor = None
        analyze_and_save_audio_logits_to_csv = None
        block_attention = None
        clear_logit_lens_hooks = None
        create_attention_token_mapping = None
        create_token_type_mapping = None
        process_mm_info = None
        register_logit_lens_hooks = None
    else:
        from qwen_omni_utils import process_mm_info
        from transformers import (
            Qwen2_5OmniForConditionalGeneration,
            Qwen2_5OmniProcessor,
        )

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
        if USE_PRECOMPUTED:
            raise RuntimeError("Live model loading is disabled in saved replay mode")
        _model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            revision=MODEL_REVISION,
            torch_dtype="auto",
            attn_implementation=attn_implementation,
        )
        # Free the talker + (float32) token2wav BEFORE moving to GPU so they never
        # occupy VRAM — this experiment only needs the thinker.
        _model.disable_talker()
        _model = _model.to(DEVICE)
        _model.eval()
        _proc = Qwen2_5OmniProcessor.from_pretrained(
            MODEL_PATH, revision=MODEL_REVISION
        )
        return _model, _proc

    # video_path/nframes are arguments, not closures: this cell must depend only
    # on the model constants, or a knob tweak would cascade into the loaders.
    def prepare_video_inputs(model, processor, prompt, token_mapping_fn, video_path, nframes):
        if USE_PRECOMPUTED:
            raise RuntimeError("Live input preparation is disabled in saved replay mode")
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
        CLIP_CHOICES,
        Counter,
        analyze_and_save_audio_logits_to_csv,
        block_attention,
        clear_logit_lens_hooks,
        create_attention_token_mapping,
        create_token_type_mapping,
        csv,
        inspect_classroom_clip,
        load_model_and_processor,
        np,
        plt,
        prepare_video_inputs,
        register_logit_lens_hooks,
        resolve_clip_selection,
    )


@app.cell(hide_code=True)
def _(USE_PRECOMPUTED, load_model_and_processor, mo):
    # One eager model serves both probes. Keeping separate SDPA + eager copies
    # exhausted a 24 GB RTX 3090 during the fixed knockout run; logit-lens hooks
    # work on the eager model too. Knob edits still reuse this one instance.
    if USE_PRECOMPUTED:
        logit_model, logit_processor = None, None
    else:
        with mo.status.spinner(title="Loading Qwen2.5-Omni-3B (eager, first run downloads ~8 GB)…"):
            logit_model, logit_processor = load_model_and_processor("eager")
    return logit_model, logit_processor


@app.cell(hide_code=True)
def _(PRECOMPUTED_DIR, USE_PRECOMPUTED, logit_model, logit_processor):
    # Alias the one eager live model for knockout + playground work. In replay
    # mode, a layer-count stub lets forms render but cannot compute.
    if USE_PRECOMPUTED:
        from src.precompute import StubModel as _StubModel
        from src.precompute import load_precompute as _load_pre

        attention_model = _StubModel(_load_pre(PRECOMPUTED_DIR)["meta"].get("n_layers", 36))
        attention_processor = None
    else:
        attention_model, attention_processor = logit_model, logit_processor
    return attention_model, attention_processor


@app.cell(hide_code=True)
def _(attention_model):
    # Submit-to-submit caches for the two playground forms, keyed on
    # (clip SHA-256, nframes, prompt): "encode" holds prepared inputs +
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
    ## 2. Guided demonstration

    ### 2.1 Observe — Shared reference

    **Required · about 15 minutes.** Replay the fixed reference before changing any
    setting. Read the provenance badge, inspect the three measures below, and write
    one observation and one limitation after each.

    #### Measure A · Raw probe score dispersion

    A multimodal forward pass; the CSV analysis focuses on `audio` token positions.

    **What this can / cannot show.** Intermediate entropy is **raw probe score
    dispersion** from an unnormalized probe; probability margin is a descriptive
    score gap. Audio positions are multimodal feature positions, not positions with
    a calibrated next-token objective. These summaries are not calibrated
    confidence, free-generation uncertainty, or causal localization.
    """)
    return


@app.cell
def _(
    LOGIT_CSV_PATH,
    LOGIT_PROMPT,
    MAX_NEW_TOKENS,
    MODEL_PATH,
    MODEL_REVISION,
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
        _validate_pre(
            _pre["meta"],
            clip=VIDEO_PATH.name,
            nframes=NFRAMES,
            logit_prompt=LOGIT_PROMPT,
            max_new_tokens=MAX_NEW_TOKENS,
            model=MODEL_PATH,
            model_revision=MODEL_REVISION,
        )
        logit_csv_written = _pre["logit_csv"]
        _logit_out = mo.vstack([
            mo.callout(mo.md("**Saved course replay** — no GPU."), kind="neutral"),
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
                if not LOGIT_CSV_PATH.is_file():
                    raise RuntimeError(
                        "The fixed logit-lens run produced no CSV (usually no audio tokens); "
                        "refusing to reuse a result from an earlier run."
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
        _logit_prompt_len = logit_inputs["input_ids"].shape[1]
        _logit_caption = logit_processor.batch_decode(
            _ids[:, _logit_prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        _logit_out = mo.md(f"**Generated caption:**\n\n> {_logit_caption}")
    _logit_out
    return (logit_csv_written,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Result: probe-token diversity by layer

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
            mo.callout(mo.md("**Saved course replay** — no GPU."), kind="neutral"),
            _fig,
        ])
    else:
        _div_out = _fig
    _div_out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Measure B · Direct attention-edge knockout

    `KNOCKOUT_RULES` are `(source_type, target_type, start_layer, end_layer)` tuples.
    The default blocks generated tokens from attending to video tokens in layers 0–35.

    This is a **direct-edge intervention**, not modality ablation: video tokens and
    their residual-stream states remain present, and information can still travel
    through unblocked layers, token types, and earlier indirect paths. A changed
    caption is evidence about this particular edge set under this prompt; an
    unchanged caption does not prove the modality was unused.
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
    MODEL_PATH,
    MODEL_REVISION,
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
        _validate_pre(
            _pre["meta"],
            clip=VIDEO_PATH.name,
            nframes=NFRAMES,
            attention_prompt=ATTENTION_PROMPT,
            knockout_rules=[list(_rule) for _rule in KNOCKOUT_RULES],
            max_new_tokens=MAX_NEW_TOKENS,
            attention_capture_layers=list(ATTENTION_CAPTURE_LAYERS),
            model=MODEL_PATH,
            model_revision=MODEL_REVISION,
        )
        baseline_text = _pre["baseline_text"]
        knockout_text = _pre["knockout_text"]
        baseline_attention_summary = _pre["baseline_attention_summary"]
        knockout_attention_summary = _pre["knockout_attention_summary"]
        attention_token_types = _pre["attention_token_types"]
        attention_inputs = None
        attention_baseline_ids = None
        _ko_rules = _pre["knockout_rules"]
        _ko_banner = mo.callout(mo.md("**Saved course replay** — no GPU."), kind="neutral")
    else:
        from src.precompute import summarize_attention as _summarize_attention

        attention_inputs, attention_token_types = prepare_video_inputs(
            attention_model, attention_processor, ATTENTION_PROMPT, create_attention_token_mapping,
            VIDEO_PATH, NFRAMES,
        )

        with block_attention(
            attention_model, [], attention_token_types, len(attention_token_types),
            track_attention=True, capture_layer_range=ATTENTION_CAPTURE_LAYERS,
        ) as _base_cap:
            with mo.status.spinner(title="Baseline generation + attention capture…"):
                with torch.no_grad():
                    # Thinker-direct generation (see the logit cell): avoids the omni
                    # wrapper's talker requirement. Greedy decoding makes the baseline
                    # caption reused by teacher forcing deterministic.
                    _base_ids = attention_model.thinker.generate(
                        **attention_inputs, max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        return_dict_in_generate=False,
                    )
            _base_captured = {layer: list(v) for layer, v in _base_cap.items()}
        baseline_attention_summary = _summarize_attention(
            _base_captured, attention_token_types, decode_only=True
        )
        del _base_captured
        attention_baseline_ids = _base_ids
        _attention_prompt_len = attention_inputs["input_ids"].shape[1]
        baseline_text = attention_processor.batch_decode(
            _base_ids[:, _attention_prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        with block_attention(
            attention_model, KNOCKOUT_RULES, attention_token_types, len(attention_token_types),
            track_attention=True, capture_layer_range=ATTENTION_CAPTURE_LAYERS,
        ) as _cap:
            with mo.status.spinner(title="Knockout generation…"):
                with torch.no_grad():
                    _ko_ids = attention_model.thinker.generate(
                        **attention_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                        return_dict_in_generate=False,
                    )
            _captured = {layer: list(v) for layer, v in _cap.items()}
        knockout_text = attention_processor.batch_decode(
            _ko_ids[:, _attention_prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        # Reduce to plot-ready matrices now, so the heatmap consumes the same
        # shape live and replayed (raw tensors are never committed).
        knockout_attention_summary = _summarize_attention(
            _captured, attention_token_types, decode_only=True
        )
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
        baseline_attention_summary,
        attention_inputs,
        attention_token_types,
        knockout_attention_summary,
        knockout_text,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Descriptive check: baseline vs knockout attention

    A **descriptive** summary (not causal importance): for each captured layer we
    average heads and **autoregressive decode steps**, then sum each generated-token
    query's attention over token groups. The prompt-prefill snapshot is excluded so
    a `generated → video` mask has the same query semantics throughout. The two
    conditions use the same scale; the delta is `knockout − baseline`. Attention
    redistribution is an expected mechanical consequence of masking and is not by
    itself evidence that a modality supplied semantic information.
    """)
    return


@app.cell
def _(baseline_attention_summary, knockout_attention_summary, mo, np, plt):
    if knockout_attention_summary is None:
        _out = mo.md("> No attention tensors were returned by this build; the text comparison above is the result.")
    elif baseline_attention_summary is None:
        _layers, _mods, _mat = knockout_attention_summary
        _mat = np.asarray(_mat, dtype=float)
        _fig, _ax = plt.subplots(
            figsize=(8, max(3, len(_layers) * 0.6)), constrained_layout=True
        )
        _im = _ax.imshow(_mat, aspect="auto", cmap="magma")
        _ax.set(
            title="Knockout attention (legacy cache has no baseline)",
            xlabel="Key modality", ylabel="Thinker layer",
            xticks=np.arange(len(_mods)), xticklabels=_mods,
            yticks=np.arange(len(_layers)), yticklabels=_layers,
        )
        _fig.colorbar(_im, ax=_ax, label="Attention mass")
        _out = _fig
    else:
        _layers, _mods, _base_mat = baseline_attention_summary
        _ko_layers, _ko_mods, _ko_mat = knockout_attention_summary
        if _layers != _ko_layers or _mods != _ko_mods:
            _out = mo.callout(
                mo.md("**Attention summaries cannot be compared:** layer/modality axes differ."),
                kind="danger",
            )
        else:
            _base_mat = np.asarray(_base_mat, dtype=float)
            _ko_mat = np.asarray(_ko_mat, dtype=float)
            _delta_mat = _ko_mat - _base_mat
            _mass_max = max(1e-9, float(max(_base_mat.max(), _ko_mat.max())))
            _delta_max = max(1e-9, float(np.abs(_delta_mat).max()))
            _fig, _axes = plt.subplots(
                1, 3, figsize=(16, max(3, len(_layers) * 0.6)), constrained_layout=True
            )
            _ims = [
                _axes[0].imshow(_base_mat, aspect="auto", cmap="magma", vmin=0, vmax=_mass_max),
                _axes[1].imshow(_ko_mat, aspect="auto", cmap="magma", vmin=0, vmax=_mass_max),
                _axes[2].imshow(
                    _delta_mat, aspect="auto", cmap="RdBu", vmin=-_delta_max, vmax=_delta_max
                ),
            ]
            for _ax, _title, _mat in zip(
                _axes,
                ("Baseline", "Knockout", "Δ knockout − baseline"),
                (_base_mat, _ko_mat, _delta_mat),
            ):
                _ax.set(
                    title=_title, xlabel="Key modality", ylabel="Thinker layer",
                    xticks=np.arange(len(_mods)), xticklabels=_mods,
                    yticks=np.arange(len(_layers)), yticklabels=_layers,
                )
                for _ri in range(_mat.shape[0]):
                    for _ci in range(_mat.shape[1]):
                        _ax.text(
                            _ci, _ri, f"{_mat[_ri, _ci]:+.2f}" if _title.startswith("Δ") else f"{_mat[_ri, _ci]:.2f}",
                            ha="center", va="center", color="white", fontsize=8,
                        )
            _fig.colorbar(_ims[1], ax=_axes[:2], label="Attention mass")
            _fig.colorbar(_ims[2], ax=_axes[2], label="Δ attention mass")
            _out = _fig
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Measure C · Teacher-forced answer-distribution change

    The string diff above is **visceral but binary** — you can't see a *small*
    effect, and it depends on how generation happens to continue. This cell asks
    the same question as a **measurement**: it feeds the baseline caption back in
    tagged `answer` and scores, per token, how its assigned probability changes
    when the answer's direct attention edges to the same target token type as
    `KNOCKOUT_RULES` (same clip, same prompt, same layers — only the source becomes
    `answer`, because the caption is now *input*, not generation).

    **Δ = knockout − baseline** per caption token; a negative value means the fixed
    answer token received lower log probability. Entropy and margin here describe
    **teacher-forced answer-distribution dispersion (uncalibrated proxy)**. Use the
    mean for cross-clip comparisons, while remembering that different clips may
    generate semantically different captions. This is not calibrated confidence,
    free-generation uncertainty, or proof of a localized causal mechanism.
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
                "**Teacher forcing needs the live model.** This measurement is not "
                "included in the saved pack. Use the execution-route form above "
                "only when your instructor has prepared a GPU runtime."
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
            _fixed_mean = fixed_tf_result["delta_mean"]
            _fixed_rule_txt = " + ".join(f"`answer→{_r[1]}` [{_r[2]},{_r[3]})" for _r in _fixed_rules)
            _fixed_out = mo.vstack([
                mo.md(f"**Knockout** {_fixed_rule_txt} &nbsp;·&nbsp; baseline caption teacher-forced as `answer`"),
                mo.hstack([
                    mo.stat(
                        value=f"{_fixed_total:+.2f}",
                        label="Σ Δ log-lik (nats)",
                        caption="knockout − baseline · negative = lower log probability",
                        direction="decrease" if _fixed_total < 0 else "increase",
                        bordered=True,
                    ),
                    mo.stat(
                        value=f"{_fixed_mean:+.3f}",
                        label="Mean Δ / token (nats)",
                        caption="length-normalized comparison",
                        direction="decrease" if _fixed_mean < 0 else "increase",
                        bordered=True,
                    ),
                    mo.stat(
                        value=str(len(_fixed_delta)),
                        label="Caption tokens scored",
                        caption="greedy baseline, teacher-forced",
                        bordered=True,
                    ),
                ], widths="equal", gap=1),
                mo.md("#### Per-token Δ log-likelihood"),
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
    def _validate_observe_reflection(_value):
        if not _value:
            return "Record the guided interpretation checkpoints."
        if any(not str(_value[_field]).strip() for _field in _value):
            return "Write one observation and one limitation for each measure."
        return None

    observe_reflection_form = mo.md(r"""
    **Measure A observation** {probe_observation}

    **Measure A limitation** {probe_limit}

    **Measure B observation** {edge_observation}

    **Measure B limitation** {edge_limit}

    **Measure C observation or “not measured”** {answer_observation}

    **Measure C limitation** {answer_limit}
    """).batch(
        probe_observation=mo.ui.text_area(rows=2, full_width=True),
        probe_limit=mo.ui.text_area(rows=2, full_width=True),
        edge_observation=mo.ui.text_area(rows=2, full_width=True),
        edge_limit=mo.ui.text_area(rows=2, full_width=True),
        answer_observation=mo.ui.text_area(rows=2, full_width=True),
        answer_limit=mo.ui.text_area(rows=2, full_width=True),
    ).form(
        submit_button_label="Commit guided checkpoints",
        validate=_validate_observe_reflection,
        bordered=True,
    )
    observe_reflection_form
    return (observe_reflection_form,)


@app.cell(hide_code=True)
def _(mo, observe_reflection_form):
    _complete = observe_reflection_form.value is not None
    _status = "Complete" if _complete else "Waiting for three checkpoints"
    mo.callout(
        mo.md(
            f"**Checkpoint · Guided demonstration — {_status}**  \n"
            "Next: choose one controlled comparison, commit a prediction, then run only after the snapshot is visible."
        ),
        kind="success" if _complete else "info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Exploratory playground

    ### 3.1 Experiment — Change one thing at a time

    **Required · one paired investigation.** The guided demonstration used one
    shared reference. Now choose one coded operation, keep the other settings fixed,
    and work in a short Prepare → Run → Reflect cycle:

    1. **Prediction before ▶** — state a directional result that could be wrong.
    2. **Intervention** — change one variable and keep the rest fixed.
    3. **Observation** — report the metric before telling a mechanism story.
    4. **Verdict** — supported, refuted, or not tested?
    5. **Next control** — name a rival explanation and a result that separates it.

    Your command key, prediction, initial explanation, and settings are committed
    through an append-only reducer before any live computation. Replaying the same
    command after a reactive rerun or event-log reload is idempotent; reusing its key
    with different content is rejected.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    _operation_labels = {
        "Original reference": "original_reference",
        "Duration-matched audio swap": "audio_swap_duration_matched",
        "Signed temporal offset with zero-fill": "temporal_offset",
        "Audio silence signal control": "audio_silence_control",
        "Neutral video signal control": "video_neutral_control",
        "True audio modality omission": "audio_omitted_model_input",
        "True video modality omission": "video_omitted_model_input",
        "Direct attention-edge knockout": "direct_attention_edge_knockout",
    }

    def _validate_experiment_plan(_value):
        if not _value:
            return "Prepare a run before continuing."
        for _field in ("command_key", "prediction", "initial_explanation", "prompt"):
            if not str(_value[_field]).strip():
                return "Command key, prediction, initial explanation, and prompt are required."
        if _value["operation"] == "temporal_offset" and _value["offset_ms"] == 0:
            return "Choose a non-zero signed offset for the temporal-offset condition."
        return None

    experiment_plan_form = mo.md(r"""
    **Stable command key** — reuse only to replay the identical snapshot {command_key}

    **Prediction before run** {prediction}

    **Initial explanation** {initial_explanation}

    **Operation** {operation}

    **Signed offset in milliseconds** — negative leads; positive delays {offset_ms}

    **Prompt** {prompt}
    """).batch(
        command_key=mo.ui.text(value="paired-run-1", full_width=True),
        prediction=mo.ui.text_area(rows=2, full_width=True),
        initial_explanation=mo.ui.text_area(rows=2, full_width=True),
        operation=mo.ui.dropdown(_operation_labels, value="Original reference"),
        offset_ms=mo.ui.slider(-2000, 2000, step=100, value=500, show_value=True),
        prompt=mo.ui.text(value="Describe what you see and hear in the video", full_width=True),
    ).form(
        submit_button_label="Commit immutable run snapshot",
        validate=_validate_experiment_plan,
        bordered=True,
    )
    experiment_plan_form
    return (experiment_plan_form,)


@app.cell(hide_code=True)
def _(
    MODEL_PATH,
    MODEL_REVISION,
    experiment_plan_form,
    get_artifact_versions,
    get_classroom_log,
    mo,
    reduce_command,
    set_classroom_log,
    sha256,
):
    _prepared = experiment_plan_form.value
    if _prepared is None:
        _run_snapshot_card = mo.callout(
            mo.md("Commit a prediction and initial explanation before running."),
            kind="info",
        )
    else:
        _artifacts = get_artifact_versions()
        _v1 = next(
            (_artifact for _artifact in _artifacts if _artifact.version_label == "V1"),
            None,
        )
        _artifact_id = _v1.artifact_id if _v1 is not None else "practice-course-reference"
        _condition = _prepared["operation"]
        _command = {
            "kind": "commit_run",
            "command_nonce": "run:" + sha256(
                _prepared["command_key"].strip().encode("utf-8")
            ).hexdigest(),
            "artifact_id": _artifact_id,
            "stimulus_id": f"condition:{_condition}",
            "condition_code": _condition,
            "model_id": MODEL_PATH,
            "model_revision": MODEL_REVISION,
            "prompt": _prepared["prompt"].strip(),
            "parameters": {
                "technical_operation": _condition,
                "signed_offset_ms": int(_prepared["offset_ms"]),
                "signal_control": _condition in {
                    "audio_silence_control",
                    "video_neutral_control",
                },
                "true_modality_omission": _condition in {
                    "audio_omitted_model_input",
                    "video_omitted_model_input",
                },
                "model_intervention": _condition == "direct_attention_edge_knockout",
            },
            "prediction": _prepared["prediction"].strip(),
            "initial_explanation": _prepared["initial_explanation"].strip(),
            "metric_versions": {
                "probe": "probe-metric/1.0.0",
                "teacher_forced": "compact-distribution/1.0.0",
            },
        }
        try:
            _reduction = reduce_command(get_classroom_log(), _command)
        except Exception as _error:  # noqa: BLE001 — reducer rejection belongs in the UI
            _run_snapshot_card = mo.callout(
                mo.md(f"**Run snapshot rejected** — `{type(_error).__name__}: {_error}`"),
                kind="danger",
            )
        else:
            set_classroom_log(_reduction.log)
            _run_id = _reduction.record_ids[0]
            _run_snapshot_card = mo.callout(
                mo.md(
                    f"**Run snapshot committed** · `{_run_id}`  \n"
                    f"Operation: `{_condition}` · replayed: `{_reduction.replayed}`"
                ),
                kind="success",
            )
    _run_snapshot_card
    return


@app.cell(hide_code=True)
def _(USE_PRECOMPUTED, knockout_text, logit_csv_written, mo):
    _ = knockout_text  # depend on the knockout run
    _ok = logit_csv_written.is_file() and logit_csv_written.stat().st_size > 0
    _teacher_forcing_status = (
        "- Teacher forcing: **skipped in GPU-free replay** (the guided logit-lens, "
        "caption knockout, and attention panels are saved).\n\n"
        if USE_PRECOMPUTED
        else "- Baseline vs knockout compared, and the caption scored under teacher forcing, above.\n\n"
    )
    mo.md(
        f"**Guided demo complete — now test your own hypotheses.**\n\n"
        f"- Logit-lens result available: **{_ok}** — `{logit_csv_written}`\n"
        f"{_teacher_forcing_status}"
        "Two Choice measurements follow:\n\n"
        "- **Diversity scoreboard** — how do the *audio positions* respond to your "
        "prompt, clip, and knockout choices?\n"
        "- **Teacher forcing** — how does caption log-likelihood change when you "
        "block a direct edge set—and on **your** clip, which tokens move most?\n\n"
        "Form the hypothesis first, then press ▶. Saved replay keeps personalized "
        "live controls unavailable rather than showing stale live output as replay."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Advanced — layer bands, attention heads, and processor-path checks": mo.md(r"""
            These optional controls are not required for any checkpoint. Use them
            only after one complete paired cycle.

            - Customize layer bands or direct attention-edge rules.
            - Inspect feature/token interventions separately from media operations.
            - A **stimulus-signal control** supplies silence or a neutral visual.
            - **True modality omission** supplies no audio or no video through a
              distinct processor path; silence or a black frame is never a substitute.
            - Position-wise trajectories may be aligned only when token-layout
              fingerprints match. Otherwise compare preregistered aggregates or
              normalized bins and display the alignment warning.
            """)
        },
        multiple=False,
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Choice measurement · Audio-position probe diversity

    Turn the **logit-lens diversity** measurement into a live experiment: pick a clip, the
    number of frames, the prompt, and (optionally) an attention knockout to apply
    **during** the forward pass, then submit to score every thinker layer by how
    many *distinct* tokens it decodes across the audio-token positions.

    Nothing runs until you press submit (the controls are wrapped in a form), and
    the eager model from the knockout experiment is reused — so runs are quick and
    need no extra VRAM.

    The score is measured at **audio** token positions, so knockouts with an `audio`
    source reshape it most directly (a `generated` source does nothing in a forward
    pass). Build one rule with the dropdowns, or enter several in the advanced field.

    #### Investigation routes

    - **Steer the prompt, touch nothing else.** Run the same clip with *"describe
      what you **hear**"* and then *"describe what you **see**"* — no knockout.
      Hypothesis first: should what the prompt asks for change what the *audio
      positions* decode, before any pathway is cut? State what result would refute
      your prompt-steering explanation.
    - **Test a candidate cross-modal edge band.** Knock out `audio → video` over
      `[0, 12)`, then `[12, 24)`, then `[24, 36)`. Which band moves the Δ trend
      most? Now try to falsify the "fusion" interpretation: can prompt sensitivity,
      attention renormalization, or a different edge rule reproduce the change?
    - **Starve the stream.** Stack `audio,video,0,36 ; audio,query_text,0,36` in the
      advanced field. Is the effect of cutting both neighbors the sum of cutting
      each alone — and what would it mean if it isn't?

    A flat Δ is a result too — record it. And be careful *reading* a big one:
    diversity is only the count of distinct argmax probe tokens. A shift does not
    prove fusion, grounding, or even improved/worsened representations. For every
    effect, write one rival explanation and one control that could make your claim
    fail. The experiment catalog in `avllm_interpretability/README.md` offers
    further contrasts after you complete one controlled cycle here.
    """)
    return


@app.cell(hide_code=True)
def _(
    CLIP_CHOICES,
    KNOCKOUT_RULES,
    LOGIT_PROMPT,
    NFRAMES,
    USE_PRECOMPUTED,
    attention_model,
    mo,
):
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

    def _validate(_value):
        if not _value or not _value["hypothesis"].strip():
            return "Write a falsifiable prediction before running."
        if not _value["prompt"].strip():
            return "Enter a non-empty prompt."
        return None

    _template = (
        "**Prediction before ▶** — state a direction or contrast that could be wrong:\n\n"
        "{hypothesis}\n\n"
        "**Clip** {clip_choice}\n\n"
        "Choose **Default**, the matched **Silent control**, or **Upload**. "
        "Uploads must be ≤250 MB, ≤120 s, ≤1080p/60 FPS, use ≤1.5 GB estimated "
        "decoded-frame memory, be decodable, and contain both video and audio "
        "(`mp4 / mov / mkv / webm / avi`):\n\n"
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
        hypothesis=mo.ui.text_area(
            placeholder=(
                "e.g. Blocking audio→video in middle layers will change diversity "
                "more than the silent-control contrast; a flat or reversed effect refutes this."
            ),
            rows=2,
            full_width=True,
        ),
        clip_choice=mo.ui.dropdown(CLIP_CHOICES, value="Default"),
        video=mo.ui.file(
            filetypes=[".mp4", ".mov", ".mkv", ".webm", ".avi"],
            multiple=False,
            kind="area",
            max_size=250_000_000,
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
        submit_button_disabled=USE_PRECOMPUTED,
        bordered=True,
        validate=_validate,
    )
    scoreboard_controls
    return (scoreboard_controls,)


@app.cell(hide_code=True)
def _(
    Counter,
    LOGIT_CSV_PATH,
    LOGIT_PROMPT,
    SILENT_VIDEO_PATH,
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
    resolve_clip_selection,
    inspect_classroom_clip,
    torch,
):
    from contextlib import nullcontext as _nullcontext

    _p = scoreboard_controls.value
    mo.stop(
        _p is None,
        mo.callout(
            mo.md("Set the parameters above and press **▶ Run logit-lens diversity**."),
            kind="info",
        ),
    )

    from qwen_omni_utils import process_mm_info as _process_mm_info

    _results_dir = LOGIT_CSV_PATH.parent

    # Resolve an explicit clip choice. Uploads are persisted under a digest-derived
    # filename; their original names never become filesystem paths or cache keys.
    _uploads = _p["video"]
    _upload = _uploads[0] if _uploads and _uploads[0].contents else None
    _clip_error = None
    try:
        _resolved_clip = resolve_clip_selection(
            _p["clip_choice"],
            default_path=VIDEO_PATH,
            silent_path=SILENT_VIDEO_PATH,
            upload_dir=_results_dir / "uploads",
            upload_name=_upload.name if _upload is not None else None,
            upload_contents=_upload.contents if _upload is not None else None,
        )
        _clip_inspection = inspect_classroom_clip(_resolved_clip.path)
    except Exception as _e:  # noqa: BLE001 — media parser errors belong in the UI
        _resolved_clip, _clip_inspection, _clip_error = None, None, str(_e)
    mo.stop(
        _clip_error is not None,
        mo.callout(mo.md(f"**Clip selection failed** — {_clip_error}"), kind="danger"),
    )
    _video_path = _resolved_clip.path
    _clip_cache_id = _resolved_clip.cache_id
    _clip_duration_text = (
        f"{_clip_inspection.duration_seconds:.1f}s"
        if _clip_inspection.duration_seconds is not None
        else "duration unknown"
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

    def _prep(video_path, clip_cache_id, nframes, prompt):
        # Encoding (video decode + feature extraction) dominates a submit when
        # only the rule/layer band changed — cache it across ▶ presses.
        _key = (clip_cache_id, nframes, prompt)
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
            _inp, _types = _prep(
                _video_path, _clip_cache_id, _nframes, _prompt
            )  # encode the clip once
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
                f"**Prediction recorded before run:** {_p['hypothesis']}  \n"
                f"**Clip** {_resolved_clip.choice}: `{_video_path.name}` "
                f"({_clip_duration_text}) "
                f"&nbsp;·&nbsp; **Frames** {_nframes} "
                f"&nbsp;·&nbsp; **Prompt** _{_prompt}_ &nbsp;·&nbsp; **Knockout** {_rule_txt}"
            ),
            mo.hstack(_stats, widths="equal", gap=1),
            _fig,
            mo.md("#### Layers ranked by decoded-prediction diversity"),
            _table,
        ]
        _scoreboard = mo.vstack(_children)
    _scoreboard
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.2 Compare — Three readings of one work

    **Required.** Import one blinded presentation packet and at least two distinct
    audience-reading JSON files created in the separate audience surface. Creator
    intention and the model reading stay hidden here until the import validates.
    After reveal, compare shared and different tags with neutral language:
    agreement, divergence, or an unrepresented reading—not truth versus error.

    Two readings are the minimum classroom activity, not a research sample-size
    claim. Consider cultural convention, accessibility barriers, prompt wording,
    training-data unknowns, and the limits of the available label vocabulary.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    def _validate_audience_import(_value):
        if not _value:
            return "Choose a packet and two reading files."
        _packet_files = _value["packet"]
        _reading_files = _value["readings"]
        if not _packet_files or not _packet_files[0].contents:
            return "Choose one blinded presentation packet."
        if not _reading_files or len(_reading_files) < 2:
            return "Choose at least two distinct audience-reading files."
        return None

    audience_import_form = mo.md(r"""
    **Blinded presentation packet** {packet}

    **Audience readings** — choose at least two JSON files {readings}
    """).batch(
        packet=mo.ui.file(filetypes=[".json"], multiple=False, kind="area"),
        readings=mo.ui.file(filetypes=[".json"], multiple=True, kind="area"),
    ).form(
        submit_button_label="Validate and reveal three readings",
        validate=_validate_audience_import,
        bordered=True,
    )
    audience_import_form
    return (audience_import_form,)


@app.cell(hide_code=True)
def _(
    AudiencePacket,
    AudienceReading,
    audience_import_form,
    json,
    mo,
    set_audience_exchange,
    validate_audience_exchange,
):
    _audience_value = audience_import_form.value
    if _audience_value is None:
        _audience_import_card = mo.callout(
            mo.md("Creator and model readings remain hidden until two blinded readings validate."),
            kind="info",
        )
    else:
        try:
            _packet_payload = json.loads(
                _audience_value["packet"][0].contents.decode("utf-8")
            )
            _packet = AudiencePacket.from_mapping(_packet_payload)
            _readings = tuple(
                AudienceReading.from_mapping(
                    json.loads(_file.contents.decode("utf-8")), packet=_packet
                )
                for _file in _audience_value["readings"]
            )
            _report = validate_audience_exchange(_packet, _readings)
            if not _report.is_complete:
                _messages = "; ".join(_issue.message for _issue in _report.issues)
                raise ValueError(_messages or "audience exchange is incomplete")
        except Exception as _error:  # noqa: BLE001 — unsafe imports belong in the UI
            _audience_import_card = mo.callout(
                mo.md(f"**Audience import rejected** — `{type(_error).__name__}: {_error}`"),
                kind="danger",
            )
        else:
            _revealed_packet = _packet.reveal(
                protocol_deviation="creator/model readings revealed after valid classroom import"
            )
            set_audience_exchange((_revealed_packet, _readings))
            _audience_import_card = mo.callout(
                mo.md(
                    f"**Audience import complete** · {len(_readings)} distinct blinded readings  \n"
                    "Creator and model readings are now available for comparison."
                ),
                kind="success",
            )
    _audience_import_card
    return


@app.cell(hide_code=True)
def _(get_artifact_versions, get_audience_exchange, knockout_text, mo):
    _packet, _readings = get_audience_exchange()
    if _packet is None or len(_readings) < 2:
        _triadic_view = mo.callout(
            mo.md("Three-reading comparison is locked until the blinded import is complete."),
            kind="neutral",
        )
    else:
        _artifacts = get_artifact_versions()
        _v1 = next(
            (_artifact for _artifact in _artifacts if _artifact.version_label == "V1"),
            None,
        )
        _creator_reading = (
            _v1.creator_intention if _v1 is not None else "Practice creator statement not registered"
        )
        _rows = [
            {"reading": "Creator", "interpretation": _creator_reading},
            *[
                {
                    "reading": f"Audience {index + 1}",
                    "interpretation": _reading.open_interpretation,
                }
                for index, _reading in enumerate(_readings)
            ],
            {"reading": "Model", "interpretation": knockout_text},
        ]
        _triadic_view = mo.vstack(
            [
                mo.md(
                    "**Revealed comparison.** Describe agreement, divergence, and "
                    "unrepresented readings without treating any reader as automatic ground truth."
                ),
                mo.ui.table(_rows, selection=None),
            ]
        )
    _triadic_view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Choice measurement · Teacher-forced edge sensitivity

    The diversity scoreboard above runs one forward pass over the **prompt**, so —
    exactly like `generated` — an **`answer`** source is inert there (there are no
    answer tokens to block). This section closes that gap. It generates the caption
    once, feeds it back in tagged **`answer`**, and measures how its token
    log-likelihood changes when answer queries are forbidden from directly
    attending to one target token type in a chosen layer band.

    The metric is **Δ log-likelihood, `knockout − baseline`** — *negative* means the
    model assigned its own caption less probability after these direct edges were
    blocked. This is consistent with reliance on the edge set, but attention
    renormalization and indirect pathways remain rival explanations. Unlike the
    free-generation string diff above it is
    **continuous** (you can see a *small* effect) and **deterministic** (greedy
    caption, forward-only scoring). Compare clips using **mean Δ/token**, not only
    total Δ; even then, their captions may differ semantically. Nothing runs until
    you press ▶.

    #### Investigation routes

    - **Sight vs. sound, in nats.** Same clip, same prompt: run `answer → audio`,
      then `answer → video`. Which Δ is more negative — and does that agree with
      which knockout changed the *free-generated* caption more in the knockout cell above?
      If the binary diff and the continuous measurement disagree, which do you
      believe, and why?
    - **Which layer band is this score sensitive to?** Keep `answer → audio` and
      narrow the layer band: `[0, 12)`, `[12, 24)`, `[24, 36)`. Which band costs the
      caption the most log-likelihood? Compare with 🎛️, then name at least one
      explanation that does **not** require a localized "fusion module."
    - **Bring your own clip — make the modalities disagree.** The most interesting
      trends come from clips where sound and sight tell different stories (narration
      over unrelated footage, a music video with off-screen audio, dubbed speech).
      Upload one, caption it, and knock out `answer → audio` vs `answer → video`.
      Which edge set changes mean Δ/token more, and can a silent or mismatched control
      falsify the tempting modality-reliance story? Hover the colored strip to find
      which words move.

    > **The control that keeps you honest.** Choose **Silent control** (the same
    > frames, but the audio track is digital silence) and run `answer → audio`:
    > audio tokens still exist, so **predict** whether Δ should approach zero. Silence
    > is not guaranteed to be a perfect null: preprocessing, positional effects, and
    > mask renormalization remain. Compare against the default clip with all other
    > settings fixed. Pair *every* interesting effect with a control and state what
    > observation would falsify your interpretation.
    """)
    return


@app.cell(hide_code=True)
def _(
    ATTENTION_PROMPT,
    CLIP_CHOICES,
    KNOCKOUT_RULES,
    NFRAMES,
    USE_PRECOMPUTED,
    attention_model,
    mo,
):
    _n_layers = len(attention_model.thinker.model.layers)
    _tf_targets = ["audio", "video", "query_text", "image"]
    _fixed_rule = KNOCKOUT_RULES[0] if KNOCKOUT_RULES else ("generated", "video", 0, _n_layers)
    _tf_default_target = _fixed_rule[1] if _fixed_rule[1] in _tf_targets else "video"
    _tf_default_layers = [
        max(0, int(_fixed_rule[2])), min(_n_layers, int(_fixed_rule[3]))
    ]

    def _validate(_value):
        if not _value or not _value["hypothesis"].strip():
            return "Write a falsifiable prediction before running."
        if not _value["prompt"].strip():
            return "Enter a non-empty prompt."
        return None

    _tf_template = (
        "*Defaults mirror the guided teacher-forcing demo above.*\n\n"
        "**Prediction before ▶** — state the expected sign or token-level contrast:\n\n"
        "{hypothesis}\n\n"
        "**Clip** {clip_choice}\n\n"
        "Choose **Default**, the matched **Silent control**, or **Upload**. "
        "Uploads must be ≤250 MB, ≤120 s, ≤1080p/60 FPS, use ≤1.5 GB estimated "
        "decoded-frame memory, be decodable, and contain both video and audio "
        "(`mp4 / mov / mkv / webm / avi`):\n\n"
        "{video}\n\n"
        "**Frames sampled from the clip** {nframes}\n\n"
        "**Prompt** {prompt}\n\n"
        "---\n\n"
        "Forbid the **answer** from attending to {target} across thinker layers {layers}\n\n"
        f"(`answer` is the model's own caption, teacher-forced back in; this thinker has "
        f"**{_n_layers}** layers, `end` exclusive.)"
    )
    teacher_forcing_controls = mo.md(_tf_template).batch(
        hypothesis=mo.ui.text_area(
            placeholder=(
                "e.g. answer→audio will reduce mean log-likelihood more on the "
                "sound clip than on silence; the opposite result refutes this."
            ),
            rows=2,
            full_width=True,
        ),
        clip_choice=mo.ui.dropdown(CLIP_CHOICES, value="Default"),
        video=mo.ui.file(
            filetypes=[".mp4", ".mov", ".mkv", ".webm", ".avi"],
            multiple=False,
            kind="area",
            max_size=250_000_000,
        ),
        nframes=mo.ui.slider(2, 32, step=2, value=NFRAMES, show_value=True, include_input=True),
        prompt=mo.ui.text(value=ATTENTION_PROMPT, full_width=True),
        target=mo.ui.dropdown(_tf_targets, value=_tf_default_target),
        layers=mo.ui.range_slider(
            0, _n_layers, step=1, value=_tf_default_layers, show_value=True
        ),
    ).form(
        submit_button_label="▶ Run teacher-forced Δ log-lik",
        submit_button_disabled=USE_PRECOMPUTED,
        bordered=True,
        validate=_validate,
    )
    teacher_forcing_controls
    return (teacher_forcing_controls,)


@app.cell(hide_code=True)
def _(
    ATTENTION_PROMPT,
    LOGIT_CSV_PATH,
    SILENT_VIDEO_PATH,
    VIDEO_PATH,
    attention_model,
    attention_processor,
    cache_put,
    create_attention_token_mapping,
    mo,
    np,
    playground_caches,
    inspect_classroom_clip,
    resolve_clip_selection,
    teacher_forcing_controls,
):
    _tp = teacher_forcing_controls.value
    mo.stop(
        _tp is None,
        mo.callout(
            mo.md("Set the parameters above and press **▶ Run teacher-forced Δ log-lik**."),
            kind="info",
        ),
    )

    from qwen_omni_utils import process_mm_info as _tf_mm_info

    from src.teacher_forcing import render_delta_strip as _render_strip
    from src.teacher_forcing import teacher_forced_delta as _tfd

    # Resolve the explicit choice with the same safe content-addressed helper
    # used by 🎛️, so the two playgrounds also share cache identities.
    _tf_uploads = _tp["video"]
    _tf_upload = _tf_uploads[0] if _tf_uploads and _tf_uploads[0].contents else None
    _tf_clip_error = None
    try:
        _tf_resolved_clip = resolve_clip_selection(
            _tp["clip_choice"],
            default_path=VIDEO_PATH,
            silent_path=SILENT_VIDEO_PATH,
            upload_dir=LOGIT_CSV_PATH.parent / "uploads",
            upload_name=_tf_upload.name if _tf_upload is not None else None,
            upload_contents=_tf_upload.contents if _tf_upload is not None else None,
        )
        _tf_clip_inspection = inspect_classroom_clip(_tf_resolved_clip.path)
    except Exception as _e:  # noqa: BLE001 — media parser errors belong in the UI
        _tf_resolved_clip, _tf_clip_inspection, _tf_clip_error = None, None, str(_e)
    mo.stop(
        _tf_clip_error is not None,
        mo.callout(
            mo.md(f"**Clip selection failed** — {_tf_clip_error}"), kind="danger"
        ),
    )
    _tf_video = _tf_resolved_clip.path
    _tf_clip_cache_id = _tf_resolved_clip.cache_id
    _tf_duration_text = (
        f"{_tf_clip_inspection.duration_seconds:.1f}s"
        if _tf_clip_inspection.duration_seconds is not None
        else "duration unknown"
    )
    _tf_nframes = int(_tp["nframes"])
    _tf_prompt = _tp["prompt"].strip() or ATTENTION_PROMPT
    _tf_lo, _tf_hi = int(_tp["layers"][0]), int(_tp["layers"][1])
    _tf_rules = [("answer", _tp["target"], _tf_lo, _tf_hi)]

    def _tf_prep(video_path, clip_cache_id, nframes, prompt):
        # Shared encode cache with the 🎛️ section: a layer-band or target sweep
        # on the same clip/prompt re-encodes nothing after the first ▶.
        _key = (clip_cache_id, nframes, prompt)
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
        # Caption cache, keyed by clip content, frame count, and prompt. The greedy
        # caption depends only on the encoded inputs, so rule/layer sweeps reuse it.
        _tf_cap_key = (_tf_clip_cache_id, _tf_nframes, _tf_prompt)
        _tf_cached_c = playground_caches["caption"].get(_tf_cap_key)
        with mo.status.spinner(
            title=f"Teacher forcing · {_tf_nframes} frames · {_tf_video.name}"
            + (" · caption cached…" if _tf_cached_c is not None else "…")
        ):
            _tf_inp, _tf_types = _tf_prep(
                _tf_video, _tf_clip_cache_id, _tf_nframes, _tf_prompt
            )
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
        _tf_mean = _tf_res["delta_mean"]
        _tf_toks = _tf_res["caption_tokens"]
        _tf_worst = int(np.argmin(_tf_delta)) if _tf_delta else 0
        _tf_rule_txt = f"`answer→{_tp['target']}` [{_tf_lo},{_tf_hi})"
        _tf_stats = [
            mo.stat(
                value=f"{_tf_total:+.2f}",
                label="Σ Δ log-lik (nats)",
                caption="knockout − baseline · negative = lower log probability",
                direction="decrease" if _tf_total < 0 else "increase",
                bordered=True,
            ),
            mo.stat(
                value=f"{_tf_mean:+.3f}",
                label="Mean Δ / token (nats)",
                caption="use for cross-clip comparisons",
                direction="decrease" if _tf_mean < 0 else "increase",
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
                f"**Prediction recorded before run:** {_tp['hypothesis']}  \n"
                f"**Clip** {_tf_resolved_clip.choice}: `{_tf_video.name}` "
                f"({_tf_duration_text}) "
                f"&nbsp;·&nbsp; **Frames** {_tf_nframes} "
                f"&nbsp;·&nbsp; **Prompt** _{_tf_prompt}_ &nbsp;·&nbsp; **Knockout** {_tf_rule_txt}"
            ),
            mo.hstack(_tf_stats, widths="equal", gap=1),
            mo.md("#### Per-token Δ log-likelihood"),
            mo.Html(f"<div style='line-height:2.1;font-family:monospace;font-size:15px'>{_render_strip(_tf_toks, _tf_delta)}</div>"),
            mo.ui.table(_tf_rows, selection=None, pagination=True, page_size=16),
        ])
    _tf_out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.3 Revise — Explanation and second cut (V2)

    **Required.** Choose the evidence trigger that changed—or strengthened—your
    account. Commit the observed result, limitation, rival explanation, revised
    explanation, and next control to the selected run. Then register a linked V2
    and name the creative decisions made between V1 and V2. Neither action mutates
    the first-cut record or the initial explanation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    def _validate_revision(_value):
        if not _value:
            return "Complete the revision record."
        _required = (
            "reflection_key",
            "observed_evidence",
            "verdict",
            "evidence_trigger",
            "limitation",
            "rival_explanation",
            "revised_explanation",
            "next_control",
            "creative_decisions",
            "v2_rationale",
        )
        if any(not str(_value[_field]).strip() for _field in _required):
            return "Every revision field needs a short response."
        if _value["v2_source"] == "Upload":
            _files = _value["video"]
            if not _files or not _files[0].contents:
                return "Choose a V2 upload or use the saved course clip for practice."
        return None

    revision_form = mo.md(r"""
    **Stable reflection key** {reflection_key}

    **Observed evidence** {observed_evidence}

    **Verdict** — supported, refuted, or not tested {verdict}

    **Evidence trigger** {evidence_trigger}

    **Limitation** {limitation}

    **Rival explanation** {rival_explanation}

    **Revised explanation** {revised_explanation}

    **Next control** {next_control}

    **V1 → V2 creative decisions** {creative_decisions}

    **V2 source** {v2_source}

    **Optional V2 upload** {video}

    **V2 change rationale** {v2_rationale}
    """).batch(
        reflection_key=mo.ui.text(value="paired-run-1-reflection", full_width=True),
        observed_evidence=mo.ui.text_area(rows=2, full_width=True),
        verdict=mo.ui.dropdown(["supported", "refuted", "not tested"], value="not tested"),
        evidence_trigger=mo.ui.text_area(rows=2, full_width=True),
        limitation=mo.ui.text_area(rows=2, full_width=True),
        rival_explanation=mo.ui.text_area(rows=2, full_width=True),
        revised_explanation=mo.ui.text_area(rows=2, full_width=True),
        next_control=mo.ui.text_area(rows=2, full_width=True),
        creative_decisions=mo.ui.text_area(rows=2, full_width=True),
        v2_source=mo.ui.dropdown(["Saved course clip", "Upload"], value="Saved course clip"),
        video=mo.ui.file(
            filetypes=[".mp4", ".mov", ".mkv", ".webm", ".avi"],
            multiple=False,
            kind="area",
            max_size=250_000_000,
        ),
        v2_rationale=mo.ui.text_area(rows=2, full_width=True),
    ).form(
        submit_button_label="Commit reflection and linked V2",
        validate=_validate_revision,
        bordered=True,
    )
    revision_form
    return (revision_form,)


@app.cell(hide_code=True)
def _(
    EditDecisionManifest,
    PRIVATE_UPLOAD_DIR,
    SILENT_VIDEO_PATH,
    VIDEO_PATH,
    get_artifact_versions,
    get_classroom_log,
    mo,
    reduce_command,
    register_artifact_version,
    revision_form,
    set_artifact_versions,
    set_classroom_log,
    sha256,
):
    from datetime import datetime as _datetime
    from datetime import timezone as _timezone

    _revision = revision_form.value
    _artifacts = get_artifact_versions()
    _v1 = next(
        (_artifact for _artifact in _artifacts if _artifact.version_label == "V1"),
        None,
    )
    _v2 = next(
        (_artifact for _artifact in _artifacts if _artifact.version_label == "V2"),
        None,
    )
    _log = get_classroom_log()
    _runs = _log.records_of_type("run")
    if _v2 is not None:
        _revision_card = mo.callout(
            mo.md(
                f"**V2 committed** · `{_v2.artifact_id}`  \n"
                f"Parent V1 remains `{_v2.parent_artifact_id}`."
            ),
            kind="success",
        )
    elif _revision is None:
        _revision_card = mo.callout(
            mo.md("Complete the revision form after one committed run."), kind="info"
        )
    elif _v1 is None or not _runs:
        _revision_card = mo.callout(
            mo.md("Register V1 and commit one run snapshot before creating V2."),
            kind="danger",
        )
    else:
        _run_id = _runs[-1].record_id
        _key_digest = sha256(_revision["reflection_key"].strip().encode("utf-8")).hexdigest()
        _result_command = {
            "kind": "attach_result",
            "command_nonce": f"result:{_key_digest}",
            "run_id": _run_id,
            "result_digest": sha256(
                _revision["observed_evidence"].strip().encode("utf-8")
            ).hexdigest(),
            "metrics": {
                "student_observation": _revision["observed_evidence"].strip(),
                "verdict": _revision["verdict"],
            },
            "metric_versions": {
                "student_evidence_summary": "classroom-reflection/1.0.0"
            },
            "status": "completed",
        }
        _reflection_command = {
            "kind": "commit_reflection",
            "command_nonce": f"reflection:{_key_digest}",
            "run_id": _run_id,
            "reflection": {
                "evidence_trigger": _revision["evidence_trigger"].strip(),
                "limitation": _revision["limitation"].strip(),
                "rival_explanation": _revision["rival_explanation"].strip(),
                "revised_explanation": _revision["revised_explanation"].strip(),
                "next_control": _revision["next_control"].strip(),
                "creative_decisions": _revision["creative_decisions"].strip(),
            },
            "tags": ["V1-to-V2", _revision["verdict"]],
        }
        try:
            _with_result = reduce_command(_log, _result_command)
            _with_reflection = reduce_command(
                _with_result.log, _reflection_command
            )
            _source_path = VIDEO_PATH
            if _revision["v2_source"] == "Upload":
                from src.playground_clips import resolve_clip_selection as _resolve_v2_clip

                _upload = _revision["video"][0]
                _resolved = _resolve_v2_clip(
                    "Upload",
                    default_path=VIDEO_PATH,
                    silent_path=SILENT_VIDEO_PATH,
                    upload_dir=PRIVATE_UPLOAD_DIR,
                    upload_name=_upload.name,
                    upload_contents=_upload.contents,
                )
                _source_path = _resolved.path
            _edit_manifest = EditDecisionManifest(
                editor_name_version=_v1.editor_name_version,
                source_assets=_v1.source_assets,
                ordering=tuple(_v1.edit_decisions.get("ordering", ())),
                trims=tuple(_v1.edit_decisions.get("trims", ())),
                mix_levels=tuple(_v1.edit_decisions.get("mix_levels", ())),
                accessibility_work=_v1.accessibility,
                assistance_disclosure=_v1.assistance_disclosure,
                export_preset_version=_v1.export_preset_version,
            )
            _new_v2 = register_artifact_version(
                _source_path,
                edit_manifest=_edit_manifest,
                version_label="V2",
                parent_artifact_id=_v1.artifact_id,
                local_registered_at_utc=_datetime.now(_timezone.utc).isoformat(),
                event_index=len(_artifacts),
                elapsed_ms=0,
                creator_intention=_v1.creator_intention,
                intended_audience=_v1.intended_audience,
                sound_image_relation=_v1.sound_image_relation,
                concept_tags=_v1.concept_tags,
                cultural_aesthetic_context=_v1.cultural_aesthetic_context,
                source_license_provenance=_v1.source_license_provenance,
                change_rationale=(
                    _revision["v2_rationale"].strip()
                    + " | creative decisions: "
                    + _revision["creative_decisions"].strip()
                ),
                processing_boundary=_v1.processing_boundary,
            )
        except Exception as _error:  # noqa: BLE001 — revision errors belong in the UI
            _revision_card = mo.callout(
                mo.md(f"**Revision needs attention** — `{type(_error).__name__}: {_error}`"),
                kind="danger",
            )
        else:
            set_classroom_log(_with_reflection.log)
            set_artifact_versions(_artifacts + (_new_v2,))
            _revision_card = mo.callout(
                mo.md(
                    f"**Reflection and V2 committed** · `{_new_v2.artifact_id}`  \n"
                    f"V1 preserved: `{_new_v2.parent_artifact_id}` · run: `{_run_id}`"
                ),
                kind="success",
            )
    _revision_card
    return


@app.cell(hide_code=True)
def _(get_artifact_versions, get_audience_exchange, get_classroom_log, mo):
    _versions = {artifact.version_label for artifact in get_artifact_versions()}
    _packet, _readings = get_audience_exchange()
    _has_reflection = bool(get_classroom_log().records_of_type("reflection"))
    _complete = {"V1", "V2"}.issubset(_versions) and len(_readings) >= 2 and _has_reflection
    _status = "Complete" if _complete else "Waiting for comparison, reflection, and V2"
    mo.callout(
        mo.md(
            f"**Checkpoint · Exploratory playground — {_status}**  \n"
            "Next: export the private portfolio and make one bounded architecture proposal."
        ),
        kind="success" if _complete else "info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Synthesis and architecture challenge

    ### 4.1 Synthesize — Portfolio and architecture challenge

    **Required.** Use one committed run and reflection to make a bounded proposal.
    Distinguish the evidence you observed from your interpretation, then name the
    measurement limits, a viable alternative, and the next test.

    Your proposal may change modality routing or fusion—for example, a learned gate,
    bottleneck token set, late fusion, or sparse cross-modal router. Predict one
    distinctive signature and one result that would make you reject the proposal.

    **Measurement boundary.** Raw-probe entropy and probability margin are
    uncalibrated descriptive summaries. Teacher-forced entropy/margin condition on a
    fixed answer. Position-wise trajectories are comparable only when token-layout
    fingerprints match; otherwise use declared aggregate or normalized-bin
    comparisons. None of these measures alone establishes calibrated confidence,
    free-generation uncertainty, or causal localization.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    def _validate_synthesis(_value):
        if not _value:
            return "Complete the synthesis card."
        if any(not str(_value[_field]).strip() for _field in _value):
            return "Every synthesis field needs a short response."
        return None

    synthesis_form = mo.md(r"""
    **Observed evidence** {evidence}

    **Bounded interpretation** {interpretation}

    **Architecture proposal and information path** {proposal}

    **Predicted signature** {signature}

    **Distribution/alignment limitation** {measurement_limit}

    **Alternative or rival account** {alternative}

    **Next test and rejection condition** {next_test}

    **Performance or interpretability trade-off** {tradeoff}

    **Exit ticket** — one sentence separating observation, interpretation, and needed evidence {exit_ticket}
    """).batch(
        evidence=mo.ui.text_area(rows=2, full_width=True),
        interpretation=mo.ui.text_area(rows=2, full_width=True),
        proposal=mo.ui.text_area(rows=3, full_width=True),
        signature=mo.ui.text_area(rows=2, full_width=True),
        measurement_limit=mo.ui.text_area(rows=2, full_width=True),
        alternative=mo.ui.text_area(rows=2, full_width=True),
        next_test=mo.ui.text_area(rows=2, full_width=True),
        tradeoff=mo.ui.text_area(rows=2, full_width=True),
        exit_ticket=mo.ui.text_area(rows=2, full_width=True),
    ).form(
        submit_button_label="Commit synthesis card",
        validate=_validate_synthesis,
        bordered=True,
    )
    synthesis_form
    return (synthesis_form,)


@app.cell(hide_code=True)
def _(
    build_private_portfolio,
    get_artifact_versions,
    get_classroom_log,
    mo,
    serialize_private_portfolio,
    validate_process_log,
):
    _log = get_classroom_log()
    _artifacts = tuple(
        _artifact.to_dict() for _artifact in get_artifact_versions()
    )
    _portfolio = build_private_portfolio(
        _log,
        artifact_versions=_artifacts,
        boundary_disclosure=(
            "Session state is processed inside the hosted Molab session/container "
            "boundary, not solely on the student's device. This private download "
            "is user-initiated and is not research data."
        ),
    )
    _issues = validate_process_log(_log)
    _portfolio_bytes = serialize_private_portfolio(_portfolio)
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(value=str(len(_artifacts)), label="Artifact versions", bordered=True),
                    mo.stat(value=str(len(_log.records)), label="Process records", bordered=True),
                    mo.stat(
                        value="Valid" if not _issues else "Needs attention",
                        label="Record links",
                        bordered=True,
                    ),
                ],
                widths="equal",
                gap=1,
            ),
            mo.download(
                data=_portfolio_bytes,
                filename="counterpoint-lens-private-portfolio.json",
                label="Download private learning portfolio",
            ),
            mo.callout(
                mo.md(
                    "This button downloads to your device. It does not create a "
                    "research submission or send the portfolio to an instructor."
                ),
                kind="neutral",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    portfolio_import_form = mo.ui.file(
        filetypes=[".json"], multiple=False, kind="area"
    ).form(
        label="Restore a private portfolio after a kernel restart",
        submit_button_label="Validate and restore private portfolio",
        bordered=True,
    )
    reset_session_button = mo.ui.run_button(
        label="Delete session state and temporary uploads",
        kind="danger",
    )
    mo.accordion(
        {
            "Choice — restore or reset this teaching session": mo.vstack(
                [portfolio_import_form, reset_session_button]
            )
        },
        multiple=False,
    )
    return portfolio_import_form, reset_session_button


@app.cell(hide_code=True)
def _(
    ArtifactVersion,
    COURSE_RELEASE_ID,
    PRIVATE_UPLOAD_DIR,
    classroom_mode,
    json,
    load_jsonl,
    mo,
    new_session,
    portfolio_import_form,
    reset_session_button,
    set_artifact_versions,
    set_audience_exchange,
    set_classroom_log,
    uuid4,
):
    _restore_card = None
    if portfolio_import_form.value:
        try:
            _upload = portfolio_import_form.value[0]
            _payload = json.loads(_upload.contents.decode("utf-8"))
            _record_lines = [
                json.dumps(
                    _record,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for _record in _payload.get("records", [])
            ]
            _restored_log = load_jsonl(("\n".join(_record_lines) + "\n").encode("utf-8"))
            _restored_artifacts = tuple(
                ArtifactVersion.from_dict(_artifact)
                for _artifact in _payload.get("artifacts", [])
            )
        except Exception as _error:  # noqa: BLE001 — unsafe restore belongs in the UI
            _restore_card = mo.callout(
                mo.md(f"**Portfolio restore rejected** — `{type(_error).__name__}: {_error}`"),
                kind="danger",
            )
        else:
            set_classroom_log(_restored_log)
            set_artifact_versions(_restored_artifacts)
            set_audience_exchange((None, tuple()))
            _restore_card = mo.callout(
                mo.md(
                    f"**Portfolio restored** · {len(_restored_log.records)} records · "
                    f"{len(_restored_artifacts)} artifact versions"
                ),
                kind="success",
            )
    if reset_session_button.value:
        _fresh_log = new_session(
            f"studio-{uuid4().hex[:12]}",
            decision=classroom_mode,
            course_release_id=COURSE_RELEASE_ID,
        )
        set_classroom_log(_fresh_log)
        set_artifact_versions(tuple())
        set_audience_exchange((None, tuple()))
        for _path in PRIVATE_UPLOAD_DIR.glob("*"):
            if _path.is_file() or _path.is_symlink():
                _path.unlink()
        _restore_card = mo.callout(
            mo.md("**Session reset complete.** In-memory records and temporary uploads were deleted."),
            kind="success",
        )
    _restore_card
    return


@app.cell(hide_code=True)
def _(mo, synthesis_form):
    _complete = synthesis_form.value is not None
    _status = "Complete" if _complete else "Waiting for synthesis"
    mo.vstack(
        [
            mo.callout(
                mo.md(
                    f"**Checkpoint · Synthesis and architecture challenge — {_status}**  \n"
                    "Stop after your private portfolio and exit ticket are saved."
                ),
                kind="success" if _complete else "info",
            ),
            mo.callout(
                mo.md(
                    "Teaching mode keeps this session local by default. Save the "
                    "private portfolio you want to keep, then reset the session "
                    "and delete temporary uploads before leaving a shared device."
                ),
                kind="info",
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
