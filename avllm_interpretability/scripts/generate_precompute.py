"""Generate the F5a precompute artifacts for GPU-free replay.

Run this ONCE on a GPU (molab or a CUDA box) to produce the committed artifacts
the notebook replays when `USE_PRECOMPUTED=True`:

    python avllm_interpretability/scripts/generate_precompute.py

It reproduces the notebook's fixed-parameter W7-W9 runs (logit-lens CSV +
caption, baseline/knockout captions, summarized attention-mass matrices) for the
sample clip and writes them to `avllm_interpretability/precomputed/`. Uses eager
attention for both experiments (one model load; the logit-lens hooks and the
knockout capture both need it). Greedy decoding is pinned so the committed
captions are reproducible. Raw attention tensors are never saved — only the
reduced matrices (see src/precompute.py).
"""
import argparse
import hashlib
import sys
from pathlib import Path

import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> avllm_interpretability/

from src.attention_knockout_experiment import block_attention
from src.attention_knockout_experiment import create_token_type_mapping as attn_type_map
from src.logitlens_experiment import (
    CaptureSpec, analyze_and_save_audio_logits_to_csv, clear_logit_lens_hooks,
    register_logit_lens_hooks,
)
from src.logitlens_experiment import create_token_type_mapping as logit_type_map
from src.precompute import save_precompute, summarize_attention
from src.probe_metrics import (
    build_analysis_cache_identity,
    build_token_layout_fingerprint,
)

KNOCKOUT_RULES = [("generated", "video", 0, 36)]
ATTENTION_CAPTURE_LAYERS = (0, 2)
DEFAULT_MODEL_REVISION = "f75b40e3da2003cdd6e1829b1f420ca70797c34e"


def _build_inputs(model, proc, prompt, video, nframes, type_fn):
    conv = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "video", "video": str(video), "nframes": nframes},
    ]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conv, use_audio_in_video=True)
    inp = proc(text=text, audio=audios, images=images, videos=videos,
               return_tensors="pt", padding=True, use_audio_in_video=True)
    inp = {k: v.to(model.device) for k, v in inp.items()}
    return inp, type_fn(inp["input_ids"], model.config.thinker_config)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-3B")
    ap.add_argument("--model_revision", default=DEFAULT_MODEL_REVISION)
    ap.add_argument("--video_path", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--nframes", type=int, default=8)
    ap.add_argument("--logit_prompt", default="Describe what you hear in the video")
    ap.add_argument("--attention_prompt", default="Describe what you see and hear in the video")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--generated_at", default="", help="optional ISO timestamp to stamp into meta.json")
    args = ap.parse_args()

    proj = Path(__file__).resolve().parents[1]
    video = Path(args.video_path) if args.video_path else proj / "assets" / "02321.mp4"
    out_dir = Path(args.out_dir) if args.out_dir else proj / "precomputed"
    assert video.is_file(), f"video not found: {video}"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"loading {args.model_path} (eager) on {device} …")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        revision=args.model_revision,
        torch_dtype="auto",
        attn_implementation="eager",
    )
    model.disable_talker()
    model = model.to(device).eval()
    proc = Qwen2_5OmniProcessor.from_pretrained(
        args.model_path, revision=args.model_revision
    )
    n_layers = len(model.thinker.model.layers)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "logit_lens_audio_token_analysis.csv"

    # --- W8: logit lens (CSV + caption) ---
    print("logit lens …")
    logit_inp, logit_types = _build_inputs(model, proc, args.logit_prompt, video, args.nframes, logit_type_map)
    audio_positions = tuple(
        position for position, modality in enumerate(logit_types) if modality == "audio"
    )
    layout_fingerprint = build_token_layout_fingerprint(
        input_ids=logit_inp["input_ids"].detach().cpu().reshape(-1).tolist(),
        modality_by_position=logit_types,
        processor_revision=args.model_revision,
        model_revision=args.model_revision,
        template_version="qwen2.5-omni-chat-template/transformers-4.52.0",
        input_shapes={
            key: list(value.shape)
            for key, value in logit_inp.items()
            if hasattr(value, "shape")
        },
    )
    register_logit_lens_hooks(
        model,
        capture_spec=CaptureSpec(selected_positions=audio_positions),
    )
    try:
        with torch.no_grad():
            model.thinker(**logit_inp, output_hidden_states=True)
        compact_probe_result = analyze_and_save_audio_logits_to_csv(
            model,
            proc,
            logit_types,
            filename=str(csv_path),
            token_layout_fingerprint=layout_fingerprint,
        )
        if compact_probe_result is None:
            raise RuntimeError("The fixed run produced no compact audio probe summaries")
    finally:
        clear_logit_lens_hooks()
    with torch.no_grad():
        _ids = model.thinker.generate(**logit_inp, max_new_tokens=args.max_new_tokens, do_sample=False)
    logit_prompt_len = logit_inp["input_ids"].shape[1]
    logit_caption = proc.batch_decode(
        _ids[:, logit_prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    # --- W9: attention knockout (baseline/knockout captions + attention summary) ---
    print("attention knockout …")
    attn_inp, attn_types = _build_inputs(model, proc, args.attention_prompt, video, args.nframes, attn_type_map)
    with block_attention(
        model, [], attn_types, len(attn_types),
        track_attention=True, capture_layer_range=ATTENTION_CAPTURE_LAYERS,
    ) as base_cap:
        with torch.no_grad():
            _base_ids = model.thinker.generate(
                **attn_inp, max_new_tokens=args.max_new_tokens, do_sample=False,
                return_dict_in_generate=False,
            )
        baseline_captured = {layer: list(values) for layer, values in base_cap.items()}
    baseline_summary = summarize_attention(
        baseline_captured, attn_types, decode_only=True
    )
    del baseline_captured
    attn_prompt_len = attn_inp["input_ids"].shape[1]
    baseline_text = proc.batch_decode(
        _base_ids[:, attn_prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    with block_attention(
        model, KNOCKOUT_RULES, attn_types, len(attn_types),
        track_attention=True, capture_layer_range=ATTENTION_CAPTURE_LAYERS,
    ) as cap:
        with torch.no_grad():
            _ko_ids = model.thinker.generate(
                **attn_inp, max_new_tokens=args.max_new_tokens, do_sample=False,
                return_dict_in_generate=False,
            )
        captured = {layer: list(v) for layer, v in cap.items()}
    knockout_text = proc.batch_decode(
        _ko_ids[:, attn_prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    knockout_summary = summarize_attention(captured, attn_types, decode_only=True)

    meta = {
        "clip": video.name, "nframes": args.nframes, "n_layers": n_layers,
        "logit_prompt": args.logit_prompt, "attention_prompt": args.attention_prompt,
        "knockout_rules": [list(r) for r in KNOCKOUT_RULES],
        "max_new_tokens": args.max_new_tokens,
        "attention_capture_layers": list(ATTENTION_CAPTURE_LAYERS),
        "model": args.model_path, "model_revision": args.model_revision,
        "generated_at": args.generated_at,
        "condition_code": "original_reference",
        "manipulation": "baseline",
        "token_layout_fingerprint": layout_fingerprint,
    }
    cache_identity = build_analysis_cache_identity(
        content_sha256=_sha256(video),
        normalized_recipe={
            "operation": "original_reference",
            "parameters": {},
            "transform_schema_version": "stimulus-variant/1.0.0",
        },
        processor_revision=args.model_revision,
        model_revision=args.model_revision,
        prompt=args.logit_prompt,
        frame_settings={"nframes": args.nframes},
        analysis_parameters={
            "layers": list(range(n_layers)),
            "positions": "audio_processor_positions",
            "top_k": 5,
            "projection_chunk_size": 32,
        },
    )
    save_precompute(
        out_dir, logit_csv_src=csv_path, logit_caption=logit_caption,
        baseline_text=baseline_text, knockout_text=knockout_text,
        knockout_rules=KNOCKOUT_RULES, attention_summary=knockout_summary,
        attention_token_types=attn_types, meta=meta,
        baseline_attention_summary=baseline_summary,
        knockout_attention_summary=knockout_summary,
        probe_distributions=compact_probe_result.to_dict(),
        cache_identity=cache_identity.to_dict(),
    )
    print(f"✅ wrote precompute artifacts to {out_dir}")
    print(f"   caption: {logit_caption!r}")
    print(f"   baseline vs knockout: {baseline_text!r}  |  {knockout_text!r}")


if __name__ == "__main__":
    main()
