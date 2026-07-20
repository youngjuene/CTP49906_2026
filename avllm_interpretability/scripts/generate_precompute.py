"""Generate the F5a precompute artifacts for GPU-free replay.

Run this ONCE on a GPU (molab or a CUDA box) to produce the committed artifacts
the notebook replays when `USE_PRECOMPUTED=True`:

    python avllm_interpretability/scripts/generate_precompute.py

It reproduces the notebook's fixed-parameter W7-W9 runs (logit-lens CSV +
caption, baseline/knockout captions, summarized attention-mass matrix) for the
sample clip and writes them to `avllm_interpretability/precomputed/`. Uses eager
attention for both experiments (one model load; the logit-lens hooks and the
knockout capture both need it). Greedy decoding is pinned so the committed
captions are reproducible. Raw attention tensors are never saved — only the
reduced matrix (see src/precompute.py).
"""
import argparse
import sys
from pathlib import Path

import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> avllm_interpretability/

from src.attention_knockout_experiment import block_attention
from src.attention_knockout_experiment import create_token_type_mapping as attn_type_map
from src.logitlens_experiment import (
    analyze_and_save_audio_logits_to_csv, clear_logit_lens_hooks,
    register_logit_lens_hooks,
)
from src.logitlens_experiment import create_token_type_mapping as logit_type_map
from src.precompute import save_precompute, summarize_attention

KNOCKOUT_RULES = [("generated", "video", 0, 36)]
ATTENTION_CAPTURE_LAYERS = (0, 2)


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-3B")
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
        args.model_path, torch_dtype="auto", attn_implementation="eager"
    )
    model.disable_talker()
    model = model.to(device).eval()
    proc = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    n_layers = len(model.thinker.model.layers)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "logit_lens_audio_token_analysis.csv"

    # --- W8: logit lens (CSV + caption) ---
    print("logit lens …")
    logit_inp, logit_types = _build_inputs(model, proc, args.logit_prompt, video, args.nframes, logit_type_map)
    register_logit_lens_hooks(model)
    try:
        with torch.no_grad():
            model.thinker(**logit_inp, output_hidden_states=True)
        analyze_and_save_audio_logits_to_csv(model, proc, logit_types, filename=str(csv_path))
    finally:
        clear_logit_lens_hooks()
    with torch.no_grad():
        _ids = model.thinker.generate(**logit_inp, max_new_tokens=args.max_new_tokens, do_sample=False)
    logit_caption = proc.batch_decode(_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # --- W9: attention knockout (baseline/knockout captions + attention summary) ---
    print("attention knockout …")
    attn_inp, attn_types = _build_inputs(model, proc, args.attention_prompt, video, args.nframes, attn_type_map)
    with torch.no_grad():
        _base = model.thinker.generate(
            **attn_inp, max_new_tokens=args.max_new_tokens, do_sample=False, return_dict_in_generate=True
        )
    baseline_text = proc.batch_decode(_base.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    with block_attention(
        model, KNOCKOUT_RULES, attn_types, len(attn_types),
        track_attention=True, capture_layer_range=ATTENTION_CAPTURE_LAYERS,
    ) as cap:
        with torch.no_grad():
            _ko = model.thinker.generate(
                **attn_inp, max_new_tokens=args.max_new_tokens, do_sample=False,
                output_attentions=True, return_dict_in_generate=True,
            )
        captured = {layer: list(v) for layer, v in cap.items()}
    knockout_text = proc.batch_decode(_ko.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    summary = summarize_attention(captured, attn_types)

    meta = {
        "clip": video.name, "nframes": args.nframes, "n_layers": n_layers,
        "logit_prompt": args.logit_prompt, "attention_prompt": args.attention_prompt,
        "knockout_rules": [list(r) for r in KNOCKOUT_RULES],
        "model": args.model_path, "generated_at": args.generated_at,
    }
    save_precompute(
        out_dir, logit_csv_src=csv_path, logit_caption=logit_caption,
        baseline_text=baseline_text, knockout_text=knockout_text,
        knockout_rules=KNOCKOUT_RULES, attention_summary=summary,
        attention_token_types=attn_types, meta=meta,
    )
    print(f"✅ wrote precompute artifacts to {out_dir}")
    print(f"   caption: {logit_caption!r}")
    print(f"   baseline vs knockout: {baseline_text!r}  |  {knockout_text!r}")


if __name__ == "__main__":
    main()
