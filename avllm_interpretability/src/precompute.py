"""F5a: GPU-free replay of the W7-W9 plots.

The notebook hard-asserts CUDA before anything renders, so a molab GPU outage
kills the class. F5a commits the *outputs* of the fixed-parameter W7-W9 cells
(the logit-lens CSV, the caption texts, and the summarized attention-mass matrix)
so those plots can be replayed with zero GPU when `USE_PRECOMPUTED=True`.

This module holds the pieces that need no model: `summarize_attention` (the
attention-mass reduction, lifted out of the notebook so the generator and the
live cell share one implementation), `save_precompute` / `load_precompute`, and
`StubModel` (a stand-in so the playground's layer-count read works with no GPU).

Only `generate_precompute.py` needs a GPU; everything here runs on CPU and is
unit-tested that way. Raw attention tensors are never committed (one eager
prefill layer is ~64 MB fp32) — only the reduced matrix.
"""
import json
import shutil
from pathlib import Path

import numpy as np

DEFAULT_MODALITY_ORDER = ["query_text", "audio", "video", "image", "generated"]

_ARTIFACTS = ("logit_lens_audio_token_analysis.csv", "captions.json",
              "attention_summary.json", "meta.json")


def summarize_attention(captured_attention, prompt_token_types, order=None):
    """Reduce captured attention to a (layers, modalities, matrix) summary.

    For each captured layer: average heads, take the final query's row, and sum
    its attention mass over each modality group. Returns
    `(layers, order, matrix)` with `matrix` a nested list of shape
    `[len(layers), len(order)]` (JSON-serializable), or `None` if nothing was
    captured. Accepts torch tensors or numpy arrays as the per-snapshot values.
    """
    order = list(order or DEFAULT_MODALITY_ORDER)
    records = []
    plen = len(prompt_token_types)
    for layer, snaps in sorted(captured_attention.items()):
        for snap in snaps:
            arr = snap.detach().float().cpu().numpy() if hasattr(snap, "detach") else np.asarray(snap, dtype=float)
            if arr.ndim == 4:
                mean = arr[0].mean(axis=0)
            elif arr.ndim == 3:
                mean = arr.mean(axis=0)
            else:
                continue
            key = mean[-1] if mean.shape[0] else mean
            ktypes = list(prompt_token_types) + ["generated"] * max(0, key.shape[-1] - plen)
            for m in order:
                idx = [i for i, tt in enumerate(ktypes) if tt == m]
                if idx:
                    records.append((layer, m, float(np.asarray(key)[idx].sum())))
    if not records:
        return None
    layers = sorted({l for l, _, _ in records})
    mat = np.zeros((len(layers), len(order)))
    for ri, l in enumerate(layers):
        for ci, m in enumerate(order):
            vals = [v for rl, rm, v in records if rl == l and rm == m]
            mat[ri, ci] = np.mean(vals) if vals else 0.0
    return layers, order, mat.tolist()


def save_precompute(out_dir, *, logit_csv_src, logit_caption, baseline_text,
                    knockout_text, knockout_rules, attention_summary,
                    attention_token_types, meta):
    """Write the committed F5a artifacts (called by the GPU generator)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _csv_dst = out / "logit_lens_audio_token_analysis.csv"
    if Path(logit_csv_src).resolve() != _csv_dst.resolve():
        shutil.copyfile(logit_csv_src, _csv_dst)
    (out / "captions.json").write_text(json.dumps({
        "logit_caption": logit_caption,
        "baseline_text": baseline_text,
        "knockout_text": knockout_text,
        "knockout_rules": [list(r) for r in knockout_rules],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    layers, mods, mat = attention_summary if attention_summary else ([], [], [])
    (out / "attention_summary.json").write_text(json.dumps({
        "layers": layers, "modalities": mods, "matrix": mat,
        "token_types": list(attention_token_types),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_precompute(precomputed_dir):
    """Load the committed artifacts; raise loudly (with the fix) if absent."""
    d = Path(precomputed_dir)
    missing = [name for name in _ARTIFACTS if not (d / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"USE_PRECOMPUTED=True but precomputed artifacts are missing in {d}: "
            f"{missing}. Generate them on a GPU with:\n"
            f"  python avllm_interpretability/scripts/generate_precompute.py"
        )
    caps = json.loads((d / "captions.json").read_text(encoding="utf-8"))
    attn = json.loads((d / "attention_summary.json").read_text(encoding="utf-8"))
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    summary = None
    if attn.get("layers"):
        summary = (attn["layers"], attn["modalities"], attn["matrix"])
    return {
        "logit_csv": d / "logit_lens_audio_token_analysis.csv",
        "logit_caption": caps.get("logit_caption", ""),
        "baseline_text": caps.get("baseline_text", ""),
        "knockout_text": caps.get("knockout_text", ""),
        "knockout_rules": caps.get("knockout_rules", []),
        "attention_summary": summary,
        "attention_token_types": attn.get("token_types", []),
        "meta": meta,
    }


class StubModel:
    """Layer-count stand-in for `USE_PRECOMPUTED` mode.

    The playground and teacher-forcing forms read
    `len(model.thinker.model.layers)` to size their layer-range sliders. With no
    GPU there is no real model, so this exposes just that. It cannot compute:
    the interactive sections fail loudly (their `try/except` surfaces the error)
    if a student submits them while replaying from cache.
    """

    class _NS:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    def __init__(self, n_layers=36):
        self.device = "cpu"
        self.thinker = StubModel._NS(model=StubModel._NS(layers=[None] * int(n_layers)))
