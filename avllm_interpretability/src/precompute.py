"""F5a: GPU-free replay of the W7-W9 plots.

The notebook hard-asserts CUDA before anything renders, so a molab GPU outage
kills the class. F5a commits the *outputs* of the fixed-parameter W7-W9 cells
(the logit-lens CSV, the caption texts, and summarized baseline/knockout
attention-mass matrices)
so those plots can be replayed with zero GPU when `USE_PRECOMPUTED=True`.

This module holds the pieces that need no model: `summarize_attention` (the
attention-mass reduction, lifted out of the notebook so the generator and the
live cell share one implementation), `save_precompute` / `load_precompute`, and
`StubModel` (a stand-in so the playground's layer-count read works with no GPU).

Only `generate_precompute.py` needs a GPU; everything here runs on CPU and is
unit-tested that way. Raw attention tensors are never committed (one eager
prefill layer is ~64 MB fp32) — only the reduced matrices.
"""
import json
import shutil
from pathlib import Path

import numpy as np

DEFAULT_MODALITY_ORDER = ["query_text", "audio", "video", "image", "generated"]

_ARTIFACTS = ("logit_lens_audio_token_analysis.csv", "captions.json",
              "attention_summary.json", "meta.json")
_PROBE_DISTRIBUTIONS = "probe_distributions.json"
PRECOMPUTE_SCHEMA_VERSION = "avllm-precompute/3.0.0"


def summarize_attention(
    captured_attention, prompt_token_types, order=None, *, decode_only=False
):
    """Reduce captured attention to a (layers, modalities, matrix) summary.

    For each captured layer: average heads, take the final query's row, and sum
    its attention mass over each modality group. Returns
    `(layers, order, matrix)` with `matrix` a nested list of shape
    `[len(layers), len(order)]` (JSON-serializable), or `None` if nothing was
    captured. If ``decode_only=True``, excludes multi-query prefill snapshots so
    every row describes autoregressive generated-token queries. Accepts torch
    tensors or numpy arrays as the per-snapshot values.
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
            if decode_only and mean.shape[0] != 1:
                continue
            key = mean[-1] if mean.shape[0] else mean
            ktypes = list(prompt_token_types) + ["generated"] * max(0, key.shape[-1] - plen)
            # Record a complete vector for every snapshot. In the prefill
            # snapshot there are no generated keys yet, but that condition must
            # contribute a zero to the temporal mean. Skipping it gives
            # generated attention a shorter denominator and makes the final
            # modality row sum exceed one.
            for modality in order:
                idx = [i for i, token_type in enumerate(ktypes) if token_type == modality]
                value = float(np.asarray(key)[idx].sum()) if idx else 0.0
                records.append((layer, modality, value))
    if not records:
        return None
    layers = sorted({layer for layer, _, _ in records})
    mat = np.zeros((len(layers), len(order)))
    for ri, layer in enumerate(layers):
        for ci, modality in enumerate(order):
            vals = [
                value
                for record_layer, record_modality, value in records
                if record_layer == layer and record_modality == modality
            ]
            mat[ri, ci] = np.mean(vals) if vals else 0.0
    return layers, order, mat.tolist()


def _summary_payload(summary):
    layers, modalities, matrix = summary if summary else ([], [], [])
    return {"layers": layers, "modalities": modalities, "matrix": matrix}


def _summary_from_payload(payload):
    if not payload or not payload.get("layers"):
        return None
    return payload["layers"], payload["modalities"], payload["matrix"]


def save_precompute(out_dir, *, logit_csv_src, logit_caption, baseline_text,
                    knockout_text, knockout_rules, attention_summary,
                    attention_token_types, meta, baseline_attention_summary=None,
                    knockout_attention_summary=None, probe_distributions=None,
                    cache_identity=None):
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
    # Schema v2 stores both conditions.  Keep the legacy top-level fields as an
    # alias for the knockout summary so older notebook revisions can still read
    # newly generated artifacts.  Conversely, ``load_precompute`` accepts v1.
    knockout = (
        knockout_attention_summary
        if knockout_attention_summary is not None
        else attention_summary
    )
    knockout_payload = _summary_payload(knockout)
    (out / "attention_summary.json").write_text(json.dumps({
        "schema_version": 2,
        **knockout_payload,
        "baseline": _summary_payload(baseline_attention_summary),
        "knockout": knockout_payload,
        "token_types": list(attention_token_types),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    distribution_path = out / _PROBE_DISTRIBUTIONS
    if probe_distributions is not None:
        if cache_identity is None:
            raise ValueError("Versioned probe distributions require a cache identity")
        missing_meta = [
            field for field in ("condition_code", "manipulation") if not meta.get(field)
        ]
        if missing_meta:
            raise ValueError(
                f"Versioned probe distributions require condition metadata: {missing_meta}"
            )
        pack = {
            "schema_version": "probe-distribution-pack/1.0.0",
            "probe_distributions": probe_distributions,
            "cache_identity": cache_identity,
        }
        _validate_distribution_pack(pack)
        distribution_path.write_text(
            json.dumps(pack, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        distribution_path.unlink(missing_ok=True)
    versioned_meta = {**meta, "precompute_schema_version": PRECOMPUTE_SCHEMA_VERSION}
    (out / "meta.json").write_text(
        json.dumps(versioned_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out


def _legacy_probe_distributions():
    return {
        "schema_version": "probe-result/legacy",
        "status": "legacy_argmax_only",
        "measurement_kind": "legacy_argmax_only",
        "caveat": (
            "Legacy argmax-only artifact; entropy, margin, and target rank were not measured."
        ),
        "uncertainty": "not_measured",
        "summaries": [],
    }


def _validate_distribution_pack(pack):
    if pack.get("schema_version") != "probe-distribution-pack/1.0.0":
        raise ValueError("Unsupported probe-distribution pack schema")
    distributions = pack.get("probe_distributions")
    if not isinstance(distributions, dict):
        raise ValueError("probe_distributions must be a mapping")
    required = {
        "schema_version",
        "measurement_kind",
        "caveat",
        "token_layout_fingerprint",
        "metric_version",
        "summaries",
    }
    missing = sorted(required.difference(distributions))
    if missing:
        raise ValueError(f"probe_distributions missing required fields: {missing}")
    if not isinstance(distributions["summaries"], list):
        raise ValueError("probe_distributions.summaries must be a list")
    cache_identity = pack.get("cache_identity")
    if not isinstance(cache_identity, dict):
        raise ValueError("cache_identity must be a mapping")
    key = cache_identity.get("key")
    if not isinstance(key, str) or len(key) != 64:
        raise ValueError("cache_identity.key must be a SHA-256 digest")
    try:
        int(key, 16)
    except ValueError as exc:
        raise ValueError("cache_identity.key must be a SHA-256 digest") from exc
    return True


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
    distributions_path = d / _PROBE_DISTRIBUTIONS
    if distributions_path.exists():
        distribution_pack = json.loads(distributions_path.read_text(encoding="utf-8"))
        _validate_distribution_pack(distribution_pack)
        probe_distributions = distribution_pack["probe_distributions"]
        cache_identity = distribution_pack["cache_identity"]
    else:
        probe_distributions = _legacy_probe_distributions()
        cache_identity = None
        meta = {**meta, "precompute_schema_version": "legacy-precompute/2.0.0"}
    legacy_summary = _summary_from_payload(attn)
    baseline_summary = _summary_from_payload(attn.get("baseline"))
    knockout_summary = (
        _summary_from_payload(attn.get("knockout"))
        if "knockout" in attn
        else legacy_summary
    )
    return {
        "logit_csv": d / "logit_lens_audio_token_analysis.csv",
        "logit_caption": caps.get("logit_caption", ""),
        "baseline_text": caps.get("baseline_text", ""),
        "knockout_text": caps.get("knockout_text", ""),
        "knockout_rules": caps.get("knockout_rules", []),
        # The alias preserves the v1 loader API for existing callers.
        "attention_summary": knockout_summary,
        "baseline_attention_summary": baseline_summary,
        "knockout_attention_summary": knockout_summary,
        "attention_token_types": attn.get("token_types", []),
        "probe_distributions": probe_distributions,
        "cache_identity": cache_identity,
        "meta": meta,
    }


def validate_precompute_meta(meta, **expected):
    """Fail rather than replay artifacts for different fixed-run parameters."""

    mismatches = {
        key: {"artifact": meta.get(key), "notebook": value}
        for key, value in expected.items()
        if meta.get(key) != value
    }
    if mismatches:
        details = "; ".join(
            f"{key}: artifact={values['artifact']!r}, notebook={values['notebook']!r}"
            for key, values in mismatches.items()
        )
        raise ValueError(
            "Precomputed artifacts do not match the fixed notebook parameters: "
            f"{details}. Regenerate the pack instead of relabeling cached results."
        )
    return True


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
