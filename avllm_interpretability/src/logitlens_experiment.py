import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
# Make sure qwen_omni_utils.py is accessible in your environment
from qwen_omni_utils import process_mm_info
from collections import Counter

try:
    from .probe_metrics import (
        MEASUREMENT_CAVEATS,
        MeasurementKind,
        metric_version,
        summarize_logits,
    )
except ImportError:  # direct ``python src/logitlens_experiment.py`` execution
    from probe_metrics import (  # type: ignore
        MEASUREMENT_CAVEATS,
        MeasurementKind,
        metric_version,
        summarize_logits,
    )


PROBE_RESULT_SCHEMA_VERSION = "probe-result/1.0.0"


@dataclass(frozen=True)
class CaptureSpec:
    """Bound the layer outputs retained by logit-lens hooks."""

    selected_layers: tuple[int, ...] | None = None
    selected_positions: tuple[int, ...] | None = None

    def validated_layers(self, layer_count: int) -> tuple[int, ...]:
        layers = (
            tuple(range(layer_count))
            if self.selected_layers is None
            else tuple(sorted(set(self.selected_layers)))
        )
        invalid = [layer for layer in layers if not 0 <= layer < layer_count]
        if invalid:
            raise ValueError(f"Selected layers outside 0..{layer_count - 1}: {invalid}")
        return layers

    def validated_positions(self, sequence_length: int) -> tuple[int, ...]:
        positions = (
            tuple(range(sequence_length))
            if self.selected_positions is None
            else tuple(sorted(set(self.selected_positions)))
        )
        invalid = [position for position in positions if not 0 <= position < sequence_length]
        if invalid:
            raise ValueError(
                f"Selected positions outside 0..{sequence_length - 1}: {invalid}"
            )
        return positions


@dataclass(frozen=True)
class CapturedLayer:
    positions: tuple[int, ...]
    hidden_states: torch.Tensor


@dataclass(frozen=True)
class CompactProbeResult:
    schema_version: str
    measurement_kind: str
    caveat: str
    token_layout_fingerprint: str
    metric_version: Mapping[str, Any]
    summaries: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


logit_lens_storage: dict[int, CapturedLayer | torch.Tensor] = {}
logit_lens_hooks: list[Any] = []


def logit_lens_hook(layer_idx, capture_spec=None):
    def hook_fn(module, input, output):
        # transformers <5 returns a tuple (hidden_states, ...) from a decoder
        # layer; >=5 returns the bare hidden_states tensor. Handle both so we
        # always store the full [batch, seq, d_model] activation.
        hidden_state = output[0] if isinstance(output, tuple) else output
        if hidden_state is not None:
            if hidden_state.dim() == 2:
                hidden_state = hidden_state.unsqueeze(0)
            if hidden_state.dim() != 3:
                raise ValueError(
                    f"Expected [batch, seq, hidden] output, got {tuple(hidden_state.shape)}"
                )
            spec = capture_spec or CaptureSpec()
            positions = spec.validated_positions(hidden_state.shape[1])
            position_index = torch.tensor(
                positions, device=hidden_state.device, dtype=torch.long
            )
            # Slice while still on-device, then detach and transfer only the
            # bounded rows. Full layer outputs never enter retained storage.
            selected = hidden_state.index_select(1, position_index)
            logit_lens_storage[layer_idx] = CapturedLayer(
                positions=positions,
                hidden_states=selected.detach().cpu(),
            )
    return hook_fn


def register_logit_lens_hooks(model, capture_spec=None):
    global logit_lens_hooks
    for hook in logit_lens_hooks:
        hook.remove()
    logit_lens_hooks.clear()
    logit_lens_storage.clear()
    llm_layers = model.thinker.model.layers
    spec = capture_spec or CaptureSpec()
    selected_layers = spec.validated_layers(len(llm_layers))
    for layer_idx in selected_layers:
        layer = llm_layers[layer_idx]
        hook = layer.register_forward_hook(logit_lens_hook(layer_idx, spec))
        logit_lens_hooks.append(hook)
    print(
        f"✅ Registered {len(logit_lens_hooks)} bounded hooks for Logit Lens analysis."
    )

def clear_logit_lens_hooks():
    global logit_lens_hooks
    for hook in logit_lens_hooks:
        hook.remove()
    logit_lens_hooks.clear()
    print("🧹 Cleared all logit lens hooks.")

def create_token_type_mapping(input_ids, config):
    token_types = []
    for token_id in input_ids.cpu().flatten():
        if token_id == config.audio_token_index:
            token_types.append("audio")
        elif token_id == config.image_token_index:
            token_types.append("image")
        elif token_id == config.video_token_index:
            token_types.append("video")
        else:
            token_types.append("text")
    return token_types


def _captured_layer(value):
    if isinstance(value, CapturedLayer):
        return value
    tensor = value if value.dim() == 3 else value.unsqueeze(0)
    return CapturedLayer(tuple(range(tensor.shape[1])), tensor)


def build_compact_probe_result(
    model,
    processor,
    token_mapping,
    *,
    top_k=5,
    target_token_ids: Iterable[int] | None = None,
    projection_chunk_size=32,
    included_modalities: Iterable[str] | None = None,
    token_layout_fingerprint="",
):
    """Project selected rows in chunks and retain compact distribution summaries."""

    if projection_chunk_size < 1:
        raise ValueError("projection_chunk_size must be >= 1")
    lm_head = model.thinker.lm_head
    lm_head_device = lm_head.weight.device
    allowed_modalities = set(included_modalities) if included_modalities else None
    kind = MeasurementKind.RAW_PROBE_SCORE_DISPERSION
    version = metric_version(kind)
    summaries = []

    def decode(token_id):
        return processor.tokenizer.decode([token_id])

    for layer_idx, stored in sorted(logit_lens_storage.items()):
        capture = _captured_layer(stored)
        if capture.hidden_states.shape[0] != 1:
            raise ValueError("Compact probe result currently requires batch size 1")
        for start in range(0, len(capture.positions), projection_chunk_size):
            positions = capture.positions[start : start + projection_chunk_size]
            hidden_chunk = capture.hidden_states[0, start : start + len(positions)]
            with torch.no_grad():
                logits_chunk = lm_head(hidden_chunk.to(lm_head_device))
            for offset, position in enumerate(positions):
                modality = (
                    token_mapping[position]
                    if 0 <= position < len(token_mapping)
                    else "unknown"
                )
                if allowed_modalities is not None and modality not in allowed_modalities:
                    continue
                distribution = summarize_logits(
                    logits_chunk[offset],
                    top_k=top_k,
                    target_token_ids=target_token_ids,
                    measurement_kind=kind,
                    token_decoder=decode,
                    version=version,
                )
                summaries.append(
                    {
                        "layer": int(layer_idx),
                        "position": int(position),
                        "modality": modality,
                        "distribution": distribution.to_dict(),
                    }
                )
            del logits_chunk
    summaries.sort(key=lambda item: (item["layer"], item["position"], item["modality"]))
    return CompactProbeResult(
        schema_version=PROBE_RESULT_SCHEMA_VERSION,
        measurement_kind=kind.value,
        caveat=MEASUREMENT_CAVEATS[kind],
        token_layout_fingerprint=token_layout_fingerprint,
        metric_version=version,
        summaries=tuple(summaries),
    )


# Temporary legacy CSV adapter for the existing notebook and replay pack.
def analyze_and_save_audio_logits_to_csv(
    model,
    processor,
    token_mapping,
    filename="logit_lens_audio_token_analysis.csv",
    *,
    top_k=5,
    target_token_ids=None,
    projection_chunk_size=32,
    token_layout_fingerprint="",
):
    """
    Apply ``lm_head`` directly to stored raw layer outputs at audio positions.

    This diagnostic intentionally omits the thinker's final RMSNorm. Audio
    positions also lack a calibrated next-token language-model objective, so
    the decoded argmax tokens are probes rather than literal intermediate
    next-token predictions. Saves those probe tokens layer-by-layer to a CSV.
    """
    print(f"\n🔬 Analyzing captured hidden states and saving to '{filename}'...")
    # A missing capture/audio stream must never leave a plausible CSV from a
    # previous fixed run behind for downstream plotting.
    output_path = Path(filename)
    output_path.unlink(missing_ok=True)
    if not logit_lens_storage:
        print("Error: Logit lens storage is empty. No hidden states were captured.")
        return

    audio_token_indices = [i for i, t_type in enumerate(token_mapping) if t_type == 'audio']
    if not audio_token_indices:
        print("Warning: No audio tokens found in the input sequence.")
        return

    result = build_compact_probe_result(
        model,
        processor,
        token_mapping,
        top_k=top_k,
        target_token_ids=target_token_ids,
        projection_chunk_size=projection_chunk_size,
        included_modalities={"audio"},
        token_layout_fingerprint=token_layout_fingerprint,
    )
    by_identity = {
        (item["position"], item["layer"]): item for item in result.summaries
    }
    layers = sorted(logit_lens_storage)
    header = ['Token_Position', 'Token_Type'] + [f'Layer_{i}' for i in layers]
    rows = []

    for token_idx in audio_token_indices:
        csv_row = [token_idx, 'audio']
        if not all((token_idx, layer_idx) in by_identity for layer_idx in layers):
            continue
        for layer_idx in layers:
            top_tokens = by_identity[(token_idx, layer_idx)]["distribution"]["top_tokens"]
            csv_row.append(top_tokens[0]["token_text"])
        rows.append(csv_row)

    if not rows:
        print("Warning: Captured rows do not include any audio-token positions.")
        return None

    try:
        with output_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"✅ Successfully saved audio token logit lens analysis to '{filename}'")
    except IOError as e:
        print(f"Error writing to CSV file: {e}")
        return None
    return result


# --- Main Execution Block (Unchanged) ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run logit lens analysis on a video using Qwen2.5 Omni model")
    parser.add_argument("--model_path", required=True, help="Path or name of the pretrained model")
    parser.add_argument("--video_path", required=True, help="Path to the input video file")
    args = parser.parse_args()

    model_path = args.model_path
    video_path = args.video_path

    # --- 1. Model and Processor Setup ---

    print("Loading model...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        attn_implementation="sdpa",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model device: {model.device}")
    model.disable_talker()

    print("Loading processor...")
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

    # --- 2. Prepare Inputs ---
    conversation = [
        {"role": "user", "content": [
            {"type": "text", "text": "Describe what you hear in the video"},
            {"type": "video", "video": video_path, "nframes": 8},
        ]},
    ]
    USE_AUDIO_IN_VIDEO = True

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    token_mapping = create_token_type_mapping(inputs['input_ids'], model.config.thinker_config)
    print("Token mapping created:", Counter(token_mapping))

    # --- 3. Perform Analysis via Direct Forward Pass ---
    print("\nRunning a direct forward pass for logit lens analysis...")
    register_logit_lens_hooks(model)

    with torch.no_grad():
        outputs = model.thinker(**inputs, output_hidden_states=True)

    analyze_and_save_audio_logits_to_csv(model, processor, token_mapping)
    clear_logit_lens_hooks()

    # --- 4. (Optional) Generate Text Output to Confirm Model Works ---
    print("\nRunning model.generate() to get the final text output...")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=64)

    print("\n--- Generated Text ---")
    prompt_len = inputs["input_ids"].shape[1]
    decoded_text = processor.batch_decode(
        generated_ids[:, prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(decoded_text[0])
    print("\nAnalysis complete.")
