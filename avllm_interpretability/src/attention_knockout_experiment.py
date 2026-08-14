import torch
import pickle
from datetime import datetime
from collections import defaultdict, Counter
from typing import Optional, List, Dict, Tuple
from contextlib import contextmanager

# `transformers` and `qwen_omni_utils` are imported inside the CLI block below,
# not here: nothing in this module's library surface touches them, and importing
# them at module scope would make the pure guards (`rule_reach`, the hook logic)
# untestable anywhere the multimodal stack is not installed.
import argparse
import time

def format_time(seconds):
    """Convert seconds to a readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


# --- 1. Type Mapping ---
# Define a consistent numeric mapping for token types
TOKEN_TYPE_MAP: Dict[str, int] = {
    "query_text": 0,
    "audio": 1,
    "video": 2,
    "image": 3,
    "generated": 4,
    # `answer` tags a teacher-forced caption appended after the prompt: the
    # tokens the model produced, fed back in as input so a single forward pass
    # can score log P(caption | prompt) under a knockout. Distinct from
    # `generated` (which only exists during autoregressive decoding) so a rule
    # like `answer -> audio` acts during the forward pass, where `generated` is
    # inert. Assigned positionally by the caller, never by the id-based mapper.
    "answer": 5,
}
# Create a reverse map for potential debugging (optional)
ID_TO_TOKEN_TYPE: Dict[int, str] = {v: k for k, v in TOKEN_TYPE_MAP.items()}

# `generated` positions are the ones past `original_input_len`; they are created
# by decoding, so they exist as queries *and* as keys only in the decode branch of
# `BlockAttentionHook`. In a single forward pass over the prompt there are none,
# which makes a `generated` rule inert on either side. `answer` is the mirror
# image: assigned positionally by `build_answer_token_types`, so it is present in
# `token_types` and live during prefill.
POSITIONAL_TOKEN_TYPES = frozenset({"generated"})


def rule_reach(rules, token_types, n_layers, context="generate"):
    """What a set of knockout rules would actually mask. Pure; no model needed.

    A rule that matches no layer or no token is silently a no-op: `block_attention`
    registers hooks only where `start <= i < end` and the hook itself only fires
    where a query is a source and a key is a target. The notebook then reports the
    unchanged output as "this band shows no effect" -- a confident false negative,
    produced precisely when a student is being careful (dragging both range-slider
    handles onto layer 12 to ask "is it exactly here?" yields `[12, 12)`).

    `context` selects which of the hook's two branches will run:
      "generate" -- `model.generate()`, so both the prefill and the decode branch
        fire and `generated` is live as source and as target.
      "forward"  -- a single forward pass over the prompt, so only the prefill
        branch fires and `generated` is inert on *both* sides. The notebook's
        diversity scoreboard and the teacher-forcing scorer are this case; the
        scoreboard's prose warns only about the source half.

    Returns `{"ok": bool, "reasons": [...], "rules": [per-rule dict]}` rather than
    raising, so a caller can render the diagnosis in-cell before spending GPU time.
    """
    counts = Counter(token_types)
    reasons = []
    per_rule = []
    for rule in rules:
        if len(rule) != 4:
            reasons.append(f"{rule!r} is not (source, target, start_layer, end_layer)")
            per_rule.append({"rule": rule, "ok": False})
            continue
        source, target, start, end = rule
        why = []
        for role, name in (("source", source), ("target", target)):
            if name not in TOKEN_TYPE_MAP:
                why.append(
                    f"unknown {role} {name!r} -- use {' / '.join(TOKEN_TYPE_MAP)}"
                )
            elif name in POSITIONAL_TOKEN_TYPES:
                if context == "forward":
                    why.append(
                        f"{role} {name!r} is inert in a forward pass: there are no "
                        f"{name} positions until the model decodes"
                    )
            elif counts.get(name, 0) == 0:
                why.append(
                    f"{role} {name!r} appears 0 times in this input "
                    f"(present: {', '.join(f'{k} {v}' for k, v in counts.most_common())})"
                )
        try:
            start_i, end_i = int(start), int(end)
        except (TypeError, ValueError):
            why.append(f"start/end must be integers, got {start!r}, {end!r}")
            start_i = end_i = 0
        else:
            layers = max(0, min(end_i, n_layers) - max(0, start_i))
            if layers == 0:
                why.append(
                    f"[{start_i}, {end_i}) masks 0 of {n_layers} layers "
                    "(`end` is exclusive)"
                )
        per_rule.append({
            "rule": tuple(rule),
            "source_tokens": counts.get(source, 0),
            "target_tokens": counts.get(target, 0),
            "layers": max(0, min(int(end_i), n_layers) - max(0, int(start_i))),
            "ok": not why,
            "reasons": why,
        })
        reasons.extend(why)
    return {"ok": not reasons, "reasons": reasons, "rules": per_rule}


def capture_vram_estimate(n_layers, seq_len, n_steps, n_heads=16, dtype_bytes=2):
    """Rough bytes held by `track_attention` capture, for the budget warning.

    Each captured layer keeps one `[heads, q, k]` attention tensor per decode step
    on CPU. Widening `ATTENTION_CAPTURE_LAYERS` from the default `(0, 2)` to every
    layer is the obvious student move and the one that can take the session down --
    `force_attention_output_hook`'s docstring already says so, but nothing measures
    it. Heads are averaged only *after* capture, so `n_heads` belongs here: the
    captured tensor is the pre-reduction one. Defaults match Qwen2.5-Omni-3B's
    thinker (16 heads, bf16); pass the real values when the caller knows them.
    """
    return (
        int(n_layers) * int(n_heads) * int(n_steps) * int(seq_len) ** 2 * int(dtype_bytes)
    )


# --- 2. The (Vectorized) Hook for Modifying Attention Masks ---
class BlockAttentionHook:
    """
    A forward pre-hook that dynamically modifies the attention mask for
    BOTH prefill (prompt processing) and generation (token-by-token).
    """
    def __init__(self, 
                 numeric_knockout_rules: torch.Tensor, 
                 numeric_token_types: torch.Tensor, 
                 original_input_len: int,
                 generated_token_id: int):
        
        self.numeric_knockout_rules = numeric_knockout_rules
        self.numeric_token_types = numeric_token_types
        self.original_input_len = original_input_len
        self.generated_token_id = generated_token_id
        
        # Optimization: only run full logic if a query token is a 'source'
        self.source_types_to_block = torch.unique(self.numeric_knockout_rules[:, 0])
        self.current_device = None # Set on first call

    def __call__(self, module, args, kwargs):
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is None or self.numeric_knockout_rules.shape[0] == 0:
            return args, kwargs

        if self.current_device is None:
            self.current_device = attention_mask.device

        k_len = attention_mask.shape[-1]
        q_len = attention_mask.shape[-2]
        
        modified_mask = None
        mask_value = torch.finfo(attention_mask.dtype).min
        
        # --- CASE 1: Generation (Autoregressive step) ---
        # Query shape is [1, K_len], where K_len is dynamic
        if q_len == 1:
            query_pos = k_len - 1 

            # 1. Determine query type
            if query_pos >= self.original_input_len:
                query_type_id = self.generated_token_id
            else:
                # This can happen in prompts with length > 1
                query_type_id = self.numeric_token_types[query_pos].item()

            # 2. OPTIMIZATION
            if query_type_id not in self.source_types_to_block:
                return args, kwargs

            # 3. Determine key types
            num_generated = k_len - self.original_input_len
            if num_generated > 0:
                generated_ids = torch.full(
                    (num_generated,), 
                    self.generated_token_id, 
                    dtype=self.numeric_token_types.dtype, 
                    device=self.numeric_token_types.device 
                )
                key_type_ids = torch.cat((self.numeric_token_types, generated_ids))
            else:
                # We are attending to a prefix of the original input
                key_type_ids = self.numeric_token_types[:k_len]

            # 4. Build block mask (1D)
            final_block_mask = torch.zeros(k_len, dtype=torch.bool, device=self.current_device)
            for rule_idx in range(self.numeric_knockout_rules.shape[0]):
                source_id, target_id = self.numeric_knockout_rules[rule_idx]
                if query_type_id == source_id:
                    key_is_target_mask = (key_type_ids == target_id)
                    final_block_mask.logical_or_(key_is_target_mask)
            
            # 5. Apply (1D)
            if torch.any(final_block_mask):
                modified_mask = attention_mask.clone()
                modified_mask[..., 0, final_block_mask] = mask_value

        # --- CASE 2: Prefill (Prompt processing) ---
        # Query and Key are both the original input
        elif q_len == k_len and k_len == self.original_input_len:
            query_type_ids = self.numeric_token_types
            key_type_ids = self.numeric_token_types

            # Build block mask (2D)
            final_block_mask = torch.zeros((q_len, k_len), dtype=torch.bool, device=self.current_device)
            
            for rule_idx in range(self.numeric_knockout_rules.shape[0]):
                source_id = self.numeric_knockout_rules[rule_idx, 0]
                target_id = self.numeric_knockout_rules[rule_idx, 1]

                # Find all queries of source type
                query_is_source_mask = (query_type_ids == source_id)
                # Find all keys of target type
                key_is_target_mask = (key_type_ids == target_id)
                
                # Create a 2D mask for (Q=src, K=tgt)
                # [Q_len, 1] & [1, K_len] -> [Q_len, K_len]
                q_k_mask = query_is_source_mask.unsqueeze(1) & key_is_target_mask.unsqueeze(0)
                final_block_mask.logical_or_(q_k_mask)

            # 5. Apply (2D)
            if torch.any(final_block_mask):
                modified_mask = attention_mask.clone()
                # Apply to all batches/heads
                modified_mask[..., final_block_mask] = mask_value
        
        if modified_mask is not None:
            kwargs["attention_mask"] = modified_mask
        
        return args, kwargs

# --- 3. The Attention *Capture* Hook ---

def attention_hook_fn(layer_idx: int, storage_dict: dict):
    """
    Creates a hook function that captures attention weights and
    stores them in the provided storage_dict.
    """
    def hook(module, input, output):
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            storage_dict[layer_idx].append(output[1].detach().cpu())
    return hook


def force_attention_output_hook(module, args, kwargs):
    """Request weights only on layers captured by our forward hook.

    Passing ``output_attentions=True`` to ``generate`` makes Transformers retain
    every layer's attention for every decoding step in its return object. On a
    24 GB classroom GPU that can OOM even though this lab only needs two layers.
    A per-module pre-hook keeps the model-level flag off: the decoder discards
    the weights immediately, while ``attention_hook_fn`` copies just the selected
    layers to CPU.
    """

    kwargs["output_attentions"] = True
    return args, kwargs


# --- 4. The Context Manager to Apply All Hooks ---

@contextmanager
def block_attention(model: torch.nn.Module,
                    knockout_rules: List[Tuple[str, str, int, int]], # <-- NEW FORMAT
                    token_types: List[str],
                    original_input_len: int,
                    track_attention: bool = False,
                    capture_layer_range: Optional[Tuple[int, int]] = None,
                    context: str = "generate"):
    """
    A context manager that applies attention hooks to the model.
    - knockout_rules: List of (source, target, start_layer, end_layer)
    - capture_layer_range: Optional (start, end) range for *attention capture*
    - context: "generate" (both hook branches fire) or "forward" (prefill only, so
      `generated` rules are inert). See `rule_reach`.
    """

    knockout_hook_handles = []
    capture_flag_handles = []
    capture_hook_handles = []
    attention_storage = defaultdict(list)
    
    # 1. Pre-process token types.
    # An unknown token type is a programming error, not a recoverable state:
    # silently yielding with zero hooks registered would run a *baseline* while
    # the caller believes a knockout is applied — a plausible-but-wrong result,
    # exactly the artifact this course teaches students to distrust. Fail loudly
    # so the notebook surfaces it as an in-cell error instead.
    device = model.device
    try:
        numeric_token_types = torch.tensor(
            [TOKEN_TYPE_MAP[t] for t in token_types], dtype=torch.long, device=device
        )
    except KeyError as e:
        raise ValueError(
            f"Unknown token type {e} in token_types; valid types are "
            f"{list(TOKEN_TYPE_MAP.keys())}"
        ) from e
    generated_token_id = TOKEN_TYPE_MAP["generated"]

    # A rule that reaches nothing is the same failure as an unknown token type:
    # the caller believes an intervention is applied and gets a baseline back.
    # Refuse it here rather than let the notebook label the unchanged output
    # "the pathway is not carried here".
    if knockout_rules:
        reach = rule_reach(
            knockout_rules, token_types, len(model.thinker.model.layers), context=context
        )
        if not reach["ok"]:
            raise ValueError(
                "These knockout rules would mask nothing: "
                + "; ".join(reach["reasons"])
                + ". A no-op rule returns a baseline run that reads as 'no effect'."
            )

    try:
        # 2. Register all hooks
        num_knockout_hooks = 0
        num_capture_hooks = 0
        
        all_layers = model.thinker.model.layers
        total_layers = len(all_layers)
        
        # --- Define the start/end for attention CAPTURE ---
        capture_start, capture_end = 0, total_layers
        if capture_layer_range:
            capture_start = max(0, capture_layer_range[0])
            capture_end = min(total_layers, capture_layer_range[1])
            print(f"Applying ATTENTION CAPTURE to layer range: [{capture_start}, {capture_end})")
        if track_attention:
            # Budget note, not a hard cap: the step count is unknown here, so this
            # prices a single prefill. Widening the range is a legitimate
            # experiment -- it just needs to be a priced one.
            n_captured = max(0, capture_end - capture_start)
            # Heads come from the model rather than the default where we can get
            # them: the captured tensor is `[heads, q, k]` and the head count
            # drives the estimate linearly.
            try:
                n_heads = int(model.config.thinker_config.text_config.num_attention_heads)
            except AttributeError:
                n_heads = 16
            est = capture_vram_estimate(
                n_captured, original_input_len, 1, n_heads=n_heads
            )
            print(
                f"Attention capture: {n_captured} layer(s) x {n_heads} heads x "
                f"{original_input_len}^2 ~= {est / 2**30:.2f} GiB per decode step "
                "(host RAM)."
            )
            if n_captured > 4:
                print(
                    f"⚠️  Capturing {n_captured} layers costs about "
                    f"{est / 2**30:.1f} GiB *per generated token*. This is the "
                    "usual way to lose both loaded models on a 24 GB classroom "
                    "GPU -- narrow ATTENTION_CAPTURE_LAYERS if the run dies."
                )


        # --- NEW: Layer-by-layer hook registration ---
        for i, layer in enumerate(all_layers):
            
            # --- Part 1: Register Knockout Hook ---
            
            # Find rules active for this layer `i`
            active_rules_for_layer = []
            for rule in knockout_rules:
                if len(rule) != 4:
                    raise ValueError(f"Rule {rule} is not in the format (src, tgt, start_layer, end_layer)")
                src, tgt, start, end = rule
                if start <= i < end:
                    active_rules_for_layer.append((src, tgt))
            
            # Determine layer device
            try:
                layer_device = next(layer.parameters()).device
            except StopIteration:
                try:
                    layer_device = next(layer.buffers()).device
                except StopIteration:
                    layer_device = model.device 
            
            hook_token_types = numeric_token_types.to(layer_device)

            # Register the knockout (pre-)hook if there are active rules
            if active_rules_for_layer:
                numeric_rules_list = [
                    (TOKEN_TYPE_MAP[src], TOKEN_TYPE_MAP[tgt]) for src, tgt in active_rules_for_layer
                ]
                numeric_knockout_rules = torch.tensor(
                    numeric_rules_list, dtype=torch.long, device=layer_device
                )
                
                knockout_hook = BlockAttentionHook(
                    numeric_knockout_rules=numeric_knockout_rules,
                    numeric_token_types=hook_token_types,
                    original_input_len=original_input_len,
                    generated_token_id=generated_token_id
                )
                handle = layer.self_attn.register_forward_pre_hook(knockout_hook, with_kwargs=True)
                knockout_hook_handles.append(handle)
                num_knockout_hooks += 1

            # --- Part 2: Register Capture Hook ---
            if track_attention and (capture_start <= i < capture_end):
                flag_handle = layer.self_attn.register_forward_pre_hook(
                    force_attention_output_hook, with_kwargs=True
                )
                capture_hook = attention_hook_fn(i, attention_storage)
                handle = layer.self_attn.register_forward_hook(capture_hook)
                capture_flag_handles.append(flag_handle)
                capture_hook_handles.append(handle)
                num_capture_hooks += 1
        
        if knockout_hook_handles:
            print(f"Applied {num_knockout_hooks} total knockout hooks across layers.")
            print(f"Rule configuration: {knockout_rules}")
        if capture_hook_handles:
            print(f"Applied {num_capture_hooks} attention capture hooks.")
            
        # 3. Yield control back to the `with` block
        yield attention_storage
    
    finally:
        # 4. Clean up all hooks
        for handle in knockout_hook_handles:
            handle.remove()
        for handle in capture_flag_handles:
            handle.remove()
        for handle in capture_hook_handles:
            handle.remove()
        
        if knockout_hook_handles:
            print(f"Removed {num_knockout_hooks} knockout hooks.")
        if capture_hook_handles:
            print(f"Removed {num_capture_hooks} capture hooks.")


# --- 5. Data Capture and Saving ---

def save_knockout_data(filename: str, 
                       generated_text: str, 
                       token_map: List[str], 
                       knockout_rules: List[Tuple[str, str, int, int]], # <-- NEW FORMAT
                       attention_storage: dict,
                       capture_layer_range: Optional[Tuple[int, int]]): # <-- NEW
    """
    Saves the results of the knockout experiment to a pickle file.
    """
    data = {
        'attention_weights_per_step': attention_storage,
        'token_mapping': token_map,
        'generated_text': generated_text,
        'metadata': {
            'knockout_rules': knockout_rules,
            'capture_layer_range': capture_layer_range,
            'model_name': model.config.name_or_path,
            'original_input_length': len(token_map),
            'num_layers_tracked': len(attention_storage) if attention_storage else 0,
            'timestamp': datetime.now().isoformat(),
        }
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"✅ Saved knockout experiment data to {filename}")


def create_token_type_mapping(input_ids: torch.Tensor, config) -> List[str]:
    """
    Maps each input ID to its modality type (str).
    """
    token_types = []
    # Assumes batch size is 1
    for token_id in input_ids[0]:
        tid = token_id.item()
        if tid == config.audio_token_index:
            token_types.append("audio")
        elif tid == config.image_token_index:
            token_types.append("image")
        elif tid == config.video_token_index:
            token_types.append("video")
        else:
            token_types.append("query_text")
    return token_types

# --- 6. Main Execution Block ---

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--knockout-rules',
        type=str,
        default='generated,video,0,35',
        help='Knockout rules as semicolon-separated pairs. Example: "generated,video,0,35;audio,text"'
    )
    
    parser.add_argument(
        '--track-attention',
        action='store_true',
        help='Enable attention tracking'
    )

    parser.add_argument(
        '--model_path',
        required=True,
        help='Path to the pretrained model directory (e.g., Qwen/Qwen2.5-Omni-3B)'
    )

    parser.add_argument(
        '--video_path',
        required=True,
        help='Path to the input video file used for evaluation (e.g., assets/02321.mp4)'
    )

    parser.add_argument(
        '--query_text',
        default='Describe what you see',
        help='Text query to prompt the model (default: "Describe what you see")'
    )

    args = parser.parse_args()
    
    # Parse knockout rules
    knockout_rules = []
    for pair in args.knockout_rules.split(';'):
        src_tok_type, tgt_tok_type, start_layer, end_layer = pair.strip().split(',')
        knockout_rules.append((src_tok_type.strip(), tgt_tok_type.strip(), int(start_layer.strip()), int(end_layer.strip())))
    print(knockout_rules)
    
    return {
        'KNOCKOUT_RULES': knockout_rules,
        'TRACK_ATTENTION': args.track_attention,
        'MODEL_PATH': args.model_path,
        'VIDEO_PATH': args.video_path,
        'QUERY_TEXT': args.query_text,
    }


if __name__ == "__main__":
    from qwen_omni_utils import process_mm_info
    from transformers import (
        Qwen2_5OmniForConditionalGeneration,
        Qwen2_5OmniProcessor,
    )

    config = parse_args()
    KNOCKOUT_RULES = config['KNOCKOUT_RULES']
    TRACK_ATTENTION = False
    MODEL_PATH = config['MODEL_PATH']
    VIDEO_PATH = config['VIDEO_PATH']
    QUERY_TEXT = config['QUERY_TEXT']

    # This range is now ONLY for attention capture
    CAPTURE_LAYER_RANGE = (0, 10)

    rules_str = "_".join([f"{s}2{t}L{start}_{end}" for s, t, start, end in KNOCKOUT_RULES]) if KNOCKOUT_RULES else "baseline"
    save_path = f"{rules_str}_single_video.json"

    print("🚀 Loading model and processor...")

    print("Loading original model...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_PATH, 
        torch_dtype="auto", 
        attn_implementation="eager", # Eager is needed for attention hooks
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model device: {model.device}")
    model.disable_talker()
    
    print("Loading processor...")

    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
    
    # You can get the total number of layers to help define rules:
    total_layers = model.thinker.model.config.num_hidden_layers
    print(f"Model has {total_layers} layers.")

    total_time = 0
    sample_idx = 0
    print("-"*10)
    print(f"Processing single video: {VIDEO_PATH}")

    start_time = time.time()
    video_path = VIDEO_PATH

    conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": QUERY_TEXT},
                    {"type": "video", "video": video_path, "nframes": 8},
                ],
            },
        ]

    USE_AUDIO_IN_VIDEO = True

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    token_mapping = create_token_type_mapping(inputs['input_ids'], model.config.thinker_config)
    original_input_len = len(token_mapping)
    print(f"Original input length: {original_input_len}")
    print(f"Token counts: {Counter(token_mapping)}")

    with block_attention(model, 
                        KNOCKOUT_RULES, 
                        token_mapping, 
                        original_input_len, 
                        track_attention=TRACK_ATTENTION,
                        capture_layer_range=CAPTURE_LAYER_RANGE # <-- MODIFIED
                        ) as attention_storage:
    
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=50,
            output_attentions=TRACK_ATTENTION,
            return_dict_in_generate=True,
        ).sequences

    generated_text = processor.batch_decode(
        output_ids[:, original_input_len:], skip_special_tokens=True
    )[0]
    
    print("\n--- Results ---")
    print(f"Video Path: {VIDEO_PATH}")
    print(f"Knockout Rules: {KNOCKOUT_RULES}")
    print(f"Capture Layer Range: {CAPTURE_LAYER_RANGE}")
    print(f"Token Mapping Length: {len(token_mapping)}")
    print(f"Generated Text:\n{generated_text}")
