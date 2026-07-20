"""Teacher-forced attention knockout (F1).

The playground runs a single forward pass, so `generated` tokens never exist and
the most intuitive student rule -- `generated -> audio` ("if the answer can't
hear, does it still describe the sound?") -- is inert there. Teacher forcing
fixes that: generate the caption `C` once, feed it back in as input tagged
`answer`, and score how much less the model believes what it said when a pathway
is knocked out.

The metric is per-token delta log-likelihood, `delta = knockout - baseline`
(negative = the model believed its own caption *less* after the knockout). Unlike
a free-generation string diff it is continuous (you can see a *small* effect) and
deterministic (greedy caption, forward-only scoring).

The pure functions (`build_answer_token_types`, `caption_logprobs`,
`delta_logprobs`, `render_delta_strip`) carry the logic and are unit-tested on
CPU with synthetic tensors; `teacher_forced_delta` is the thin orchestration that
needs the model.
"""

from contextlib import nullcontext

import torch

from .attention_knockout_experiment import block_attention


def build_answer_token_types(prompt_token_types, n_answer):
    """Types for the extended sequence `[prompt, C]`.

    The appended caption tokens are tagged `answer` **positionally** -- they are
    ordinary text ids, so the id-based mappers would type them `query_text`; the
    knockout has to distinguish "the answer" from "the instruction", so the type
    must come from position (>= prompt length), not from the token id.
    """
    if n_answer < 0:
        raise ValueError(f"n_answer must be >= 0, got {n_answer}")
    return list(prompt_token_types) + ["answer"] * n_answer


def caption_logprobs(logits, input_ids, prompt_len):
    """Per-token log P(C_t | C_<t, prompt) for the caption tokens.

    `logits[p]` predicts the token at position `p + 1`, so the log-prob of the
    caption token at extended position `p` (for `prompt_len <= p < seq`) is read
    from `logits[p - 1]`. Returns a 1-D tensor of length `seq - prompt_len`.
    """
    if logits.dim() == 3:
        logits = logits[0]
    if input_ids.dim() == 2:
        input_ids = input_ids[0]
    seq = input_ids.shape[0]
    if not (1 <= prompt_len <= seq):
        raise ValueError(f"prompt_len {prompt_len} out of range for seq {seq}")
    if prompt_len == seq:
        return logits.new_zeros(0)
    pred_positions = torch.arange(prompt_len - 1, seq - 1, device=logits.device)
    targets = input_ids[prompt_len:seq].to(logits.device)
    logp = torch.log_softmax(logits[pred_positions].float(), dim=-1)
    return logp.gather(1, targets.unsqueeze(1)).squeeze(1)


def delta_logprobs(knockout_logprobs, baseline_logprobs):
    """`delta = knockout - baseline` per token (negative = believed less)."""
    return knockout_logprobs - baseline_logprobs


def render_delta_strip(caption_tokens, delta, cmap_name="RdBu"):
    """Minimal per-token colored caption (F1's readable output; F2 polishes it).

    Diverging scale centered at 0. Convention is pinned to `delta = knockout -
    baseline`, so the **negative** side is the hot color -- `RdBu` maps the most
    negative value to red, matching the notebook's existing
    "delta diversity (knockout - baseline)" panel. Per-token nat value on hover.

    NOTE (F2 follow-up): renders one span per *token*, so subword pieces
    (` saxophone` -> `sax`+`ophone`) show fragment boundaries; F2 will join to
    word level for display while keeping per-token hover values.
    """
    import matplotlib

    vals = [float(x) for x in delta]
    if not vals:
        return "<em>(empty caption)</em>"
    vmax = max(1e-6, max(abs(v) for v in vals))
    norm = matplotlib.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = matplotlib.colormaps[cmap_name]
    spans = []
    for tok, val in zip(caption_tokens, vals):
        bg = matplotlib.colors.to_hex(cmap(norm(val)))
        text = (tok or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace(" ", "&nbsp;") or "&nbsp;"
        spans.append(
            f'<span title="Δ={val:+.2f} nats" '
            f'style="background:{bg};padding:1px 2px;border-radius:2px">{text}</span>'
        )
    return "".join(spans)


def teacher_forced_delta(
    model,
    processor,
    inputs,
    prompt_token_types,
    rules,
    max_new_tokens=32,
    cached_caption_ids=None,
):
    """Score a caption under `rules` vs baseline in two forward passes.

    Args:
        model: the (eager) Qwen thinker-bearing model, already on device.
        processor: its processor (for decoding caption tokens).
        inputs: the encoded *prompt* inputs dict (input_ids, attention_mask, and
            the multimodal feature tensors). Not mutated.
        prompt_token_types: per-position types for the prompt, from
            `create_token_type_mapping` (the attention mapper -> `query_text`,
            never the logit-lens mapper which emits `text`).
        rules: knockout rules as `(source, target, start, end)`, e.g.
            `[("answer", "audio", 0, 36)]`. Empty -> baseline only.
        max_new_tokens: greedy caption length when not cached.
        cached_caption_ids: reuse a previously generated `C` (shape `[1, n]`) so
            an unchanged (clip, prompt, nframes) submit skips regeneration.

    Returns a dict with per-token `delta` (= knockout - baseline), `delta_total`,
    the caption token strings, and both log-prob vectors.
    """
    device = inputs["input_ids"].device
    prompt_len = inputs["input_ids"].shape[1]

    # 1. Caption C, greedy (deterministic) unless supplied from cache. Slice the
    #    raw generated ids -- never re-tokenize the decoded string, which drops
    #    the audio/video placeholders and would break feature scatter.
    if cached_caption_ids is None:
        with torch.no_grad():
            gen = model.thinker.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        caption_ids = gen[:, prompt_len:]
    else:
        caption_ids = cached_caption_ids.to(device)
    n_answer = caption_ids.shape[1]
    if n_answer == 0:
        raise ValueError("empty caption; nothing to teacher-force")

    # 2. Extend: append caption ids + attention over them. The multimodal
    #    feature tensors describe placeholder positions in the *prompt* and are
    #    unchanged (we appended only text tokens), so they carry through as-is.
    ext = dict(inputs)
    ext["input_ids"] = torch.cat([inputs["input_ids"], caption_ids], dim=1)
    if inputs.get("attention_mask") is not None:
        am = inputs["attention_mask"]
        ext["attention_mask"] = torch.cat(
            [am, am.new_ones((am.shape[0], n_answer))], dim=1
        )

    # 3. Positional `answer` types for the extended sequence.
    ext_types = build_answer_token_types(prompt_token_types, n_answer)
    ext_len = ext["input_ids"].shape[1]

    # 4. One forward pass per condition. `block_attention` needs
    #    original_input_len = the full extended length so the prefill branch
    #    fires (q_len == k_len == ext_len).
    def _logits(active_rules):
        ctx = (
            block_attention(model, active_rules, ext_types, ext_len, track_attention=False)
            if active_rules
            else nullcontext()
        )
        with ctx, torch.no_grad():
            return model.thinker(**ext).logits

    base_logp = caption_logprobs(_logits([]), ext["input_ids"], prompt_len)
    ko_logp = caption_logprobs(_logits(rules), ext["input_ids"], prompt_len) if rules else base_logp

    caption_tokens = [processor.tokenizer.decode([t]) for t in caption_ids[0].tolist()]
    delta = delta_logprobs(ko_logp, base_logp)
    return {
        "caption_ids": caption_ids,
        "caption_tokens": caption_tokens,
        "baseline_logprobs": base_logp,
        "knockout_logprobs": ko_logp,
        "delta": delta,
        "delta_total": float(delta.sum()),
    }
