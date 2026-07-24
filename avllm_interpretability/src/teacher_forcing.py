"""Teacher-forced attention knockout (F1).

The playground runs a single forward pass, so `generated` tokens never exist and
the tempting student rule `generated -> audio` is inert there. Teacher forcing
makes answer queries observable: generate caption `C` once, feed it back in as
input tagged `answer`, and measure how its assigned log-probability changes when
selected direct attention edges are blocked.

The metric is per-token delta log-likelihood, `delta = knockout - baseline`
(negative = the model assigns its own caption less probability after knockout). Unlike
a free-generation string diff it is continuous (you can see a *small* effect) and
deterministic (greedy caption, forward-only scoring).

The pure functions (`build_answer_token_types`, `caption_logprobs`,
`delta_logprobs`, `render_delta_strip`) carry the logic and are unit-tested on
CPU with synthetic tensors; `teacher_forced_delta` is the thin orchestration that
needs the model.
"""

from contextlib import nullcontext
from typing import Any

import torch

from .attention_knockout_experiment import block_attention
from .probe_metrics import (
    MEASUREMENT_CAVEATS,
    MeasurementKind,
    metric_version,
    summarize_logits,
)


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


def answer_distribution_summaries(
    logits,
    input_ids,
    prompt_len,
    *,
    top_k=5,
    target_token_set_version=None,
):
    """Compact final answer-token distributions for teacher-forced positions."""

    if logits.dim() == 3:
        logits = logits[0]
    if input_ids.dim() == 2:
        input_ids = input_ids[0]
    sequence_length = int(input_ids.shape[0])
    if not 1 <= prompt_len <= sequence_length:
        raise ValueError(
            f"prompt_len {prompt_len} out of range for seq {sequence_length}"
        )
    kind = MeasurementKind.TEACHER_FORCED_ANSWER_DISTRIBUTION_DISPERSION
    version = metric_version(
        kind,
        target_token_set_version=target_token_set_version,
    )
    positions = []
    for answer_position in range(prompt_len, sequence_length):
        target_token_id = int(input_ids[answer_position])
        distribution = summarize_logits(
            logits[answer_position - 1],
            top_k=top_k,
            target_token_ids=[target_token_id],
            measurement_kind=kind,
            version=version,
        )
        positions.append(
            {
                "answer_position": answer_position,
                "target_token_id": target_token_id,
                "target_log_probability": distribution.targets[0].log_probability,
                "distribution": distribution.to_dict(),
            }
        )
    return {
        "schema_version": "teacher-forced-distribution/1.0.0",
        "measurement_kind": kind.value,
        "caveat": MEASUREMENT_CAVEATS[kind],
        "metric_version": version,
        "positions": positions,
    }


def delta_logprobs(knockout_logprobs, baseline_logprobs):
    """`delta = knockout - baseline` per token (negative = believed less)."""
    return knockout_logprobs - baseline_logprobs


def group_tokens_into_words(caption_tokens, delta):
    """Group subword pieces into display words (F2).

    A token that starts with whitespace (or a newline) begins a new word --
    Qwen's byte-level BPE marks word starts with a leading space, so
    ` saxophone` -> [` sax`, `ophone`] regroups into one word. Returns a list of
    `(word_text, word_delta, [(piece, piece_delta), ...])` where `word_delta` is
    the **sum** of the pieces' deltas (log-probs add: the word's log-likelihood
    change).
    """
    words: list[list[Any]] = []
    for tok, val in zip(caption_tokens, [float(x) for x in delta]):
        text = tok or ""
        starts_word = (not words) or text[:1].isspace()
        if starts_word:
            words.append([text, val, [(text, val)]])
        else:
            words[-1][0] += text
            words[-1][1] += val
            words[-1][2].append((text, val))
    return [tuple(w) for w in words]


def render_delta_strip(
    caption_tokens, delta, cmap_name="RdBu", word_level=True, highlight_below=None
):
    """Colored caption strip: word-level display, token-level values (F2).

    Diverging scale centered at 0. Convention is pinned to `delta = knockout -
    baseline`, so the **negative** side is the hot color -- `RdBu` maps the most
    negative value to red, matching the notebook's existing
    "delta diversity (knockout - baseline)" panel.

    Subword pieces are joined into words for display (` saxophone` renders as
    one span, not `sax`+`ophone`); a word's color comes from its **summed**
    delta and its hover shows the sum plus the per-token breakdown when the word
    has several pieces. Pass `word_level=False` for the raw one-span-per-token
    view.

    `highlight_below`, when set to a threshold `t >= 0`, outlines every display
    unit whose delta is strictly below `-t` -- an outline (not a color change)
    so the diverging background scale stays readable underneath. The notebook's
    draggable threshold drives this.
    """
    import matplotlib

    vals = [float(x) for x in delta]
    if not vals:
        return "<em>(empty caption)</em>"

    if word_level:
        units = [
            (text, val, pieces if len(pieces) > 1 else None)
            for text, val, pieces in group_tokens_into_words(caption_tokens, vals)
        ]
    else:
        units = [(tok or "", val, None) for tok, val in zip(caption_tokens, vals)]

    vmax = max(1e-6, max(abs(v) for _, v, _ in units))
    norm = matplotlib.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = matplotlib.colormaps[cmap_name]

    def _esc(s):
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    spans = []
    for text, val, pieces in units:
        bg = matplotlib.colors.to_hex(cmap(norm(val)))
        title = f"Δ={val:+.2f} nats"
        if pieces:
            title += " (" + ", ".join(f"{_esc(p.strip()) or '·'}: {v:+.2f}" for p, v in pieces) + ")"
        shown = _esc(text).replace(" ", "&nbsp;") or "&nbsp;"
        flagged = highlight_below is not None and val < -abs(highlight_below)
        outline = "outline:2px solid #333;outline-offset:1px;" if flagged else ""
        spans.append(
            f'<span title="{title}" '
            f'style="background:{bg};{outline}padding:1px 2px;border-radius:2px">{shown}</span>'
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
    distribution_top_k=5,
    target_token_set_version=None,
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
    length-normalized `delta_mean`, the caption token strings, and both log-prob
    vectors.
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

    base_logits = _logits([])
    base_logp = caption_logprobs(base_logits, ext["input_ids"], prompt_len)
    baseline_distribution = answer_distribution_summaries(
        base_logits,
        ext["input_ids"],
        prompt_len,
        top_k=distribution_top_k,
        target_token_set_version=target_token_set_version,
    )
    if rules:
        knockout_logits = _logits(rules)
        ko_logp = caption_logprobs(knockout_logits, ext["input_ids"], prompt_len)
        knockout_distribution = answer_distribution_summaries(
            knockout_logits,
            ext["input_ids"],
            prompt_len,
            top_k=distribution_top_k,
            target_token_set_version=target_token_set_version,
        )
        del knockout_logits
    else:
        ko_logp = base_logp
        knockout_distribution = baseline_distribution
    del base_logits

    caption_tokens = [processor.tokenizer.decode([t]) for t in caption_ids[0].tolist()]
    delta = delta_logprobs(ko_logp, base_logp)
    return {
        "caption_ids": caption_ids,
        "caption_tokens": caption_tokens,
        "baseline_logprobs": base_logp,
        "knockout_logprobs": ko_logp,
        "baseline_distribution": baseline_distribution,
        "knockout_distribution": knockout_distribution,
        "delta": delta,
        "delta_total": float(delta.sum()),
        "delta_mean": float(delta.mean()),
    }
