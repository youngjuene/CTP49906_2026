"""Lightweight, lossless caption display and classroom cache identity helpers.

No model or tensor imports: replay and notebook configuration can use these
helpers before loading the GPU stack.
"""

import hashlib
import json
import math
from pathlib import Path

SPECIAL_TOKEN_DISPLAY = "⟨special⟩"


def decode_caption_tokens(tokenizer, caption_ids):
    """Return ``(caption_text, pieces, kinds)`` with one slot per scored ID.

    Decode the full sequence once, with special tokens skipped and cleanup
    disabled. Each cumulative decode contributes only its prefix that agrees
    with that intact text. Incomplete UTF-8 tokens can therefore have an empty
    display slot; a later token displays the completed text. These are display
    boundaries, **not character-level probability attributions**. The original
    per-token scores remain unchanged and must be summed when grouping pieces.

    ``kinds`` marks empty slots as ``byte_continuation`` or ``special``. A final
    incomplete UTF-8 sequence retains the full decoder's replacement character;
    it is not silently deleted. Minimal legacy decoder adapters without the
    standard decode options retain their historic independent-token fallback.
    """
    ids = [int(token_id) for token_id in caption_ids]
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    decode_options = {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
    try:
        caption_text = tokenizer.decode(ids, **decode_options)
    except TypeError:
        pieces = ["" if token_id in special_ids else tokenizer.decode([token_id]) for token_id in ids]
        kinds = ["special" if token_id in special_ids else "text" for token_id in ids]
        return "".join(pieces), pieces, kinds

    pieces, kinds = [], []
    emitted = 0
    for position, token_id in enumerate(ids):
        prefix = tokenizer.decode(ids[:position + 1], **decode_options)
        stable = 0
        for observed, final in zip(prefix, caption_text):
            if observed != final:
                break
            stable += 1
        # Standard byte tokenizers are monotonic once incomplete bytes are
        # excluded. Retaining the previous boundary also tolerates decoder
        # prefix revisions without deleting text from an earlier display slot.
        stable = max(emitted, stable)
        piece = caption_text[emitted:stable]
        pieces.append(piece)
        kinds.append("special" if token_id in special_ids else "text" if piece else "byte_continuation")
        emitted = stable
    return caption_text, pieces, kinds


def group_tokens_into_words(caption_tokens, delta, token_kinds=None):
    """Whitespace display groups, with summed token Δ and original pieces.

    These groups are not linguistic words or character-level scores. Blank
    byte slots join the next completed text, including a new whitespace group.
    Supply ``token_kinds`` to keep each special ID in its own visible scored
    unit, never adding its Δ to lexical text. Byte slots may cross a special
    unit to join their next completed Unicode text; their display ordering is
    therefore not a replacement for the original per-token evidence table.
    Without kinds, the historic grouping of trailing blank slots is retained.
    """
    tokens = list(caption_tokens)
    values = [float(value) for value in delta]
    if len(tokens) != len(values):
        raise ValueError("caption_tokens and delta must have the same length")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("delta must contain only finite values")
    kinds = list(token_kinds) if token_kinds is not None else [None] * len(tokens)
    if len(kinds) != len(tokens):
        raise ValueError("token_kinds and caption_tokens must have the same length")
    words, pending = [], []
    last_is_special = False
    for token, value, kind in zip(tokens, values, kinds):
        text = token or ""
        if kind == "special":
            words.append([SPECIAL_TOKEN_DISPLAY, value, [(text, value)]])
            last_is_special = True
            continue
        if not text:
            pending.append((text, value))
            continue
        pieces = pending + [(text, value)]
        pending = []
        total = sum(piece_value for _, piece_value in pieces)
        if not words or last_is_special or text[:1].isspace():
            words.append([text, total, pieces])
        else:
            words[-1][0] += text
            words[-1][1] += total
            words[-1][2].extend(pieces)
        last_is_special = False
    if pending:
        total = sum(value for _, value in pending)
        if words and not last_is_special:
            words[-1][1] += total
            words[-1][2].extend(pending)
        else:
            words.append(["", total, pending])
    return [tuple(word) for word in words]


def selected_drop_share(caption_tokens, delta, threshold, token_kinds=None):
    """Percent of negative *grouped* Δ mass below ``-abs(threshold)``.

    Positive groups cannot cancel the denominator. Group before measuring
    negative mass, matching the displayed strip. Returns 0 when no group drops
    and otherwise a percentage in [0, 100]; it is not share of signed net Δ.
    Pass the same ``token_kinds`` as the strip so special-token units stay
    separate in both the numerator and denominator.
    """
    threshold = abs(float(threshold))
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    drops = [-value for _, value, _ in group_tokens_into_words(caption_tokens, delta, token_kinds=token_kinds) if value < 0]
    total = sum(drops)
    if not total:
        return 0.0
    return min(100.0, max(0.0, 100 * (sum(drop for drop in drops if drop > threshold) / total)))


def caption_cache_key(clip_path, nframes, prompt, max_new_tokens, model_identity):
    """SHA-256 identity for clip contents and greedy generation settings.

    ``model_identity`` is a JSON-serializable model/revision identity supplied by
    the caller. Hash file contents each time so same-name/same-size replacement
    uploads cannot reuse a different clip's caption. No file is changed.
    """
    clip_hash = hashlib.sha256()
    with Path(clip_path).open("rb") as clip:
        for chunk in iter(lambda: clip.read(1024 * 1024), b""):
            clip_hash.update(chunk)
    identity = {
        "schema": "caption-cache/2", "clip_sha256": clip_hash.hexdigest(),
        "nframes": nframes, "prompt": prompt, "max_new_tokens": max_new_tokens,
        "model_identity": model_identity, "do_sample": False,
    }
    serialized = json.dumps(identity, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
