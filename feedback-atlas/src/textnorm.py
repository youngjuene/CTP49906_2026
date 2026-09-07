"""Text normalisation, in one place, because Korean makes it load-bearing.

Three call sites need three different normal forms and they must not be
confused:

  normalize_text  what gets stored and shown.
  text_hash       the embedding cache key. Two submissions of the same sentence
                  must hash the same or the cache misses forever and every
                  server restart re-encodes the whole corpus.
  match_key       roster comparison only (PRD 5.1: trim and case, nothing more).

The reason all three start with NFC: Hangul has two Unicode representations. A
macOS client sends "한국" decomposed (NFD, one jamo per code point); Android and
Windows send it composed (NFC). They render identically, compare unequal, and
hash differently. Without this the cache silently doubles and a roster entry
typed on a Mac never matches the same name typed on a phone.
"""

import hashlib
import re
import unicodedata

# Collapse runs of any whitespace, including the ideographic space U+3000, which
# a Korean IME emits and str.split() already treats as whitespace.
_WS = re.compile(r"\s+")


def normalize_text(raw: str) -> str:
    """NFC, trimmed, with internal whitespace runs collapsed to one space."""
    return _WS.sub(" ", unicodedata.normalize("NFC", raw)).strip()


def text_hash(text: str) -> str:
    """Cache key for one opinion's embedding.

    Hashes the *normalised* form, so "  좋았어요  " and "좋았어요" share a
    cached vector, and so do the NFC and NFD spellings of the same sentence.
    """
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def match_key(raw: str) -> str:
    """Roster comparison key: NFKC, trimmed, casefolded.

    casefold() rather than lower(): lower() is a no-op on the Turkish dotless i
    and several other cases where two spellings should compare equal. Korean has
    no case, but the roster holds Latin ids ("kim.seoyeon") that students retype
    from a slide, and that is where the trailing space and the capital K come
    from.

    NFKC here, where the other two functions use NFC. The compatibility fold is
    what turns full-width Latin into ASCII: a Korean IME left in 전각 mode emits
    "ｋｉｍ.ｓｅｏｙｅｏｎ", which is visibly the right id and matches nothing under
    NFC. That is precisely the repeated-failure-at-the-gate case PRD 5.1 is
    written against. NFKC is also the conventional fold for identifier matching.

    It stays out of normalize_text and text_hash on purpose: those carry what a
    person actually wrote, and NFKC rewrites content (ﬁ becomes fi, ① becomes 1).
    Folding is right for comparing ids and wrong for storing opinions.

    Deliberately not fuzzy. PRD 5.1 asks for trim and case, and nothing else:
    an edit-distance match would let one student's id resolve to another's.
    """
    return _WS.sub(" ", unicodedata.normalize("NFKC", raw)).strip().casefold()
