"""Unicode normalisation, which decides who gets in and what the cache hits.

These three functions look interchangeable and are not, and the differences are
exactly where the bugs live. Each stands for a real failure:

  normalize_text  a student's opinion arrived with a trailing newline from a
                  phone keyboard and sorted as a different string.
  text_hash       the embedding cache never hit for anyone on a Mac, so every
                  restart re-encoded the whole semester -- silently, just slowly.
  match_key       a Korean IME left in full-width mode produced an id that was
                  visibly correct on screen and matched nothing on the server.

The last one is why match_key folds with NFKC while the other two use NFC: an
identifier should be folded for comparison, and a person's words should not be
rewritten for storage.

Run:  python -m pytest feedback-atlas/tests/test_textnorm.py
  or:  python feedback-atlas/tests/test_textnorm.py
"""
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> feedback-atlas/

from src.textnorm import match_key, normalize_text, text_hash  # noqa: E402


def test_normalize_text_trims_and_collapses_including_the_ideographic_space():
    """U+3000 comes from a Korean IME and is invisible on screen."""
    assert normalize_text("  숨이   막혔어요.  ") == "숨이 막혔어요."
    assert normalize_text("좋았어요\n\t정말") == "좋았어요 정말"
    assert normalize_text("좋았어요　　정말") == "좋았어요 정말"
    assert normalize_text("") == "" and normalize_text("   ") == ""


def test_normalize_text_does_not_rewrite_what_the_person_wrote():
    """Only whitespace is touched. Compatibility characters are content."""
    assert normalize_text("①번 소리") == "①번 소리"
    assert normalize_text("ﬁ") == "ﬁ"
    assert "​" in normalize_text("a​b")  # zero-width space is not whitespace


def test_the_same_sentence_hashes_the_same_from_a_mac_and_a_phone():
    """The cache-never-hits defect. NFD and NFC Hangul render identically."""
    nfc = "물의 기억"
    nfd = unicodedata.normalize("NFD", nfc)
    assert nfc != nfd
    assert text_hash(nfc) == text_hash(nfd)
    assert text_hash("  물의 기억  ") == text_hash(nfc)


def test_different_sentences_hash_differently():
    assert text_hash("좋았어요") != text_hash("싫었어요")
    assert len(text_hash("x")) == 64


def test_text_hash_keeps_compatibility_characters_distinct():
    """text_hash is NFC, not NFKC: 'ﬁ' and 'fi' are different text and must not
    share a cached embedding, because they are not the same input to the model."""
    assert text_hash("ﬁ") != text_hash("fi")


def test_match_key_folds_case_whitespace_and_full_width_latin():
    """The IME defect. All four of these are the same person."""
    canonical = match_key("kim.seoyeon")
    for typed in ("kim.seoyeon", "  KIM.SeoYeon \n", "Kim.SeoYeon",
                  "ｋｉｍ.ｓｅｏｙｅｏｎ"):
        assert match_key(typed) == canonical, f"{typed!r} should fold to the same key"


def test_match_key_folds_hangul_across_normal_forms():
    assert match_key(unicodedata.normalize("NFD", "김서연")) == match_key("김서연")


def test_match_key_does_not_merge_genuinely_different_ids():
    """Folding must not become fuzzy matching: one student resolving to another's
    id is worse than a failed login."""
    for a, b in (("kim.a", "kim.b"), ("park", "parks"), ("김서연", "김서연2"),
                 ("lee.h", "lee_h")):
        assert match_key(a) != match_key(b), f"{a!r} and {b!r} collided"


def test_the_three_functions_are_not_interchangeable():
    """Pins the deliberate difference, so a later 'simplification' that routes all
    three through one normal form has to break a test that says why."""
    assert match_key("ｋｉｍ") == "kim"          # NFKC: folded
    assert normalize_text("ｋｉｍ") == "ｋｉｍ"      # NFC: left alone
    assert text_hash("ｋｉｍ") != text_hash("kim")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    fails = 0
    for fn in fns:
        try:
            fn()
            print("PASS", fn.__name__)
        except Exception as e:  # noqa: BLE001
            fails += 1
            print("FAIL", fn.__name__, "->", type(e).__name__, e)
    print(f"\n{len(fns) - fails} passed, {fails} failed")
    sys.exit(1 if fails else 0)
