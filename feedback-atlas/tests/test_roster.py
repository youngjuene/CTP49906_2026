"""The roster gate: relaxed matching, strict membership. CPU only.

Two defects this file stands for. The first shipped as a usability failure: a
student pasted their id from a slide, brought a trailing space with it, and was
bounced three times in front of the room while the presentation waited. The
second is the one that would not have been noticed at all -- a helper that
admitted an unlisted id "just for now" and quietly turned a bounded set of
writers into an unbounded one.

PRD 5.1 asks for trim and case-folding, and nothing more. The tests below pin
both halves: sloppy input resolves, unlisted input does not, and there is no code
path that creates a roster entry at runtime.

Run:  python -m pytest feedback-atlas/tests/test_roster.py
  or:  python feedback-atlas/tests/test_roster.py
"""
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> feedback-atlas/

from src.roster import Roster, RosterEntry  # noqa: E402

CSV = (
    "id,display_name,role\n"
    "kim.seoyeon,김서연,student\n"
    "park.junho,박준호,student\n"
    "noh.kyungjin,노경진,auditor\n"
)


def test_a_pasted_id_with_stray_space_and_capitals_still_resolves():
    """The three-bounces-in-front-of-the-room defect.

    Everything here is what a real paste from a slide looks like.
    """
    roster = Roster.from_csv_text(CSV)
    for typed in ("kim.seoyeon", "  kim.seoyeon", "kim.seoyeon\n", "Kim.SeoYeon",
                  " KIM.SEOYEON \t"):
        entry = roster.resolve(typed)
        assert entry is not None, f"{typed!r} should have resolved"
        assert entry.id == "kim.seoyeon"


def test_resolve_returns_the_canonical_entry_not_the_typed_string():
    """Callers must store entry.id, never what the person typed.

    Storing the typed form files one person's opinions under three spellings, and
    admin mode's author filter -- the only reason admin mode exists -- silently
    stops working.
    """
    entry = Roster.from_csv_text(CSV).resolve("  KIM.SeoYeon ")
    assert isinstance(entry, RosterEntry)
    assert entry.id == "kim.seoyeon"
    assert entry.display_name == "김서연"


def test_an_unlisted_id_is_refused_and_does_not_grow_the_roster():
    """A miss must be a miss, not a lazily-created entry."""
    roster = Roster.from_csv_text(CSV)
    before = len(roster)
    assert roster.resolve("wanderer") is None
    assert "wanderer" not in roster
    assert len(roster) == before


def test_the_roster_exposes_no_way_to_add_an_entry_at_runtime():
    """Strict pre-registration is the policy; this absence is how it is enforced.

    Named explicitly because the tempting fix for a walk-in auditor mid-class is
    a one-line `roster.add(...)`, and that decision should have to survive
    deleting this test.
    """
    roster = Roster.from_csv_text(CSV)
    for attr in ("add", "register", "insert", "append", "update", "__setitem__",
                 "add_entry", "admit"):
        assert not hasattr(roster, attr), \
            f"Roster.{attr} exists: strict registration (PRD 11) is no longer enforced"


def test_a_name_typed_on_macos_matches_the_same_name_typed_on_android():
    """NFD versus NFC Hangul.

    macOS hands over decomposed Hangul. It renders identically and compares
    unequal, so without normalisation a Mac user is permanently locked out by an
    id that looks exactly right on screen.
    """
    roster = Roster.from_csv_text("id,display_name,role\n김서연,김서연,student\n")
    decomposed = unicodedata.normalize("NFD", "김서연")
    assert decomposed != "김서연"  # the premise: they really are different strings
    assert roster.resolve(decomposed) is not None


def test_two_ids_differing_only_in_case_are_a_load_error():
    """Silent shadowing would misattribute one person's feedback to another.

    Both rows answer to the same typed id, so one of them can never be reached --
    and which one wins depends on file order. Refusing to load is the only
    honest option.
    """
    try:
        Roster.from_csv_text(
            "id,display_name,role\nkim.a,A,student\nKim.A ,B,student\n")
    except ValueError as e:
        assert "kim.a" in str(e)
    else:
        raise AssertionError("duplicate ids after folding must not load")


def test_targets_are_students_only():
    """Auditors write feedback but are not presenting, so they are not targets."""
    targets = Roster.from_csv_text(CSV).targets()
    assert [t["id"] for t in targets] == ["kim.seoyeon", "park.junho"]
    assert all(set(t) == {"id", "display_name"} for t in targets)


def test_a_hand_edited_csv_with_a_bom_crlf_and_a_blank_last_line_loads():
    """What a roster actually looks like after a round-trip through Excel."""
    roster = Roster.from_csv_text(
        "﻿id,display_name,role\r\nkim.seoyeon,김서연,student\r\n\r\n")
    assert len(roster) == 1
    assert roster.resolve("kim.seoyeon").display_name == "김서연"


def test_an_unknown_role_and_an_empty_roster_both_refuse_to_load():
    """Both mean nobody can get in, which is never what was meant."""
    try:
        Roster.from_csv_text("id,display_name,role\na,A,ta\n")
    except ValueError as e:
        assert "ta" in str(e)
    else:
        raise AssertionError("an unknown role must not load")

    try:
        Roster.from_csv_text("id,display_name,role\n")
    except ValueError as e:
        assert "no entries" in str(e)
    else:
        raise AssertionError("an empty roster must not load")


def test_from_path_loads_the_file_the_server_actually_starts_from(tmp_path):
    """The production entry point. Every other test here builds a Roster from a
    string, so without this the file-reading path ships untested."""
    path = tmp_path / "roster.csv"
    path.write_text(CSV, encoding="utf-8")
    roster = Roster.from_path(path)
    assert len(roster) == 3
    assert roster.resolve("park.junho").display_name == "박준호"
    assert Roster.from_path(str(path)).resolve("kim.seoyeon") is not None


def test_an_id_typed_with_a_full_width_ime_still_resolves():
    """Found by QA. A Korean IME left in 전각 mode emits "ｋｉｍ.ｓｅｏｙｅｏｎ" --
    visibly the right id, matching nothing. Exactly the repeated-failure case
    PRD 5.1 is written against."""
    assert Roster.from_csv_text(CSV).resolve("ｋｉｍ.ｓｅｏｙｅｏｎ") is not None


def test_membership_and_length_read_naturally():
    """`in` and len() are part of the interface the server uses."""
    roster = Roster.from_csv_text(CSV)
    assert "kim.seoyeon" in roster and "  KIM.SeoYeon " in roster
    assert "nobody" not in roster and 123 not in roster
    assert len(roster) == 3 and len(roster.entries()) == 3


def test_a_missing_display_name_falls_back_to_the_id():
    """An instructor filling the file in a hurry should not produce blank rows in
    the admin panel."""
    roster = Roster.from_csv_text("id,display_name,role\nkim.seoyeon,,student\n")
    assert roster.resolve("kim.seoyeon").display_name == "kim.seoyeon"


if __name__ == "__main__":
    import tempfile

    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    fails = 0
    for fn in fns:
        try:
            if fn.__code__.co_varnames[: fn.__code__.co_argcount] == ("tmp_path",):
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print("PASS", fn.__name__)
        except Exception as e:  # noqa: BLE001
            fails += 1
            print("FAIL", fn.__name__, "->", type(e).__name__, e)
    print(f"\n{len(fns) - fails} passed, {fails} failed")
    sys.exit(1 if fails else 0)
