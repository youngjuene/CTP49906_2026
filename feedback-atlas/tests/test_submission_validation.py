"""What becomes an Opinion, and what authorship it gets. CPU only.

Two defects. The first: an empty submit created a point at the origin that
nobody could delete, because validation checked `"text" in body` rather than
whether the text had any content. The second is the one that matters -- a client
supplying its own `reviewer_id` and being believed, which would make the admin
author view worse than useless, since it would look authoritative while being
forgeable by anyone who can open devtools.

The rest of the file pins PRD 5.2's deliberate leniency: exactly one field can
block a submission, and everything else falls back to a default. A form that
argues with somebody mid-presentation just gets abandoned.

Run:  python -m pytest feedback-atlas/tests/test_submission_validation.py
  or:  python feedback-atlas/tests/test_submission_validation.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> feedback-atlas/

from src.models import DEFAULT_SOURCE, SOURCES, WEEKS, validate_submission  # noqa: E402
from src.roster import Roster  # noqa: E402

ROSTER = Roster.from_csv_text(
    "id,display_name,role\n"
    "kim.seoyeon,김서연,student\n"
    "kang.minsu,강민수,student\n"
    "noh.kyungjin,노경진,observer\n"
)
WRITER = ROSTER.resolve("kang.minsu")
OBSERVER = ROSTER.resolve("noh.kyungjin")


def _submit(**body):
    body.setdefault("target_id", "kim.seoyeon")
    body.setdefault("text", "좋았어요.")
    return validate_submission(body, reviewer=WRITER, roster=ROSTER)


def test_a_submitted_reviewer_id_is_ignored_in_favour_of_the_connection():
    """Authorship comes from the socket, never from the payload.

    Not rejected -- ignored. Rejecting would tell a prober that the field is
    interesting; ignoring means there is no code path where it can matter.
    """
    op, err = validate_submission(
        {"target_id": "kim.seoyeon", "text": "좋았어요.",
         "reviewer_id": "kim.seoyeon", "reviewerId": "kim.seoyeon"},
        reviewer=WRITER, roster=ROSTER)
    assert err is None
    assert op.reviewer_id == "kang.minsu"


def test_the_target_is_stored_canonically_not_as_typed():
    """Otherwise one project accumulates feedback under several spellings and the
    per-target colour on the map splits into two."""
    op, err = _submit(target_id="  KIM.SeoYeon \n")
    assert err is None and op.target_id == "kim.seoyeon"


def test_text_that_is_empty_or_only_whitespace_is_refused():
    """The origin-point defect. U+3000 is included because a Korean IME emits it
    and it looks exactly like nothing on screen."""
    for blank in ("", "   ", "\n\t", "　", "  　 \n"):
        op, err = _submit(text=blank)
        assert op is None and err == "EMPTY_TEXT", f"{blank!r} should be empty"


def test_source_defaults_to_ai_and_week_defaults_rather_than_blocking():
    """PRD 5.3: recorded as AI unless the writer says otherwise."""
    op, err = _submit()
    assert err is None
    assert op.source == DEFAULT_SOURCE == "ai"
    assert op.week in WEEKS


def test_an_out_of_range_week_is_refused_and_never_clamped():
    """Clamping a 5 to a 4 invents data indistinguishable from data somebody
    entered, in the field the whole week filter is built on."""
    for bad in (0, 5, -1, 99):
        op, err = _submit(week=bad)
        assert op is None and err == "BAD_WEEK", f"week={bad}"
    op, err = _submit(week="not a number")
    assert op is None and err == "BAD_WEEK"


def test_a_week_arriving_as_a_numeric_string_is_accepted():
    """JSON from a <select> is a string. Refusing it would break the real form
    while the tests, which pass ints, kept passing."""
    op, err = _submit(week="3")
    assert err is None and op.week == 3


def test_an_auditor_may_submit_and_may_tag_an_opinion_as_ai():
    """PRD 5.2 puts no role restriction on either. Anyone in the room can add an
    AI reading, which is what makes the AI-versus-human comparison collectable at
    all."""
    op, err = validate_submission(
        {"target_id": "kim.seoyeon", "text": "차갑게 느껴졌습니다.", "source": "ai"},
        reviewer=OBSERVER, roster=ROSTER)
    assert err is None
    assert op.reviewer_id == "noh.kyungjin" and op.source == "ai"


def test_feedback_aimed_at_an_auditor_or_an_unlisted_id_is_refused():
    """Unlike week and source this gets no default: filing an opinion against the
    wrong project is worse than refusing it, because it is invisible afterwards."""
    for target in ("noh.kyungjin", "nobody.here", "", "   "):
        op, err = _submit(target_id=target)
        assert op is None and err == "UNKNOWN_TARGET", f"target={target!r}"


def test_text_longer_than_the_cap_is_refused_before_it_can_be_stored():
    """Every stored character rides in every future snapshot and every reconnect
    for the rest of the semester. One 20 KB paste is a permanent tax."""
    op, err = _submit(text="가" * 1001)
    assert op is None and err == "TEXT_TOO_LONG"
    op, err = _submit(text="가" * 1000)
    assert err is None and len(op.text) == 1000


def test_an_unknown_source_is_refused_rather_than_silently_becoming_ai():
    """The default applies to an *absent* source. A present-but-wrong one is a
    client bug, and defaulting it would file human feedback as AI -- which is the
    one confusion the whole map is built to measure."""
    op, err = _submit(source="bot")
    assert op is None and err == "BAD_SOURCE"
    for good in SOURCES:
        op, err = _submit(source=good)
        assert err is None and op.source == good


def test_stored_text_is_normalised_but_not_otherwise_rewritten():
    """Trim and collapse whitespace; leave the person's words alone."""
    op, err = _submit(text="  숨이   막혔어요.  ")
    assert err is None and op.text == "숨이 막혔어요."


def test_a_boolean_is_not_a_week():
    """Found by QA. isinstance(True, int) is True and True == 1, so a bare
    `week in WEEKS` accepted True and stored it. It serialises as JSON `true`, and
    a client comparing week === 1 silently never matches that point."""
    op, err = _submit(week=True)
    assert op is None and err == "BAD_WEEK"
    from src.models import new_opinion
    direct = new_opinion(reviewer_id="kang.minsu", target_id="kim.seoyeon",
                         text="x", week=True)
    assert direct.week is not True and isinstance(direct.week, int)


def test_every_accepted_submission_gets_a_distinct_id():
    ids = {_submit()[0].id for _ in range(50)}
    assert len(ids) == 50


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
