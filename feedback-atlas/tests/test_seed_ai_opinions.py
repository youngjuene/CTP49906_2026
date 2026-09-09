"""Bulk AI import writes to the store, and refuses what it cannot place. CPU only.

The defect: an instructor pasted thirty model readings between presentations and
twenty of them vanished. The submit path is rate-limited to about one message a
second per connection -- right for people, wrong for a script -- and the refusals
were invisible from the pasting end. Hence a script that writes to the database
directly (PRD 4.3).

Writing straight to the store means skipping the server's validation, so this
file exists to check the script did not skip the parts that matter: authorship
still resolves to a real roster entry, and a comment whose target cannot be
placed is refused rather than guessed at.

Run:  python -m pytest feedback-atlas/tests/test_seed_ai_opinions.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> feedback-atlas/

from scripts.seed_ai_opinions import main as seed  # noqa: E402
from src.store import Store  # noqa: E402

ROSTER = (
    "id,display_name,role\n"
    "kim.seoyeon,김서연,student\n"
    "park.junho,박준호,student\n"
    "instructor,담당교수,observer\n"
)


def _setup(tmp_path, rows, name="comments.csv"):
    roster = tmp_path / "roster.csv"
    roster.write_text(ROSTER, encoding="utf-8")
    src = tmp_path / name
    src.write_text(rows, encoding="utf-8")
    return src, roster, tmp_path / "atlas.db"


def _opinions(db):
    store = Store(str(db))
    store.migrate()
    out = store.all_opinions()
    store.close()
    return out


def _run(src, roster, db, *extra):
    return seed([str(src), "--db", str(db), "--roster", str(roster),
                 "--reviewer", "instructor", *extra])


def test_comments_land_in_the_store_tagged_ai_by_default(tmp_path):
    """PRD 5.3's default, and the entire point of the script."""
    src, roster, db = _setup(tmp_path,
        "target_id,text\n"
        "kim.seoyeon,화면 중앙의 피사체가 반복 진동합니다.\n"
        "park.junho,명도 대비가 낮아 식별이 어렵습니다.\n")
    assert _run(src, roster, db) == 0
    ops = _opinions(db)
    assert len(ops) == 2
    assert {o.source for o in ops} == {"ai"}
    assert {o.reviewer_id for o in ops} == {"instructor"}
    assert {o.target_id for o in ops} == {"kim.seoyeon", "park.junho"}


def test_a_comment_aimed_at_an_unknown_target_is_skipped_not_guessed(tmp_path):
    """Filing a model's reading against the wrong project is invisible once it is
    on the map, so it is refused rather than placed somewhere plausible."""
    src, roster, db = _setup(tmp_path,
        "target_id,text\nnobody,이 줄은 건너뛰어야 합니다.\nkim.seoyeon,이 줄은 들어갑니다.\n")
    assert _run(src, roster, db) == 1          # non-zero: something was skipped
    ops = _opinions(db)
    assert len(ops) == 1 and ops[0].text == "이 줄은 들어갑니다."


def test_empty_and_over_long_text_are_skipped(tmp_path):
    src, roster, db = _setup(tmp_path,
        "target_id,text\nkim.seoyeon,\nkim.seoyeon,   \n"
        f"kim.seoyeon,{'가' * 1001}\nkim.seoyeon,좋습니다.\n")
    assert _run(src, roster, db) == 1
    assert [o.text for o in _opinions(db)] == ["좋습니다."]


def test_a_reviewer_who_is_not_on_the_roster_stops_the_import(tmp_path):
    """Authorship has to resolve to a real entry, or admin mode's author view
    files these under a name that does not exist."""
    src, roster, db = _setup(tmp_path, "target_id,text\nkim.seoyeon,좋습니다.\n")
    try:
        seed([str(src), "--db", str(db), "--roster", str(roster),
              "--reviewer", "ghost"])
    except SystemExit as e:
        assert "roster" in str(e)
    else:
        raise AssertionError("an unlisted reviewer must not import")
    assert not db.exists(), "a refused import must not create a database"


def test_the_target_is_stored_canonically(tmp_path):
    """A CSV hand-edited from a slide carries the same stray spaces a student's
    paste does."""
    src, roster, db = _setup(tmp_path,
        "target_id,text\n  KIM.SeoYeon ,대소문자와 공백이 섞인 대상.\n")
    assert _run(src, roster, db) == 0
    assert _opinions(db)[0].target_id == "kim.seoyeon"


def test_a_per_row_week_and_source_override_the_defaults(tmp_path):
    src, roster, db = _setup(tmp_path,
        "target_id,text,week,source\n"
        "kim.seoyeon,사람이 쓴 것으로 표시합니다.,3,human\n"
        "kim.seoyeon,기본값을 씁니다.,,\n")
    assert _run(src, roster, db, "--week", "2") == 0
    by_text = {o.text: o for o in _opinions(db)}
    assert by_text["사람이 쓴 것으로 표시합니다."].week == 3
    assert by_text["사람이 쓴 것으로 표시합니다."].source == "human"
    assert by_text["기본값을 씁니다."].week == 2
    assert by_text["기본값을 씁니다."].source == "ai"


def test_an_out_of_range_week_is_skipped_rather_than_clamped(tmp_path):
    src, roster, db = _setup(tmp_path,
        "target_id,text,week\nkim.seoyeon,주차가 5입니다.,5\nkim.seoyeon,정상입니다.,2\n")
    assert _run(src, roster, db) == 1
    assert [o.text for o in _opinions(db)] == ["정상입니다."]


def test_jsonl_input_works_too(tmp_path):
    src, roster, db = _setup(tmp_path,
        '{"target_id":"kim.seoyeon","text":"제이슨 줄 하나."}\n'
        '{"target_id":"park.junho","text":"두 번째 줄.","week":4}\n',
        name="comments.jsonl")
    assert _run(src, roster, db) == 0
    ops = _opinions(db)
    assert len(ops) == 2
    assert {o.week for o in ops} == {1, 4}


def test_a_dry_run_writes_nothing(tmp_path):
    """The instructor's chance to see what would be skipped before it is stored."""
    src, roster, db = _setup(tmp_path, "target_id,text\nkim.seoyeon,좋습니다.\n")
    assert _run(src, roster, db, "--dry-run") == 0
    assert not db.exists()


def test_importing_the_same_file_twice_duplicates_it(tmp_path):
    """Documented, not prevented. Opinion ids are fresh per row, so a second run
    is a second set of points -- which is why the README says to use --dry-run
    first and not to re-run a file."""
    src, roster, db = _setup(tmp_path, "target_id,text\nkim.seoyeon,좋습니다.\n")
    _run(src, roster, db)
    _run(src, roster, db)
    assert len(_opinions(db)) == 2


if __name__ == "__main__":
    import tempfile

    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    fails = 0
    for fn in fns:
        try:
            with tempfile.TemporaryDirectory() as d:
                fn(Path(d))
            print("PASS", fn.__name__)
        except Exception as e:  # noqa: BLE001
            fails += 1
            print("FAIL", fn.__name__, "->", type(e).__name__, e)
    print(f"\n{len(fns) - fails} passed, {fails} failed")
    sys.exit(1 if fails else 0)
