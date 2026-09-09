"""The post-semester archive must not be able to damage the live data. CPU only.

The defect this file stands for is the one you get exactly once: a
pseudonymisation script that overwrote the database it was reading, replacing
the only copy of the semester's reviewer ids with codes and destroying the
mapping in the same pass.

So the first test hashes the source before and after. The rest pin the property
that makes the output worth calling de-identified at all -- that the code cannot
be reversed without the mapping file, and that the mapping file is somewhere
else.

Run:  python -m pytest feedback-atlas/tests/test_deidentify.py
"""
import csv
import hashlib
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> feedback-atlas/

from scripts.deidentify import main as deidentify  # noqa: E402
from src.models import new_opinion  # noqa: E402
from src.store import Store  # noqa: E402
from src.textnorm import text_hash  # noqa: E402

ROSTER = (
    "id,display_name,role\n"
    "kim.seoyeon,김서연,student\n"
    "park.junho,박준호,student\n"
    "kang.minsu,강민수,observer\n"
    "seo.jiwoo,서지우,observer\n"
    "han.doyun,한도윤,observer\n"
    "noh.kyungjin,노경진,observer\n"
)
WRITERS = ["kang.minsu", "seo.jiwoo", "han.doyun", "noh.kyungjin"]
TARGETS = ["kim.seoyeon", "park.junho"]


def _build(tmp_path, n=12):
    roster = tmp_path / "roster.csv"
    roster.write_text(ROSTER, encoding="utf-8")
    db = tmp_path / "atlas.db"
    store = Store(str(db))
    store.migrate()
    coords = {}
    for i in range(n):
        op = new_opinion(reviewer_id=WRITERS[i % len(WRITERS)],
                         target_id=TARGETS[i % len(TARGETS)],
                         text=f"의견 {i} — 소리가 몸 안쪽에서 나는 것처럼 들렸어요.")
        store.insert_opinion(op, text_hash(op.text))
        coords[op.id] = (0.1 * i, -0.2 * i)
    store.put_coords(coords, 1)
    store.close()
    return db, roster


def _rows(path):
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_the_source_database_is_byte_identical_afterwards(tmp_path):
    """The defect this file is named for. Run it, then prove nothing moved."""
    db, roster = _build(tmp_path)
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    assert deidentify(["--db", str(db), "--roster", str(roster),
                       "--out", str(tmp_path / "archive")]) == 0
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before


def test_the_archive_holds_no_real_id_and_no_display_name(tmp_path):
    """A substring search over the whole file, not a column check: a real name
    smuggled into any field is the same leak as one in the id column."""
    db, roster = _build(tmp_path)
    out = tmp_path / "archive"
    deidentify(["--db", str(db), "--roster", str(roster), "--out", str(out),
                "--targets"])
    text = (out / "archive.csv").read_text(encoding="utf-8")
    for leak in WRITERS + ["강민수", "서지우", "한도윤", "노경진",
                           "김서연", "박준호"] + TARGETS:
        assert leak not in text, f"the archive still contains {leak!r}"


def test_one_reviewer_gets_exactly_one_code_and_codes_do_not_collide(tmp_path):
    """Otherwise two people merge into one row of the analysis, or one person
    splits into two, and neither is visible from the archive alone."""
    db, roster = _build(tmp_path)
    out = tmp_path / "archive"
    deidentify(["--db", str(db), "--roster", str(roster), "--out", str(out)])

    mapping = [r for r in _rows(out / "mapping.csv") if r["kind"] == "reviewer"]
    codes = [r["code"] for r in mapping]
    ids = [r["id"] for r in mapping]
    assert len(set(codes)) == len(codes), "two reviewers share a code"
    assert len(set(ids)) == len(ids), "a reviewer appears twice"
    assert set(ids) == set(WRITERS)

    by_code = {r["code"]: r["id"] for r in mapping}
    archive = _rows(out / "archive.csv")
    assert {r["reviewer_code"] for r in archive} == set(by_code)
    assert len(archive) == 12


def test_the_mapping_is_a_separate_file_the_archive_does_not_reference(tmp_path):
    """The archive is what you keep; the mapping is what you destroy. If the
    archive named it, deleting the mapping would leave a dangling pointer to the
    thing you were trying to remove."""
    db, roster = _build(tmp_path)
    out = tmp_path / "archive"
    deidentify(["--db", str(db), "--roster", str(roster), "--out", str(out)])
    assert (out / "archive.csv").exists() and (out / "mapping.csv").exists()
    text = (out / "archive.csv").read_text(encoding="utf-8")
    assert "mapping" not in text
    # And the archive stays readable on its own once the mapping is gone.
    (out / "mapping.csv").unlink()
    assert len(_rows(out / "archive.csv")) == 12


def test_codes_are_not_derived_from_the_ids(tmp_path):
    """A hash of the id, or a number in roster order, is reversible by anyone who
    has the class list -- which is everyone the archive would be shared with.

    Run repeatedly and require the assignment to differ at least once. With four
    reviewers, six identical shuffles by chance is about one in ten million.
    """
    db, roster = _build(tmp_path)
    seen = set()
    for i in range(6):
        out = tmp_path / f"archive{i}"
        deidentify(["--db", str(db), "--roster", str(roster), "--out", str(out)])
        mapping = {r["id"]: r["code"] for r in _rows(out / "mapping.csv")}
        seen.add(tuple(sorted(mapping.items())))
    assert len(seen) > 1, "the code assignment is deterministic, so it is reversible"


def test_running_twice_into_the_same_directory_is_refused(tmp_path):
    """A second run assigns different codes. Silently overwriting would leave the
    earlier mapping decoding nothing, with no sign that it had stopped working."""
    db, roster = _build(tmp_path)
    out = tmp_path / "archive"
    deidentify(["--db", str(db), "--roster", str(roster), "--out", str(out)])
    try:
        deidentify(["--db", str(db), "--roster", str(roster), "--out", str(out)])
    except SystemExit as e:
        assert "already exists" in str(e)
    else:
        raise AssertionError("a second run must not overwrite the first")


def test_target_pseudonymisation_is_opt_in(tmp_path):
    """PRD 7 makes it optional: an archive that keeps project ids is still useful
    for per-project analysis, and the presenters are not the anonymous party."""
    db, roster = _build(tmp_path)
    plain, coded = tmp_path / "a", tmp_path / "b"
    deidentify(["--db", str(db), "--roster", str(roster), "--out", str(plain)])
    deidentify(["--db", str(db), "--roster", str(roster), "--out", str(coded),
                "--targets"])
    assert {r["target"] for r in _rows(plain / "archive.csv")} == set(TARGETS)
    assert not {r["target"] for r in _rows(coded / "archive.csv")} & set(TARGETS)


def test_the_script_runs_as_a_command_and_reports_where_things_went(tmp_path):
    """It is run once, by hand, months after anyone last looked at this code."""
    db, roster = _build(tmp_path, n=4)
    out = tmp_path / "archive"
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "deidentify.py"),
         "--db", str(db), "--roster", str(roster), "--out", str(out)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "archive.csv" in result.stdout and "mapping.csv" in result.stdout
    assert "unchanged" in result.stdout
    assert "pseudonymous, not anonymous" in result.stdout


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
