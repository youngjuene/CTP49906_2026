"""CPU logic tests for the run ledger. No GPU, no model, no marimo.

Guards the properties the ledger exists for: idempotence under marimo's
re-execution (the same run appended twice is still one row), the changed-keys
diff that makes "one variable at a time" visible, the control accounting the
course grades on, and the escaping/units discipline in the rendered table --
student prompts reach the renderer unfiltered, and a similarity and a Δ in nats
must never end up under one shared column header.

Run:  python -m pytest avllm_interpretability/tests/test_run_ledger.py
  or:  python avllm_interpretability/tests/test_run_ledger.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> avllm_interpretability/

from src.run_ledger import (  # noqa: E402
    append_run, apply_verdict, build_worksheet_md, ledger_counts, load_log,
    render_ledger_html, run_record,
)


def _record(**kw):
    base = dict(
        kind="band_sweep",
        condition="generated→video [0,12)",
        metric_name="caption similarity",
        metric_value=0.96,
        metric_unit="cos",
        config={"clip": "02321.mp4", "layers": "0-12", "prompt": "describe"},
        prediction="if I block video the caption should lose the saxophone",
    )
    base.update(kw)
    return run_record(**base)


def test_append_does_not_mutate_and_is_idempotent():
    prev = []
    rec = _record()
    once = append_run(prev, rec)
    assert prev == []  # caller's list untouched
    twice = append_run(once, _record())  # marimo re-executed the cell
    assert len(once) == 1 and len(twice) == 1
    assert twice[0]["run_id"] == once[0]["run_id"]
    assert twice[0]["seq"] == once[0]["seq"]  # no renumbering under the student
    assert once[0] is not rec  # record copied in, not aliased


def test_changed_keys_name_the_one_variable_that_moved():
    runs = append_run([], _record())
    assert runs[0]["changed"] == []  # first run of a kind has nothing to differ from
    runs = append_run(runs, _record(config={
        "clip": "02321.mp4", "layers": "12-24", "prompt": "describe",
    }))
    assert runs[1]["changed"] == ["layers"]
    # A different kind diffs against its own family, not the previous row.
    runs = append_run(runs, _record(kind="diversity", metric_name="distinct-2",
                                    metric_value=0.31, metric_unit="ratio"))
    assert runs[2]["changed"] == []


def test_fifo_cap_holds():
    runs = []
    for i in range(10):
        runs = append_run(runs, _record(condition=f"run {i}"), max_runs=4)
    assert len(runs) == 4
    assert [r["condition"] for r in runs] == [f"run {i}" for i in range(6, 10)]


def test_counts_turn_on_whether_the_family_has_a_control():
    runs = append_run([], _record())
    counts = ledger_counts(runs)
    assert counts["n"] == 1 and counts["n_uncontrolled_claims"] == 1
    assert counts["n_controlled"] == 0 and counts["n_unresolved"] == 1
    runs = append_run(runs, _record(condition="silent clip", is_control=True,
                                    config={"clip": "02321_silent.mp4"}))
    counts = ledger_counts(runs)
    assert counts["n_uncontrolled_claims"] == 0  # the family now has a control
    assert counts["n_controlled"] == 2
    # A second family with no control of its own is still counted.
    runs = append_run(runs, _record(kind="diversity", metric_name="distinct-2",
                                    metric_value=0.31, metric_unit="ratio"))
    assert ledger_counts(runs)["n_uncontrolled_claims"] == 1


def test_render_escapes_student_input():
    runs = append_run([], _record(
        condition='<script>alert("x")</script> & more',
        prediction='he said "block it" & it broke <b>',
    ))
    html = render_ledger_html(runs)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html and "&amp;" in html and "&quot;" in html
    assert "alert" in html  # escaped, not dropped


def test_render_keeps_unit_glued_to_value_and_names_each_metric():
    runs = append_run([], _record())
    runs = append_run(runs, _record(kind="teacher_forcing", condition="answer→audio",
                                    metric_name="Σ Δ log-likelihood",
                                    metric_value=-4.2, metric_unit="nats"))
    html = render_ledger_html(runs)
    assert "0.96 cos" in html and "-4.2 nats" in html  # unit adjacent to its value
    header = html.split("</tr>")[0]
    for name in ("caption similarity", "Σ Δ log-likelihood"):
        assert name in html  # the metric's name is visible...
        assert name not in header  # ...in its own row, never as a shared column head
    assert html.count("<th") == html.split("</tr>")[0].count("<th")  # one header row


def test_render_marks_highlighted_runs_and_shouts_about_missing_controls():
    runs = append_run([], _record())
    html = render_ledger_html(runs, highlight_ids=(runs[0]["run_id"],))
    assert "▸" in html
    assert "NO control" in html
    quiet = render_ledger_html(
        append_run(runs, _record(condition="silent", is_control=True,
                                 config={"clip": "silent.mp4"}))
    )
    assert "NO control" not in quiet
    assert render_ledger_html([]).startswith("<div")  # friendly empty state, not ""
    assert "No runs logged yet" in render_ledger_html([])


def test_apply_verdict_hits_one_run_and_ignores_unknown_ids():
    runs = append_run([], _record())
    runs = append_run(runs, _record(condition="other", config={"clip": "b.mp4"}))
    out = apply_verdict(runs, runs[0]["run_id"], "refuted", rival="frames dropped")
    assert out[0]["verdict"] == "refuted" and out[0]["rival"] == "frames dropped"
    assert out[1]["verdict"] == ""
    assert runs[0]["verdict"] == ""  # input list untouched
    assert ledger_counts(out)["n_unresolved"] == 1
    assert apply_verdict(runs, "deadbeef", "supported") == runs  # no-op


def test_worksheet_md_carries_the_prediction_and_flags_the_uncontrolled_run():
    runs = append_run([], _record(prediction="blocking video kills the saxophone"))
    md = build_worksheet_md(runs)
    assert "blocking video kills the saxophone" in md
    assert "No control" in md and "#1" in md
    assert "caption similarity: 0.96 cos" in md
    runs = append_run(runs, _record(condition="silent clip", is_control=True,
                                    config={"clip": "silent.mp4"}))
    assert "No control" not in build_worksheet_md(runs)
    # only_ids narrows the paste to the runs the student wants to report
    assert build_worksheet_md(runs, only_ids=[runs[1]["run_id"]]).count("| 2 |") == 1


def test_log_round_trips_and_an_unwritable_path_never_raises(tmp_path):
    log = tmp_path / "runs.jsonl"
    runs = append_run([], _record(), log_path=str(log))
    append_run(runs, _record(condition="second", config={"clip": "b.mp4"}),
               log_path=str(log))
    back = load_log(str(log))
    assert [r["condition"] for r in back] == [runs[0]["condition"], "second"]
    assert back[0]["run_id"] == runs[0]["run_id"]

    # A truncated final line (kernel killed mid-write) must not lose the rest.
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": "band_sweep"})[:12])
    assert len(load_log(str(log))) == 2

    # A GPU result must survive a broken log destination.
    append_run([], _record(), log_path=str(tmp_path / "no" / "such" / "dir.jsonl"))
    assert load_log(str(tmp_path / "missing.jsonl")) == []


def test_exporting_one_run_does_not_call_a_controlled_run_uncontrolled():
    # A per-run export exists precisely to leave the rest behind. Deriving
    # coverage from the narrowed slice would stamp the alarm banner onto a graded
    # document about a run whose control is sitting in the ledger it came from.
    control = run_record(
        kind="teacher_forcing", condition="answer→audio, silent clip",
        metric_name="delta_per_token", metric_value=-0.004, metric_unit="nats/token",
        config={"clip": "02321_silent.mp4"}, is_control=True,
        prediction="silent should be ~0",
    )
    experiment = run_record(
        kind="teacher_forcing", condition="answer→audio, real clip",
        metric_name="delta_per_token", metric_value=-0.212, metric_unit="nats/token",
        config={"clip": "02321.mp4"}, prediction="real should be far more negative",
    )
    runs = append_run(append_run([], control), experiment)

    whole = build_worksheet_md(runs)
    assert "No control" not in whole

    just_the_experiment = build_worksheet_md(runs, only_ids=[experiment["run_id"]])
    assert experiment["condition"] in just_the_experiment
    assert control["condition"] not in just_the_experiment
    assert "No control" not in just_the_experiment

    # And a genuinely uncontrolled family still raises the banner.
    lone = append_run([], run_record(
        kind="band_sweep", condition="generated→video [0,12)",
        metric_name="caption_similarity", metric_value=0.61, metric_unit="ratio",
        config={"target": "video"}, prediction="early layers matter",
    ))
    assert "No control" in build_worksheet_md(lone)


def test_worksheet_rows_survive_backslashes_and_pipes():
    # A prediction is free text. Escaping the pipe but not the backslash turns
    # `\|` into `\\|` — a literal backslash followed by a live cell break, so the
    # row still splits, one character later.
    nasty = r"block A \| B, then C | D"
    rec = run_record(
        kind="band_sweep", condition="generated→video [0,12)",
        metric_name="caption_similarity", metric_value=0.61, metric_unit="ratio",
        config={"target": "video"}, prediction=nasty,
    )
    md = build_worksheet_md(append_run([], rec))
    row = next(li for li in md.splitlines()
               if li.startswith("|") and "generated" in li)
    # 7 columns -> 8 delimiters, and not one more.
    assert row.count("|") - row.count("\\|") == 8, row
    assert "block A" in row


def test_a_verdict_survives_a_kernel_restart(tmp_path):
    # The ledger's "runs without a verdict" count is the thing the section tells
    # students to drive to zero. Held only in memory, it silently reset to N
    # every time a molab kernel died, while the runs themselves came back.
    log = tmp_path / "lab_log.jsonl"
    rec = run_record(
        kind="band_sweep", condition="generated→video [0,12)",
        metric_name="caption_similarity", metric_value=0.61, metric_unit="ratio",
        config={"target": "video"}, prediction="early layers matter",
    )
    runs = append_run([], rec, log_path=log)
    runs = apply_verdict(runs, rec["run_id"], "refuted",
                         rival="the caption just got shorter", log_path=log)
    assert ledger_counts(runs)["n_unresolved"] == 0

    reseeded = load_log(log)  # what the state cell does on a fresh kernel
    assert len(reseeded) == 1, reseeded
    assert reseeded[0]["verdict"] == "refuted"
    assert reseeded[0]["rival"] == "the caption just got shorter"
    assert ledger_counts(reseeded)["n_unresolved"] == 0


def test_reloading_folds_repeated_ids_last_wins(tmp_path):
    # `run_id` digests kind/condition/config/metric but not the prediction text,
    # so rewording a prediction and re-running writes a second line for the same
    # id. On reload that must be one row carrying the newer text, not two rows
    # with a stale copy on top.
    log = tmp_path / "lab_log.jsonl"

    def rec(prediction):
        return run_record(
            kind="band_sweep", condition="generated→video [0,12)",
            metric_name="caption_similarity", metric_value=0.61,
            metric_unit="ratio", config={"target": "video"}, prediction=prediction,
        )

    first, second = rec("early layers matter"), rec("actually, late layers matter")
    assert first["run_id"] == second["run_id"]
    runs = append_run([], first, log_path=log)
    runs = append_run(runs, second, log_path=log)
    assert len(runs) == 1

    reseeded = load_log(log)
    assert len(reseeded) == 1, reseeded
    assert reseeded[0]["prediction"] == "actually, late layers matter"
    assert reseeded[0]["seq"] == 1


def test_reloading_respects_the_same_cap_as_the_ledger(tmp_path):
    # The JSONL grows unbounded; the in-memory ledger caps at max_runs. Seeding
    # from an uncapped read would start a session longer than the cap allows.
    log = tmp_path / "lab_log.jsonl"
    runs = []
    for i in range(8):
        runs = append_run(runs, run_record(
            kind="band_sweep", condition=f"band {i}", metric_name="caption_similarity",
            metric_value=i / 10, metric_unit="ratio", config={"i": i},
        ), max_runs=4, log_path=log)
    assert len(runs) == 4
    assert len(load_log(log, max_runs=4)) == 4
    assert len(load_log(log, max_runs=None)) == 8


def test_log_mirrors_the_ledger_across_cell_re_execution(tmp_path):
    # The JSONL is what a student reloads after a molab kernel restart, so a
    # re-executed marimo cell must not put a duplicate in it — while a re-run
    # that produced a *different* number must still land as its own row.
    log = tmp_path / "lab_log.jsonl"

    def rec(value):
        return run_record(
            kind="teacher_forcing", condition="answer→audio [0,36)",
            metric_name="delta_per_token", metric_value=value,
            metric_unit="nats/token", config={"clip": "02321.mp4"},
        )

    runs = append_run([], rec(-0.2), log_path=log)
    runs = append_run(runs, rec(-0.2), log_path=log)
    assert len(runs) == 1 and len(load_log(log)) == 1

    runs = append_run(runs, rec(-0.9), log_path=log)
    assert len(runs) == 2 and len(load_log(log)) == 2


if __name__ == "__main__":
    import tempfile

    fns = [
        v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)
    ]
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
