"""Regressions for control matching, durable history and reviewable exports."""

import json
import re

import pytest

from src import run_ledger as ledger


def record(*, clip="02321.mp4", control=False, config=None, **overrides):
    settings = {
        "comparison_key": "builtin-02321-av-pair-v1",
        "clip": clip,
        "clip_sha256": "silent-hash" if control else "original-hash",
        "nframes": 8,
        "prompt": "Describe sound",
        "target": "audio",
        "start": 0,
        "end": 36,
        "rules": [["generated", "audio", 0, 36]],
        "max_new_tokens": 32,
        "model_id": "Qwen/Omni",
        "model_revision": "model-commit",
        "repo_revision": "repo-commit",
    }
    settings.update(config or {})
    fields = dict(
        kind="teacher_forcing", condition="answer→audio [0,36)",
        metric_name="delta_per_token", metric_value=0.004 if control else -0.212,
        metric_unit="nats/token", config=settings, is_control=control,
    )
    fields.update(overrides)
    return ledger.run_record(**fields)


@pytest.mark.parametrize("changed", [
    {"prompt": "Describe visuals"}, {"nframes": 4}, {"target": "video"},
    {"start": 12}, {"end": 24}, {"rules": []}, {"max_new_tokens": 8},
    {"model_id": "other/model"}, {"model_revision": "different"},
    {"repo_revision": "different"}, {"comparison_key": "another-clip-pair"},
])
def test_a_control_with_different_settings_leaves_the_claim_unmatched(changed):
    experiment = record()
    control = record(clip="02321_silent.mp4", control=True, config=changed)
    assert ledger.ledger_counts([experiment, control])["n_uncontrolled_claims"] == 1


def test_matching_exposes_exact_control_ids_without_using_clip_hash_or_metric():
    experiment = record()
    control = record(clip="02321_silent.mp4", control=True)
    assert ledger.matched_control_ids(experiment, [experiment, control]) == [control["run_id"]]
    assert ledger.ledger_counts([experiment, control])["n_controlled"] == 2
    assert ledger.matched_control_ids(experiment, [experiment]) == []


@pytest.mark.parametrize("clip,condition", [
    ("02321.mp4", "answer→audio [12,36)"),
    ("uploaded-holiday.mp4", "answer→audio [0,36)"),
])
def test_legacy_records_do_not_pair_different_conditions_or_arbitrary_uploads(clip, condition):
    experiment = record(config={"comparison_key": None}, clip=clip, condition=condition)
    control = record(config={"comparison_key": None}, clip="02321_silent.mp4", control=True)
    for run in (experiment, control):
        run["config"].pop("clip_sha256")
    assert ledger.ledger_counts([experiment, control])["n_uncontrolled_claims"] == 1


def test_legacy_pair_requires_all_non_clip_settings_and_does_not_self_match():
    experiment = record(config={"clip": "02321.mp4"})
    control = record(clip="02321_silent.mp4", control=True)
    for run in (experiment, control):
        run["config"].pop("comparison_key")
        run["config"].pop("clip_sha256")
    assert ledger.matched_control_ids(experiment, [experiment, control]) == [control["run_id"]]
    declared_null = record(control=True)
    assert ledger.matched_control_ids(declared_null, [declared_null]) == []
    same_clip = record(control=True, metric_value=0.01)
    assert ledger.matched_control_ids(record(), [same_clip]) == []


def test_history_and_default_reload_retain_the_earliest_control_after_40_runs(tmp_path):
    path = tmp_path / "runs.jsonl"
    control = record(clip="02321_silent.mp4", control=True)
    runs = ledger.append_run([], control, log_path=path)
    for value in range(45):
        runs = ledger.append_run(runs, record(metric_value=value), log_path=path)
    assert len(runs) == 46
    assert runs[0]["run_id"] == control["run_id"]
    assert len(ledger.load_log(path)) == 46
    assert ledger.ledger_counts(runs)["n_uncontrolled_claims"] == 0


def test_failed_writes_retain_results_report_error_and_retry_identical_runs(tmp_path):
    path = tmp_path / "missing" / "runs.jsonl"
    original = record(extra={"caption": "A saxophone plays."})
    runs = ledger.append_run([], original, log_path=path)
    assert runs[0]["extra"]["caption"] == "A saxophone plays."
    assert runs[0]["save_status"]["state"] == "failed"
    assert runs[0]["save_status"]["error"]
    assert runs[0]["save_status"]["path"] == str(path)
    path.parent.mkdir()
    recovered = ledger.append_run(runs, original, log_path=path)
    assert recovered[0]["save_status"]["state"] == "saved"
    assert len(ledger.load_log(path)) == 1
    ledger.append_run(recovered, original, log_path=path)
    assert len(path.read_text().splitlines()) == 1


@pytest.mark.parametrize("changed_evidence", [
    {"caption": "A violin plays."},
    {"token_ids": [7, 14]},
    {"deltas": [-0.124, -0.300]},
])
def test_distinct_scientific_evidence_never_overwrites_a_prior_run_or_its_verdict(changed_evidence):
    evidence = {"caption": "A saxophone plays.", "token_ids": [7, 13], "deltas": [-0.123, -0.301]}
    first = record(extra=evidence)
    second = record(extra={**evidence, **changed_evidence})
    runs = ledger.append_run([], first)
    runs = ledger.apply_verdict(runs, first["run_id"], "supported", "alternative cause")
    runs = ledger.append_run(runs, second)
    assert len(runs) == 2
    assert first["run_id"] != second["run_id"]
    assert runs[0]["extra"] == evidence
    assert runs[0]["verdict"] == "supported"
    assert runs[1]["extra"] == {**evidence, **changed_evidence}
    assert runs[1]["verdict"] == ""


def test_annotation_edits_preserve_identity_and_verdict_for_the_same_scientific_evidence():
    evidence = {"caption": "A saxophone plays.", "token_ids": [7, 13], "deltas": [-0.123, -0.301]}
    original = record(extra=evidence, prediction="first prediction", note="first note")
    revised = record(extra=evidence, prediction="reworded prediction", note="reworded note", is_control=True)
    assert original["run_id"] == revised["run_id"]
    runs = ledger.append_run([], original)
    runs = ledger.apply_verdict(runs, original["run_id"], "untested", "alternative cause")
    updated = ledger.append_run(runs, revised)
    assert len(updated) == 1
    assert updated[0]["prediction"] == "reworded prediction"
    assert updated[0]["note"] == "reworded note"
    assert updated[0]["is_control"] is True
    assert updated[0]["verdict"] == "untested"
    assert updated[0]["rival"] == "alternative cause"


@pytest.mark.parametrize("tail", ['{"run_id":"crash', ''])
def test_append_after_an_unterminated_log_preserves_existing_and_new_runs(tmp_path, tail):
    path = tmp_path / "runs.jsonl"
    first, second = record(), record(metric_value=-0.3)
    runs = ledger.append_run([], first, log_path=path)
    if tail:
        with path.open("a") as stream:
            stream.write(tail)
    else:
        # An intact JSON record without its final newline also needs separation.
        path.write_text(path.read_text().rstrip("\n"))
    updated = ledger.append_run(runs, second, log_path=path)
    recovered = ledger.load_log(path)
    assert updated[1]["save_status"]["state"] == "saved"
    assert [run["run_id"] for run in recovered] == [first["run_id"], second["run_id"]]
    if tail:
        assert tail + "\n" in path.read_text()  # preserve the corrupt fragment for inspection


def test_verdict_validation_reports_unknown_id_and_invalid_values(tmp_path):
    runs = ledger.append_run([], record(prediction="Blocking audio lowers caption likelihood"))
    for run_id, verdict, error in [
        ("not-a-run", "supported", "unknown_run_id"),
        (runs[0]["run_id"], "guess", "invalid_verdict"),
    ]:
        updated, status = ledger.apply_verdict_checked(runs, run_id, verdict)
        assert updated == runs
        assert status["ok"] is False
        assert status["error"] == error
    updated, status = ledger.apply_verdict_checked(
        runs, runs[0]["run_id"], "refuted", rival="caption content changed",
        log_path=tmp_path / "missing" / "runs.jsonl",
    )
    assert updated[0]["verdict"] == "refuted"
    assert updated[0]["rival"] == "caption content changed"
    assert status["ok"] is True  # accepted in memory; durable status is separate
    assert status["save_status"]["state"] == "failed"
    assert runs[0]["verdict"] == ""


@pytest.mark.parametrize("verdict", ["supported", "refuted"])
@pytest.mark.parametrize("claim", [None, "", " \n "])
def test_checked_conclusive_verdict_requires_a_written_claim(verdict, claim):
    runs = ledger.append_run([], record())
    updated, status = ledger.apply_verdict_checked(runs, runs[0]["run_id"], verdict, claim=claim)
    assert updated == runs
    assert status["ok"] is False
    assert status["error"] == "missing_claim"


def test_checked_claim_survives_reload_and_full_exports_without_changing_identity(tmp_path):
    path = tmp_path / "runs.jsonl"
    original = record(extra={"caption": "색소폰이 들립니다", "token_ids": [7, 13]})
    claim = "오디오 경로 차단 후 고정 캡션의 우도가 낮아졌다."
    runs = ledger.append_run([], original, log_path=path)
    updated, status = ledger.apply_verdict_checked(
        runs, original["run_id"], "supported", rival="토큰별 차이를 더 확인해야 함",
        log_path=path, claim=claim,
    )
    assert status["ok"] is True and status["save_status"]["state"] == "saved"
    assert updated[0]["run_id"] == original["run_id"]
    assert updated[0]["extra"] == original["extra"]
    assert runs[0]["prediction"] == ""
    recovered = ledger.load_log(path)
    assert recovered[0]["prediction"] == claim
    assert recovered[0]["verdict"] == "supported"
    assert claim in ledger.build_worksheet_md(recovered)
    assert json.loads(ledger.build_evidence_json(recovered))["runs"][0]["prediction"] == claim


def test_identical_rerun_preserves_a_claim_and_explicit_prediction_can_replace_it():
    original = record()
    runs = ledger.append_run([], original)
    runs, _ = ledger.apply_verdict_checked(runs, original["run_id"], "refuted", claim="관측한 차이가 없다")
    rerun = ledger.append_run(runs, record())
    assert len(rerun) == 1
    assert rerun[0]["prediction"] == "관측한 차이가 없다"
    assert rerun[0]["verdict"] == "refuted"
    revised = ledger.append_run(rerun, record(prediction="수정한 주장"))
    assert len(revised) == 1
    assert revised[0]["prediction"] == "수정한 주장"


def test_checked_verdict_can_reuse_existing_prediction_and_leave_untested_blank():
    runs = ledger.append_run([], record(prediction="Previously written claim"))
    updated, status = ledger.apply_verdict_checked(runs, runs[0]["run_id"], "supported", claim="")
    assert status["ok"] is True
    assert updated[0]["prediction"] == "Previously written claim"
    blank = ledger.append_run([], record())
    untested, status = ledger.apply_verdict_checked(blank, blank[0]["run_id"], "untested", claim="")
    assert status["ok"] is True and untested[0]["prediction"] == ""
    # The English notebook's older caller remains deliberately permissive.
    assert ledger.apply_verdict(blank, blank[0]["run_id"], "supported")[0]["verdict"] == "supported"


def test_full_json_export_preserves_evidence_and_provenance_without_mutation():
    experiment = record(
        prediction="A small drop", note="Use fixed settings",
        extra={"caption": "색소폰 연주", "token_ids": [7, 13], "deltas": [-0.123456789, 0.2]},
    )
    control = record(clip="02321_silent.mp4", control=True)
    runs = ledger.append_run(ledger.append_run([], control), experiment)
    runs = ledger.apply_verdict(runs, experiment["run_id"], "untested", "different captions")
    provenance = {"runtime": "CPU regression", "versions": {"python": "3.x"}}
    exported = json.loads(ledger.build_evidence_json(runs, provenance=provenance))
    assert exported["provenance"] == provenance
    assert len(exported["runs"]) == 2
    evidence = exported["runs"][1]
    for key, value in runs[1].items():
        assert evidence[key] == value
    assert evidence["matched_control_ids"] == [control["run_id"]]
    assert "matched_control_ids" not in runs[1]


def test_markdown_contains_complete_run_details_with_safe_fenced_evidence():
    attack = '<script>alert("x")</script> [link](javascript:alert(1)) | `code`\n```\n# fake'
    experiment = record(prediction=attack, note="note retained", extra={"caption": attack})
    control = record(clip="02321_silent.mp4", control=True)
    runs = ledger.append_run(ledger.append_run([], control), experiment)
    md = ledger.build_worksheet_md(runs, only_ids=[experiment["run_id"]])
    assert "No control" not in md
    assert control["run_id"] in md
    assert experiment["run_id"] in md and "teacher_forcing" in md
    # The fenced JSON is lossless evidence and cannot be closed by student input.
    match = re.search(r"(?m)^(`{3,})json\n(.*?)\n\1$", md, re.S)
    assert match is not None
    evidence = json.loads(match.group(2))
    assert evidence["config"] == experiment["config"]
    assert evidence["prediction"] == attack
    assert evidence["extra"]["caption"] == attack
    assert evidence["note"] == "note retained"
    summary = md[:match.start()]
    assert "<script>" not in summary
    assert "[link](javascript:" not in summary


def test_korean_renderer_shows_matching_and_failed_save_status(tmp_path):
    runs = ledger.append_run([], record(), log_path=tmp_path / "absent" / "runs.jsonl")
    html = ledger.render_ledger_html(runs, lang="ko")
    assert "저장 실패" in html
    assert "대조군" in html and "일치" in html
    assert "NO control" not in html


def test_markdown_escapes_untrusted_sequence_labels_from_imported_logs():
    run = record()
    run["seq"] = '<img src="x" onerror="alert(1)">'
    md = ledger.build_worksheet_md([run])
    summary = md.split("```json", 1)[0]
    assert "<img" not in summary
    assert "&lt;img" in summary


def test_html_does_not_present_declared_control_as_a_validated_null():
    html = ledger.render_ledger_html(ledger.append_run([], record(control=True)))
    assert "declared control" in html
    assert "matched settings" in html
