"""CPU logic tests for the ledger's parallel-coordinates rows. No GPU, no model.

`ledger_pc_rows` feeds `wigglystuff.ParallelCoordinates`. What has to hold:

* rows are JSON-native with identical keys (the widget syncs them to the
  browser; a numpy scalar, a `None`, or a ragged row breaks the plot),
* one plot is one kind and one metric -- a similarity ratio and a Δ in nats never
  share an axis, and runs left out are counted, not silently dropped,
* the "대조군 짝" column agrees with `matched_control_ids`, the rule the ledger
  table and the worksheet already use,
* an unusual run (student-invented kind, baseline-only diversity, NaN metric)
  degrades to a plot rather than an exception.

Run:  python -m pytest avllm_interpretability/tests/test_run_ledger_pc.py
"""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> avllm_interpretability/

from src.run_ledger import (  # noqa: E402
    append_run, apply_verdict, ledger_pc_rows, matched_control_ids, run_record,
)

PAIR_KEY = "builtin-02321-av-pair-v1"


def _config(clip="02321.mp4", sha="a", **kw):
    base = {
        "clip": clip, "clip_sha256": sha, "comparison_key": PAIR_KEY,
        "nframes": 8, "prompt": "p", "target": "video", "start": 0, "end": 12,
        "max_new_tokens": 32,
    }
    base.update(kw)
    return base


def _band(value, *, control=False, **cfg):
    clip, sha = ("02321_silent.mp4", "b") if control else ("02321.mp4", "a")
    start, end = cfg.get("start", 0), cfg.get("end", 12)
    return run_record(
        kind="band_sweep", condition=f"generated→video [{start},{end})",
        metric_name="caption_similarity", metric_value=value, metric_unit="ratio",
        config=_config(clip=clip, sha=sha, **cfg), is_control=control,
    )


def _ledger(*records):
    runs = []
    for record in records:
        runs = append_run(runs, record)
    return runs


def test_rows_are_json_native_uniform_and_traceable_to_the_ledger():
    runs = _ledger(_band(0.91), _band(0.40, start=12, end=24), _band(0.97, start=24, end=36))
    rows, ids, other = ledger_pc_rows(runs, "band_sweep")
    assert other == 0
    assert ids == [r["run_id"] for r in runs]
    assert len({tuple(sorted(r)) for r in rows}) == 1, "ragged rows"
    json.dumps(rows)  # what the widget does when it syncs
    assert all(v is not None for row in rows for v in row.values())
    assert [r["시작 레이어"] for r in rows] == [0, 12, 24]
    assert [r["차단 레이어 수"] for r in rows] == [12, 12, 12]
    assert [r["캡션 유사도 (비율)"] for r in rows] == [0.91, 0.40, 0.97]


def test_matching_the_ledgers_own_control_rule():
    paired = _band(0.80)
    control = _band(0.99, control=True)
    lonely = _band(0.55, start=12, end=24)  # no silent twin for this band
    runs = _ledger(paired, control, lonely)
    assert matched_control_ids(paired, runs) == [control["run_id"]]
    rows, _, _ = ledger_pc_rows(runs, "band_sweep")
    assert [(r["소리"], r["대조군 짝"]) for r in rows] == [
        ("원본", "짝 있음"), ("무음", "대조군"), ("원본", "짝 없음"),
    ]


def test_metrics_are_never_mixed_and_left_out_runs_are_counted():
    tf = run_record(
        kind="teacher_forcing", condition="answer→video [0,12)",
        metric_name="delta_per_token", metric_value=-0.31, metric_unit="nats/token",
        config=_config(),
    )
    div_delta = run_record(
        kind="diversity", condition="audio→video [0,36)",
        metric_name="mean_delta_diversity", metric_value=-4.2, metric_unit="unique preds",
        config={"clip": "02321.mp4", "nframes": 8, "prompt": "p",
                "rules": [["audio", "video", 0, 36]], "compare": True},
    )
    div_base = run_record(
        kind="diversity", condition="기준선 (연결 차단 없음)",
        metric_name="mean_unique_per_layer", metric_value=88.5, metric_unit="unique preds",
        config={"clip": "02321.mp4", "nframes": 8, "prompt": "p", "rules": [], "compare": False},
    )
    runs = _ledger(_band(0.9), tf, div_delta, div_base)

    rows, _, other = ledger_pc_rows(runs, "band_sweep")
    assert other == 0 and len(rows) == 1
    assert "Δ 로그 확률 (nats/토큰)" not in rows[0]  # a different kind's metric

    # Within diversity there are two metrics; by default the newest run's wins
    # and the other is reported rather than plotted on the same axis.
    rows, ids, other = ledger_pc_rows(runs, "diversity")
    assert [r for r in rows[0] if "레이어 평균" in r] == ["다양성 (레이어 평균)"]
    assert other == 1 and ids == [div_base["run_id"]]
    rows, _, other = ledger_pc_rows(runs, "diversity", metric_name="mean_delta_diversity")
    assert [r for r in rows[0] if "레이어 평균" in r] == ["다양성 Δ (레이어 평균)"]
    assert other == 1


def test_baseline_only_diversity_run_is_a_zero_layer_row_not_an_error():
    base = run_record(
        kind="diversity", condition="기준선 (연결 차단 없음)",
        metric_name="mean_unique_per_layer", metric_value=88.5, metric_unit="unique preds",
        config={"clip": "scene02.mp4", "nframes": 4, "prompt": "p", "rules": [], "compare": False},
    )
    rows, _, _ = ledger_pc_rows(_ledger(base), "diversity")
    assert rows[0]["타깃"] == "없음" and rows[0]["소스"] == "없음"
    assert (rows[0]["시작 레이어"], rows[0]["차단 레이어 수"]) == (0, 0)
    assert rows[0]["장면"] == "장면 2"


def test_several_diversity_rules_span_their_union():
    rec = run_record(
        kind="diversity", condition="audio→video [0,12) + audio→query_text [6,36)",
        metric_name="mean_delta_diversity", metric_value=-1.0, metric_unit="unique preds",
        config={"clip": "02321.mp4", "nframes": 8, "prompt": "p",
                "rules": [["audio", "video", 0, 12], ["audio", "query_text", 6, 36]], "compare": True},
    )
    rows, _, _ = ledger_pc_rows(_ledger(rec), "diversity")
    assert rows[0]["타깃"] == "query_text+video"
    assert (rows[0]["시작 레이어"], rows[0]["차단 레이어 수"]) == (0, 36)


def test_verdicts_use_the_ledgers_korean_labels_and_unresolved_is_explicit():
    runs = _ledger(_band(0.8), _band(0.5, start=12, end=24))
    runs = apply_verdict(runs, runs[0]["run_id"], "refuted", "다른 설명")
    rows, _, _ = ledger_pc_rows(runs, "band_sweep")
    assert [r["판정"] for r in rows] == ["반박됨", "미판정"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, None, "0.9"])
def test_unplottable_metric_values_are_skipped_not_fatal(bad):
    record = _band(0.8)
    record = dict(record, metric_value=bad, run_id="bad00000")
    good = _band(0.5)
    rows, ids, other = ledger_pc_rows(_ledger(good, record), "band_sweep")
    assert ids == [good["run_id"]] and other == 0
    assert rows[0]["캡션 유사도 (비율)"] == 0.5 and not math.isnan(rows[0]["캡션 유사도 (비율)"])


def test_a_student_invented_kind_and_metric_still_plots():
    rec = run_record(
        kind="my_probe", condition="자유 탐색", metric_name="entropy", metric_value=2.5,
        metric_unit="bits", config={"clip": "upload_0123456789ab_my.mp4"},
    )
    rows, _, _ = ledger_pc_rows(_ledger(rec), "my_probe")
    assert rows[0]["entropy (bits)"] == 2.5
    assert rows[0]["장면"] == "업로드"
    assert "프레임 수" not in rows[0]  # not recorded, so no axis rather than a fake 0


def test_empty_ledger_or_unknown_kind_returns_nothing():
    assert ledger_pc_rows([], "band_sweep") == ([], [], 0)
    assert ledger_pc_rows(_ledger(_band(0.9)), "teacher_forcing") == ([], [], 0)


def test_rows_build_a_real_widget():
    pytest.importorskip("wigglystuff", reason="needs the classroom widget package")
    from wigglystuff import ParallelCoordinates

    runs = _ledger(_band(0.9), _band(0.4, start=12, end=24), _band(0.99, control=True))
    rows, _, _ = ledger_pc_rows(runs, "band_sweep")
    widget = ParallelCoordinates(rows, color_by="대조군 짝", height=360)
    assert len(widget.data) == 3 and widget.filtered_indices == [0, 1, 2]
    widget.filtered_uids = ["1", "2"]  # what the browser sends after a Keep
    assert [d["시작 레이어"] for d in widget.filtered_data] == [12, 0]
