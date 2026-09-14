"""Run ledger: the notebook's memory of what was tried, and what it settled.

Deliberately **not** a widget. In marimo, a synced trait that changes from the
browser re-runs every cell that refers to the widget, so a ledger the GPU cells
pushed rows into would let one chip click re-trigger a 60-second generation.
The ledger is therefore plain data (a list of dicts living in one cell) plus
pure functions that render it to an HTML string for `mo.Html` -- the same shape
`teacher_forcing.render_delta_strip` uses.

Why it exists at all: the course's success criterion is "zero submissions
claiming an effect without a control". That is a property of a *history* of
runs, not of any single run, so something has to remember them. Two consequences
shape the data model:

* `metric_name`, `metric_value` and `metric_unit` travel together, always. A 96%
  caption similarity and a -4.2 nat Σ are incommensurable; stacking them under
  one "headline metric" column is precisely the error the lab teaches against,
  so the renderer keeps the name and unit welded to the number.
* Each record carries the `config` keys that moved since the previous run of the
  same kind, which is what makes "change one variable at a time" visible rather
  than merely advised.

Nothing heavy is imported: the module must load on a CPU-only Python 3.10
checkout with no torch, marimo or matplotlib installed.
"""

import hashlib
import json
import re
from html import escape

# The vocabulary the notebook's dropdowns offer. Not enforced here -- a record
# with an unlisted kind still renders -- because a student inventing a fourth
# experiment family should not hit a validation error mid-lab.
KINDS = ("band_sweep", "diversity", "teacher_forcing")
VERDICTS = ("supported", "refuted", "untested")

# Neutral, theme-agnostic surfaces. marimo ships both a light and a dark theme
# and the ledger is rendered inside it, so every color here is either
# `currentColor` or a translucent gray that darkens light backgrounds and
# lightens dark ones. A hardcoded white background would strand the text.
_HAIRLINE = "1px solid rgba(127,127,127,0.35)"
_SOFT_BG = "rgba(127,127,127,0.10)"
_ALARM = "#e5484d"  # readable on both themes; used only when a claim has no control


def _canonical(payload):
    """Stable JSON for hashing: sorted keys, no incidental whitespace."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def run_record(
    *,
    kind,
    condition,
    metric_name,
    metric_value,
    metric_unit,
    config,
    prediction="",
    is_control=False,
    note="",
    extra=None,
):
    """Build one immutable run record.

    `run_id` is a content digest rather than a counter because marimo re-executes
    cells freely: the same experiment re-run must produce the same id so
    `append_run` can replace instead of appending a duplicate. `seq` and
    `changed` are placeholders here -- both are positional facts that only
    `append_run` can know.

    `extra` contains scientific result evidence (captions, scored token IDs,
    token-level deltas, distributions) and contributes to the digest. Different
    evidence must never overwrite a prior result just because its headline
    metric agrees. Keep annotations in `prediction`, `note`, `is_control`,
    `verdict` and `rival`; those can change without creating a new experiment.
    """
    config = dict(config or {})
    extra = dict(extra or {})
    identity = {
        "kind": kind,
        "condition": condition,
        "config": config,
        "metric": [metric_name, metric_value, metric_unit],
    }
    # Preserve legacy IDs for runs with no extra evidence. Existing log IDs
    # remain readable; new evidence-bearing runs receive distinct identities.
    if extra:
        identity["extra"] = extra
    run_id = hashlib.sha1(
        _canonical(identity).encode("utf-8")
    ).hexdigest()[:8]
    return {
        "run_id": run_id,
        "seq": 0,
        "kind": kind,
        "condition": condition,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "metric_unit": metric_unit,
        "config": config,
        "prediction": prediction,
        "is_control": bool(is_control),
        "note": note,
        "extra": extra,
        "changed": [],
        "verdict": "",
        "rival": "",
    }


def _changed_keys(record, prior):
    """Config keys that moved between `prior` and `record` (added/removed count)."""
    if prior is None:
        return []
    old, new = prior.get("config", {}), record.get("config", {})
    return sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))


def _persist_record(record, log_path):
    """Retain the result and expose whether this version reached the local log."""
    record["save_status"] = {"state": "memory_only", "path": "", "error": ""}
    if log_path is None:
        return
    record["save_status"] = {"state": "saved", "path": str(log_path), "error": ""}
    try:
        with open(log_path, "ab+") as fh:
            # A killed kernel may leave a fragment without its final newline.
            # Retain that fragment for inspection, but never concatenate a new
            # valid record onto it: load_log would discard both as one bad row.
            fh.seek(0, 2)
            needs_separator = False
            if fh.tell():
                fh.seek(-1, 2)
                needs_separator = fh.read(1) != b"\n"
            payload = ("\n" if needs_separator else "") + _canonical(record) + "\n"
            fh.write(payload.encode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - keep the expensive result in memory
        record["save_status"] = {
            "state": "failed", "path": str(log_path),
            "error": f"{type(exc).__name__}: {exc}",
        }


def append_run(prev, record, max_runs=None, log_path=None):
    """Return a new run list with `record` added (or refreshed in place).

    Idempotent by `run_id`: a re-executed marimo cell hands us the same digest,
    and replacing keeps the original `seq` so the ledger does not renumber
    itself under the student. `changed` is computed against the most recent run
    of the same kind *that precedes this one*, so it too survives re-execution.
    All unique runs are retained by default. An explicit `max_runs` limits only
    the returned list; the JSONL remains append-only. Persistence failure is
    returned in `record['save_status']` and never discards the experiment.
    """
    runs = [dict(r) for r in prev]
    record = dict(record)

    index = next(
        (i for i, r in enumerate(runs) if r.get("run_id") == record.get("run_id")), None
    )
    before = runs[:index] if index is not None else runs
    same_kind = [r for r in before if r.get("kind") == record.get("kind")]
    record["changed"] = _changed_keys(record, same_kind[-1] if same_kind else None)

    if index is not None:
        record["seq"] = runs[index].get("seq", index + 1)
        # A claim may be written after the result in the verdict form. Fresh
        # records usually have an empty prediction; preserve the annotation on
        # an identical rerun unless a new nonblank prediction was supplied.
        if not str(record.get("prediction") or "").strip():
            record["prediction"] = runs[index].get("prediction", "")
        # `verdict` and `rival` are resolved *after* the run, by the verdict form.
        # A re-executed cell mints a fresh record with both empty, so overwriting
        # blindly would destroy the student's own conclusion on any upstream edit,
        # Run-all, or molab reconnect. Carrying them across also restores the
        # equality below, which is what keeps a re-execution from writing a second
        # JSONL line for the same run.
        for _post_hoc in ("verdict", "rival"):
            if not record.get(_post_hoc):
                record[_post_hoc] = runs[index].get(_post_hoc, "")
        # A digest match means the same kind/condition/config produced the same
        # metric and scientific extra evidence. Annotation fields (prediction,
        # control flag, note)
        # can still differ -- rewording a prediction and pressing ▶ again lands
        # here -- so compare the whole record, and log only when something
        # actually changed. `load_log` folds by `run_id`, last-wins, so a second
        # line for an amended run reloads as one row rather than two.
        previous = runs[index]
        unchanged = (
            {k: v for k, v in previous.items() if k != "save_status"}
            == {k: v for k, v in record.items() if k != "save_status"}
        )
        status = previous.get("save_status", {})
        already_logged = unchanged and (
            log_path is None or (
                status.get("state") == "saved" and status.get("path") == str(log_path)
            )
        )
        if already_logged:
            record["save_status"] = dict(status) or {
                "state": "memory_only", "path": "", "error": "",
            }
        runs[index] = record
    else:
        already_logged = False
        record["seq"] = max((r.get("seq", 0) for r in runs), default=0) + 1
        runs.append(record)
    if max_runs and len(runs) > max_runs:
        runs = runs[-max_runs:]

    if not already_logged:
        _persist_record(record, log_path)
    return runs


def apply_verdict(prev, run_id, verdict, rival="", log_path=None):
    """Set `verdict`/`rival` on one run; unknown ids are a no-op.

    `log_path` re-appends the amended record. A verdict is the student's own
    conclusion and the thing the ledger's "runs without a verdict" count is
    about, so leaving it only in memory made that count reset to N on every
    kernel restart while the runs themselves came back. `load_log` folds by
    `run_id` keeping the last occurrence, which is what makes the re-append the
    authoritative copy rather than a duplicate row.
    """
    if not any(r.get("run_id") == run_id for r in prev):
        return prev
    runs = []
    amended = None
    for r in prev:
        r = dict(r)
        if r.get("run_id") == run_id:
            r["verdict"] = verdict
            r["rival"] = rival
            amended = r
        runs.append(r)
    if amended is not None:
        _persist_record(amended, log_path)
    return runs


def apply_verdict_checked(prev, run_id, verdict, rival="", log_path=None, *, claim=None):
    """Return `(runs, status)` with explicit validation for notebook feedback.

    `ok` means the verdict was accepted in memory. Check `save_status` separately
    before claiming local persistence. `apply_verdict` retains its legacy no-op
    behavior for unknown IDs; new interfaces should use this checked helper.
    A nonblank `claim` updates the existing `prediction` annotation without
    changing scientific identity. Omitted or blank claims reuse that annotation.
    Supported/refuted verdicts require a written claim; untested may stay blank.
    """
    status = {"ok": False, "run_id": run_id, "error": "", "save_status": None}
    original = next((r for r in prev if r.get("run_id") == run_id), None)
    if original is None:
        status["error"] = "unknown_run_id"
        return prev, status
    if verdict not in VERDICTS:
        status["error"] = "invalid_verdict"
        return prev, status
    supplied_claim = str(claim).strip() if claim is not None else ""
    effective_claim = supplied_claim or str(original.get("prediction") or "")
    if verdict in ("supported", "refuted") and not effective_claim.strip():
        status["error"] = "missing_claim"
        return prev, status
    annotated = [
        {**r, "prediction": effective_claim} if r.get("run_id") == run_id else r
        for r in prev
    ]
    runs = apply_verdict(annotated, run_id, verdict, rival, log_path=log_path)
    amended = next(r for r in runs if r.get("run_id") == run_id)
    status.update(ok=True, save_status=dict(amended["save_status"]))
    return runs, status


def matched_control_ids(run, runs):
    """IDs of declared controls with matching recorded settings, never a verdict.

    New records identify a deliberate clip pair with `config.comparison_key`.
    Kind, condition and every other config field must still agree, except the
    clip filename and its SHA-256 (the original and silent files differ).
    Legacy keyless records may pair only 02321.mp4 with 02321_silent.mp4 and
    require identical remaining config. Arbitrary uploads are never inferred to
    have a control. This checks recorded settings, not scientific validity or
    whether a declared control actually had a null effect.
    """
    if run.get("is_control"):
        return []
    config = run.get("config") or {}
    comparison_key = config.get("comparison_key")
    ignored = {"clip", "clip_sha256"} if comparison_key else {"clip"}
    settings = {k: v for k, v in config.items() if k not in ignored}
    matches = []
    for control in runs:
        if not control.get("is_control") or control.get("run_id") == run.get("run_id"):
            continue
        if (run.get("kind"), run.get("condition")) != (
            control.get("kind"), control.get("condition")
        ):
            continue
        other = control.get("config") or {}
        if comparison_key:
            if comparison_key != other.get("comparison_key"):
                continue
            if not config.get("clip") or not other.get("clip") or config["clip"] == other["clip"]:
                continue
        elif (
            other.get("comparison_key")
            or config.get("clip") != "02321.mp4"
            or other.get("clip") != "02321_silent.mp4"
        ):
            continue
        if _canonical(settings) != _canonical({
            k: v for k, v in other.items() if k not in ignored
        }):
            continue
        control_id = control.get("run_id")
        if control_id and control_id not in matches:
            matches.append(control_id)
    return matches


def ledger_counts(runs):
    """Count declared controls and runs participating in matched settings pairs."""
    matches = [matched_control_ids(r, runs) for r in runs]
    used_controls = {run_id for ids in matches for run_id in ids}
    return {
        "n": len(runs),
        "n_controls": sum(bool(r.get("is_control")) for r in runs),
        "n_matched_claims": sum(bool(ids) for ids in matches),
        "n_controlled": sum(
            bool(ids) or r.get("run_id") in used_controls for r, ids in zip(runs, matches)
        ),
        "n_uncontrolled_claims": sum(
            1
            for r, ids in zip(runs, matches)
            if not r.get("is_control") and not ids
        ),
        "n_unresolved": sum(1 for r in runs if not r.get("verdict")),
    }


def _esc(value):
    return escape(str(value), quote=True)


def _fmt_value(value):
    """Numbers short enough to scan, without lying about precision."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _fmt_metric(record):
    """`value unit` as one unbreakable phrase -- the unit never drifts away."""
    unit = str(record.get("metric_unit", "") or "").strip()
    shown = _fmt_value(record.get("metric_value", ""))
    return _esc(f"{shown} {unit}".strip())


def _truncate(text, limit=52):
    text = str(text or "")
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def render_ledger_html(runs, highlight_ids=(), lang="en"):
    """Compact HTML table of the ledger, for `mo.Html`.

    Everything is escaped: `condition` and `prediction` are student input that
    reaches this function straight from a text box. Styling is inline only --
    a `<style>` block or class names would collide with marimo's own CSS.
    """
    ko = lang == "ko"

    def label(english, korean):
        return korean if ko else english

    if not runs:
        return (
            f'<div style="border:{_HAIRLINE};border-radius:6px;padding:10px 12px;'
            f'background:{_SOFT_BG};font-size:0.9em;opacity:0.85">'
            + "<em>" + label(
                "No runs logged yet. Write a prediction, press ▶, and the run "
                "lands here — with or without a control.",
                "아직 실행 기록이 없습니다. 실행하면 결과와 설정, 대조군 연결 여부가 여기에 표시됩니다.",
            ) + "</em></div>"
        )

    counts = ledger_counts(runs)
    highlight = set(highlight_ids or ())

    summary = label(
        f'{counts["n"]} run(s) · {counts["n_controlled"]} with matched settings '
        f'· {counts["n_unresolved"]} still unresolved',
        f'실행 {counts["n"]}개 · 대조군과 설정이 일치하는 실행 {counts["n_controlled"]}개 '
        f'· 미판정 {counts["n_unresolved"]}개',
    )
    if counts["n_uncontrolled_claims"] > 0:
        # Loud on purpose: this is the one number the success criterion is about.
        summary += (
            f' · <strong style="color:{_ALARM}">'
            + label(
                f'{counts["n_uncontrolled_claims"]} claim(s) with NO control with matched settings',
                f'설정이 일치하는 대조군 없는 실행 {counts["n_uncontrolled_claims"]}개',
            ) + '</strong>'
        )

    th = (
        f'text-align:left;padding:4px 8px;border-bottom:{_HAIRLINE};'
        "font-weight:600;white-space:nowrap"
    )
    td = f"padding:4px 8px;border-bottom:{_HAIRLINE};vertical-align:top"
    chip = (
        f"display:inline-block;padding:0 5px;border-radius:3px;"
        f"background:{_SOFT_BG};border:{_HAIRLINE};font-size:0.85em"
    )

    # Column headers stay generic ("metric"); the metric's *name* lives in the
    # cell, so two runs measuring different things can never be read as one
    # column of comparable numbers.
    # A lab that does not ask for a hypothesis would otherwise render the red
    # "none written" alarm on every row -- flagging as a defect exactly the thing
    # the notebook deliberately stopped collecting. Drop the column instead, and
    # keep it the moment any run carries one.
    show_prediction = any((r.get("prediction") or "").strip() for r in runs)
    heads = [
        # The id is shown, not just the sequence number: `apply_verdict` matches
        # on `run_id`, so an id the student cannot read off the table makes the
        # whole verdict half of the ledger unreachable.
        label("# / id", "번호 / 실행 ID"), label("kind", "실험 종류"),
        label("condition", "조건"), label("metric", "측정값"),
        label("changed", "바뀐 설정"), label("control", "대조군 / 설정 일치"),
        *([label("prediction", "예측 / 주장·관측")] if show_prediction else []),
        label("verdict", "판정"), label("save status", "로컬 저장 상태"),
    ]
    rows = [
        "<tr>" + "".join(f'<th style="{th}">{_esc(h)}</th>' for h in heads) + "</tr>"
    ]

    for r in runs:
        marked = r.get("run_id") in highlight
        row_style = (
            f'background:{_SOFT_BG};box-shadow:inset 3px 0 0 currentColor'
            if marked
            else ""
        )
        changed = r.get("changed") or []
        changed_cell = (
            f'<span style="{chip}">Δ {_esc(", ".join(map(str, changed)))}</span>'
            if changed
            else '<span style="opacity:0.45">—</span>'
        )
        if r.get("is_control"):
            control_cell = (
                f'<span style="{chip};font-weight:600">'
                + label("declared control", "대조군으로 지정") + '</span>'
            )
        else:
            ids = matched_control_ids(r, runs)
            control_cell = (
                label("matched settings: ", "설정 일치: ") + _esc(", ".join(ids))
                if ids else '<span style="opacity:0.55">'
                + label("no matched control", "설정 일치 대조군 없음") + '</span>'
            )
        prediction = str(r.get("prediction") or "")
        prediction_cell = (
            f'<span title="{_esc(prediction)}">{_esc(_truncate(prediction))}</span>'
            if prediction
            else f'<span style="color:{_ALARM}">'
            + label("none written", "미작성") + '</span>'
        )
        verdict = str(r.get("verdict") or "")
        rival = str(r.get("rival") or "")
        shown_verdict = (
            {"supported": "지지됨", "refuted": "반박됨", "untested": "아직 검증하지 않음"}.get(verdict, verdict)
            if ko else verdict
        )
        verdict_cell = (
            f'<span title="{label("rival", "경쟁 설명")}: {_esc(rival)}">{_esc(shown_verdict)}</span>'
            if verdict
            else '<span style="opacity:0.55">' + label("unresolved", "미판정") + '</span>'
        )
        save_status = r.get("save_status") or {}
        save_state = save_status.get("state", "unknown")
        save_label = {
            "saved": label("saved locally", "로컬 저장됨"),
            "failed": label("save failed — retained in memory", "저장 실패 — 메모리에 유지됨"),
            "memory_only": label("memory only", "메모리에만 있음"),
        }.get(save_state, label("not verified", "확인되지 않음"))
        save_detail = " ".join(filter(None, (save_status.get("path"), save_status.get("error"))))
        save_cell = (
            f'<span title="{_esc(save_detail)}"'
            + (f' style="color:{_ALARM}"' if save_state == "failed" else "")
            + f'>{_esc(save_label)}</span>'
        )
        prediction_td = (
            f'<td style="{td}">{prediction_cell}</td>' if show_prediction else ""
        )
        condition_title = r.get("note") or _canonical(r.get("config", {}))
        rows.append(
            f'<tr style="{row_style}">'
            f'<td style="{td}">{"▸ " if marked else ""}{_esc(r.get("seq", ""))}'
            f'<br><code style="opacity:0.6;font-size:0.9em">'
            f'{_esc(r.get("run_id", ""))}</code></td>'
            f'<td style="{td}">{_esc(r.get("kind", ""))}</td>'
            f'<td style="{td}" title="{_esc(condition_title)}">'
            f'{_esc(r.get("condition", ""))}</td>'
            f'<td style="{td}"><span style="opacity:0.7">'
            f'{_esc(r.get("metric_name", ""))}</span><br>'
            f"<strong>{_fmt_metric(r)}</strong></td>"
            f'<td style="{td}">{changed_cell}</td>'
            f'<td style="{td}">{control_cell}</td>'
            f"{prediction_td}"
            f'<td style="{td}">{verdict_cell}</td>'
            f'<td style="{td}">{save_cell}</td>'
            "</tr>"
        )

    return (
        f'<div style="border:{_HAIRLINE};border-radius:6px;padding:8px 10px;'
        'font-size:0.85em;color:currentColor">'
        f'<div style="margin-bottom:6px">{summary}</div>'
        '<div style="overflow-x:auto">'
        '<table style="border-collapse:collapse;width:100%">'
        + "".join(rows)
        + "</table></div></div>"
    )


def _md_cell(text):
    """Escape student text as plain Markdown table content, including HTML.

    Backslashes are escaped *first*: escaping only the pipe turns `\\|` into
    `\\\\|`, which renders as a literal backslash followed by a live cell break --
    the row splits anyway, one character later.
    """
    value = escape(str(text if text not in (None, "") else ""), quote=False)
    value = value.replace("\\", "\\\\")
    for character in "|`*_[]!#~":
        value = value.replace(character, "\\" + character)
    return value.replace("\r", " ").replace("\n", " ")


def _evidence_record(run, runs):
    return {**run, "matched_control_ids": matched_control_ids(run, runs)}


def build_evidence_json(runs, provenance=None):
    """Full JSON evidence bundle; no row limits, rounding, or field removal.

    Matching is recomputed from the supplied full history. It describes recorded
    settings only. Callers may attach runtime/model/repository provenance; an
    empty value records that no provenance was supplied.
    """
    runs = list(runs)
    return json.dumps({
        "schema_version": 1,
        "provenance": provenance if provenance is not None else {},
        "runs": [_evidence_record(run, runs) for run in runs],
    }, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n"


def build_worksheet_md(runs, only_ids=None):
    """Readable summary plus complete per-run JSON for pasting into WORKSHEET.md.

    The header states which runs lack a control *before* the table, because the
    student pastes this into a graded document and the missing control is the
    thing a reader must not have to derive for themselves.
    """
    # Resolve linked control IDs from the full history before selecting rows.
    all_runs = list(runs)
    runs = all_runs
    if only_ids is not None:
        keep = set(only_ids)
        runs = [r for r in runs if r.get("run_id") in keep]
    if not runs:
        return "_No runs to report yet._"

    counts = ledger_counts(runs)
    uncontrolled = [
        r
        for r in runs
        if not r.get("is_control") and not matched_control_ids(r, all_runs)
    ]

    lines = [f"### Run ledger — {counts['n']} run(s)", ""]
    if uncontrolled:
        seqs = ", ".join(f"#{_md_cell(r.get('seq', '?'))}" for r in uncontrolled)
        lines.append(
            f"> **No control with matching settings:** {len(uncontrolled)} run(s) ({seqs}). "
            "Add a control with the same recorded settings or leave the claim untested."
        )
    else:
        lines.append("> Each experiment has a declared control with matching recorded settings.")
    lines.append(
        "> Matching settings do not establish scientific validity or prove a null effect. "
        "A local save does not guarantee that the runtime will retain files after the session ends."
    )
    show_prediction = any((r.get("prediction") or "").strip() for r in runs)
    cols = (
        ["#", "Run ID", "Kind", "Condition"]
        + (["Prediction / claim / observation"] if show_prediction else [])
        + ["Metric", "Control", "Matched control IDs", "Verdict", "Rival explanation"]
    )
    lines += ["", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in runs:
        metric = (
            f"{r.get('metric_name', '')}: {_fmt_value(r.get('metric_value', ''))} "
            f"{r.get('metric_unit', '') or ''}"
        ).strip()
        cells = [
            _md_cell(r.get("seq", "")), _md_cell(r.get("run_id", "")),
            _md_cell(r.get("kind", "")), _md_cell(r.get("condition", "")),
        ]
        if show_prediction:
            cells.append(_md_cell(r.get("prediction", "")) or "_(none written)_")
        cells += [
            _md_cell(metric),
            "declared control" if r.get("is_control") else "—",
            _md_cell(", ".join(matched_control_ids(r, all_runs))) or "—",
            _md_cell(r.get("verdict", "")) or "unresolved",
            _md_cell(r.get("rival", "")) or "—",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    for run in runs:
        evidence = json.dumps(
            _evidence_record(run, all_runs), ensure_ascii=False,
            sort_keys=True, indent=2, default=str,
        )
        # A student's caption can contain Markdown fences. The enclosing fence
        # must be longer, so every original byte stays inert, readable evidence.
        fence = "`" * max(3, 1 + max(map(len, re.findall(r"`+", evidence)), default=0))
        lines.extend([
            "", f"#### Run {_md_cell(run.get('run_id', ''))} — complete evidence", "",
            fence + "json", evidence, fence,
        ])
    return "\n".join(lines) + "\n"


def load_log(log_path, max_runs=None):
    """Read a JSONL ledger back, tolerating a truncated final line.

    The log is appended to during live GPU work, so a session that ended with a
    kernel kill mid-write leaves half a line behind. That is not a reason to lose
    the runs above it.

    The file is append-only and a `run_id` can legitimately appear more than once
    -- a verdict recorded later, or a re-run whose non-digested fields (the
    prediction text, the control flag) changed. **Last occurrence wins**, keeping
    the position of the first, so a reload reproduces what was on screen rather
    than showing the same run twice with a stale copy above it. `max_runs`
    is optional, as in `append_run`; the default restores all unique runs.
    """
    rows = []
    try:
        with open(log_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue  # truncated / corrupt line: skip, keep the rest
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        return []

    order = []
    latest = {}
    for row in rows:
        key = row.get("run_id")
        if key is None:
            order.append(len(latest))  # unkeyed rows can never collide
            latest[len(latest)] = row
            continue
        if key not in latest:
            order.append(key)
        latest[key] = row
    runs = [latest[key] for key in order]
    if max_runs and len(runs) > max_runs:
        runs = runs[-max_runs:]
    return runs
