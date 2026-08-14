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
    """
    config = dict(config or {})
    run_id = hashlib.sha1(
        _canonical(
            {
                "kind": kind,
                "condition": condition,
                "config": config,
                "metric": [metric_name, metric_value, metric_unit],
            }
        ).encode("utf-8")
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
        "extra": dict(extra or {}),
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


def append_run(prev, record, max_runs=40, log_path=None):
    """Return a new run list with `record` added (or refreshed in place).

    Idempotent by `run_id`: a re-executed marimo cell hands us the same digest,
    and replacing keeps the original `seq` so the ledger does not renumber
    itself under the student. `changed` is computed against the most recent run
    of the same kind *that precedes this one*, so it too survives re-execution.
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
        # metric. The non-digested fields (prediction text, control flag, note)
        # can still differ -- rewording a prediction and pressing ▶ again lands
        # here -- so compare the whole record, and log only when something
        # actually changed. `load_log` folds by `run_id`, last-wins, so a second
        # line for an amended run reloads as one row rather than two.
        already_logged = runs[index] == record
        runs[index] = record
    else:
        already_logged = False
        record["seq"] = max((r.get("seq", 0) for r in runs), default=0) + 1
        runs.append(record)
    if max_runs and len(runs) > max_runs:
        runs = runs[-max_runs:]

    if log_path and not already_logged:
        # Best effort, always. The record in hand may have cost a minute of GPU
        # time; a read-only directory or a full disk must not be able to throw
        # it away, so every failure mode here is swallowed on purpose.
        try:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(_canonical(record) + "\n")
        except Exception:  # noqa: BLE001 - logging must never break the experiment
            pass
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
    if log_path and amended is not None:
        # Best effort, as in `append_run`: a logging failure must never cost the
        # student the verdict they just recorded.
        try:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(_canonical(amended) + "\n")
        except Exception:  # noqa: BLE001
            pass
    return runs


def ledger_counts(runs):
    """Coverage summary of the ledger.

    "Controlled" is a property of the *kind*, not of the individual run: a
    knockout sweep is controlled because the silent-clip run exists somewhere in
    the same family, not because that particular row ran twice. So
    `n_uncontrolled_claims` -- the number the course grades on -- counts
    non-control runs whose family contains no control at all.
    """
    controlled_kinds = {r.get("kind") for r in runs if r.get("is_control")}
    return {
        "n": len(runs),
        "n_controlled": sum(1 for r in runs if r.get("kind") in controlled_kinds),
        "n_uncontrolled_claims": sum(
            1
            for r in runs
            if not r.get("is_control") and r.get("kind") not in controlled_kinds
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


def render_ledger_html(runs, highlight_ids=()):
    """Compact HTML table of the ledger, for `mo.Html`.

    Everything is escaped: `condition` and `prediction` are student input that
    reaches this function straight from a text box. Styling is inline only --
    a `<style>` block or class names would collide with marimo's own CSS.
    """
    if not runs:
        return (
            f'<div style="border:{_HAIRLINE};border-radius:6px;padding:10px 12px;'
            f'background:{_SOFT_BG};font-size:0.9em;opacity:0.85">'
            "<em>No runs logged yet. Write a prediction, press ▶, and the run "
            "lands here — with or without a control.</em></div>"
        )

    counts = ledger_counts(runs)
    highlight = set(highlight_ids or ())

    summary = (
        f'{counts["n"]} run(s) · {counts["n_controlled"]} in a family that has a '
        f'control · {counts["n_unresolved"]} still unresolved'
    )
    if counts["n_uncontrolled_claims"] > 0:
        # Loud on purpose: this is the one number the success criterion is about.
        summary += (
            f' · <strong style="color:{_ALARM}">'
            f'{counts["n_uncontrolled_claims"]} claim(s) with NO control</strong>'
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
    heads = [
        # The id is shown, not just the sequence number: `apply_verdict` matches
        # on `run_id`, so an id the student cannot read off the table makes the
        # whole verdict half of the ledger unreachable.
        "# / id", "kind", "condition", "metric", "changed", "control", "prediction",
        "verdict",
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
                f'<span style="{chip};font-weight:600">control</span>'
            )
        else:
            control_cell = '<span style="opacity:0.45">—</span>'
        prediction = str(r.get("prediction") or "")
        prediction_cell = (
            f'<span title="{_esc(prediction)}">{_esc(_truncate(prediction))}</span>'
            if prediction
            else f'<span style="color:{_ALARM}">none written</span>'
        )
        verdict = str(r.get("verdict") or "")
        rival = str(r.get("rival") or "")
        verdict_cell = (
            f'<span title="rival: {_esc(rival)}">{_esc(verdict)}</span>'
            if verdict
            else '<span style="opacity:0.55">unresolved</span>'
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
            f'<td style="{td}">{prediction_cell}</td>'
            f'<td style="{td}">{verdict_cell}</td>'
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
    """Markdown tables end a cell at `|`, and student text contains pipes.

    Backslashes are escaped *first*: escaping only the pipe turns `\\|` into
    `\\\\|`, which renders as a literal backslash followed by a live cell break --
    the row splits anyway, one character later.
    """
    return (
        str(text if text not in (None, "") else "")
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\n", " ")
    )


def build_worksheet_md(runs, only_ids=None):
    """Markdown table for pasting into WORKSHEET.md.

    The header states which runs lack a control *before* the table, because the
    student pastes this into a graded document and the missing control is the
    thing a reader must not have to derive for themselves.
    """
    # Coverage is a property of the whole ledger, not of the slice being pasted.
    # Computed after the filter, exporting just the experiment -- leaving its
    # control behind, which is the point of a per-run export -- would stamp
    # "claims an effect with no control" onto a graded document about a run that
    # *is* controlled.
    controlled_kinds = {r.get("kind") for r in runs if r.get("is_control")}
    if only_ids is not None:
        keep = set(only_ids)
        runs = [r for r in runs if r.get("run_id") in keep]
    if not runs:
        return "_No runs to report yet._"

    counts = ledger_counts(runs)
    uncontrolled = [
        r
        for r in runs
        if not r.get("is_control") and r.get("kind") not in controlled_kinds
    ]

    lines = [f"### Run ledger — {counts['n']} run(s)", ""]
    if uncontrolled:
        seqs = ", ".join(f"#{r.get('seq', '?')}" for r in uncontrolled)
        lines.append(
            f"> **No control:** {len(uncontrolled)} run(s) ({seqs}) claim an effect "
            "with no control run in the same family. Either add one or say plainly "
            "that the claim is untested."
        )
    else:
        lines.append("> Every run here belongs to a family that includes a control.")
    lines += [
        "",
        "| # | Condition | Prediction (before ▶) | Metric | Control | Verdict "
        "| Rival explanation |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        metric = (
            f"{r.get('metric_name', '')}: {_fmt_value(r.get('metric_value', ''))} "
            f"{r.get('metric_unit', '') or ''}"
        ).strip()
        lines.append(
            "| {seq} | {cond} | {pred} | {metric} | {control} | {verdict} "
            "| {rival} |".format(
                seq=_md_cell(r.get("seq", "")),
                cond=_md_cell(r.get("condition", "")),
                pred=_md_cell(r.get("prediction", "")) or "_(none written)_",
                metric=_md_cell(metric),
                control="control" if r.get("is_control") else "—",
                verdict=_md_cell(r.get("verdict", "")) or "unresolved",
                rival=_md_cell(r.get("rival", "")) or "—",
            )
        )
    return "\n".join(lines) + "\n"


def load_log(log_path, max_runs=40):
    """Read a JSONL ledger back, tolerating a truncated final line.

    The log is appended to during live GPU work, so a session that ended with a
    kernel kill mid-write leaves half a line behind. That is not a reason to lose
    the runs above it.

    The file is append-only and a `run_id` can legitimately appear more than once
    -- a verdict recorded later, or a re-run whose non-digested fields (the
    prediction text, the control flag) changed. **Last occurrence wins**, keeping
    the position of the first, so a reload reproduces what was on screen rather
    than showing the same run twice with a stale copy above it. `max_runs`
    mirrors `append_run`'s cap so the seeded list cannot start out longer than
    the in-memory one.
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
