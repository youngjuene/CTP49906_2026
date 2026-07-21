from __future__ import annotations

import ast
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from curriculum_common.session_records import (  # noqa: E402
    load_jsonl,
    new_session,
    reduce_command,
    teaching_mode,
)


NOTEBOOK = Path(__file__).resolve().parents[1] / "CTP49906_avllm_molab.py"


def _source() -> str:
    return NOTEBOOK.read_text(encoding="utf-8")


def _numbered_headings(source: str, level: int) -> list[str]:
    marker = "#" * level
    return [
        match.group(1).strip()
        for match in re.finditer(
            rf"^\s*{re.escape(marker)}\s+((?:[1-4]\.)?\d*\.?\d*\s*[^\n]+)$",
            source,
            flags=re.MULTILINE,
        )
        if match.group(1).strip()[0].isdigit()
    ]


def _student_markdown_literals(source: str) -> str:
    tree = ast.parse(source)
    chunks: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function = node.func
        if not (
            isinstance(function, ast.Attribute)
            and function.attr == "md"
            and isinstance(function.value, ast.Name)
            and function.value.id == "mo"
        ):
            continue
        value = node.args[0]
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            chunks.append(value.value)
    return "\n".join(chunks)


def test_exact_four_section_eight_stage_route() -> None:
    source = _source()
    assert _numbered_headings(source, 2) == [
        "1. Prepare your project",
        "2. Guided demonstration",
        "3. Exploratory playground",
        "4. Synthesis and architecture challenge",
    ]
    assert _numbered_headings(source, 3) == [
        "1.1 Orient — Make, test, revise",
        "1.2 Compose — Plan your first cut",
        "1.3 Register — First cut (V1)",
        "2.1 Observe — Shared reference",
        "3.1 Experiment — Change one thing at a time",
        "3.2 Compare — Three readings of one work",
        "3.3 Revise — Explanation and second cut (V2)",
        "4.1 Synthesize — Portfolio and architecture challenge",
    ]
    assert source.count("**Checkpoint ·") == 4


def test_progressive_disclosure_and_student_text_boundary() -> None:
    source = _source()
    student_text = _student_markdown_literals(source)
    assert student_text.count("**Required") >= 8
    assert "**Choice" in student_text
    assert "Advanced —" in source
    assert "mo.accordion(" in source
    assert "lazy=True" in source
    forbidden = re.compile(
        r"\b(?:PRD|AC-\d+|WP-\d+|reviewer|developer annotation|"
        r"implementation history|roadmap)\b",
        flags=re.IGNORECASE,
    )
    assert not forbidden.search(student_text)


def test_replay_privacy_and_expensive_operation_gates_are_visible() -> None:
    source = _source()
    assert 'value="Saved course replay"' in source
    assert 'execution_mode_form.value != "Live model"' in source
    assert "if not USE_PRECOMPUTED:" in source
    assert "submit_button_disabled=USE_PRECOMPUTED" in source
    assert "Teaching is the default" in source
    assert "no automatic student-data egress" in source
    assert "research destination or collection action" in source
    assert "automatic student-data egress" not in source.lower().replace(
        "no automatic student-data egress", ""
    )


def test_interventions_and_epistemic_labels_stay_distinct() -> None:
    source = _source()
    lowered = source.lower()
    for operation in (
        "audio_swap_duration_matched",
        "temporal_offset",
        "audio_silence_control",
        "video_neutral_control",
        "audio_omitted_model_input",
        "video_omitted_model_input",
        "direct_attention_edge_knockout",
    ):
        assert operation in source
    assert "stimulus-signal control" in source
    assert "True modality omission" in source
    assert "silence or a black frame is never a substitute" in source
    assert "raw probe score dispersion" in lowered
    assert "teacher-forced answer-distribution dispersion" in lowered
    assert "token-layout fingerprints match" in source
    assert "not calibrated confidence" in source
    assert "causal localization" in source


def test_notebook_uses_exactly_once_commands_and_immutable_versions() -> None:
    source = _source()
    assert "reduce_command" in source
    assert '"kind": "commit_run"' in source
    assert '"kind": "attach_result"' in source
    assert '"kind": "commit_reflection"' in source
    assert "command_nonce" in source
    assert 'version_label="V1"' in source
    assert 'version_label="V2"' in source
    assert "parent_artifact_id=_v1.artifact_id" in source

    log = new_session("wp4-contract", decision=teaching_mode())
    command = {
        "kind": "commit_run",
        "command_nonce": "wp4-run-1",
        "artifact_id": "practice-v1",
        "stimulus_id": "condition:temporal_offset",
        "condition_code": "temporal_offset",
        "model_id": "model",
        "model_revision": "immutable-revision",
        "prompt": "Describe what you see and hear",
        "parameters": {"signed_offset_ms": -500},
        "prediction": "The offset will change the answer trajectory.",
        "initial_explanation": "Timing may alter cross-modal alignment.",
    }
    first = reduce_command(log, command)
    repeated = reduce_command(first.log, command)
    reloaded = load_jsonl(repeated.log.to_jsonl())
    replayed_after_reload = reduce_command(reloaded, command)

    assert len(first.log.records_of_type("run")) == 1
    assert repeated.replayed is True
    assert replayed_after_reload.replayed is True
    assert replayed_after_reload.log.to_jsonl() == first.log.to_jsonl()


def test_private_restore_and_audience_reveal_are_explicit() -> None:
    source = _source()
    assert "build_private_portfolio" in source
    assert "serialize_private_portfolio" in source
    assert "Download private learning portfolio" in source
    assert "Validate and restore private portfolio" in source
    assert "load_jsonl" in source
    assert "Delete session state and temporary uploads" in source
    assert "validate_audience_exchange" in source
    assert "Creator and model readings remain hidden" in source
    assert "at least two distinct" in source
