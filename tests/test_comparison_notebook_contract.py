from __future__ import annotations

import ast
from pathlib import Path
import re

from digital_storytelling.workflow import CONTENT, ROUTE_MANIFEST, ROUTE_STAGES


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "digital_storytelling" / "CTP49906_storytelling_molab.py"


def _source() -> str:
    return NOTEBOOK.read_text(encoding="utf-8")


def _root_imports(source: str) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_comparison_surface_has_inward_only_model_free_imports() -> None:
    roots = _root_imports(_source())
    assert roots <= {
        "__future__",
        "json",
        "marimo",
        "pathlib",
        "sys",
        "uuid",
        "curriculum_common",
        "digital_storytelling",
    }
    assert roots.isdisjoint(
        {
            "avllm_interpretability",
            "src",
            "audience",
            "torch",
            "torchvision",
            "transformers",
            "accelerate",
            "qwen_omni_utils",
            "openai",
        }
    )


def test_pre_post_source_has_no_treatment_contamination() -> None:
    source = _source()
    marker = "post_heading"
    before_post = source[: source.index(marker)].lower()
    forbidden = (
        "counterpoint lens",
        "interpretability",
        "logit lens",
        "layer probe",
        "attention knockout",
        "internal ablation",
        "avllm_interpretability",
        "src.playground_clips",
    )
    assert not any(term in before_post for term in forbidden)
    assert "condition_assignment" not in source
    assert not re.search(r"\b(?:PRD|FR|AC|WP)[-_ ]?\d+\b", source)


def test_creator_imports_use_bounded_duplicate_safe_json_parser() -> None:
    source = _source()

    assert "from curriculum_common.json_import import parse_json_object" in source
    assert source.count("parse_json_object(") >= 2
    assert "json.loads(_audience_value" not in source
    assert "json.loads(_file.contents" not in source


def test_notebook_has_teaching_privacy_and_audience_boundaries() -> None:
    source = _source()
    required_symbols = (
        "build_blinded_packet",
        "require_complete_audience_exchange",
        "create_story_artifact",
        "build_private_story_bundle",
        'version_label="V1"',
        'version_label="V2"',
        "parent_artifact_id=_first.artifact_id",
        "permission_confirmed",
        "audience_complete",
        "COMPARISON_ROUTE_ID",
    )
    for symbol in required_symbols:
        assert symbol in source
    assert "requests" not in source
    assert "urlopen" not in source
    assert "subprocess" not in source


def test_all_marimo_ui_controls_have_nonempty_labels() -> None:
    tree = ast.parse(_source())
    ui_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not isinstance(function, ast.Attribute):
            continue
        owner = function.value
        if not (
            isinstance(owner, ast.Attribute)
            and owner.attr == "ui"
            and isinstance(owner.value, ast.Name)
            and owner.value.id == "mo"
        ):
            continue
        ui_calls.append(node)
    assert ui_calls
    for call in ui_calls:
        labels = [keyword.value for keyword in call.keywords if keyword.arg == "label"]
        assert labels, f"missing label at line {call.lineno}"
        label = labels[0]
        if isinstance(label, ast.Constant):
            assert isinstance(label.value, str) and label.value.strip()
    assert "autoplay" not in _source().lower()


def test_route_manifest_preserves_unresolved_fidelity_truthfully() -> None:
    assert ROUTE_MANIFEST["activity_sequence"] == list(ROUTE_STAGES)
    assert ROUTE_MANIFEST["research_ready"] is False
    assert ROUTE_MANIFEST["numeric_time_budget_minutes"] is None
    assert ROUTE_MANIFEST["concrete_fidelity_status"] == (
        "blocked_pending_human_freeze"
    )
    for dimension in ROUTE_MANIFEST["matched_dimensions"].values():
        assert dimension == {
            "value": None,
            "tolerance": None,
            "approval_status": "unresolved",
        }


def test_localized_student_catalog_is_exactly_parallel() -> None:
    assert set(CONTENT) == {"en", "ko"}
    assert set(CONTENT["en"]) == set(CONTENT["ko"])
    assert all(value.strip() for catalog in CONTENT.values() for value in catalog.values())
    for catalog in CONTENT.values():
        rendered = "\n".join(catalog.values()).lower()
        assert "counterpoint lens" not in rendered
        assert "logit lens" not in rendered
        assert "condition assignment" not in rendered
