from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
WP6 = ROOT / "study_materials" / "wp6"

PAIRED_GUIDES = (
    "instructor_runbook",
    "student_quick_start",
    "worksheet_migration",
    "privacy_data_dictionary",
    "troubleshooting",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_wp6_has_bilingual_classroom_document_set() -> None:
    for stem in PAIRED_GUIDES:
        english = WP6 / f"{stem}.en.md"
        korean = WP6 / f"{stem}.ko.md"
        assert english.is_file(), english
        assert korean.is_file(), korean
        assert english.read_text(encoding="utf-8").strip()
        assert korean.read_text(encoding="utf-8").strip()

    content_manifest = json.loads(
        (WP6 / "content_manifest.json").read_text(encoding="utf-8")
    )
    assert content_manifest["content_version"]
    assert set(content_manifest["documents"]) == set(PAIRED_GUIDES)
    for entry in content_manifest["documents"].values():
        assert set(entry) == {"en", "ko"}
        assert (WP6 / entry["en"]).is_file()
        assert (WP6 / entry["ko"]).is_file()


def test_replay_manifest_uses_immutable_identity_and_matching_checksums() -> None:
    manifest = json.loads((WP6 / "replay_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "classroom-replay-manifest/1.0.0"
    assert len(manifest["model_revision"]) == 40
    assert manifest["model_revision"] not in {"main", "master", "latest"}
    assert "--model_revision" in manifest["generation_command"]
    assert manifest["release_claim"] == "candidate_only"

    for relative_path, expected in manifest["files"].items():
        path = ROOT / relative_path
        assert path.is_file(), path
        assert _sha256(path) == expected["sha256"]
        assert path.stat().st_size == expected["bytes"]


def test_visual_alternatives_are_bilingual_and_downloads_are_checksum_bound() -> None:
    alternatives = json.loads(
        (WP6 / "visual_alternatives.json").read_text(encoding="utf-8")
    )
    assert alternatives["schema_version"] == "visual-alternatives/1.0.0"
    visual_ids = {entry["visual_id"] for entry in alternatives["visuals"]}
    assert visual_ids == {
        "guided.probe_distribution",
        "guided.attention_summary",
        "guided.caption_comparison",
        "audience.reading_matrix",
    }
    for entry in alternatives["visuals"]:
        assert set(entry["text_alternative"]) == {"en", "ko"}
        assert entry["text_alternative"]["en"].strip()
        assert entry["text_alternative"]["ko"].strip()
        data_path = ROOT / entry["download"]["path"]
        assert data_path.is_file(), data_path
        assert _sha256(data_path) == entry["download"]["sha256"]


def test_readmes_link_classroom_release_guidance_in_both_languages() -> None:
    for relative_path in (
        "README.md",
        "README.ko.md",
        "avllm_interpretability/README.md",
        "avllm_interpretability/README.ko.md",
    ):
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "study_materials/wp6" in text or "../study_materials/wp6" in text


def test_docs_do_not_overclaim_research_or_accessibility_approval() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in WP6.glob("*.md"))
    lowered = combined.lower()
    assert "research-ready" in lowered
    assert "not research-ready" in lowered
    assert "wcag 2.2 conformant" not in lowered
    assert "automatically send" in lowered
