"""Treatment-neutral production-manifest contract tests."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from curriculum_common.production_manifest import (  # noqa: E402
    ArtifactVersion,
    EditDecisionManifest,
    create_artifact_version,
    dump_artifact_version,
    load_artifact_version,
)


CONTENT = "sha256:" + "a" * 64


def _edit_manifest():
    return EditDecisionManifest(
        editor_name_version="common-editor/1.0",
        source_assets=({"asset_id": "source-a", "license": "synthetic-fixture"},),
        ordering=("source-a",),
        trims=({"asset_id": "source-a", "in_ms": 0, "out_ms": 1000},),
        mix_levels=({"asset_id": "source-a", "gain_db": -3.0},),
        accessibility_work={"captions": "creator-authored"},
        assistance_disclosure={"human": True, "ai": False},
        export_preset_version="common-export/1.0",
    )


def _artifact(version_label="V1", parent=None, content=CONTENT):
    return create_artifact_version(
        content_sha256=content,
        version_label=version_label,
        parent_artifact_id=parent,
        local_registered_at_utc="2026-07-21T00:00:00Z",
        event_index=1 if version_label == "V1" else 2,
        elapsed_ms=100 if version_label == "V1" else 200,
        media_facts={"duration_ms": 1000, "local_filename_alias": "private.mp4"},
        creator_intention="contrast image rhythm with a steady pulse",
        intended_audience="class peers",
        sound_image_relation="counterpoint",
        concept_tags=("rhythm", "contrast"),
        cultural_aesthetic_context="creator-described fixture context",
        source_license_provenance={"status": "synthetic_fixture"},
        edit_manifest=_edit_manifest(),
        change_rationale="initial cut" if version_label == "V1" else "tightened timing",
        processing_boundary="browser upload to Molab-hosted session",
    )


def test_edit_decisions_round_trip_identically_across_arms():
    manifest = _edit_manifest()
    treatment_fields = manifest.to_artifact_fields()
    comparison = EditDecisionManifest.from_artifact_fields(treatment_fields)
    assert comparison.to_artifact_fields() == treatment_fields


def test_v1_v2_ids_are_stable_content_derived_and_lineage_is_immutable():
    v1 = _artifact()
    renamed_v1 = _artifact()
    v2 = _artifact("V2", v1.artifact_id, "sha256:" + "b" * 64)
    assert v1.artifact_id == renamed_v1.artifact_id
    assert v2.artifact_id != v1.artifact_id
    assert v2.parent_artifact_id == v1.artifact_id
    assert v1.parent_artifact_id is None
    with pytest.raises(ValueError, match="require"):
        _artifact("V2", None, "sha256:" + "b" * 64)


def test_artifact_schema_round_trip_rejects_unknown_or_retargeted_identity():
    artifact = _artifact()
    payload = dump_artifact_version(artifact)
    assert dump_artifact_version(load_artifact_version(payload)) == payload
    changed = artifact.to_dict()
    changed["artifact_id"] = "artifact:sha256:" + "0" * 64
    with pytest.raises(ValueError, match="identity"):
        ArtifactVersion.from_dict(changed)
    changed = artifact.to_dict()
    changed["future_field"] = True
    with pytest.raises(ValueError, match="unknown"):
        ArtifactVersion.from_dict(changed)
