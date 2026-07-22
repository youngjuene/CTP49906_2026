"""Fail-closed contract for the full static target-lock candidate."""

from __future__ import annotations

from copy import deepcopy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dependency_lock import (  # noqa: E402
    REQUIRED_IDENTITY_BINDINGS,
    canonical_object_identity,
    compute_lock_identity,
    parse_direct_requirements,
    verify_target_dependency_lock,
)


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "target_dependency_lock.json"
REQUIREMENTS = (ROOT / "requirements.txt").read_bytes()


def committed_lock() -> dict:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def resign(lock: dict) -> dict:
    lock["lock_identity"] = compute_lock_identity(lock)
    return lock


def observed_inventory(lock: dict) -> dict[str, dict]:
    return {item["name"]: deepcopy(item) for item in lock["installed_distributions"]}


def observed_bindings(lock: dict) -> dict[str, str]:
    return {name: item["identity"] for name, item in lock["bindings"].items()}


def test_committed_candidate_enumerates_full_hashed_installed_environment() -> None:
    lock = committed_lock()
    serialized = json.dumps(lock, sort_keys=True)
    packages = lock["installed_distributions"]
    direct = parse_direct_requirements(REQUIREMENTS)

    assert lock["status"] == "STATIC_CANDIDATE_UNVERIFIED"
    assert lock["readiness"] is False
    assert lock["inventory_role"] == (
        "local_environment_observation_not_target_dependency_closure"
    )
    assert "/home/" not in serialized and "/mnt/" not in serialized
    assert len(packages) >= 170
    assert len(packages) == lock["distribution_count"]
    assert {item["name"] for item in packages if item["direct"]} == direct
    assert all(item["installed_record_sha256"].startswith("sha256:") for item in packages)
    assert all(item["installed_tree_sha256"].startswith("sha256:") for item in packages)
    assert all(item["record_entry_count"] > 0 for item in packages)
    assert all(item["wheel_artifact_sha256"] is None for item in packages)
    assert lock["distribution_inventory_identity"] == canonical_object_identity(packages)
    closure = lock["course_required_closure"]
    assert direct <= set(closure["root_packages"])
    assert {"marimo", "numpy", "matplotlib", "av"} <= set(closure["root_packages"])
    assert 0 < closure["distribution_count"] < lock["distribution_count"]
    assert closure["inventory_identity"] == canonical_object_identity(
        [item for item in packages if item["name"] in set(closure["distribution_names"])]
    )
    notebook_binding = lock["bindings"]["notebook_dependencies"]
    assert notebook_binding["content"]["block_sha256"].startswith("sha256:")
    assert any(item.startswith("marimo") for item in notebook_binding["content"]["dependencies"])
    assert notebook_binding["content"]["pyav_runtime_package"] == "av"

    report = verify_target_dependency_lock(lock, requirements_bytes=REQUIREMENTS)
    assert report.identity_valid is True
    assert report.source_requirements_valid is True
    assert report.candidate_valid is True
    assert report.target_ready is False
    assert report.target_blockers


def test_missing_direct_or_transitive_distribution_fails_closed() -> None:
    lock = committed_lock()
    observed = observed_inventory(lock)
    direct_name = next(item["name"] for item in lock["installed_distributions"] if item["direct"])
    transitive_name = next(item["name"] for item in lock["installed_distributions"] if not item["direct"])

    for missing in (direct_name, transitive_name):
        changed = deepcopy(lock)
        changed["installed_distributions"] = [
            item for item in changed["installed_distributions"] if item["name"] != missing
        ]
        changed["distribution_count"] = len(changed["installed_distributions"])
        changed["distribution_inventory_identity"] = canonical_object_identity(
            changed["installed_distributions"]
        )
        resign(changed)
        report = verify_target_dependency_lock(
            changed,
            requirements_bytes=REQUIREMENTS,
            observed_installed_distributions=observed,
        )
        assert report.candidate_valid is False
        assert any(missing in issue and "missing" in issue for issue in report.issues)


@pytest.mark.parametrize("field", ["installed_record_sha256", "installed_tree_sha256"])
def test_unhashed_direct_or_transitive_distribution_fails_closed(field: str) -> None:
    lock = committed_lock()
    for package in (
        next(item for item in lock["installed_distributions"] if item["direct"]),
        next(item for item in lock["installed_distributions"] if not item["direct"]),
    ):
        changed = deepcopy(lock)
        target = next(item for item in changed["installed_distributions"] if item["name"] == package["name"])
        target[field] = None
        changed["distribution_inventory_identity"] = canonical_object_identity(
            changed["installed_distributions"]
        )
        resign(changed)
        report = verify_target_dependency_lock(changed, requirements_bytes=REQUIREMENTS)
        assert report.candidate_valid is False
        assert any(package["name"] in issue and field in issue for issue in report.issues)


def test_wheel_artifacts_and_exact_target_runtime_remain_hard_readiness_blockers() -> None:
    report = verify_target_dependency_lock(committed_lock(), requirements_bytes=REQUIREMENTS)
    assert report.candidate_valid is True
    assert report.target_ready is False
    assert any("wheel_artifact_sha256" in issue for issue in report.target_blockers)
    assert any("target_runtime" in issue for issue in report.target_blockers)


def test_pyav_or_linked_ffmpeg_mismatch_fails_closed() -> None:
    lock = committed_lock()
    for mutation in ("pyav", "ffmpeg"):
        observed = deepcopy(lock["local_runtime"])
        if mutation == "pyav":
            observed["pyav_version"] = "0.0.0"
        else:
            observed["linked_ffmpeg"]["libavcodec"] = [0, 0, 0]
        report = verify_target_dependency_lock(lock, observed_runtime=observed)
        assert report.candidate_valid is False
        assert any("runtime" in issue for issue in report.issues)


@pytest.mark.parametrize("binding_name", sorted(REQUIRED_IDENTITY_BINDINGS))
def test_every_required_identity_binding_rejects_drift(binding_name: str) -> None:
    lock = committed_lock()
    observed = observed_bindings(lock)
    observed[binding_name] = "sha256:" + "0" * 64
    report = verify_target_dependency_lock(lock, observed_bindings=observed)
    assert report.candidate_valid is False
    assert f"identity mismatch: {binding_name}" in report.issues


def test_missing_required_identity_and_internal_metadata_disagreement_fail_closed() -> None:
    lock = committed_lock()
    missing = deepcopy(lock)
    missing["bindings"].pop("stimulus_manifest")
    resign(missing)
    report = verify_target_dependency_lock(missing)
    assert report.candidate_valid is False
    assert "missing identity binding: stimulus_manifest" in report.issues

    disagreement = deepcopy(lock)
    disagreement["local_runtime"]["python_abi"] = "cp999"
    resign(disagreement)
    report = verify_target_dependency_lock(disagreement)
    assert report.candidate_valid is False
    assert "local_runtime_identity does not match local_runtime metadata" in report.issues


def test_lock_identity_detects_unsigned_payload_tampering() -> None:
    lock = committed_lock()
    lock["installed_distributions"][0]["version"] = "tampered"
    report = verify_target_dependency_lock(lock)
    assert report.identity_valid is False
    assert report.candidate_valid is False
    assert "lock_identity does not match canonical lock content" in report.issues
