import hashlib
import json
from pathlib import Path

from avllm_interpretability.src.dependency_lock import (
    canonical_object_identity,
    parse_pep723_dependencies,
)

ROOT = Path(__file__).parents[2]
NOTEBOOK = ROOT / "avllm_interpretability/CTP49906_avllm_molab.py"
PROFILE = ROOT / "study_materials/wp0/course_release_profile.json"
LOCK = ROOT / "avllm_interpretability/target_dependency_lock.json"


def test_final_notebook_profile_and_lock_identity() -> None:
    source = NOTEBOOK.read_text()
    profile = json.loads(PROFILE.read_text())
    lock = json.loads(LOCK.read_text())
    assert profile["notebook"]["sha256"] == hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    block, specs, _ = parse_pep723_dependencies(source)
    binding = lock["bindings"]["notebook_dependencies"]
    assert binding["content"]["block_sha256"] == "sha256:" + hashlib.sha256(block.encode()).hexdigest()
    assert list(specs) == binding["content"]["dependencies"]
    assert binding["identity"] == canonical_object_identity(binding["content"])


def test_controlled_media_privacy_contracts_present() -> None:
    source = NOTEBOOK.read_text()
    for token in (
        "PRIVATE_MEDIA_DIR",
        "get_private_media_state",
        "set_private_media_state(None)",
        "processor_nframes",
        "donor_content_sha256",
        "source_artifact_id",
        "cleanup failed",
        "is_symlink",
    ):
        assert token in source
    download_cell = source[source.find("def _(PRIVATE_MEDIA_DIR, get_private_media_state") :]
    assert "ControlledOperationResult" not in download_cell[:3000]
