from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import subprocess

import pytest

from curriculum_common.release_bootstrap import (
    BootstrapVerificationError,
    PilotBootstrapExpectation,
    bootstrap_pilot_checkout,
)


PROTECTED_REF = "refs/tags/teaching-pilot-candidate"
NOTEBOOK_PATH = "avllm_interpretability/CTP49906_avllm_molab.py"
PROFILE_PATH = "study_materials/wp0/course_release_profile.json"


def _git(*args: object, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *(str(arg) for arg in args)],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _write_fixture(repo: Path, *, marker: str) -> None:
    notebook = repo / NOTEBOOK_PATH
    profile = repo / PROFILE_PATH
    notebook.parent.mkdir(parents=True, exist_ok=True)
    profile.parent.mkdir(parents=True, exist_ok=True)
    notebook.write_text(f"PILOT_MARKER = {marker!r}\n", encoding="utf-8")
    profile.write_text(
        '{"candidate_observation_only":true,"research_enabled":false,'
        f'"marker":"{marker}"}}\n',
        encoding="utf-8",
    )


def _commit(repo: Path, *, marker: str) -> tuple[str, str]:
    _write_fixture(repo, marker=marker)
    _git("add", ".", cwd=repo)
    _git(
        "-c",
        "user.name=Pilot Fixture",
        "-c",
        "user.email=pilot@example.invalid",
        "commit",
        "-m",
        f"fixture {marker}",
        cwd=repo,
    )
    return _git("rev-parse", "HEAD", cwd=repo), _git("rev-parse", "HEAD^{tree}", cwd=repo)


def _expectation(repo: Path, commit: str, tree: str) -> PilotBootstrapExpectation:
    return PilotBootstrapExpectation(
        commit=commit,
        tree=tree,
        notebook_path=NOTEBOOK_PATH,
        notebook_sha256=hashlib.sha256((repo / NOTEBOOK_PATH).read_bytes()).hexdigest(),
        profile_path=PROFILE_PATH,
        profile_sha256=hashlib.sha256((repo / PROFILE_PATH).read_bytes()).hexdigest(),
    )


def _remote_fixture(tmp_path: Path) -> tuple[Path, Path, str, str]:
    source = tmp_path / "source"
    remote = tmp_path / "remote.git"
    source.mkdir()
    _git("init", cwd=source)
    commit, tree = _commit(source, marker="first")
    _git("init", "--bare", remote)
    _git("remote", "add", "origin", remote, cwd=source)
    _git("push", "origin", f"HEAD:{PROTECTED_REF}", cwd=source)
    return source, remote, commit, tree


def _fresh_hosted_bootstrap():
    notebook = Path(__file__).resolve().parents[1] / "CTP49906_avllm_molab.py"
    tree = ast.parse(notebook.read_text(encoding="utf-8"))
    cell = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(child, ast.FunctionDef)
            and child.name == "bootstrap_pilot_checkout"
            for child in node.body
        )
    )
    allowed_names = {
        "_OBJECT_ID",
        "_SHA256",
        "_PILOT_TAG",
        "_git",
        "bootstrap_pilot_checkout",
    }
    helper_body = [
        node
        for node in cell.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        or (
            isinstance(node, (ast.Assign, ast.FunctionDef))
            and (
                (isinstance(node, ast.FunctionDef) and node.name in allowed_names)
                or (
                    isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Name) and target.id in allowed_names
                        for target in node.targets
                    )
                )
            )
        )
    ]
    namespace: dict[str, object] = {}
    exec(
        compile(ast.fix_missing_locations(ast.Module(helper_body, type_ignores=[])), str(notebook), "exec"),
        namespace,
    )
    return namespace["bootstrap_pilot_checkout"]


def test_bootstrap_fetches_protected_ref_verifies_before_import_and_detaches(
    tmp_path: Path,
) -> None:
    source, remote, commit, tree = _remote_fixture(tmp_path)
    checkout = tmp_path / "checkout"

    report = bootstrap_pilot_checkout(
        remote_url=str(remote),
        destination=checkout,
        protected_ref=PROTECTED_REF,
        expectation=_expectation(source, commit, tree),
    )

    assert report.verified is True
    assert report.commit == commit
    assert report.tree == tree
    assert _git("rev-parse", "HEAD", cwd=checkout) == commit
    assert subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"],
        cwd=checkout,
        check=False,
        capture_output=True,
        text=True,
    ).returncode == 1
    assert (checkout / NOTEBOOK_PATH).read_text(encoding="utf-8").strip()


def test_fresh_hosted_notebook_verifier_uses_only_stdlib_and_git(tmp_path: Path) -> None:
    source, remote, commit, tree = _remote_fixture(tmp_path)
    expectation = _expectation(source, commit, tree)

    report = _fresh_hosted_bootstrap()(
        remote_url=str(remote),
        destination=tmp_path / "fresh-hosted-checkout",
        protected_ref=PROTECTED_REF,
        **expectation.__dict__,
    )

    assert report["verified"] is True
    assert report["commit"] == commit
    assert report["tree"] == tree


def test_bootstrap_rejects_stale_protected_ref_before_checkout(tmp_path: Path) -> None:
    source, remote, commit, tree = _remote_fixture(tmp_path)
    stale = _expectation(source, commit, tree)
    _commit(source, marker="second")
    _git("push", "--force", "origin", f"HEAD:{PROTECTED_REF}", cwd=source)

    with pytest.raises(BootstrapVerificationError, match="commit"):
        bootstrap_pilot_checkout(
            remote_url=str(remote),
            destination=tmp_path / "stale-checkout",
            protected_ref=PROTECTED_REF,
            expectation=stale,
        )


def test_bootstrap_refuses_nonempty_or_dirty_destinations(tmp_path: Path) -> None:
    source, remote, commit, tree = _remote_fixture(tmp_path)
    expectation = _expectation(source, commit, tree)
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "student-work.txt").write_text("preserve me", encoding="utf-8")

    with pytest.raises(BootstrapVerificationError, match="empty"):
        bootstrap_pilot_checkout(
            remote_url=str(remote),
            destination=nonempty,
            protected_ref=PROTECTED_REF,
            expectation=expectation,
        )

    dirty = tmp_path / "dirty"
    dirty.mkdir()
    _git("init", cwd=dirty)
    _git("remote", "add", "origin", remote, cwd=dirty)
    (dirty / "student-work.txt").write_text("preserve me", encoding="utf-8")
    with pytest.raises(BootstrapVerificationError, match="dirty"):
        bootstrap_pilot_checkout(
            remote_url=str(remote),
            destination=dirty,
            protected_ref=PROTECTED_REF,
            expectation=expectation,
        )


def test_bootstrap_does_not_repoint_or_force_checkout_unrelated_worktree(
    tmp_path: Path,
) -> None:
    source, remote, commit, tree = _remote_fixture(tmp_path)
    expectation = _expectation(source, commit, tree)
    unrelated_remote = tmp_path / "unrelated.git"
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    _git("init", cwd=unrelated)
    _git("init", "--bare", unrelated_remote)
    _git("remote", "add", "origin", unrelated_remote, cwd=unrelated)
    _commit(unrelated, marker="unrelated")

    with pytest.raises(BootstrapVerificationError, match="unrelated"):
        bootstrap_pilot_checkout(
            remote_url=str(remote),
            destination=unrelated,
            protected_ref=PROTECTED_REF,
            expectation=expectation,
        )
    assert _git("remote", "get-url", "origin", cwd=unrelated) == str(unrelated_remote)
    assert (unrelated / NOTEBOOK_PATH).read_text(encoding="utf-8") == (
        "PILOT_MARKER = 'unrelated'\n"
    )


@pytest.mark.parametrize("field", ["notebook_sha256", "profile_sha256"])
def test_bootstrap_rejects_content_hash_mismatch_before_checkout(
    tmp_path: Path,
    field: str,
) -> None:
    source, remote, commit, tree = _remote_fixture(tmp_path)
    values = _expectation(source, commit, tree).__dict__.copy()
    values[field] = "0" * 64

    with pytest.raises(BootstrapVerificationError, match="sha256"):
        bootstrap_pilot_checkout(
            remote_url=str(remote),
            destination=tmp_path / f"bad-{field}",
            protected_ref=PROTECTED_REF,
            expectation=PilotBootstrapExpectation(**values),
        )


def test_notebook_verifies_identity_before_install_or_course_import() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "CTP49906_avllm_molab.py"
    ).read_text(encoding="utf-8")

    verified = source.index("BOOTSTRAP_REPORT = bootstrap_pilot_checkout(")
    live_guard = source.index(
        'if not USE_PRECOMPUTED and BOOTSTRAP_REPORT.get("verified") is not True:'
    )
    installed = source.index("_ensure_packages([")
    course_imported = source.index("from curriculum_common.audience_packets import")
    model_imported = source.index("from transformers import")
    assert verified < live_guard < installed < course_imported < model_imported


def test_local_unverified_source_cannot_enter_live_model_path() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "CTP49906_avllm_molab.py"
    ).read_text(encoding="utf-8")

    assert '"verified": False' in source
    assert "LOCAL_CHECKED_OUT_SOURCE_NOT_HOSTED_VERIFICATION" in source
    assert (
        'if not USE_PRECOMPUTED and BOOTSTRAP_REPORT.get("verified") is not True:'
        in source
    )
    assert "Live model execution requires verified protected pilot" in source
