"""Verify a protected teaching-pilot ref before checking out or importing it."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
import re
import subprocess


_OBJECT_ID = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROTECTED_REF = re.compile(r"^refs/tags/teaching-pilot-[a-z0-9._/-]+$")


class BootstrapVerificationError(RuntimeError):
    """The fetched candidate did not match its pre-import identity contract."""


@dataclass(frozen=True)
class PilotBootstrapExpectation:
    commit: str
    tree: str
    notebook_path: str
    notebook_sha256: str
    profile_path: str
    profile_sha256: str

    def __post_init__(self) -> None:
        for field in ("commit", "tree"):
            if not _OBJECT_ID.fullmatch(getattr(self, field)):
                raise ValueError(f"{field} must be a full lowercase git object ID")
        for field in ("notebook_sha256", "profile_sha256"):
            if not _SHA256.fullmatch(getattr(self, field)):
                raise ValueError(f"{field} must be a lowercase SHA-256 digest")
        for field in ("notebook_path", "profile_path"):
            path = PurePosixPath(getattr(self, field))
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError(f"{field} must be a safe repository-relative path")


@dataclass(frozen=True)
class PilotBootstrapReport:
    destination: str
    protected_ref: str
    commit: str
    tree: str
    notebook_sha256: str
    profile_sha256: str
    verified: bool = True


def _git_text(repo: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        raise BootstrapVerificationError(f"git bootstrap failed: {detail}") from exc
    return completed.stdout.strip()


def _git_blob(repo: Path, commit: str, relative_path: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "show", f"{commit}:{relative_path}"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", None)
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        raise BootstrapVerificationError(
            f"cannot read protected candidate path {relative_path!r}: {detail or exc}"
        ) from exc
    return completed.stdout


def _verify_sha256(label: str, payload: bytes, expected: str) -> str:
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected:
        raise BootstrapVerificationError(
            f"{label} sha256 mismatch: expected {expected}, observed {observed}"
        )
    return observed


def bootstrap_pilot_checkout(
    *,
    remote_url: str,
    destination: str | Path,
    protected_ref: str,
    expectation: PilotBootstrapExpectation,
) -> PilotBootstrapReport:
    """Fetch one protected ref, verify its objects/blobs, then detach at the commit."""

    if not _PROTECTED_REF.fullmatch(protected_ref):
        raise BootstrapVerificationError(
            "protected_ref must use refs/tags/teaching-pilot-*"
        )
    if not isinstance(remote_url, str) or not remote_url.strip():
        raise BootstrapVerificationError("remote_url must be non-empty")

    repo = Path(destination).resolve()
    repo.mkdir(parents=True, exist_ok=True)
    existing_checkout = (repo / ".git").is_dir()
    if not existing_checkout:
        if any(repo.iterdir()):
            raise BootstrapVerificationError(
                "bootstrap destination must be empty or an existing git worktree"
            )
        try:
            subprocess.run(
                ["git", "init", str(repo)],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise BootstrapVerificationError(f"cannot initialize checkout: {exc}") from exc
    if existing_checkout:
        if _git_text(repo, "status", "--porcelain"):
            raise BootstrapVerificationError(
                "refusing dirty existing checkout; preserve local or student work"
            )
        try:
            origin_url = _git_text(repo, "remote", "get-url", "origin")
        except BootstrapVerificationError as exc:
            raise BootstrapVerificationError(
                "refusing unrelated existing worktree without the expected origin"
            ) from exc
        if origin_url != remote_url:
            raise BootstrapVerificationError(
                "refusing unrelated existing worktree or origin repoint"
            )
        symbolic = subprocess.run(
            ["git", "-C", str(repo), "symbolic-ref", "-q", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        try:
            current_commit = _git_text(repo, "rev-parse", "HEAD^{commit}")
        except BootstrapVerificationError as exc:
            raise BootstrapVerificationError(
                "refusing unrelated existing worktree without a verified HEAD"
            ) from exc
        if symbolic.returncode == 0 or current_commit != expectation.commit:
            raise BootstrapVerificationError(
                "refusing to force-checkout an unrelated existing worktree"
            )
    else:
        _git_text(repo, "remote", "add", "origin", remote_url)
    _git_text(repo, "fetch", "--no-tags", "--depth", "1", "origin", protected_ref)

    commit = _git_text(repo, "rev-parse", "FETCH_HEAD^{commit}")
    if commit != expectation.commit:
        raise BootstrapVerificationError(
            f"protected ref commit mismatch: expected {expectation.commit}, observed {commit}"
        )
    tree = _git_text(repo, "show", "-s", "--format=%T", commit)
    if tree != expectation.tree:
        raise BootstrapVerificationError(
            f"protected ref tree mismatch: expected {expectation.tree}, observed {tree}"
        )

    notebook_sha256 = _verify_sha256(
        "notebook",
        _git_blob(repo, commit, expectation.notebook_path),
        expectation.notebook_sha256,
    )
    profile_sha256 = _verify_sha256(
        "profile",
        _git_blob(repo, commit, expectation.profile_path),
        expectation.profile_sha256,
    )

    _git_text(repo, "checkout", "--detach", "--force", commit)
    if _git_text(repo, "rev-parse", "HEAD") != commit:
        raise BootstrapVerificationError("detached checkout did not preserve verified commit")
    symbolic = subprocess.run(
        ["git", "-C", str(repo), "symbolic-ref", "-q", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if symbolic.returncode == 0:
        raise BootstrapVerificationError("pilot checkout must remain detached")

    return PilotBootstrapReport(
        destination=str(repo),
        protected_ref=protected_ref,
        commit=commit,
        tree=tree,
        notebook_sha256=notebook_sha256,
        profile_sha256=profile_sha256,
    )
