"""Startup helper contracts for the Korean Studio notebook.

The tests extract only nested helper functions from the setup cell. They never
execute package installation, media shims, git, or marimo.
"""

import ast
import os
import subprocess
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT / "CTP49906_avllm_molab_kr.py"
PIN = "57054857f379658928aed49ee6bf286ca7271434"


def _setup_source() -> str:
    source = NOTEBOOK.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and "def _resolve_project_dir" in ast.get_source_segment(source, node):
            return ast.get_source_segment(source, node) or ""
    raise AssertionError("setup cell with _resolve_project_dir not found")


def _helpers():
    source = _setup_source()
    tree = ast.parse(source)
    names = {
        "_notebook_dir_from_location",
        "_run_text",
        "_has_project_files",
        "_resolve_project_dir",
    }
    chunks = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            chunks.append(textwrap.dedent(ast.get_source_segment(source, node) or ""))
    namespace = {
        "Path": Path,
        "RuntimeError": RuntimeError,
        "ValueError": ValueError,
        "REPO_REF": PIN,
        "_repo_url": "https://github.com/youngjuene/CTP49906_2026.git",
        "subprocess": subprocess,
    }
    exec("\n\n".join(chunks), namespace)  # noqa: S102 -- trusted helper source, mocked git
    return namespace


class FakeRunner:
    def __init__(self, tmp_path: Path, head: str | None = None, status: str = ""):
        self.tmp_path = tmp_path
        self.head = head
        self.status = status
        self.calls: list[list[str]] = []

    def __call__(self, cmd, check=True, capture_output=False, text=False):
        cmd = [str(part) for part in cmd]
        self.calls.append(cmd)
        if cmd[:2] == ["git", "clone"]:
            repo_dir = Path(cmd[-1])
            (repo_dir / ".git").mkdir(parents=True)
            return SimpleNamespace(stdout="")
        if len(cmd) >= 5 and cmd[0:2] == ["git", "-C"] and cmd[3:5] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout=(self.head or "") + "\n")
        if len(cmd) >= 5 and cmd[0:2] == ["git", "-C"] and cmd[3:5] == ["status", "--porcelain"]:
            return SimpleNamespace(stdout=self.status)
        if len(cmd) >= 4 and cmd[0:2] == ["git", "-C"] and cmd[3] == "checkout":
            repo_dir = Path(cmd[2])
            (repo_dir / "avllm_interpretability" / "src").mkdir(parents=True, exist_ok=True)
            (repo_dir / "avllm_interpretability" / "assets").mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(stdout="")
        return SimpleNamespace(stdout="")


def test_notebook_location_derives_parent_directory(tmp_path: Path) -> None:
    helper = _helpers()["_notebook_dir_from_location"]
    notebook = tmp_path / "lesson.py"
    notebook.write_text("# notebook\n", encoding="utf-8")
    assert helper(str(notebook)) == tmp_path.resolve()
    assert helper(tmp_path) == tmp_path.resolve()
    dotted_directory = tmp_path / "lesson.v1"
    dotted_directory.mkdir()
    assert helper(dotted_directory) == dotted_directory.resolve()
    with pytest.raises(ValueError):
        helper(None)


def test_startup_uses_local_project_without_git_and_ignores_cwd(tmp_path: Path) -> None:
    resolve = _helpers()["_resolve_project_dir"]
    local = tmp_path / "uploaded"
    (local / "src").mkdir(parents=True)
    (local / "assets").mkdir()
    runner = FakeRunner(tmp_path)
    cwd = os.getcwd()
    try:
        os.chdir("/")
        project_dir, status = resolve(local, runner=runner)
    finally:
        os.chdir(cwd)
    assert project_dir == local.resolve()
    assert status["mode"] == "local"
    assert runner.calls == []


def test_first_bootstrap_clone_fetches_exact_pinned_commit(tmp_path: Path) -> None:
    resolve = _helpers()["_resolve_project_dir"]
    runner = FakeRunner(tmp_path)
    project_dir, status = resolve(tmp_path, runner=runner)
    assert project_dir == (tmp_path / "CTP49906_2026" / "avllm_interpretability").resolve()
    assert status["mode"] == "checkout-cloned"
    assert runner.calls[0][:4] == ["git", "clone", "--depth", "1"]
    assert "--no-checkout" in runner.calls[0]
    assert runner.calls[1] == ["git", "-C", str(tmp_path / "CTP49906_2026"), "fetch", "--depth", "1", "origin", PIN]
    assert runner.calls[2] == ["git", "-C", str(tmp_path / "CTP49906_2026"), "checkout", "--detach", PIN]


def test_same_pinned_checkout_skips_network(tmp_path: Path) -> None:
    resolve = _helpers()["_resolve_project_dir"]
    repo = tmp_path / "CTP49906_2026"
    (repo / ".git").mkdir(parents=True)
    (repo / "avllm_interpretability" / "src").mkdir(parents=True)
    (repo / "avllm_interpretability" / "assets").mkdir()
    runner = FakeRunner(tmp_path, head=PIN)
    project_dir, status = resolve(tmp_path, runner=runner)
    assert project_dir == (repo / "avllm_interpretability").resolve()
    assert status["mode"] == "checkout-current"
    assert runner.calls == [["git", "-C", str(repo), "rev-parse", "HEAD"]]


def test_dirty_existing_checkout_is_refused_before_fetch(tmp_path: Path) -> None:
    resolve = _helpers()["_resolve_project_dir"]
    repo = tmp_path / "CTP49906_2026"
    (repo / ".git").mkdir(parents=True)
    runner = FakeRunner(tmp_path, head="old", status=" M avllm_interpretability/CTP49906_avllm_molab_kr.py\n")
    with pytest.raises(RuntimeError, match="커밋되지 않은 변경"):
        resolve(tmp_path, runner=runner)
    assert not any("fetch" in call for call in runner.calls)
    assert not any("checkout" in call for call in runner.calls)


def test_unowned_existing_bootstrap_directory_is_refused(tmp_path: Path) -> None:
    resolve = _helpers()["_resolve_project_dir"]
    (tmp_path / "CTP49906_2026").mkdir()
    runner = FakeRunner(tmp_path)
    with pytest.raises(RuntimeError, match="git checkout이 아닙니다"):
        resolve(tmp_path, runner=runner)
    assert runner.calls == []


def test_replay_env_override_is_developer_opt_in_only() -> None:
    source = NOTEBOOK.read_text(encoding="utf-8")
    assert 'USE_PRECOMPUTED = __import__("os").environ.get("CTP49906_REPLAY") == "1"' in source
    assert "CTP49906_REPLAY=1" not in source
