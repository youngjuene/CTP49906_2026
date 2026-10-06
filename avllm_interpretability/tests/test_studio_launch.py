"""A single uploaded notebook needs its view before the server discovers it."""

import asyncio
import importlib.util
import shutil
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("launch_studio", PROJECT / "scripts/launch_studio.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def test_single_file_is_discoverable_before_any_notebook_cell_runs(tmp_path):
    from marimo_studio.authoring import open_workspace

    notebook = tmp_path / "lesson.py"
    shutil.copyfile(PROJECT / "CTP49906_avllm_molab_kr.py", notebook)
    sentinel = tmp_path / "executed"
    notebook.write_text(notebook.read_text() + f'\nraise RuntimeError({str(sentinel)!r})\n')
    assert not (tmp_path / "studio").exists()
    before = asyncio.run(open_workspace(str(notebook)).status())
    assert before.state == "needs-view"
    assert before.views == ()
    result = launcher.prepare_notebook(notebook)
    assert len(result["written"]) == 5
    assert not sentinel.exists()
    workspace = open_workspace(str(notebook))
    status = asyncio.run(workspace.status())
    assert status.state == "ready"
    assert [view.name for view in status.views] == ["explore"]
    assert status.default_view == "explore"
    assert launcher.prepare_notebook(notebook)["written"] == []


def test_nonliteral_payload_is_rejected_without_executing_it(tmp_path):
    notebook = tmp_path / "lesson.py"
    notebook.write_text('def studio_files(mo):\n    _payload = print("must not run")\n'
                        '    _restore_studio_bundle(None, _payload, "digest")\n')
    with pytest.raises(ValueError, match="literal"):
        launcher.prepare_notebook(notebook)
    assert not (tmp_path / "studio").exists()


def test_prepare_preserves_student_view_edits(tmp_path):
    notebook = tmp_path / "lesson.py"
    shutil.copyfile(PROJECT / "CTP49906_avllm_molab_kr.py", notebook)
    launcher.prepare_notebook(notebook)
    page = tmp_path / "studio/ctp49906-kr/explore/index.html"
    page.write_text("student edit")
    with pytest.raises(FileExistsError):
        launcher.prepare_notebook(notebook)
    assert page.read_text() == "student edit"


def test_runtime_mismatch_fails_before_launch(monkeypatch):
    versions = {"marimo": "0.25.1", "marimo-studio": "0.2.3"}
    monkeypatch.setattr(launcher.metadata, "version", versions.__getitem__)
    with pytest.raises(RuntimeError, match="marimo==0.25.0"):
        launcher.check_runtime()


@pytest.mark.parametrize("mode", ["edit", "run"])
def test_command_keeps_environment_and_authentication(tmp_path, mode):
    command = launcher.server_command(tmp_path / "lesson.py", mode, 8765)
    assert command[:4] == [launcher.sys.executable, "-m", "marimo", mode]
    assert "--no-sandbox" in command
    assert "--token" in command
    assert "--no-token" not in command
    assert command[command.index("--host") + 1] == "127.0.0.1"
