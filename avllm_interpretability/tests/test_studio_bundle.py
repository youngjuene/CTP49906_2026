"""Single-file delivery must be deterministic and preserve edited view files."""

import base64
import hashlib
import json
import zlib

import pytest

from src.studio_bundle import restore_studio_bundle


def bundle(files):
    raw = json.dumps({"schema": 1, "files": files}, sort_keys=True).encode()
    return base64.b64encode(zlib.compress(raw)).decode(), hashlib.sha256(raw).hexdigest()


def test_uploaded_notebook_materializes_view_and_reruns_without_rewrites(tmp_path):
    payload, digest = bundle({"explore/view.toml": 'schema = 1\nprovider = "marimo-studio/vanilla"',
                              "explore/index.html": "<main id='app-shell'>한국어</main>"})
    first = restore_studio_bundle(tmp_path, payload, digest)
    page = tmp_path / "studio/ctp49906-kr/explore/index.html"
    assert page.read_text() == "<main id='app-shell'>한국어</main>"
    before = page.stat().st_mtime_ns
    again = restore_studio_bundle(tmp_path, payload, digest)
    assert first["written"]
    assert again["written"] == []
    assert page.stat().st_mtime_ns == before


def test_new_bundle_updates_owned_unedited_files(tmp_path):
    restore_studio_bundle(tmp_path, *bundle({"explore/index.html": "old"}))
    result = restore_studio_bundle(tmp_path, *bundle({"explore/index.html": "new"}))
    assert result["written"] == ["explore/index.html"]
    assert (tmp_path / "studio/ctp49906-kr/explore/index.html").read_text() == "new"


def test_student_edit_is_not_replaced_or_partially_updated(tmp_path):
    restore_studio_bundle(tmp_path, *bundle({"explore/index.html": "old"}))
    page = tmp_path / "studio/ctp49906-kr/explore/index.html"
    page.write_text("student edit")
    with pytest.raises(FileExistsError, match="index.html"):
        restore_studio_bundle(tmp_path, *bundle({"explore/index.html": "new", "explore/main.js": "new"}))
    assert page.read_text() == "student edit"
    assert not page.with_name("main.js").exists()


@pytest.mark.parametrize("path", ["../escape", "/tmp/escape", "explore/../../escape"])
def test_unsafe_paths_rejected_before_writes(tmp_path, path):
    with pytest.raises(ValueError):
        restore_studio_bundle(tmp_path, *bundle({path: "bad"}))
    assert not list(tmp_path.iterdir())


def test_existing_symlink_not_followed(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "studio").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        restore_studio_bundle(tmp_path, *bundle({"explore/index.html": "bad"}))
    assert not list(outside.iterdir())


def test_corrupted_bundle_fails_before_writes(tmp_path):
    payload, _ = bundle({"explore/index.html": "test"})
    with pytest.raises(ValueError, match="digest"):
        restore_studio_bundle(tmp_path, payload, "0" * 64)
    assert not list(tmp_path.iterdir())
