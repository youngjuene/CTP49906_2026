"""Regression tests for separating the current example corpus into demo data."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.separate_demo_data import separate  # noqa: E402
from src.models import Opinion  # noqa: E402
from src.store import Store  # noqa: E402
from src.textnorm import text_hash  # noqa: E402


def _seed(path: Path, n: int = 3) -> None:
    store = Store(str(path))
    try:
        store.migrate()
        for index in range(n):
            op = Opinion(
                id=f"old_demo_{index}",
                reviewer_id="kang.minsu",
                target_id="kim.seoyeon",
                text=f"기존 데모 의견 {index}",
                source="human",
                week=(index % 4) + 1,
                timestamp=f"2026-09-08T00:00:0{index}.000Z",
            )
            store.insert_opinion(op, text_hash(op.text))
    finally:
        store.close()


def _opinion_count(path: Path) -> int:
    with sqlite3.connect(path) as db:
        return db.execute("SELECT count(*) FROM opinions").fetchone()[0]


def test_separate_copies_the_full_source_to_demo_and_creates_empty_classroom(tmp_path):
    source = tmp_path / "atlas.db"
    demo = tmp_path / "demo" / "atlas.db"
    classroom = tmp_path / "classroom" / "atlas.db"
    env = tmp_path / ".env"
    _seed(source, 3)
    env.write_text(
        "ATLAS_ADMIN_CODE=keep-secret\n"
        "ATLAS_ACCESS_CODES=.run/access-codes.json\n"
        f"ATLAS_DB={source}\n",
        encoding="utf-8",
    )

    separate(source, demo, classroom, env, expected_opinions=3)

    assert source.exists()
    assert _opinion_count(source) == 3
    assert _opinion_count(demo) == 3
    assert _opinion_count(classroom) == 0


def test_separate_preserves_existing_env_secrets_and_codes(tmp_path):
    source = tmp_path / "atlas.db"
    demo = tmp_path / "demo.db"
    classroom = tmp_path / "classroom.db"
    env = tmp_path / ".env"
    _seed(source, 1)
    env.write_text(
        "ATLAS_ADMIN_CODE=do-not-change\n"
        "ATLAS_ACCESS_CODES=/private/codes.json\n",
        encoding="utf-8",
    )

    separate(source, demo, classroom, env, expected_opinions=1)
    text = env.read_text(encoding="utf-8")

    assert "ATLAS_ADMIN_CODE=do-not-change" in text
    assert "ATLAS_ACCESS_CODES=/private/codes.json" in text
    assert f"ATLAS_DB={classroom.resolve()}" in text
    assert f"ATLAS_DEMO_DB={demo.resolve()}" in text
    assert "ATLAS_ENABLE_DEMO=1" in text


def test_separate_refuses_to_overwrite_existing_destinations(tmp_path):
    source = tmp_path / "atlas.db"
    demo = tmp_path / "demo.db"
    classroom = tmp_path / "classroom.db"
    env = tmp_path / ".env"
    _seed(source, 1)
    demo.write_text("already here", encoding="utf-8")
    env.write_text("ATLAS_DB=old.db\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        separate(source, demo, classroom, env, expected_opinions=1)

    assert env.read_text(encoding="utf-8") == "ATLAS_DB=old.db\n"
    assert not classroom.exists()


def test_separate_refuses_an_unexpected_source_opinion_count(tmp_path):
    source = tmp_path / "atlas.db"
    demo = tmp_path / "demo.db"
    classroom = tmp_path / "classroom.db"
    env = tmp_path / ".env"
    _seed(source, 2)
    env.write_text("ATLAS_DB=old.db\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected 30 example opinions, found 2"):
        separate(source, demo, classroom, env, expected_opinions=30)

    assert not demo.exists()
    assert not classroom.exists()


def test_missing_env_does_not_create_destination_files(tmp_path):
    source, demo, classroom = (tmp_path / name for name in ("source.db", "demo.db", "classroom.db"))
    _seed(source, 1)
    with pytest.raises(FileNotFoundError):
        separate(source, demo, classroom, tmp_path / "missing.env", 1)
    assert not demo.exists() and not classroom.exists()
    assert _opinion_count(source) == 1


def test_env_switch_failure_leaves_source_unchanged_and_is_retryable(tmp_path, monkeypatch):
    source, demo, classroom = (tmp_path / name for name in ("source.db", "demo.db", "classroom.db"))
    env = tmp_path / ".env"
    _seed(source, 1)
    env.write_text("ATLAS_DB=source.db\n")

    def fail_replace(*args):
        raise OSError("simulated configuration write failure")

    with monkeypatch.context() as patch:
        patch.setattr("scripts.separate_demo_data.os.replace", fail_replace)
        with pytest.raises(OSError, match="simulated"):
            separate(source, demo, classroom, env, 1)
    assert env.read_text() == "ATLAS_DB=source.db\n"
    assert not demo.exists() and not classroom.exists()
    assert _opinion_count(source) == 1
    separate(source, demo, classroom, env, 1)
    assert _opinion_count(demo) == 1 and _opinion_count(classroom) == 0
