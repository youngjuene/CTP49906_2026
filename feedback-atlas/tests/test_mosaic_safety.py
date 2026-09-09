"""Safety tests for the SQL surface used by the Embedding Atlas viewer."""

import threading
import time

import pytest

from src import mosaic_db
from src.mosaic_db import MosaicDatabase

pytestmark = [
    pytest.mark.needs_duckdb,
    pytest.mark.needs_pyarrow,
    pytest.mark.needs_embedding_atlas,
]


POINTS = [
    {
        "id": f"o{i}",
        "target_id": "target1",
        "text": f"opinion {i}",
        "source": "human",
        "week": 2,
        "timestamp": f"2026-09-08T00:00:0{i}Z",
        "x": float(i),
        "y": float(-i),
        "neighbors": {"ids": [], "distances": []},
    }
    for i in range(5)
]


@pytest.fixture
def db():
    database = MosaicDatabase()
    database.replace(POINTS, {"target1": "Target One"})
    try:
        yield database
    finally:
        database.close()


def _json(db, sql, *, kind="json"):
    payload, content_type = db.query(sql, kind=kind)
    assert content_type == "application/json"
    return payload


def test_select_describe_and_show_remain_queryable(db):
    assert b'"text":"opinion 0"' in _json(
        db, "SELECT id, text FROM dataset ORDER BY id LIMIT 1"
    )

    described = _json(db, "DESCRIBE SELECT id, text FROM dataset")
    assert b'"column_name":"id"' in described
    assert b'"column_name":"text"' in described

    shown = _json(db, "SHOW TABLES")
    assert b'"name":"dataset"' in shown


def test_arrow_queries_keep_their_schema(db):
    import pyarrow as pa

    payload, content_type = db.query(
        "SELECT id, x, y FROM dataset ORDER BY id",
        kind="arrow",
    )

    assert content_type == "application/octet-stream"
    table = pa.ipc.open_stream(pa.BufferReader(payload)).read_all()
    assert table.column_names == ["id", "x", "y"]
    assert table.num_rows == len(POINTS)


@pytest.mark.parametrize("kind", ["exec", "json", "arrow"])
def test_write_statements_cannot_mutate_the_relation(db, kind):
    with pytest.raises(ValueError, match="read-only SELECT"):
        db.query("UPDATE dataset SET text='tampered'", kind=kind)

    assert b"tampered" not in _json(db, "SELECT text FROM dataset ORDER BY id")
    assert b"opinion 0" in _json(db, "SELECT text FROM dataset ORDER BY id")


def test_multi_statement_queries_are_refused_before_execution(db):
    with pytest.raises(ValueError, match="exactly one statement"):
        db.query("SELECT 1; UPDATE dataset SET text='tampered'", kind="json")

    assert b"tampered" not in _json(db, "SELECT text FROM dataset ORDER BY id")


def test_json_and_arrow_outputs_are_bounded(db, monkeypatch):
    monkeypatch.setattr(mosaic_db, "MAX_RESULT_ROWS", 3)

    with pytest.raises(ValueError, match="more than 3 rows"):
        db.query("SELECT * FROM dataset ORDER BY id", kind="json")

    with pytest.raises(ValueError, match="more than 3 rows"):
        db.query("SELECT * FROM dataset ORDER BY id", kind="arrow")


def test_interrupted_query_recovers_for_followup_query_and_replace(db, monkeypatch):
    monkeypatch.setattr(mosaic_db, "QUERY_TIMEOUT_SECONDS", 0.05)

    with pytest.raises(Exception, match="Interrupted|interrupted|interrupt"):
        db.query("SELECT sum(sin(i)) FROM range(100000000000) AS r(i)")

    assert b'"count_star()":5' in _json(db, "SELECT count(*) FROM dataset")

    replacement = [{**POINTS[0], "id": "replacement", "text": "after interrupt"}]
    db.replace(replacement, {"target1": "Target One"})
    assert b"after interrupt" in _json(db, "SELECT text FROM dataset")


def test_query_lock_wait_is_bounded(db, monkeypatch):
    monkeypatch.setattr(mosaic_db, "QUERY_LOCK_TIMEOUT_SECONDS", 0.05)
    assert db._lock.acquire(timeout=0.1)
    try:
        error = []

        def run_query():
            try:
                db.query("SELECT 1")
            except Exception as exc:
                error.append(exc)

        thread = threading.Thread(target=run_query)
        thread.start()
        thread.join(timeout=1.0)
        assert not thread.is_alive()
        assert error and isinstance(error[0], TimeoutError)
    finally:
        db._lock.release()

    time.sleep(0.01)
    assert b'"1":1' in _json(db, "SELECT 1")
