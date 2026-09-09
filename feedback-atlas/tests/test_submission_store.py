"""Durable original/derived records; these are data-loss and privacy regressions."""
import dataclasses
import sqlite3

import pytest

from src.store import Store
from src.models import new_opinion
from src.textnorm import text_hash


def opened(tmp_path):
    db = Store(str(tmp_path / 'atlas.db'))
    db.migrate()
    return db


def request(**changes):
    from src.submissions import SubmissionRequest
    fields = dict(reviewer_id='writer1', target_id='target1', week=2, source='human',
                  raw_text='  조명이 좋아요.\n\n음악은 너무 큽니다. 🎵  ', nonce='nonce-1',
                  owner_capability='a' * 43, context_revision=0)
    fields.update(changes)
    return SubmissionRequest(**fields)


def test_acceptance_is_durable_but_not_a_public_point(tmp_path):
    db = opened(tmp_path)
    assert hasattr(db, 'accept_submission'), 'parent intake is missing'
    req = request()
    receipt = db.accept_submission(req)
    assert receipt.state == 'queued'
    assert db.all_opinions() == []
    db.close()
    db = opened(tmp_path)
    assert db.accept_submission(req).submission_id == receipt.submission_id
    parent = db.submission_detail(receipt.submission_id, 'writer1', req.owner_capability)
    assert parent['raw_text'] == req.raw_text
    assert parent['source'] == 'human'
    assert db.public_submission(receipt.submission_id) is None


def test_replay_conflict_and_owner_check(tmp_path):
    db = opened(tmp_path)
    assert hasattr(db, 'accept_submission')
    from src.submissions import SubmissionError
    req = request()
    receipt = db.accept_submission(req)
    with pytest.raises(SubmissionError, match='NONCE_CONFLICT'):
        db.accept_submission(dataclasses.replace(req, raw_text='다른 내용'))
    with pytest.raises(SubmissionError, match='NOT_AUTHORIZED'):
        db.submission_detail(receipt.submission_id, 'writer2', req.owner_capability)
    with pytest.raises(SubmissionError, match='NOT_AUTHORIZED'):
        db.submission_detail(receipt.submission_id, 'writer1', 'b' * 43)
    other = db.accept_submission(dataclasses.replace(req, reviewer_id='writer2'))
    assert other.submission_id != receipt.submission_id


def test_staged_units_publish_atomically_and_preserve_spans(tmp_path):
    db = opened(tmp_path)
    assert hasattr(db, 'stage_revision')
    from src.submissions import FeedbackSpan, SplitResult
    req = request()
    receipt = db.accept_submission(req)
    cut = req.raw_text.index('음악')
    split = SplitResult((FeedbackSpan(0, cut), FeedbackSpan(cut, len(req.raw_text))), 'test')
    ids = db.stage_revision(receipt.submission_id, 1, split, ('human', 'human'))
    assert db.stage_revision(receipt.submission_id, 1, split, ('human', 'human')) == ids
    assert db.all_opinions() == []
    rev = db.commit_publication(db.data_rev, {receipt.submission_id: 1},
                                {oid: (float(i), 2.0) for i, oid in enumerate(ids)}, 1)
    ops = db.all_opinions()
    assert len(ops) == 2 and rev == db.data_rev
    assert ''.join(op.text for op in ops) == req.raw_text
    assert all(op.source == 'human' for op in ops)
    public = db.public_submission(receipt.submission_id)
    assert public['raw_text'] == req.raw_text
    assert not {'reviewer_id', 'owner_capability', 'capability_hash'} & public.keys()


def test_withdrawal_cannot_be_undone_by_delayed_publication(tmp_path):
    db = opened(tmp_path)
    assert hasattr(db, 'withdraw_submission')
    from src.submissions import FeedbackSpan, SplitResult, SubmissionError
    req = request()
    receipt = db.accept_submission(req)
    ids = db.stage_revision(receipt.submission_id, 1,
                            SplitResult((FeedbackSpan(0, len(req.raw_text)),), 'test'), ('human',))
    old_rev = db.data_rev
    revision = db.withdraw_submission(receipt.submission_id, 1, 'withdraw-1',
                                      req.reviewer_id, req.owner_capability)
    assert db.withdraw_submission(receipt.submission_id, 1, 'withdraw-1',
                                  req.reviewer_id, req.owner_capability) == revision
    with pytest.raises(SubmissionError, match='REVISION_CONFLICT'):
        db.commit_publication(old_rev, {receipt.submission_id: 1}, {ids[0]: (0, 0)}, 1)
    assert db.all_opinions() == []
    assert db.public_submission(receipt.submission_id) is None


def test_legacy_rows_survive_repeated_migration(tmp_path):
    db = opened(tmp_path)
    op = new_opinion(reviewer_id='writer1', target_id='target1', text='기존 의견', source='ai')
    db.insert_opinion_with_receipt(op, text_hash(op.text), 'old-nonce')
    db.put_coords({op.id: (1, 2)}, 4)
    db.migrate()
    db.close()
    db = opened(tmp_path)
    assert [o.id for o in db.all_opinions()] == [op.id]
    assert db.submission_receipt('writer1', 'old-nonce') == op.id
    assert db.all_coords()[op.id] == (1, 2)
    assert hasattr(db, 'data_rev')
    assert db.all_opinions()[0].submission_id


def test_future_schema_is_not_modified(tmp_path):
    path = tmp_path / 'future.db'
    with sqlite3.connect(path) as con:
        con.execute('CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT)')
        con.execute("INSERT INTO meta VALUES('schema_version','999')")
    db = Store(str(path))
    with pytest.raises(RuntimeError):
        db.migrate()
    with sqlite3.connect(path) as con:
        assert con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == [('meta',)]


def make_actual_v1(tmp_path):
    from src.store import _SCHEMA
    path = tmp_path/'v1.db'
    with sqlite3.connect(path) as con:
        con.executescript(_SCHEMA)
        con.execute("INSERT INTO meta VALUES('schema_version','1')")
        con.execute("INSERT INTO meta VALUES('layout_rev','7')")
        con.execute('INSERT INTO opinions VALUES(?,?,?,?,?,?,?,?)',
            ('old-id','writer1','target1','기존 정규화 의견','ai',2,'2026-01-01T00:00:00Z','old-hash'))
        con.execute('INSERT INTO coords VALUES(?,?,?,?)',('old-id',1.25,-2.5,7))
        con.execute('INSERT INTO submission_receipts VALUES(?,?,?)',('writer1','legacy-nonce','old-id'))
    return path


def test_actual_v1_upgrade_creates_restorable_backup_and_preserves_ids(tmp_path):
    path = make_actual_v1(tmp_path)
    db=Store(str(path));db.migrate();db.migrate()
    assert db.data_rev == 1 and db.get_meta('schema_version') == '2'
    assert [(o.id,o.source,o.week) for o in db.all_opinions()] == [('old-id','ai',2)]
    assert db.all_coords() == {'old-id':(1.25,-2.5)}
    assert db.submission_receipt('writer1','legacy-nonce') == 'old-id'
    assert len(db.recovery_records()) == 1
    with sqlite3.connect(str(path)+'.pre-v2-backup') as old:
        assert old.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == '1'
        assert old.execute('SELECT id,text FROM opinions').fetchall() == [('old-id','기존 정규화 의견')]
        assert old.execute("SELECT name FROM sqlite_master WHERE name='submissions'").fetchall() == []
    db.close()


def test_interrupted_v1_migration_rolls_back_all_new_tables(tmp_path,monkeypatch):
    path = make_actual_v1(tmp_path)
    db=Store(str(path))
    def interrupted(*args):
        raise RuntimeError('injected migration interruption')
    monkeypatch.setattr(db,'_legacy_parent',interrupted)
    with pytest.raises(RuntimeError,match='interruption'):
        db.migrate()
    assert db.get_meta('schema_version') == '1'
    assert db._db.execute("SELECT name FROM sqlite_master WHERE name='submissions'").fetchall() == []
    assert db._db.execute('SELECT text FROM opinions').fetchone()[0] == '기존 정규화 의견'
    db.close()
