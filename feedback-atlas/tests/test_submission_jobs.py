"""Crash/retry persistence and stale worker results, exercised through SQLite jobs."""
from src.submissions import FeedbackSpan, SplitResult
from test_submission_store import opened, request


def test_interrupted_split_and_embedding_recover_same_parent_and_unit_ids(tmp_path):
    db = opened(tmp_path)
    req = request(); sid = db.accept_submission(req).submission_id
    assert db.job_started(sid, 1) is False
    db.close(); db = opened(tmp_path); db.recover_jobs()
    assert [j['id'] for j in db.pending_jobs()] == [sid]
    db.job_started(sid, 1)
    result = SplitResult((FeedbackSpan(0, len(req.raw_text)),), 'crash-fixture')
    ids = db.stage_revision(sid, 1, result, ('human',))
    db.close(); db = opened(tmp_path); db.recover_jobs()
    assert db.job_started(sid, 1) is True
    assert db.stage_revision(sid, 1, result, ('human',)) == ids
    db.commit_publication(db.data_rev, {sid: 1}, {ids[0]: (0, 0)}, 1)
    assert len(db.all_opinions()) == 1
    assert db.submission_detail(sid, req.reviewer_id, req.owner_capability)['raw_text'] == req.raw_text


def test_three_transient_failures_stop_automatic_retry_but_owner_can_retry(tmp_path):
    db = opened(tmp_path)
    req = request(); sid = db.accept_submission(req).submission_id
    for attempt in range(1, 4):
        db.job_started(sid, 1); db.job_failed(sid, 1)
        detail = db.submission_detail(sid, req.reviewer_id, req.owner_capability)
        assert detail['attempts'] == attempt
        assert detail['state'] == ('failed' if attempt == 3 else 'queued')
    assert not db.pending_jobs()
    db.retry_submission(sid, req.reviewer_id, req.owner_capability)
    detail = db.submission_detail(sid, req.reviewer_id, req.owner_capability)
    assert detail['state'] == 'queued' and detail['attempts'] == 0
    assert detail['raw_text'] == req.raw_text and db.all_opinions() == []


def test_queue_backpressure_never_discards_accepted_originals(tmp_path):
    import pytest
    from src.submissions import SubmissionError
    db = opened(tmp_path)
    first = db.accept_submission(request(), max_pending=1)
    with pytest.raises(SubmissionError, match='QUEUE_FULL'):
        db.accept_submission(request(nonce='second'), max_pending=1)
    assert db.accept_submission(request(), max_pending=1).submission_id == first.submission_id
    assert len(db.recovery_records()) == 1


def test_publication_conflicts_do_not_exhaust_processing_retries(tmp_path, monkeypatch):
    import asyncio
    from src.submission_worker import SubmissionWorker
    from src.model_runtime import ModelRuntime
    from src.hub import Hub
    from src.submissions import SubmissionError
    from test_recompute_pipeline import _state
    state = _state(tmp_path)
    runtime = ModelRuntime(state.embedder)
    worker = SubmissionWorker(state, Hub(), runtime)
    req = request(raw_text='한 의견입니다.')
    sid = state.store.accept_submission(req).submission_id
    def stale(*args, **kwargs):
        raise SubmissionError('REVISION_CONFLICT')
    monkeypatch.setattr(state.store, 'commit_publication', stale)
    async def scenario():
        try:
            for _ in range(3):
                await worker.process_wave([dict(state.store._parent(sid))])
            detail = state.store.submission_detail(sid, req.reviewer_id, req.owner_capability)
            assert detail['state'] == 'queued' and detail['attempts'] == 0
            assert state.store.pending_jobs()
        finally:
            await worker.aclose()
    asyncio.run(scenario())


def test_metadata_correction_reuses_all_coordinates_and_fitted_model(tmp_path, monkeypatch):
    import asyncio
    import numpy as np
    from src.submission_worker import SubmissionWorker
    from src.model_runtime import ModelRuntime
    from src.hub import Hub
    from test_recompute_pipeline import _state
    state = _state(tmp_path)
    runtime = ModelRuntime(state.embedder); worker = SubmissionWorker(state, Hub(), runtime)
    req = request(raw_text='조명이 좋습니다.')
    sid = state.store.accept_submission(req).submission_id
    async def scenario():
        try:
            await worker.process_wave(state.store.pending_jobs())
            old = state.opinions[0]; old_coords = dict(state.server_coords)
            fitted = object()
            state.layout._reducer = fitted
            state.layout._ids = [old.id]; state.layout.knn_ids = [old.id]
            state.layout._coords = np.array([old_coords[old.id]])
            state.store.correct_submission(sid, 1, 'source-only', req.reviewer_id, req.owner_capability,
                (FeedbackSpan(0,len(req.raw_text)),), ('ai',), 'target1', 2)
            def forbidden(*args, **kwargs):
                raise AssertionError('metadata correction called the projector')
            monkeypatch.setattr(state.layout, 'project', forbidden)
            await worker.process_wave(state.store.pending_jobs())
            new = state.opinions[0]
            assert new.id != old.id and new.source == 'ai' and new.revision == 2
            assert state.server_coords[new.id] == old_coords[old.id]
            assert state.layout._reducer is fitted
            assert state.layout._ids == state.layout.knn_ids == [new.id]
            assert state._row_ids == [new.id]
        finally:
            await worker.aclose()
    asyncio.run(scenario())
