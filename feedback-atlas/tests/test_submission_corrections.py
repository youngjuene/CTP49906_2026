from dataclasses import replace

import pytest

from src.store import Store
from src.submissions import FeedbackSpan, SplitResult, SubmissionError
from test_submission_store import request


def published(tmp_path):
    store = Store(str(tmp_path / 'atlas.db'))
    store.migrate()
    req = request(raw_text='좋은 조명.\n시끄러운 음악.')
    receipt = store.accept_submission(req)
    ids = store.stage_revision(receipt.submission_id, 1,
        SplitResult((FeedbackSpan(0, len(req.raw_text)),), 'fixture'), ('human',))
    store.commit_publication(0, {receipt.submission_id: 1}, {ids[0]: (1, 2)}, 1)
    return store, req, receipt


def test_correction_is_private_until_complete_and_replay_is_idempotent(tmp_path):
    store, req, receipt = published(tmp_path)
    cut = req.raw_text.index('시끄러운')
    spans = (FeedbackSpan(0, cut), FeedbackSpan(cut, len(req.raw_text)))
    old = store.all_opinions()[0]
    args = (receipt.submission_id, 1, 'edit1', req.reviewer_id, req.owner_capability,
            spans, ('human', 'ai'), 'target2', 4)
    new_rev = store.correct_submission(*args)
    assert store.correct_submission(*args) == new_rev == 2
    assert store.all_opinions()[0] == old
    assert store.public_submission(receipt.submission_id)['target_id'] == 'target1'
    pending = store.staged_opinions({receipt.submission_id: 2})
    store.commit_publication(store.data_rev, {receipt.submission_id: 2},
                              {op.id: (i, 3) for i, op in enumerate(pending)}, 2)
    assert [o.source for o in store.all_opinions()] == ['human', 'ai']
    assert {o.week for o in store.all_opinions()} == {4}
    assert {o.target_id for o in store.all_opinions()} == {'target2'}
    assert old.id not in store.all_coords()
    assert store.data_rev == 2


def test_invalid_manifest_rolls_back_the_entire_correction(tmp_path):
    store, req, receipt = published(tmp_path)
    with pytest.raises(SubmissionError, match='INVALID_SEGMENTS'):
        store.correct_submission(receipt.submission_id, 1, 'bad-edit', req.reviewer_id,
            req.owner_capability, (FeedbackSpan(0, 4), FeedbackSpan(3, len(req.raw_text))),
            ('ai', 'human'), 'target2', 3)
    detail = store.submission_detail(receipt.submission_id, req.reviewer_id, req.owner_capability)
    assert detail['revision'] == 1 and detail['target_id'] == 'target1'


def test_read_only_public_id_is_not_correction_authority(tmp_path):
    store, req, receipt = published(tmp_path)
    with pytest.raises(SubmissionError, match='NOT_AUTHORIZED'):
        store.withdraw_submission(receipt.submission_id, 1, 'steal', req.reviewer_id, 'z' * 43)
    assert len(store.all_opinions()) == 1


def test_publication_rejects_missing_coordinates_without_partial_visibility(tmp_path):
    store = Store(str(tmp_path / 'atlas.db')); store.migrate()
    req = request(); receipt = store.accept_submission(req)
    store.stage_revision(receipt.submission_id, 1,
        SplitResult((FeedbackSpan(0, len(req.raw_text)),), 'fixture'), ('human',))
    with pytest.raises(SubmissionError, match='INVALID_LAYOUT'):
        store.commit_publication(0, {receipt.submission_id: 1}, {}, 1)
    assert store.all_opinions() == []
    assert store.data_rev == 0
