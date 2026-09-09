import csv
import io

from src.models import new_opinion
from src.textnorm import text_hash
from test_server_routes import client, CODE


def seed(client):
    atlas = client.app.state.atlas
    ids = []
    for i in range(3):
        op = new_opinion(reviewer_id='writer1', target_id='target1', text=f'내보낼 의견 {i}', week=1 if i < 2 else 2)
        atlas.store.insert_opinion(op, text_hash(op.text)); ids.append(op.id)
    atlas.runtime.run_sync(atlas.state.load)
    return ids


def rows(response):
    return list(csv.DictReader(io.StringIO(response.content.decode('utf-8-sig'))))


def test_visible_scope_matches_ids_and_keeps_parent_provenance(client):
    ids = seed(client)
    response = client.post('/api/export.csv', json=dict(code=CODE, scope='visible', ids=ids[:2],
                           expected_data_rev=client.app.state.atlas.state.rev, format='units'))
    assert response.status_code == 200
    actual = rows(response)
    assert {r['id'] for r in actual} == set(ids[:2])
    assert all(r['submission_id'] and r['revision'] == '1' for r in actual)


def test_empty_selection_never_falls_back_to_everything(client):
    seed(client)
    response = client.post('/api/export.csv', json=dict(code=CODE, scope='selected', ids=[],
                           expected_data_rev=client.app.state.atlas.state.rev))
    assert response.status_code == 200 and rows(response) == []


def test_export_rejects_stale_revision_and_missing_visible_ids(client):
    seed(client)
    response = client.post('/api/export.csv', json=dict(code=CODE, scope='all', expected_data_rev=0))
    assert response.status_code == 409
    response = client.post('/api/export.csv', json=dict(code=CODE, scope='visible',
                           expected_data_rev=client.app.state.atlas.state.rev))
    assert response.status_code == 400


def test_original_export_is_one_row_per_parent(client):
    seed(client)
    response = client.post('/api/export.csv', json=dict(code=CODE, scope='all', format='originals',
                           expected_data_rev=client.app.state.atlas.state.rev))
    assert response.status_code == 200
    actual = rows(response)
    assert len(actual) == 3 and all('raw_text' in row and 'source' in row for row in actual)


def test_recovery_export_includes_pending_original_without_owner_secrets(client):
    from test_submission_intake_v2 import CAP
    from src.submissions import SubmissionRequest
    atlas = client.app.state.atlas
    raw = '  아직 처리되지 않은 원문\n'
    receipt = atlas.store.accept_submission(SubmissionRequest('writer1', 'target1', 1, 'ai', raw,
                                           'recovery', CAP, 0))
    response = client.post('/api/export.csv', json=dict(code=CODE, scope='all', format='recovery',
                           expected_data_rev=atlas.state.rev))
    actual = rows(response)
    assert len(actual) == 1 and actual[0]['raw_text'] == raw
    assert actual[0]['submission_id'] == receipt.submission_id
    assert 'capability' not in response.text and CAP not in response.text
