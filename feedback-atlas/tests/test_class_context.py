from dataclasses import asdict

from test_server_routes import client, CODE, _hello, _drain_until
from test_submission_intake_v2 import envelope


def test_context_changes_require_confirmation_without_retargeting(client):
    with client.websocket_connect('/ws/admin') as admin, client.websocket_connect('/ws') as writer:
        _hello(admin, code=CODE); _drain_until(admin, 'snapshot')
        _hello(writer, id='writer1'); _drain_until(writer, 'snapshot')
        admin.send_json(dict(t='class_context_update', expected_revision=0,
                             week=3, target_id='target2', accepting=True))
        assert _drain_until(admin, 'class_context')['context']['revision'] == 1
        assert _drain_until(writer, 'class_context')['context']['target_id'] == 'target2'
        writer.send_json(envelope())
        assert _drain_until(writer, 'error')['code'] == 'STALE_CONTEXT'
        writer.send_json(envelope(metadata_confirmed=True))
        accepted = _drain_until(writer, 'submission_accepted')
        store = client.app.state.atlas.store
        detail = store.submission_detail(accepted['submission_id'], admin=True)
        assert detail['target_id'] == 'target1' and detail['week'] == 2


def test_closing_intake_still_returns_accepted_receipts(client):
    with client.websocket_connect('/ws/admin') as admin, client.websocket_connect('/ws') as writer:
        _hello(admin, code=CODE); _drain_until(admin, 'snapshot')
        _hello(writer, id='writer1'); _drain_until(writer, 'snapshot')
        writer.send_json(envelope(text='짧은 의견'))
        first = _drain_until(writer, 'submission_accepted')
        admin.send_json(dict(t='class_context_update', expected_revision=0, week=2,
                             target_id='target1', accepting=False))
        _drain_until(writer, 'class_context')
        writer.send_json(envelope(text='짧은 의견'))
        replay = _drain_until(writer, 'submission_accepted')
        assert replay['submission_id'] == first['submission_id']
        writer.send_json(envelope(nonce='new-one'))
        assert _drain_until(writer, 'error')['code'] == 'CLASS_CLOSED'


def test_participant_cannot_change_class_context(client):
    with client.websocket_connect('/ws') as writer:
        _hello(writer, id='writer1'); _drain_until(writer, 'snapshot')
        writer.send_json(dict(t='class_context_update', expected_revision=0, week=3,
                             target_id='target2', accepting=False))
        assert _drain_until(writer, 'error')['code'] == 'NOT_AUTHORIZED'
