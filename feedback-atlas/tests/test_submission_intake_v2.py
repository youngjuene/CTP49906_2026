import time

from test_server_routes import client, _hello, _drain_until, CODE

CAP = 'x' * 43


def envelope(**changes):
    body = dict(t='submit', nonce='new-parent', target_id='target1', source='human',
                week=2, text='  좋은 조명입니다.\n\n음악이 너무 큽니다. 🎵  ',
                owner_capability=CAP, context_revision=0)
    body.update(changes)
    return body


def joined(client):
    return client.websocket_connect('/ws')


def test_v2_accepts_long_raw_text_and_returns_a_parent(client):
    with joined(client) as ws:
        hello = _hello(ws, id='writer1'); _drain_until(ws, 'snapshot')
        assert hello['protocol'] == 2
        assert hello['max_text_chars'] == 20000
        assert hello['dataset_id']
        raw = '  긴 의견입니다.\n' * 110
        ws.send_json(envelope(text=raw))
        accepted = ws.receive_json()
        assert accepted['t'] == 'submission_accepted'
        assert accepted['state'] == 'queued'
        sid = accepted['submission_id']
        ws.send_json(dict(t='submission_status', submission_id=sid,
                          owner_capability=CAP, request_id='status-1'))
        detail = None
        for _ in range(8):
            message = ws.receive_json()
            if message.get('request_id') == 'status-1':
                detail = message
                break
        assert detail is not None
        assert detail['raw_text'] == raw and detail['source'] == 'human'
        assert detail['request_id'] == 'status-1'
        assert CAP not in str(detail)


def test_missing_owner_capability_is_refused(client):
    with joined(client) as ws:
        _hello(ws, id='writer1'); _drain_until(ws, 'snapshot')
        ws.send_json(envelope(owner_capability=''))
        reply = ws.receive_json()
        assert reply['t'] == 'error' and reply['code'] == 'NOT_AUTHORIZED'


def test_same_nonce_returns_same_parent_and_changed_text_conflicts(client):
    with joined(client) as ws:
        _hello(ws, id='writer1'); _drain_until(ws, 'snapshot')
        ws.send_json(envelope())
        first = ws.receive_json()
        assert first['t'] == 'submission_accepted'
        ws.send_json(envelope())
        second = _drain_until(ws, 'submission_accepted')
        assert second['submission_id'] == first['submission_id']
        ws.send_json(envelope(text='같은 전송 ID로 다른 내용'))
        assert _drain_until(ws, 'error')['code'] == 'NONCE_CONFLICT'


def test_unit_delivery_and_original_reader_do_not_reveal_submitter(client):
    with joined(client) as writer, joined(client) as reader:
        _hello(writer, id='writer1'); _drain_until(writer, 'snapshot')
        _hello(reader, id='writer2'); _drain_until(reader, 'snapshot')
        writer.send_json(envelope(text='공간이 편안했습니다.'))
        accepted = writer.receive_json()
        assert accepted['t'] == 'submission_accepted'
        sid = accepted['submission_id']
        frame = reader.receive_json()
        assert frame['t'] in ('delta', 'snapshot')
        points = frame.get('added', frame.get('points'))
        assert len(points) == 1 and points[0]['submission_id'] == sid
        assert 'writer1' not in str(frame) and CAP not in str(frame)
        reader.send_json(dict(t='submission_original', submission_id=sid, request_id='original'))
        original = _drain_until(reader, 'submission_original')
        assert original['raw_text'] == '공간이 편안했습니다.'
        assert not {'reviewer_id','history','owner_capability'} & original.keys()
        reader.send_json(dict(t='submission_status', submission_id=sid, owner_capability=CAP))
        assert _drain_until(reader, 'error')['code'] == 'NOT_AUTHORIZED'


def test_malformed_identifiers_are_refused_without_dropping_socket(client):
    with joined(client) as ws:
        _hello(ws, id='writer1'); _drain_until(ws, 'snapshot')
        for body in [envelope(nonce={}), dict(t='submission_original', submission_id=[]),
                     dict(t='submission_status', submission_id={}, owner_capability=CAP),
                     dict(t='submission_receipt', nonce=[], owner_capability=CAP)]:
            ws.send_json(body)
            assert _drain_until(ws, 'error')['code'] == 'MALFORMED'
        ws.send_json(dict(t='ping'))
        assert _drain_until(ws, 'pong')['t'] == 'pong'


def test_fractional_week_is_not_silently_truncated(client):
    with joined(client) as ws:
        _hello(ws, id='writer1'); _drain_until(ws, 'snapshot')
        ws.send_json(envelope(week=1.5))
        assert _drain_until(ws, 'error')['code'] == 'BAD_WEEK'
