"""V2 websocket commands. Identity comes from the resolved connection."""
from dataclasses import asdict
from types import SimpleNamespace

from src.class_context import current_context, update_context
from src.hub import Channel
from src.models import validate_submission
from src.protocol import error
from src.submissions import FeedbackSpan, SubmissionRequest, SubmissionError

COMMANDS = {'submit', 'submission_status', 'submission_original', 'submission_retry',
            'submission_correct', 'submission_withdraw', 'submission_receipt', 'class_context_update'}


def intake_request(body, entry, atlas):
    raw = body.get('text')
    if not isinstance(raw, str) or not raw.strip():
        raise SubmissionError('EMPTY_TEXT')
    if len(raw) > atlas.cfg.max_text_chars:
        raise SubmissionError('TEXT_TOO_LONG')
    if type(body.get('week', atlas.cfg.default_week)) is not int:
        raise SubmissionError('BAD_WEEK')
    op, err = validate_submission({**body, 'text': 'metadata validation'}, reviewer=entry,
        roster=atlas.roster, default_week=atlas.cfg.default_week)
    if err:
        raise SubmissionError(err)
    if type(body.get('context_revision')) is not int or type(body.get('metadata_confirmed', False)) is not bool:
        raise SubmissionError('MALFORMED')
    return SubmissionRequest(entry.id, op.target_id, op.week, op.source, raw,
        body.get('nonce'), body.get('owner_capability'), body['context_revision'],
        body.get('metadata_confirmed', False))


def hello_metadata(atlas):
    return {'dataset_id': atlas.store.dataset_id,
            'projection_method': atlas.state.projection_method,
            'context': asdict(current_context(atlas.store, atlas.cfg.default_week)),
            'processing': atlas.store.processing_counts(),
            'segmentation_reviewed': atlas.loop.policy.classroom_enabled}


async def handle_submission(atlas, socket, conn_id, entry, channel, kind, body):
    if kind not in COMMANDS:
        return False
    store = atlas.store
    admin = channel is Channel.ADMIN
    identity = entry.id if entry else None
    request_id = body.get('request_id')
    extra = {'request_id': request_id} if isinstance(request_id, str) and len(request_id) <= 128 else {}

    async def reply(message):
        await atlas.hub.send(conn_id, {**message, **extra})

    def subscribe(sid):
        conn = atlas.hub.get(conn_id)
        if conn:
            conn.meta.setdefault('submissions', set()).add(sid)

    try:
        if kind == 'class_context_update':
            if not admin:
                raise SubmissionError('NOT_AUTHORIZED')
            ctx = update_context(store, atlas.roster, expected_revision=body.get('expected_revision'),
                week=body.get('week'), target_id=body.get('target_id'), accepting=body.get('accepting'),
                default_week=atlas.cfg.default_week)
            message = {'t': 'class_context', 'context': asdict(ctx)}
            await reply(message)
            await atlas.hub.broadcast(Channel.PARTICIPANT, message)
            await atlas.hub.broadcast(Channel.ADMIN, message)
        elif kind == 'submit':
            if entry is None or admin:
                raise SubmissionError('NOT_AUTHENTICATED')
            req = intake_request(body, entry, atlas)
            existing = store.find_receipt(identity, req.nonce, req.owner_capability)
            if existing is None:
                ctx = current_context(store, atlas.cfg.default_week)
                if not ctx.accepting:
                    raise SubmissionError('CLASS_CLOSED')
                if req.context_revision != ctx.revision and not req.metadata_confirmed:
                    raise SubmissionError('STALE_CONTEXT')
                if not atlas.limiter.allow(conn_id):
                    raise SubmissionError('RATE_LIMITED')
            receipt = store.accept_submission(req)
            subscribe(receipt.submission_id)
            await reply({'t': 'submission_accepted', **asdict(receipt)})
            atlas.loop.request()
        elif kind == 'submission_receipt':
            if not identity:
                raise SubmissionError('NOT_AUTHENTICATED')
            receipt = store.find_receipt(identity, body.get('nonce'), body.get('owner_capability'))
            if receipt:
                subscribe(receipt.submission_id)
            await reply({'t': 'submission_receipt', 'receipt': asdict(receipt) if receipt else None})
        elif kind == 'submission_original':
            detail = store.public_submission(body.get('submission_id'))
            if detail is None:
                raise SubmissionError('UNKNOWN_SUBMISSION')
            await reply({'t': 'submission_original', **detail})
        else:
            sid = body.get('submission_id')
            cap = body.get('owner_capability')
            # Authorize before mutation or subscription. Public IDs confer no authority.
            detail = store.submission_detail(sid, identity, cap, admin=admin)
            subscribe(sid)
            if kind == 'submission_retry':
                if not atlas.limiter.allow(conn_id):
                    raise SubmissionError('RATE_LIMITED')
                store.retry_submission(sid, identity, cap, admin=admin)
                atlas.loop.request()
            elif kind in ('submission_correct', 'submission_withdraw'):
                if type(body.get('expected_revision')) is not int:
                    raise SubmissionError('MALFORMED')
                if not atlas.limiter.allow(conn_id):
                    raise SubmissionError('RATE_LIMITED')
                args = (sid, body['expected_revision'], body.get('action_nonce'), identity, cap)
                if kind == 'submission_withdraw':
                    store.withdraw_submission(*args, admin=admin)
                    await atlas.loop.refresh_published()
                    atlas.loop.request(reload=True)
                else:
                    units = body.get('units')
                    if not isinstance(units, list) or not 1 <= len(units) <= 64 or any(not isinstance(u, dict) for u in units):
                        raise SubmissionError('INVALID_SEGMENTS')
                    validator = SimpleNamespace(id=detail['reviewer_id']) if admin else entry
                    if type(body.get('week')) is not int:
                        raise SubmissionError('BAD_WEEK')
                    op, err = validate_submission({'text': 'metadata validation', 'target_id': body.get('target_id'),
                        'week': body.get('week'), 'source': detail['source']}, reviewer=validator, roster=atlas.roster)
                    if err:
                        raise SubmissionError(err)
                    spans = tuple(FeedbackSpan(u.get('start'), u.get('end')) for u in units)
                    store.correct_submission(*args, spans, tuple(u.get('source') for u in units), op.target_id, op.week, admin=admin)
                    atlas.loop.request()
            detail = store.submission_detail(sid, identity, cap, admin=admin)
            await reply({'t': 'submission_detail', **detail})
    except SubmissionError as exc:
        message = error(exc.code)
        if isinstance(body.get('nonce'), str):
            message['nonce'] = body['nonce'][:128]
        await reply(message)
    except (TypeError, ValueError, KeyError):
        await reply(error('MALFORMED'))
    return True
