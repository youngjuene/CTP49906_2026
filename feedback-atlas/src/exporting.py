"""Revision-bound CSV exports using published units and explicit scope."""
import csv
import io
import json

from src.submissions import SubmissionError


def csv_cell(value):
    if isinstance(value, str) and (value.lstrip().startswith(('=', '+', '-', '@'))
                                   or value.startswith(('\t', '\r', '\n'))):
        return "'" + value
    return value


def build_export(atlas, body):
    state, store = atlas.state, atlas.store
    revision = body.get('expected_data_rev')
    if type(revision) is not int:
        raise SubmissionError('MALFORMED')
    if revision != state.rev or revision != store.data_rev:
        raise SubmissionError('REVISION_CONFLICT')
    scope = body.get('scope')
    fmt = body.get('format', 'units')
    if scope not in ('all', 'visible', 'selected') or fmt not in ('units', 'originals', 'recovery'):
        raise SubmissionError('MALFORMED')
    wanted = body.get('ids')
    if scope != 'all':
        if not isinstance(wanted, list) or any(not isinstance(i, str) for i in wanted):
            raise SubmissionError('MALFORMED')
        if set(wanted) - {o.id for o in state.opinions}:
            raise SubmissionError('REVISION_CONFLICT')
    elif wanted is not None:
        raise SubmissionError('MALFORMED')
    keep = set(wanted) if scope != 'all' else None
    ops = [o for o in state.opinions if keep is None or o.id in keep]
    names = {t['id']: t['display_name'] for t in atlas.roster.targets()}
    buf = io.StringIO()
    writer = csv.writer(buf)
    if fmt == 'units':
        writer.writerow(['id', 'reviewer_id', 'reviewer_name', 'target_id', 'target_name',
                         'text', 'source', 'week', 'timestamp', 'x', 'y', 'submission_id',
                         'ordinal', 'revision', 'start_cp', 'end_cp', 'algorithm_key', 'data_rev'])
        parents = {}
        for op in ops:
            detail = parents.setdefault(op.submission_id, None)
            if detail is None:
                detail = store.submission_detail(op.submission_id, admin=True)
                parents[op.submission_id] = detail
            algorithm = next(h['algorithm_key'] for h in detail['history'] if h['revision'] == op.revision)
            reviewer = atlas.roster.resolve(op.reviewer_id)
            x, y = state.server_coords[op.id]
            writer.writerow([csv_cell(v) for v in [op.id, op.reviewer_id,
                reviewer.display_name if reviewer else op.reviewer_id, op.target_id,
                names.get(op.target_id, op.target_id), op.text, op.source, op.week,
                op.timestamp, x, y, op.submission_id, op.ordinal, op.revision,
                op.start_cp, op.end_cp, algorithm, revision]])
    elif fmt == 'originals':
        writer.writerow(['submission_id', 'reviewer_id', 'target_id', 'week', 'raw_text',
                         'source', 'effective_sources', 'revision', 'timestamp', 'data_rev'])
        for sid in dict.fromkeys(o.submission_id for o in ops):
            public = store.public_submission(sid)
            private = store.submission_detail(sid, admin=True)
            writer.writerow([csv_cell(v) for v in [sid, private['reviewer_id'], public['target_id'],
                public['week'], public['raw_text'], private['source'],
                ','.join(sorted({u['source'] for u in public['units']})), public['revision'],
                private['timestamp'], revision]])
    else:
        # Explicit instructor recovery archive includes accepted but unpublished work.
        if scope != 'all':
            raise SubmissionError('MALFORMED')
        writer.writerow(['submission_id', 'reviewer_id', 'target_id', 'week', 'raw_text',
                         'source', 'state', 'revision', 'published_revision', 'error_code',
                         'history_json', 'timestamp', 'data_rev'])
        for d in store.recovery_records():
            writer.writerow([csv_cell(v) for v in [d['submission_id'], d['reviewer_id'],
                d['target_id'], d['week'], d['raw_text'], d['source'], d['state'], d['revision'],
                d['published_revision'], d['error_code'], json.dumps(d['history'], ensure_ascii=False),
                d['timestamp'], revision]])
    return ('\ufeff' + buf.getvalue()).encode('utf-8')
