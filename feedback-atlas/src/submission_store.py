"""SQLite parent submissions, staged revisions, and ownership checks.

Used by Store; transaction/lock ownership stays with the same SQLite connection.
No model work, socket writes, or private fields in public readers live here.
"""
import hashlib
import hmac
import json
import time
import uuid
from dataclasses import asdict

from src.models import Opinion, _now_iso
from src.submissions import (FeedbackSpan, SplitResult, SubmissionError,
                             SubmissionReceipt, SubmissionRequest, validate_partition)
from src.textnorm import text_hash

SUBMISSION_SCHEMA = """
CREATE TABLE IF NOT EXISTS submissions (
 id TEXT PRIMARY KEY, reviewer_id TEXT NOT NULL, target_id TEXT NOT NULL,
 week INTEGER NOT NULL CHECK(week BETWEEN 1 AND 4),
 source TEXT NOT NULL CHECK(source IN ('ai','human')), raw_text TEXT NOT NULL,
 nonce TEXT, request_hash TEXT, capability_hash TEXT, timestamp TEXT NOT NULL,
 revision INTEGER NOT NULL DEFAULT 1, published_revision INTEGER NOT NULL DEFAULT 0,
 state TEXT NOT NULL CHECK(state IN ('queued','splitting','embedding','ready','failed','withdrawn')),
 attempts INTEGER NOT NULL DEFAULT 0, next_attempt REAL NOT NULL DEFAULT 0,
 error_code TEXT, UNIQUE(reviewer_id,nonce)
);
CREATE TABLE IF NOT EXISTS submission_revisions (
 submission_id TEXT NOT NULL REFERENCES submissions(id), revision INTEGER NOT NULL,
 target_id TEXT NOT NULL, week INTEGER NOT NULL CHECK(week BETWEEN 1 AND 4),
 manifest TEXT NOT NULL, algorithm_key TEXT NOT NULL, actor_kind TEXT NOT NULL,
 timestamp TEXT NOT NULL, PRIMARY KEY(submission_id,revision)
);
CREATE TABLE IF NOT EXISTS submission_units (
 submission_id TEXT NOT NULL, revision INTEGER NOT NULL, ordinal INTEGER NOT NULL,
 opinion_id TEXT NOT NULL REFERENCES opinions(id), start_cp INTEGER NOT NULL,
 end_cp INTEGER NOT NULL, source TEXT NOT NULL CHECK(source IN ('ai','human')),
 PRIMARY KEY(submission_id,revision,ordinal),
 UNIQUE(submission_id,revision,start_cp,end_cp), UNIQUE(opinion_id),
 FOREIGN KEY(submission_id,revision) REFERENCES submission_revisions(submission_id,revision)
);
CREATE TABLE IF NOT EXISTS submission_actions (
 submission_id TEXT NOT NULL REFERENCES submissions(id), action_nonce TEXT NOT NULL,
 request_hash TEXT NOT NULL, result_revision INTEGER NOT NULL,
 PRIMARY KEY(submission_id,action_nonce)
);
CREATE INDEX IF NOT EXISTS submissions_pending ON submissions(state,next_attempt,timestamp);
"""


def _digest(value) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def _capability(value: str) -> str:
    if not isinstance(value, str) or not 32 <= len(value) <= 128:
        raise SubmissionError('NOT_AUTHORIZED')
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _receipt(row) -> SubmissionReceipt:
    return SubmissionReceipt(row['id'], row['nonce'] or '', row['revision'], row['state'])


class SubmissionStoreMixin:
    @property
    def data_rev(self) -> int:
        return int(self.get_meta('data_rev') or 0)

    @property
    def dataset_id(self) -> str:
        return self.get_meta('dataset_id')

    def _bump_rev(self) -> int:
        rev = self.data_rev + 1
        self._db.execute("INSERT INTO meta VALUES('data_rev',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(rev),))
        return rev

    def _parent(self, sid):
        if not isinstance(sid, str) or not 1 <= len(sid) <= 128:
            raise SubmissionError("MALFORMED")
        row = self._db.execute('SELECT * FROM submissions WHERE id=?', (sid,)).fetchone()
        if row is None:
            raise SubmissionError('UNKNOWN_SUBMISSION')
        return row

    def _authorize(self, row, reviewer_id, capability, admin=False):
        if admin:
            return
        given = _capability(capability)
        if row['reviewer_id'] != reviewer_id or not row['capability_hash'] or not hmac.compare_digest(given, row['capability_hash']):
            raise SubmissionError('NOT_AUTHORIZED')

    def accept_submission(self, request: SubmissionRequest, *, max_pending=300,
                          initial_split: SplitResult | None = None) -> SubmissionReceipt:
        fields = asdict(request)
        capability_hash = _capability(fields.pop('owner_capability'))
        if initial_split is not None:
            fields['initial_split'] = asdict(initial_split)
        fingerprint = _digest(fields)
        if not isinstance(request.raw_text, str) or not request.raw_text.strip():
            raise SubmissionError('EMPTY_TEXT')
        if len(request.raw_text) > 20000:
            raise SubmissionError('TEXT_TOO_LONG')
        if not isinstance(request.nonce, str) or not 1 <= len(request.nonce) <= 128:
            raise SubmissionError('MALFORMED')
        with self.transaction():
            old = self._db.execute('SELECT * FROM submissions WHERE reviewer_id=? AND nonce=?',
                                   (request.reviewer_id, request.nonce)).fetchone()
            if old:
                if old['request_hash'] != fingerprint or not old['capability_hash'] or not hmac.compare_digest(old['capability_hash'], capability_hash):
                    raise SubmissionError('NONCE_CONFLICT')
                return _receipt(old)
            count = self._db.execute("SELECT COUNT(*) FROM submissions WHERE state IN ('queued','splitting','embedding')").fetchone()[0]
            if count >= max_pending:
                raise SubmissionError('QUEUE_FULL')
            sid = 's_' + uuid.uuid4().hex
            self._db.execute('''INSERT INTO submissions
                (id,reviewer_id,target_id,week,source,raw_text,nonce,request_hash,capability_hash,timestamp,state)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)''',
                (sid, request.reviewer_id, request.target_id, request.week, request.source,
                 request.raw_text, request.nonce, fingerprint, capability_hash, _now_iso(), 'queued'))
            if initial_split is not None:
                self._stage(self._parent(sid), 1, initial_split,
                            (request.source,) * len(initial_split.spans), actor='admin')
            return _receipt(self._parent(sid))

    def find_receipt(self, reviewer_id, nonce, capability):
        if not isinstance(nonce, str) or not 1 <= len(nonce) <= 128:
            raise SubmissionError("MALFORMED")
        _capability(capability)
        with self._lock:
            row = self._db.execute('SELECT * FROM submissions WHERE reviewer_id=? AND nonce=?',
                                   (reviewer_id, nonce)).fetchone()
            if row is None:
                return None
            self._authorize(row, reviewer_id, capability)
            return _receipt(row)

    def read_receipt(self, submission_id, reviewer_id, capability):
        with self._lock:
            row = self._parent(submission_id)
            self._authorize(row, reviewer_id, capability)
            return _receipt(row)

    def _legacy_parent(self, op, nonce=None):
        sid = 's_legacy_' + op.id
        if self._db.execute('SELECT 1 FROM submissions WHERE id=?', (sid,)).fetchone():
            return
        self._db.execute('''INSERT INTO submissions
            (id,reviewer_id,target_id,week,source,raw_text,nonce,timestamp,published_revision,state)
            VALUES(?,?,?,?,?,?,?,?,1,'ready')''',
            (sid, op.reviewer_id, op.target_id, op.week, op.source, op.text, nonce, op.timestamp))
        manifest = json.dumps([{'start': 0, 'end': len(op.text), 'source': op.source}])
        self._db.execute('INSERT INTO submission_revisions VALUES(?,?,?,?,?,?,?,?)',
                         (sid, 1, op.target_id, op.week, manifest, 'legacy_normalized', 'legacy', op.timestamp))
        self._db.execute('INSERT INTO submission_units VALUES(?,?,?,?,?,?,?)',
                         (sid, 1, 0, op.id, 0, len(op.text), op.source))

    def _stage(self, row, revision, result, sources, *, actor='system', target_id=None, week=None):
        validate_partition(row['raw_text'], result.spans)
        if len(sources) != len(result.spans) or any(s not in ('ai', 'human') for s in sources):
            raise SubmissionError('BAD_SOURCE')
        target_id = target_id or row['target_id']
        week = week if week is not None else row['week']
        manifest = json.dumps([{'start': s.start, 'end': s.end, 'source': source}
                               for s, source in zip(result.spans, sources, strict=True)])
        old = self._db.execute('SELECT * FROM submission_revisions WHERE submission_id=? AND revision=?',
                               (row['id'], revision)).fetchone()
        if old and (old['manifest'] != manifest or old['target_id'] != target_id or old['week'] != week):
            raise SubmissionError('REVISION_CONFLICT')
        self._db.execute('INSERT OR IGNORE INTO submission_revisions VALUES(?,?,?,?,?,?,?,?)',
                         (row['id'], revision, target_id, week, manifest, result.algorithm_key, actor, _now_iso()))
        ids = []
        for ordinal, (span, source) in enumerate(zip(result.spans, sources, strict=True)):
            oid = 'o_' + uuid.uuid5(uuid.NAMESPACE_URL, f"atlas:{row['id']}:{revision}:{ordinal}").hex
            text = row['raw_text'][span.start:span.end]
            self._db.execute('INSERT OR IGNORE INTO opinions VALUES(?,?,?,?,?,?,?,?)',
                (oid, row['reviewer_id'], target_id, text, source, week, row['timestamp'], text_hash(text)))
            self._db.execute('INSERT OR IGNORE INTO submission_units VALUES(?,?,?,?,?,?,?)',
                             (row['id'], revision, ordinal, oid, span.start, span.end, source))
            ids.append(oid)
        self._db.execute("UPDATE submissions SET state='embedding',error_code=NULL WHERE id=?", (row['id'],))
        return tuple(ids)

    def stage_revision(self, submission_id, expected_revision, result, sources):
        with self.transaction():
            row = self._parent(submission_id)
            if row['revision'] != expected_revision or row['state'] == 'withdrawn':
                raise SubmissionError('REVISION_CONFLICT')
            return self._stage(row, expected_revision, result, sources)

    def staged_opinions(self, parent_revisions):
        """Immutable candidate corpus with only specified staged replacements."""
        with self._lock:
            active = [o for o in self.all_opinions() if o.submission_id not in parent_revisions]
            for sid, rev in parent_revisions.items():
                rows = self._db.execute('''SELECT o.*,u.submission_id,u.ordinal,u.revision,u.start_cp,u.end_cp
                    FROM opinions o JOIN submission_units u ON u.opinion_id=o.id
                    WHERE u.submission_id=? AND u.revision=? ORDER BY u.ordinal''', (sid, rev)).fetchall()
                active.extend(self._opinion_row(row) for row in rows)
            return sorted(active, key=lambda o: (o.timestamp, o.submission_id or o.id, o.ordinal))

    def commit_publication(self, expected_data_rev, parent_revisions, coords, layout_rev):
        import math
        with self.transaction():
            if self.data_rev != expected_data_rev:
                raise SubmissionError('REVISION_CONFLICT')
            for sid, rev in parent_revisions.items():
                row = self._parent(sid)
                if row['revision'] != rev or row['state'] == 'withdrawn':
                    raise SubmissionError('REVISION_CONFLICT')
            candidate = self.staged_opinions(parent_revisions)
            if set(coords) != {o.id for o in candidate} or any(
                    len(xy) != 2 or not all(math.isfinite(float(x)) for x in xy) for xy in coords.values()):
                raise SubmissionError('INVALID_LAYOUT')
            for sid, rev in parent_revisions.items():
                self._db.execute("UPDATE submissions SET published_revision=?,state='ready',error_code=NULL,attempts=0 WHERE id=?", (rev, sid))
            self._db.execute('DELETE FROM coords')
            self._db.executemany('INSERT INTO coords VALUES(?,?,?,?)',
                                 [(oid, float(xy[0]), float(xy[1]), layout_rev) for oid, xy in coords.items()])
            self._db.execute("INSERT INTO meta VALUES('layout_rev',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(layout_rev),))
            return self._bump_rev()

    @staticmethod
    def _opinion_row(row):
        return Opinion(**{key: row[key] for key in ('id', 'reviewer_id', 'target_id', 'text', 'source',
                                                     'week', 'timestamp', 'submission_id', 'ordinal',
                                                     'revision', 'start_cp', 'end_cp')})

    def _units(self, sid, revision):
        return [dict(row) for row in self._db.execute('''SELECT u.opinion_id AS id,u.ordinal,
            u.start_cp AS start,u.end_cp AS end,u.source,o.text FROM submission_units u
            JOIN opinions o ON o.id=u.opinion_id WHERE u.submission_id=? AND u.revision=? ORDER BY u.ordinal''',
            (sid, revision)).fetchall()]

    def public_submission(self, submission_id):
        if not isinstance(submission_id, str) or not 1 <= len(submission_id) <= 128:
            raise SubmissionError("MALFORMED")
        with self._lock:
            row = self._db.execute('SELECT * FROM submissions WHERE id=?', (submission_id,)).fetchone()
            if not row or not row['published_revision'] or row['state'] == 'withdrawn':
                return None
            rev = self._db.execute('SELECT * FROM submission_revisions WHERE submission_id=? AND revision=?',
                                   (submission_id, row['published_revision'])).fetchone()
            return {'submission_id': row['id'], 'raw_text': row['raw_text'],
                    'revision': rev['revision'], 'target_id': rev['target_id'], 'week': rev['week'],
                    'units': self._units(row['id'], rev['revision'])}

    def submission_detail(self, submission_id, reviewer_id=None, capability=None, *, admin=False):
        with self._lock:
            row = self._parent(submission_id)
            self._authorize(row, reviewer_id, capability, admin)
            result = {key: row[key] for key in ('raw_text', 'source', 'target_id', 'week', 'revision',
                       'published_revision', 'state', 'attempts', 'next_attempt', 'error_code', 'timestamp')}
            result.update(submission_id=row['id'], nonce=row['nonce'], units=self._units(row['id'], row['revision']))
            result['history'] = [dict(x) for x in self._db.execute('''SELECT revision,manifest,algorithm_key,target_id,week,
                actor_kind,timestamp FROM submission_revisions WHERE submission_id=? ORDER BY revision''', (row['id'],))]
            if admin:
                result['reviewer_id'] = row['reviewer_id']
            return result

    def recovery_records(self):
        """Admin-only caller; exclude owner secrets and request hashes by construction."""
        with self._lock:
            ids = [r[0] for r in self._db.execute('SELECT id FROM submissions ORDER BY timestamp,id')]
            return [self.submission_detail(sid, admin=True) for sid in ids]

    def _action(self, row, nonce, fingerprint):
        if not isinstance(nonce, str) or not 1 <= len(nonce) <= 128:
            raise SubmissionError('MALFORMED')
        previous = self._db.execute('SELECT * FROM submission_actions WHERE submission_id=? AND action_nonce=?',
                                    (row['id'], nonce)).fetchone()
        if previous and previous['request_hash'] != fingerprint:
            raise SubmissionError('NONCE_CONFLICT')
        return previous

    def withdraw_submission(self, submission_id, expected_revision, action_nonce, reviewer_id, capability, *, admin=False):
        fingerprint = _digest(['withdraw', expected_revision])
        with self.transaction():
            row = self._parent(submission_id)
            self._authorize(row, reviewer_id, capability, admin)
            previous = self._action(row, action_nonce, fingerprint)
            if previous:
                return previous['result_revision']
            if row['revision'] != expected_revision or row['state'] == 'withdrawn':
                raise SubmissionError('REVISION_CONFLICT')
            revision = expected_revision + 1
            self._db.execute("UPDATE submissions SET state='withdrawn',revision=? WHERE id=?", (revision, submission_id))
            self._db.execute('DELETE FROM coords WHERE opinion_id IN (SELECT opinion_id FROM submission_units WHERE submission_id=?)', (submission_id,))
            self._db.execute('INSERT INTO submission_actions VALUES(?,?,?,?)', (submission_id, action_nonce, fingerprint, revision))
            self._bump_rev()
            return revision

    def correct_submission(self, submission_id, expected_revision, action_nonce, reviewer_id, capability,
                           spans, sources, target_id, week, *, admin=False):
        """Stage one complete immutable replacement; the old revision stays public."""
        manifest = tuple(spans)
        digest = _digest(['correct', expected_revision,
                          [(s.start, s.end) for s in manifest], sources, target_id, week])
        with self.transaction():
            parent = self._parent(submission_id)
            self._authorize(parent, reviewer_id, capability, admin)
            previous = self._action(parent, action_nonce, digest)
            if previous:
                return previous['result_revision']
            if parent['revision'] != expected_revision or parent['state'] == 'withdrawn':
                raise SubmissionError('REVISION_CONFLICT')
            validate_partition(parent['raw_text'], manifest)
            revision = expected_revision + 1
            self._stage(parent, revision, SplitResult(manifest, 'manual'), tuple(sources),
                        actor='admin' if admin else 'owner', target_id=target_id, week=week)
            self._db.execute('''UPDATE submissions SET revision=?,target_id=?,week=?,
                state='embedding',attempts=0,next_attempt=0,error_code=NULL WHERE id=?''',
                (revision, target_id, week, submission_id))
            self._db.execute('INSERT INTO submission_actions VALUES(?,?,?,?)',
                             (submission_id, action_nonce, digest, revision))
            return revision

    def pending_jobs(self, limit=4):
        with self._lock:
            return [dict(r) for r in self._db.execute("""SELECT * FROM submissions
                WHERE state IN ('queued','embedding') AND next_attempt<=? ORDER BY timestamp,id LIMIT ?""", (time.time(), limit))]

    def job_started(self, sid, revision):
        with self.transaction():
            row = self._parent(sid)
            if row['revision'] != revision or row['state'] == 'withdrawn':
                raise SubmissionError('REVISION_CONFLICT')
            staged = self._db.execute('SELECT 1 FROM submission_revisions WHERE submission_id=? AND revision=?', (sid, revision)).fetchone()
            self._db.execute('UPDATE submissions SET state=?,attempts=attempts+1 WHERE id=?',
                             ('embedding' if staged else 'splitting', sid))
            return bool(staged)

    def job_deferred(self, sid, revision):
        """A stale publication is scheduling contention, not a model failure."""
        with self.transaction():
            row = self._parent(sid)
            if row['revision'] == revision and row['state'] not in ('withdrawn', 'ready'):
                self._db.execute("UPDATE submissions SET state='queued',attempts=MAX(0,attempts-1),next_attempt=0 WHERE id=?", (sid,))

    def job_failed(self, sid, revision, code='PROCESSING_FAILED', *, transient=True):
        with self.transaction():
            row = self._parent(sid)
            if row['revision'] != revision or row['state'] == 'withdrawn':
                return
            retry = transient and row['attempts'] < 3
            delay = 1 if row['attempts'] <= 1 else 5
            self._db.execute('UPDATE submissions SET state=?,next_attempt=?,error_code=? WHERE id=?',
                             ('queued' if retry else 'failed', time.time() + delay if retry else 0, code, sid))

    def retry_submission(self, sid, reviewer_id=None, capability=None, *, admin=False):
        with self.transaction():
            row = self._parent(sid)
            self._authorize(row, reviewer_id, capability, admin)
            if row['state'] == 'failed':
                self._db.execute("UPDATE submissions SET state='queued',attempts=0,next_attempt=0,error_code=NULL WHERE id=?", (sid,))
            return _receipt(self._parent(sid))

    def recover_jobs(self):
        with self.transaction():
            self._db.execute("UPDATE submissions SET state='queued',next_attempt=0 WHERE state IN ('splitting','embedding')")

    def processing_counts(self):
        with self._lock:
            return {r['state']: r['n'] for r in self._db.execute('SELECT state,COUNT(*) AS n FROM submissions GROUP BY state')}

    def queue_diagnostics(self):
        from datetime import datetime, timezone
        with self._lock:
            oldest = self._db.execute("SELECT MIN(timestamp) FROM submissions WHERE state IN ('queued','splitting','embedding')").fetchone()[0]
            age = max(0, (datetime.now(timezone.utc) - datetime.fromisoformat(oldest.replace('Z','+00:00'))).total_seconds()) if oldest else 0
            failed = {'splitting': 0, 'embedding': 0}
            for row in self._db.execute("""SELECT CASE WHEN EXISTS (
                    SELECT 1 FROM submission_revisions r WHERE r.submission_id=s.id AND r.revision=s.revision)
                    THEN 'embedding' ELSE 'splitting' END AS stage,COUNT(*) AS n
                    FROM submissions s WHERE state='failed' GROUP BY stage"""):
                failed[row['stage']] = row['n']
            return {'oldest_pending_seconds': round(age, 1), 'failed_by_stage': failed}
