#!/usr/bin/env python
"""Offline import of exact raw feedback into the v2 durable parent queue.

Default: each input row is explicitly already unitized. --segment requests
semantic boundaries. Both modes preserve raw text and wait for the application's
shared model worker to embed/project/publish; no model is loaded by this script.
Stop the server for administrative offline imports, then start it to drain the
queue. Online imports require a separate authenticated intake API.

CSV/JSONL fields: target_id,text[,source][,week]. AI is the source default.
Identical file bytes, row positions, effective metadata and mode replay safely;
changing any of those denotes a new import. Separate identical rows stay distinct.
"""
import argparse
import csv
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import secrets
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.models import SOURCES,WEEKS
from src.roster import Roster
from src.store import Store
from src.submission_store import _digest
from src.submissions import FeedbackSpan,SplitResult,SubmissionError,SubmissionRequest


def read_rows(path: Path, *, content: bytes | None = None) -> list[dict]:
    text=(content if content is not None else path.read_bytes()).decode('utf-8-sig')
    if path.suffix.lower()=='.jsonl':
        rows=[json.loads(line) for line in text.splitlines() if line.strip()]
        if any(not isinstance(row,dict) for row in rows):
            raise SystemExit('JSONL rows must be objects')
        return rows
    reader=csv.DictReader(io.StringIO(text,newline=''))
    if reader.fieldnames is None:
        raise SystemExit(f'{path} is empty')
    reader.fieldnames=[(field or '').strip() for field in reader.fieldnames]
    missing={'target_id','text'}-set(reader.fieldnames)
    if missing:
        raise SystemExit(f'{path} is missing column(s): {", ".join(sorted(missing))}')
    return list(reader)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('input',type=Path)
    ap.add_argument('--db',type=Path,required=True)
    ap.add_argument('--roster',type=Path,required=True)
    ap.add_argument('--reviewer',required=True,help='canonical roster submitter (usually instructor)')
    ap.add_argument('--week',type=int,default=1,choices=WEEKS)
    ap.add_argument('--source',default='ai',choices=SOURCES)
    ap.add_argument('--max-chars',type=int,default=20000)
    ap.add_argument('--max-pending',type=int,default=300)
    ap.add_argument('--segment',action='store_true',help='request semantic segmentation; default rows are already unitized')
    ap.add_argument('--dry-run',action='store_true',help='validate without opening/creating a database; capacity not checked')
    args=ap.parse_args(argv)
    if not 1<=args.max_chars<=20000 or not 1<=args.max_pending<=300:
        ap.error('max-chars must be 1..20000 and max-pending 1..300')
    roster=Roster.from_path(args.roster)
    reviewer=roster.resolve(args.reviewer)
    if reviewer is None:
        raise SystemExit(f'--reviewer {args.reviewer!r} is not on the roster')
    targets={row['id'] for row in roster.targets()}
    content=args.input.read_bytes()
    file_digest=hashlib.sha256(content).hexdigest()
    prepared=[];skipped=[]
    for index,row in enumerate(read_rows(args.input,content=content),1):
        raw=row.get('text')
        target_raw=row.get('target_id')
        entry=roster.resolve(target_raw) if isinstance(target_raw,str) else None
        target=entry.id if entry and entry.id in targets else None
        source=str(row.get('source') or args.source).strip().lower()
        try: week=int(row.get('week') or args.week)
        except (ValueError,TypeError): week=None
        if not isinstance(raw,str) or not raw.strip(): reason='empty or non-string text'
        elif len(raw)>args.max_chars: reason=f'longer than {args.max_chars} code points'
        elif target is None: reason='unknown target'
        elif source not in SOURCES: reason='bad source'
        elif week not in WEEKS: reason='bad week'
        else: reason=None
        if reason:
            skipped.append((index,reason));continue
        identity=[file_digest,index,reviewer.id,target,week,source,'segment' if args.segment else 'unitized']
        nonce='import:'+hashlib.sha256(json.dumps(identity,separators=(',',':')).encode()).hexdigest()
        prepared.append((index,dict(reviewer_id=reviewer.id,target_id=target,week=week,source=source,
            raw_text=raw,nonce=nonce,context_revision=0,metadata_confirmed=True)))
    if args.dry_run:
        for index,why in skipped: print(f'  skipped row {index}: {why}',file=sys.stderr)
        print(f'would import {len(prepared)} parent(s), skipping {len(skipped)}; replay/capacity not checked')
        return int(bool(skipped))
    written=replayed=0
    store=Store(str(args.db))
    try:
        store.migrate()
        for index,fields in prepared:
            initial=None if args.segment else SplitResult((FeedbackSpan(0,len(fields['raw_text'])),),'import-unitized-v1')
            # Privileged offline replay lookup. Owner capabilities are random,
            # stored only as hashes, and deliberately never retained or printed.
            old=store._db.execute('SELECT * FROM submissions WHERE reviewer_id=? AND nonce=?',
                                  (fields['reviewer_id'],fields['nonce'])).fetchone()
            if old:
                expected=dict(fields)
                if initial is not None:
                    expected['initial_split']=asdict(initial)
                if old['request_hash'] != _digest(expected):
                    skipped.append((index,'NONCE_CONFLICT'));continue
                replayed+=1;continue
            request=SubmissionRequest(**fields,owner_capability=secrets.token_urlsafe(48))
            try:
                store.accept_submission(request,max_pending=args.max_pending,initial_split=initial)
                written+=1
            except SubmissionError as exc:
                skipped.append((index,exc.code))
    finally:
        store.close()
    for index,why in skipped: print(f'  skipped row {index}: {why}',file=sys.stderr)
    print(f'imported {written} parent(s); replayed {replayed}; skipped {len(skipped)}')
    if written:
        print('Raw originals saved. Start the server to process/publish the durable queue; no model ran here.')
    if any(why=='QUEUE_FULL' for _,why in skipped):
        print('Queue full: accepted rows remain saved. Drain the queue, then rerun the identical file and options.')
    return int(bool(skipped))


if __name__=='__main__':
    raise SystemExit(main())
