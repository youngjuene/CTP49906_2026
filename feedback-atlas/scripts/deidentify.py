#!/usr/bin/env python
"""Create a pseudonymous v2 SQLite archive and CSVs without modifying the source.

archive.db preserves every original, revision and unit, including pending/failed
work; archive.csv contains currently published units; originals.csv contains all
parents. mapping.csv is separate and identifies only the randomized roster codes.
Owner/nonces/action handles are removed. Free text is unchanged in archive.db and
may identify people: this is pseudonymous, not anonymous. CSV cells are escaped
for spreadsheet safety; archive.db is authoritative for exact raw strings.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import secrets
import sys
import tempfile
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.backup_database import backup_database
from src.exporting import csv_cell
from src.roster import Roster
from src.store import Store
from src.submissions import FeedbackSpan,validate_partition


def assign_codes(ids,prefix,rng):
    shuffled=sorted(set(ids));rng.shuffle(shuffled)
    width=max(2,len(str(len(shuffled))))
    return {oid:f'{prefix}{i:0{width}d}' for i,oid in enumerate(shuffled,1)}


def sanitize_snapshot(path,*,code_targets):
    store=Store(str(path));store.migrate()
    db=store._db
    known={'opinions','submission_receipts','embeddings','coords','meta','submissions',
           'submission_revisions','submission_units','submission_actions'}
    tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
    if tables-known:
        store.close();raise RuntimeError('unknown tables in archive schema; refusing incomplete de-identification')
    parents=[dict(r) for r in db.execute('SELECT * FROM submissions')]
    opinions=[dict(r) for r in db.execute('SELECT * FROM opinions')]
    revisions=[dict(r) for r in db.execute('SELECT * FROM submission_revisions')]
    if not parents:
        store.close();raise SystemExit(f'{path} holds no submissions')
    rng=secrets.SystemRandom()
    reviewers=assign_codes([r['reviewer_id'] for r in parents+opinions],'R',rng)
    targets=assign_codes([r['target_id'] for r in parents+opinions+revisions],'P',rng) if code_targets else {}
    parent_ids={r['id']:'s_archive_'+uuid.uuid4().hex for r in parents}
    opinion_ids={r['id']:'o_archive_'+uuid.uuid4().hex for r in opinions}
    preserved_meta={key:store.get_meta(key) for key in ('schema_version','data_rev','layout_rev')}
    db.execute('PRAGMA journal_mode=DELETE')
    db.execute('PRAGMA secure_delete=ON')
    with store.transaction():
        db.execute('PRAGMA defer_foreign_keys=ON')
        db.execute('DELETE FROM submission_receipts');db.execute('DELETE FROM submission_actions')
        db.execute('UPDATE submissions SET nonce=NULL,request_hash=NULL,capability_hash=NULL')
        for row in parents:
            old,new=row['id'],parent_ids[row['id']]
            db.execute('UPDATE submissions SET id=?,reviewer_id=?,target_id=? WHERE id=?',
                       (new,reviewers[row['reviewer_id']],targets.get(row['target_id'],row['target_id']),old))
            db.execute('UPDATE submission_revisions SET submission_id=? WHERE submission_id=?',(new,old))
            db.execute('UPDATE submission_units SET submission_id=? WHERE submission_id=?',(new,old))
        for row in opinions:
            old,new=row['id'],opinion_ids[row['id']]
            db.execute('UPDATE opinions SET id=?,reviewer_id=?,target_id=? WHERE id=?',
                       (new,reviewers[row['reviewer_id']],targets.get(row['target_id'],row['target_id']),old))
            db.execute('UPDATE submission_units SET opinion_id=? WHERE opinion_id=?',(new,old))
            db.execute('UPDATE coords SET opinion_id=? WHERE opinion_id=?',(new,old))
        for row in revisions:
            db.execute('UPDATE submission_revisions SET target_id=? WHERE submission_id=? AND revision=?',
                       (targets.get(row['target_id'],row['target_id']),
                        parent_ids[row['submission_id']],row['revision']))
        db.execute("UPDATE submission_revisions SET actor_kind='redacted' WHERE actor_kind NOT IN ('system','admin','owner','legacy')")
        db.execute("UPDATE submissions SET error_code='PROCESSING_FAILED' WHERE error_code IS NOT NULL")
        db.execute('DELETE FROM meta')
        db.executemany('INSERT INTO meta VALUES(?,?)',[(k,v) for k,v in preserved_meta.items() if v is not None])
        db.execute("INSERT INTO meta VALUES('dataset_id',?)",('d_archive_'+uuid.uuid4().hex,))
        db.execute("INSERT INTO meta VALUES('archive_pseudonymous','true')")
    for row in db.execute('SELECT r.manifest,s.raw_text FROM submission_revisions r JOIN submissions s ON s.id=r.submission_id'):
        validate_partition(row['raw_text'],tuple(FeedbackSpan(u['start'],u['end']) for u in json.loads(row['manifest'])))
    for row in db.execute('SELECT o.text,s.raw_text,u.start_cp,u.end_cp FROM submission_units u JOIN opinions o ON o.id=u.opinion_id JOIN submissions s ON s.id=u.submission_id'):
        if row['text']!=row['raw_text'][row['start_cp']:row['end_cp']]:
            raise RuntimeError('archive unit text does not match original span')
    if db.execute('PRAGMA foreign_key_check').fetchall(): raise RuntimeError('archive foreign-key failure')
    db.execute('VACUUM')  # purge deleted credential/identifier bytes, not only rows
    if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok': raise RuntimeError('archive integrity failure')
    published=store.all_opinions();coords=store.all_coords();originals=store.recovery_records()
    store.close()
    return reviewers,targets,published,coords,originals


def write_csv(path,header,rows):
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(header)
        writer.writerows([[csv_cell(value) for value in row] for row in rows])
    path.chmod(0o600)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--db',type=Path,required=True)
    ap.add_argument('--roster',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--targets',action='store_true')
    args=ap.parse_args(argv)
    if not args.db.exists(): raise SystemExit(f'{args.db} does not exist')
    names=('archive.db','archive.csv','originals.csv','mapping.csv')
    args.out.mkdir(parents=True,exist_ok=True,mode=0o700)
    for name in names:
        if (args.out/name).exists(): raise SystemExit(f'{args.out/name} already exists; refusing to overwrite')
    roster=Roster.from_path(args.roster)
    with tempfile.TemporaryDirectory(prefix='.archive-',dir=args.out) as directory:
        temporary=Path(directory)
        backup_database(args.db,temporary/'archive.db')
        reviewers,targets,opinions,coords,originals=sanitize_snapshot(temporary/'archive.db',code_targets=args.targets)
        write_csv(temporary/'archive.csv',
            ['id','reviewer_code','target','text','source','week','timestamp','x','y','submission_id','ordinal','revision','start_cp','end_cp'],
            ([o.id,o.reviewer_id,o.target_id,o.text,o.source,o.week,o.timestamp,*coords.get(o.id,('','')),
              o.submission_id,o.ordinal,o.revision,o.start_cp,o.end_cp] for o in opinions))
        write_csv(temporary/'originals.csv',
            ['submission_id','reviewer_code','target','raw_text','source','week','state','revision','published_revision','history_json','timestamp'],
            ([d['submission_id'],d['reviewer_id'],d['target_id'],d['raw_text'],d['source'],d['week'],d['state'],d['revision'],d['published_revision'],
              json.dumps(d['history'],ensure_ascii=False),d['timestamp']] for d in originals))
        mapping=[]
        for kind,entries in [('reviewer',reviewers),('target',targets)]:
            for oid,code in sorted(entries.items(),key=lambda item:item[1]):
                entry=roster.resolve(oid)
                mapping.append([code,kind,oid,entry.display_name if entry else ''])
        write_csv(temporary/'mapping.csv',['code','kind','id','display_name'],mapping)
        published=[]
        try:
            for name in names:
                os.link(temporary/name,args.out/name);published.append(args.out/name)
        except BaseException:
            for path in published: path.unlink()
            raise
    print(f'archive : {args.out/"archive.csv"} ({len(opinions)} published units)')
    print(f'originals : {args.out/"originals.csv"} ({len(originals)} parents, including unpublished work)')
    print(f'database : {args.out/"archive.db"} (sanitized full lineage)')
    print(f'mapping : {args.out/"mapping.csv"}')
    print(f'source : {args.db} unchanged')
    print('Keep the mapping separate. The archive is pseudonymous, not anonymous; unchanged free text may identify people.')
    return 0


if __name__=='__main__': raise SystemExit(main())
