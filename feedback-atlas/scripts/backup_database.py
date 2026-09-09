#!/usr/bin/env python
"""Create a private, consistent SQLite recovery snapshot, including pending work."""
import argparse
from contextlib import closing
import os
from pathlib import Path
import sqlite3
import tempfile


def backup_database(source: Path, destination: Path):
    source,destination=Path(source),Path(destination)
    if not source.is_file(): raise FileNotFoundError(source)
    if destination.exists(): raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    fd,temporary=tempfile.mkstemp(prefix='.atlas-backup-',dir=destination.parent)
    os.close(fd)
    try:
        with closing(sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True)) as original:
            with closing(sqlite3.connect(temporary)) as snapshot:
                original.backup(snapshot)
                if snapshot.execute('PRAGMA integrity_check').fetchone()[0]!='ok':
                    raise RuntimeError('SQLite snapshot integrity check failed')
        os.chmod(temporary,0o600)
        with open(temporary,'rb') as handle: os.fsync(handle.fileno())
        # Publishing by link refuses an existing target even in a racing process.
        os.link(temporary,destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args(argv)
    backup_database(args.db,args.out)
    print(f'Private recovery backup: {args.out}; includes originals, history and owner credentials.')
    print('Source unchanged. Keep this backup private; it is not a de-identified archive.')
    return 0


if __name__=='__main__': raise SystemExit(main())
