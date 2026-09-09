import sqlite3
import pytest


def test_backup_includes_uncheckpointed_wal_rows_and_refuses_overwrite(tmp_path):
    from scripts.backup_database import backup_database
    source=tmp_path/'live.db'
    conn=sqlite3.connect(source)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('CREATE TABLE submissions(raw_text,state,capability_hash)')
    conn.execute('INSERT INTO submissions VALUES(?,?,?)',('queued original','queued','recovery-secret-hash'))
    conn.commit()
    target=tmp_path/'backup.db'
    backup_database(source,target)
    with sqlite3.connect(target) as copied:
        assert copied.execute('SELECT * FROM submissions').fetchone()==('queued original','queued','recovery-secret-hash')
        assert copied.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    assert target.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError): backup_database(source,target)
    assert conn.execute('SELECT count(*) FROM submissions').fetchone()[0]==1
    conn.close()
