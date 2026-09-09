"""Durable state: opinions, cached embeddings, and the last layout.

stdlib sqlite3, four tables, no ORM and no vector extension. The reasoning:

* PRD 6 requires survival across restart. The repo's existing idiom is
  append-only JSONL (notebook_results/lab_log.jsonl), but that would split text
  and vectors across two files with a desync hazard -- crash between the two
  writes and the Nth vector no longer belongs to the Nth text. WAL-mode SQLite
  gives the same append durability, atomically.
* Vectors are float32 BLOBs. 768 dims is 3072 bytes; the same vector as a JSON
  array of floats is ~5x that and has to be reparsed on every start.
* No sqlite-vec. At a thousand rows the whole corpus is one (1000, 768) array in
  RAM and any similarity operation is a matmul, so an ANN index buys nothing and
  costs a loadable extension that some Python builds disable.

Embeddings are keyed by (text_hash, cache_key), not by opinion id. Two people
writing the same sentence share one cached vector, and swapping the model *adds*
rows rather than replacing them -- so swapping back later is free rather than a
full re-encode of the semester.
"""

import sqlite3
import threading
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from src.models import Opinion

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS opinions (
  id          TEXT PRIMARY KEY,
  reviewer_id TEXT NOT NULL,
  target_id   TEXT NOT NULL,
  text        TEXT NOT NULL,
  source      TEXT NOT NULL,
  week        INTEGER NOT NULL,
  timestamp   TEXT NOT NULL,
  text_hash   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS opinions_order ON opinions(timestamp, id);

CREATE TABLE IF NOT EXISTS submission_receipts (
  reviewer_id TEXT NOT NULL,
  nonce       TEXT NOT NULL,
  opinion_id  TEXT NOT NULL,
  PRIMARY KEY (reviewer_id, nonce),
  FOREIGN KEY (opinion_id) REFERENCES opinions(id)
);

CREATE TABLE IF NOT EXISTS embeddings (
  text_hash TEXT NOT NULL,
  cache_key TEXT NOT NULL,
  dim       INTEGER NOT NULL,
  vec       BLOB NOT NULL,
  PRIMARY KEY (text_hash, cache_key)
);

CREATE TABLE IF NOT EXISTS coords (
  opinion_id TEXT PRIMARY KEY,
  x          REAL NOT NULL,
  y          REAL NOT NULL,
  layout_rev INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def vector_to_blob(v) -> bytes:
    """float32, always. Models emit float64 as often as not, and a cache holding
    both silently doubles its size and breaks the fixed-width read back."""
    return np.asarray(v, dtype=np.float32).tobytes()


def blob_to_vector(b: bytes, dim: int) -> np.ndarray:
    arr = np.frombuffer(b, dtype=np.float32)
    if arr.shape[0] != dim:
        raise ValueError(f"cached vector has {arr.shape[0]} dims, expected {dim}")
    return arr


class Store:
    def __init__(self, path: str):
        self.path = str(path)
        # check_same_thread=False because the recompute runs in a worker thread
        # (sentence-transformers blocks); the lock below is what actually makes
        # that safe, not the flag.
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()

    def migrate(self) -> None:
        with self._lock:
            # WAL so a read during an in-flight write does not block. Note for
            # deployment: WAL is unreliable on NFS/SMB -- keep the db on local disk.
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.executescript(_SCHEMA)
            self._db.commit()
            found = self.get_meta("schema_version")
            if found is None:
                self.set_meta("schema_version", str(SCHEMA_VERSION))
            elif int(found) != SCHEMA_VERSION:
                raise RuntimeError(
                    f"database at {self.path} is schema v{found}, code expects "
                    f"v{SCHEMA_VERSION}; move it aside rather than letting it be "
                    "read with the wrong shape"
                )

    def close(self) -> None:
        with self._lock:
            self._db.commit()
            self._db.close()

    # --- meta ---------------------------------------------------------------
    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
            self._db.commit()

    def compare_and_set_meta(self, key: str, expected: str | None, value: str) -> bool:
        """Atomically save settings only if another admin has not changed them."""
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
                if (row["value"] if row else None) != expected:
                    self._db.rollback()
                    return False
                self._db.execute(
                    "INSERT INTO meta(key,value) VALUES(?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
                self._db.commit()
                return True
            except BaseException:
                self._db.rollback()
                raise

    # --- opinions -----------------------------------------------------------
    def insert_opinion(self, op: Opinion, text_hash: str) -> bool:
        """False when this id is already stored -- a retry over a flaky tunnel,
        not a second opinion."""
        with self._lock:
            cur = self._db.execute(
                "INSERT OR IGNORE INTO opinions"
                "(id,reviewer_id,target_id,text,source,week,timestamp,text_hash)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (op.id, op.reviewer_id, op.target_id, op.text, op.source,
                 op.week, op.timestamp, text_hash))
            self._db.commit()
            return cur.rowcount > 0

    def submission_receipt(self, reviewer_id: str, nonce) -> str | None:
        nonce = _bounded_nonce(nonce)
        if nonce is None:
            return None
        with self._lock:
            row = self._db.execute(
                "SELECT opinion_id FROM submission_receipts "
                "WHERE reviewer_id=? AND nonce=?",
                (reviewer_id, nonce)).fetchone()
        return row["opinion_id"] if row else None

    def insert_opinion_with_receipt(
        self, op: Opinion, text_hash: str, nonce
    ) -> tuple[bool, str]:
        """Insert an opinion and its replay receipt in one transaction.

        Returns (inserted, opinion_id). Empty or oversized nonces keep the older
        best-effort insert behavior so stale clients are still accepted.
        """
        nonce = _bounded_nonce(nonce)
        if nonce is None:
            return self.insert_opinion(op, text_hash), op.id

        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute(
                    "SELECT opinion_id FROM submission_receipts "
                    "WHERE reviewer_id=? AND nonce=?",
                    (op.reviewer_id, nonce)).fetchone()
                if row:
                    self._db.commit()
                    return False, row["opinion_id"]

                cur = self._db.execute(
                    "INSERT OR IGNORE INTO opinions"
                    "(id,reviewer_id,target_id,text,source,week,timestamp,text_hash)"
                    " VALUES(?,?,?,?,?,?,?,?)",
                    (op.id, op.reviewer_id, op.target_id, op.text, op.source,
                     op.week, op.timestamp, text_hash))
                if cur.rowcount > 0:
                    self._db.execute(
                        "INSERT INTO submission_receipts"
                        "(reviewer_id,nonce,opinion_id) VALUES(?,?,?)",
                        (op.reviewer_id, nonce, op.id))
                self._db.commit()
                return cur.rowcount > 0, op.id
            except Exception:
                self._db.rollback()
                raise

    def all_opinions(self) -> list[Opinion]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM opinions ORDER BY timestamp, id").fetchall()
        return [Opinion(id=r["id"], reviewer_id=r["reviewer_id"],
                        target_id=r["target_id"], text=r["text"], source=r["source"],
                        week=r["week"], timestamp=r["timestamp"]) for r in rows]

    # --- embeddings ---------------------------------------------------------
    def get_embeddings(
        self, hashes: Iterable[str], cache_key: str
    ) -> dict[str, np.ndarray]:
        hashes = list(dict.fromkeys(hashes))
        out: dict[str, np.ndarray] = {}
        if not hashes:
            return out
        with self._lock:
            for chunk in (hashes[i:i + 400] for i in range(0, len(hashes), 400)):
                q = ("SELECT text_hash,dim,vec FROM embeddings WHERE cache_key=? "
                     f"AND text_hash IN ({','.join('?' * len(chunk))})")
                for r in self._db.execute(q, (cache_key, *chunk)).fetchall():
                    out[r["text_hash"]] = blob_to_vector(r["vec"], r["dim"])
        return out

    def put_embeddings(
        self, cache_key: str, dim: int, items: Mapping[str, Sequence[float]]
    ) -> None:
        if not items:
            return
        for text_hash, vector in items.items():
            width = len(np.asarray(vector).reshape(-1))
            if width != dim:
                # Caught on the way in, where the model and its spec are still in
                # scope. Left to the read, this surfaces as a width error hundreds
                # of opinions later, with nothing to say which swap caused it.
                raise ValueError(
                    f"vector for {text_hash[:12]}... is {width} wide but dim={dim} "
                    "was declared; the model and its registry entry disagree")
        with self._lock:
            self._db.executemany(
                "INSERT OR REPLACE INTO embeddings(text_hash,cache_key,dim,vec)"
                " VALUES(?,?,?,?)",
                [(h, cache_key, dim, vector_to_blob(v)) for h, v in items.items()])
            self._db.commit()

    def missing_embeddings(self, cache_key: str) -> list[tuple[str, str]]:
        """[(text_hash, text)] absent under *this* cache key -- the backfill set.

        After a model swap this is the whole corpus, which is exactly right: the
        old vectors stay on disk under their own key, and nothing mixes.
        """
        with self._lock:
            rows = self._db.execute(
                "SELECT DISTINCT o.text_hash, o.text FROM opinions o "
                "LEFT JOIN embeddings e"
                "  ON e.text_hash = o.text_hash AND e.cache_key = ? "
                "WHERE e.text_hash IS NULL", (cache_key,)).fetchall()
        return [(r["text_hash"], r["text"]) for r in rows]

    # --- coordinates --------------------------------------------------------
    def all_coords(self) -> dict[str, tuple[float, float]]:
        with self._lock:
            rows = self._db.execute("SELECT opinion_id,x,y FROM coords").fetchall()
        return {r["opinion_id"]: (r["x"], r["y"]) for r in rows}

    def put_coords(
        self, coords: Mapping[str, Sequence[float]], layout_rev: int
    ) -> None:
        if not coords:
            return
        with self._lock:
            self._db.executemany(
                "INSERT INTO coords(opinion_id,x,y,layout_rev) VALUES(?,?,?,?) "
                "ON CONFLICT(opinion_id) DO UPDATE SET x=excluded.x, y=excluded.y,"
                " layout_rev=excluded.layout_rev",
                [(k, float(v[0]), float(v[1]), layout_rev) for k, v in coords.items()])
            self._db.commit()


def _bounded_nonce(value) -> str | None:
    if not isinstance(value, str):
        return None
    nonce = value.strip()
    if not nonce or len(nonce) > 128:
        return None
    return nonce
