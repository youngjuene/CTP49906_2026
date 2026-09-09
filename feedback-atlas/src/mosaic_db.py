"""The DuckDB that Embedding Atlas's viewer queries, and the boundary around it.

Embedding Atlas's viewer is a Mosaic application: it does not receive data, it
issues SQL. Every chart, the table, the cross-filter, the search box and the
embedding view itself are queries against one relation. So adopting the viewer
means standing up a database for it, and `embedding_atlas.server` does exactly
that -- `make_duckdb_connection` builds an in-memory DuckDB, loads the frame, and
then locks the connection down. This module is that function with two changes,
both of which the classroom forces.

**Two databases, not one.** src/payloads.py refuses to put `reviewer_id` on a
participant frame, and does it structurally: two serialisers, two channels, no
flag to get wrong. A SQL endpoint would hand all of that back, because the client
writes the query -- `SELECT reviewer_id FROM dataset` is a perfectly ordinary
Mosaic-shaped request. No filter on the way out can be trusted to hold, because
the surface is a language rather than a set of fields.

So the participant database does not contain the column. Two connections, built
from two schemas, chosen by the endpoint before a single character of client SQL
is parsed. A participant asking for `reviewer_id` gets a binder error, which is
the correct answer and is produced by DuckDB rather than by our vigilance.

**The table is replaced, not appended.** The corpus is re-projected on every
submission, so x and y change for rows that already exist; `CREATE OR REPLACE`
from one Arrow table is simpler than reconciling, and at a thousand rows it costs
under a millisecond. Only trusted server refreshes replace these tables.

**What the hardening does and does not buy.** `enable_external_access = false`
plus `lock_configuration = true` are Apple's own two lines, and they are the
important ones: without them `read_csv('/etc/passwd')` is a valid query and this
server is reachable over a public tunnel. They stop file and network reads and
stop themselves being turned back on. Client queries also pass a single-SELECT
guard and execute in read-only transactions with time, memory and result limits.
The vendored viewer computes categories as inline expressions through a pinned
build adapter, so its normal setup requires no client table mutations.
"""

import threading
from collections.abc import Iterable, Mapping, Sequence

# The relation name Embedding Atlas's own server uses. Kept identical so a query
# copied from their docs, or from their MCP tooling, runs here unchanged.
TABLE = "dataset"
MAX_RESULT_ROWS = 1000
QUERY_TIMEOUT_SECONDS = 2.0
QUERY_LOCK_TIMEOUT_SECONDS = 2.0
DUCKDB_MEMORY_LIMIT = "256MB"

# Columns every audience may query. Deliberately the same list as
# payloads.PARTICIPANT_KEYS, plus the display name the charts group by.
PARTICIPANT_COLUMNS = (
    "id", "target_id", "target_name", "text", "source", "week", "timestamp",
    "x", "y", "neighbors",
)
ADMIN_COLUMNS = PARTICIPANT_COLUMNS + ("reviewer_id", "reviewer_name")


def available() -> bool:
    from importlib.util import find_spec

    try:
        return (find_spec("duckdb") is not None
                and find_spec("pyarrow") is not None)
    except (ImportError, ValueError):
        return False


def _schema(columns: Sequence[str]):
    """An explicit Arrow schema, never inference.

    Inference would read the types off the first batch, and the first batch of a
    session is one opinion whose `neighbors` lists are empty -- from which Arrow
    infers `list<null>`, and every later insert of real neighbour ids fails
    against a schema nobody chose. The nested type is the whole reason this is
    spelled out: it is embedding-atlas's contract, and its viewer reads
    `neighbors.ids` and `neighbors.distances` by name.
    """
    import pyarrow as pa

    neighbors = pa.struct([
        pa.field("ids", pa.list_(pa.string())),
        pa.field("distances", pa.list_(pa.float64())),
    ])
    types = {
        "id": pa.string(), "target_id": pa.string(), "target_name": pa.string(),
        "text": pa.string(), "source": pa.string(), "week": pa.int32(),
        "timestamp": pa.string(), "x": pa.float64(), "y": pa.float64(),
        "neighbors": neighbors,
        "reviewer_id": pa.string(), "reviewer_name": pa.string(),
    }
    return pa.schema([pa.field(c, types[c]) for c in columns])


class MosaicDatabase:
    """One audience's view of the corpus, as a relation Mosaic can query."""

    def __init__(self, columns: Sequence[str] = PARTICIPANT_COLUMNS):
        import duckdb

        self.columns = tuple(columns)
        self._schema = _schema(self.columns)
        # A single connection with a lock rather than a pool. Queries are short
        # and the corpus is small; a pool would mean N locked-down connections to
        # keep in step through every reload, for no measurable gain at this size.
        self._con = duckdb.connect(":memory:")
        self._con.execute(f"SET memory_limit = '{DUCKDB_MEMORY_LIMIT}'")
        self._lock = threading.RLock()
        self.rows = 0
        self._hardened = False
        self.replace([])

    # --- loading ------------------------------------------------------------
    def _record_batch(self, points: Iterable[Mapping], names: Mapping[str, str]):
        import pyarrow as pa

        cols: dict[str, list] = {c: [] for c in self.columns}
        for p in points:
            for c in self.columns:
                if c == "target_name":
                    # Resolved here rather than carried on the wire. The websocket
                    # clients already receive the roster in hello_ok and look names
                    # up themselves, so putting it on every point would repeat a
                    # display name in every frame for no client that needs it. The
                    # viewer has no such table, and grouping a chart by an opaque
                    # id would make the one column a human reads unreadable.
                    tid = str(p.get("target_id") or "")
                    cols[c].append(names.get(tid, tid))
                elif c == "neighbors":
                    nb = p.get("neighbors") or {}
                    cols[c].append({
                        "ids": list(nb.get("ids") or []),
                        "distances": [float(d) for d in (nb.get("distances") or [])],
                    })
                elif c == "week":
                    cols[c].append(int(p.get("week") or 0))
                elif c in ("x", "y"):
                    cols[c].append(float(p.get(c) or 0.0))
                else:
                    value = p.get(c)
                    cols[c].append(None if value is None else str(value))
        return pa.Table.from_pydict(cols, schema=self._schema)

    def replace(self, points: Iterable[Mapping],
                names: Mapping[str, str] | None = None) -> None:
        """Swap in a whole corpus. Safe to call from the event loop's thread pool."""
        table = self._record_batch(points, names or {})
        with self._lock:
            self._con.register("_incoming", table)
            try:
                self._con.execute(
                    f"CREATE OR REPLACE TABLE {TABLE} AS SELECT * FROM _incoming")
            finally:
                self._con.unregister("_incoming")
            self.rows = table.num_rows
            # Hardened after the first load, not before it: registering an Arrow
            # table is an in-process handoff rather than a filesystem read, but
            # locking the configuration first would leave no way to recover if a
            # future DuckDB decided otherwise. Apple's server sets both lines
            # after loading for the same reason.
            if not self._hardened:
                self._con.execute("SET enable_external_access = false")
                self._con.execute("SET lock_configuration = true")
                self._hardened = True

    # --- querying -----------------------------------------------------------
    def _select_statement(self, sql: str) -> str:
        import duckdb

        if not isinstance(sql, str) or not sql.strip():
            raise ValueError("query must be a single SELECT statement")
        statements = duckdb.extract_statements(sql)
        if len(statements) != 1:
            raise ValueError("query must contain exactly one statement")
        statement = statements[0]
        if statement.type != duckdb.StatementType.SELECT:
            raise ValueError("query must be read-only SELECT")
        query = sql.strip()
        if query.endswith(";"):
            query = query[:-1].rstrip()
        return query

    def _limited_query(self, sql: str) -> str:
        return (
            "SELECT * FROM ("
            f"{self._select_statement(sql)}"
            f") AS _mosaic_result LIMIT {MAX_RESULT_ROWS + 1}"
        )

    def query(self, sql: str, kind: str = "arrow"):
        """-> (payload, content_type). Mosaic's three command types, as upstream.

        `arrow` is what the viewer uses for anything it plots; `json` for small
        metadata reads. `exec` is accepted for upstream connector compatibility,
        but it is intentionally subject to the same read-only guard as every
        other command.
        """
        from embedding_atlas.utils import arrow_to_bytes

        if kind not in {"arrow", "json", "exec"}:
            raise ValueError(f"unknown command {kind!r}")
        bounded_sql = self._limited_query(sql)

        if not self._lock.acquire(timeout=QUERY_LOCK_TIMEOUT_SECONDS):
            raise TimeoutError("query lock wait timed out")
        cursor = self._con.cursor()
        def interrupt_query() -> None:
            try:
                cursor.interrupt()
            except Exception:
                pass

        timer = threading.Timer(
            QUERY_TIMEOUT_SECONDS,
            interrupt_query,
        )
        try:
            cursor.execute("BEGIN TRANSACTION READ ONLY")
            timer.start()
            result = cursor.sql(bounded_sql)
            table_fn = getattr(result, "to_arrow_table", None) or result.fetch_arrow_table
            table = table_fn()
            if table.num_rows > MAX_RESULT_ROWS:
                raise ValueError(f"query returned more than {MAX_RESULT_ROWS} rows")
            if kind == "exec":
                return b"{}", "application/json"
            if kind == "arrow":
                return arrow_to_bytes(table.to_reader()), "application/octet-stream"
            if kind == "json":
                return (table.to_pandas().to_json(orient="records").encode("utf-8"),
                        "application/json")
            raise AssertionError("unreachable query kind")
        finally:
            timer.cancel()
            timer.join(timeout=0.1)
            try:
                cursor.execute("ROLLBACK")
            except Exception:
                pass
            cursor.close()
            self._lock.release()

    def close(self) -> None:
        with self._lock:
            self._con.close()


class MosaicService:
    """The pair of databases, and the one place that decides which is which."""

    def __init__(self):
        self.participant = MosaicDatabase(PARTICIPANT_COLUMNS)
        self.admin = MosaicDatabase(ADMIN_COLUMNS)

    def replace(self, participant_points: Sequence[Mapping],
                admin_points: Sequence[Mapping],
                names: Mapping[str, str] | None = None) -> None:
        """Load both. Called with the output of the two existing serialisers.

        Reusing them is the point: the viewer's relation and the websocket frame
        are then the same data by construction, so a column can not appear in one
        and be missing from the other, and the privacy rule is enforced once.

        `names` maps a target id to the display name the charts group by. Only
        student display names belong in it: they are the projects being presented,
        which every participant already sees in the compose form, and an id that
        is not in the map falls back to itself rather than to a blank.
        """
        self.participant.replace(participant_points, names)
        self.admin.replace(admin_points, names)

    def database(self, *, admin: bool) -> MosaicDatabase:
        return self.admin if admin else self.participant

    def close(self) -> None:
        self.participant.close()
        self.admin.close()
