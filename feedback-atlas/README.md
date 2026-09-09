# Real-time Feedback Atlas

<p align="center">
  <b>English</b> · <a href="README_kr.md">한국어</a>
</p>

A classroom tool for the CTP49906 *Multimodal AI* workshop. During a student
presentation, attendees open a link on their phones and write what they think.
Participants can paste a whole response of up to 20,000 Unicode code points.
The original is saved durably before processing; derived topic units are embedded
**locally**, projected to 2D, and published together. Processing time depends on
input length and the queue; the UI distinguishes saved, processing and ready.

Colour encodes **which student the feedback is about**; shape encodes **feedback
source: AI-generated or human-written**, independently of who submits it. AI is
the intentional default; using AI to process human-written feedback does not
change its source. The session asks how far a model can
describe subjective experience, and this puts both kinds of comment in one
semantic space so the room can see whether they land in the same place.

The classroom map serves phones, laptops and projectors. The optional Embedding
Atlas analysis viewer remains disabled until it supports the same filters and
selection semantics. Automatic semantic splitting is also **not approved for
classroom rollout**: see the [implementation and release evidence](docs/qa/feedback-atlas-release.md).

The specification is [`realtime-feedback-atlas-prd.md`](realtime-feedback-atlas-prd.md).

## What it is not

Not a login system. There are no passwords: a landing page checks the id you type
against a roster the instructor registered in advance, and that is the whole gate.
Registration is strict — an id that is not on the list does not get in.

Not anonymous, and not pretending to be. `reviewer_id` is stored on the server
and is excluded from public map payloads. Each participant receives their own
private identity and ownership records; admin mode, behind a fixed access code,
can attribute other people's submissions.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and **Python ≥ 3.12** — not a
preference: numpy 2.5 and scipy 1.18 both declare it, so a 3.11 venv silently
resolves to older ones.

```bash
cd feedback-atlas
uv venv --python 3.12 --seed .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

### Accept the Gemma licence before the first run

The default model, `google/embeddinggemma-300m`, is a **gated** download. A fresh
clone fails at model load with a 401 or 403, not at install time — so do this
before the day, not during it:

1. Accept the licence at <https://huggingface.co/google/embeddinggemma-300m>
2. Log in as **that same account**: `huggingface-cli login`, or set `HF_TOKEN`.
   A token from a different account than the one that accepted the licence fails
   in exactly the same way as no token at all.
3. Pre-fetch the weights so the first start is not a 1.2 GB download:
   `hf download google/embeddinggemma-300m`

> **If it still says "gated" after you have accepted the licence, check `HF_TOKEN`.**
> The environment variable takes precedence over whatever `huggingface-cli login`
> stored, so an invalid or revoked `HF_TOKEN` shadows a working login — and the hub
> reports that as a gated-repo error, which sends you back to the licence page where
> everything already looks correct. `env -u HF_TOKEN python -m uvicorn …` is the
> quickest way to tell the two apart. This cost real time on the first run here.

If the gate is in your way, `ATLAS_EMBEDDING_MODEL=e5-small-ko` is ungated,
Apache-2.0, and 384-dimensional. The server refuses to start with a readable
error rather than falling back on its own — a map built by a model nobody chose
is worse than no map.

### The roster

`roster.csv`, registered before class. See [`roster.example.csv`](roster.example.csv).

```csv
id,display_name,role
kim.seoyeon,김서연,student
kang.minsu,강민수,auditor
ta.youngjune,영준,ta
```

Three roles. `student` (수강생) rows are also the **targets** — the projects
feedback can be *about*. `auditor` (청강생) and `ta` (조교) write feedback and are
never targets, which is what the course actually looks like: a TA comments on
work, nobody comments on the TA.

PRD 7's table names only `student` and `auditor`; `ta` is a deliberate addition,
because PRD 3 already lists 조교 as a user who may submit — and submitting needs a
roster entry. Without a role of their own a TA has to be filed as 청강생, and the
admin panel, which exists to attribute writing correctly, would then label them as
something they are not. Matching is forgiving about spaces, case and
full-width characters (a Korean IME left in 전각 mode emits `ｋｉｍ`), and about
nothing else: an edit-distance match would let one student's id resolve to
another's.

Adding someone later means editing the file and sending `SIGHUP` — no restart,
no dropped connections:

```bash
kill -HUP $(pgrep -f 'uvicorn.*src.server')
```

### Configuration

Copy [`.env.example`](.env.example) to `.env`. Two values are required and have
no defaults: `ATLAS_ADMIN_CODE` and `ATLAS_ROSTER`. A default admin code is a
published admin code.

## Running it

```bash
source .venv/bin/activate
export $(grep -v '^#' .env | xargs)
python -m uvicorn --factory src.server:create_app \
  --host 0.0.0.0 --port 8100 --proxy-headers --forwarded-allow-ips='*'
```

Start it **five minutes early**. The first run loads the model and warms UMAP's
numba kernels; done at the bell, that cost lands on the first submission of the
class instead.

`GET /healthz` reports the model's cache key, the corpus size, the projector in
use and the connection counts. It never reports the access code.

### Getting the room connected

**Try the LAN first.** Lowest latency, no quotas, no third party: share
`http://<your-ip>:8100` and a QR code. Test it from a phone on the same SSID *the
day before* — many campus and guest networks enable client isolation, which
silently blocks phone-to-laptop traffic and looks exactly like the server being
down.

**An SSH tunnel is not a classroom path.** `ssh -L 8100:127.0.0.1:8100 …` forwards
the port to *one* machine — the laptop you typed it on. It is the right way to
preview the map yourself over SSH, and it does nothing for thirty phones.

### On the day

```bash
scripts/class.sh start     # server + tunnel, both detached; prints URL and QR
scripts/class.sh url       # the URL and QR again, for the slide
scripts/class.sh status    # roster/context, limits, queue, model and semantic readiness
scripts/class.sh backup    # private consistent SQLite snapshot, including unpublished work
scripts/class.sh stop      # requires closed intake and resolved queue; backs up before stopping
```

The current semantic release gate is **NO-GO**: synthetic held-out single-topic preservation is only 37.5%, and independent human judgments are absent. `start` refuses an unreviewed public classroom tunnel. `ATLAS_ALLOW_UNREVIEWED_SEMANTICS=1 scripts/class.sh start` is an explicit development override, not classroom approval. See [segmentation evaluation](docs/qa/segmentation-evaluation.md).

Use one database per class and a stable HTTPS origin. A changed quick-tunnel origin cannot access the previous origin's saved browser draft/outbox. Recover from the original browser origin or preserve the draft text before changing URLs.

Once classroom release gates pass, run it five minutes early. `setsid` on both processes is the load-bearing part:
started from a plain shell, `cloudflared` dies the moment the laptop sleeps or the
wifi drops, and the address the room is looking at disappears mid-session. The
server survives that on its own; the tunnel does not.

The address is four random words, which is fine to scan and hopeless to read from
the back of a room — so `start` also prints a QR to the terminal and writes
`.run/qr.png` for the slide. Put the QR up; nobody should be typing this.

An `ssh -L` forward is **not** part of this. Once the tunnel is up its HTTPS URL
works from your laptop too, so the forward has nothing left to do.

A short `http://<host-ip>:8100` is tempting instead, and on this host it is the
wrong trade: the address is publicly routable, so binding to `0.0.0.0` publishes
the app to the internet rather than to the room, over plain HTTP, with the admin
endpoint on the same port. The tunnel gives HTTPS and opens no port, and the QR
makes its length irrelevant.

**A Cloudflare quick tunnel is useful for temporary checks.** It needs no account and provides HTTPS, but a stable configured HTTPS endpoint is preferable for classroom draft continuity:

```bash
cloudflared tunnel --url http://localhost:8100
# → https://<random-words>.trycloudflare.com   share this, or a QR of it
```

The earlier v1 tunnel check covered HTTPS, WebSocket connection and submission
from a phone-sized client. The v2 flow has not been checked on a physical phone
through the actual classroom URL. Quick-tunnel URLs change on restart, so a new
origin also loses access to the old origin's browser storage.

This is also why the app speaks WebSocket rather than Server-Sent Events:
`trycloudflare.com` buffers `text/event-stream`, so SSE never arrives through it.
Tunnelled traffic is HTTPS, and the page connects with `wss://` accordingly.

Quick tunnels drop an idle socket at around 100 seconds — the quiet middle of a
presentation. The server heartbeats every 25 s and the client reconnects with
backoff and resyncs, so this is handled; it is worth knowing why the code is
there.

## Admin mode

Open `/#admin` and enter the access code. The code buys a **different websocket**,
not a UI flag — a participant connection has no path to become an admin one,
because the two endpoints use different serialisers.

Four tools, all admin-only:

| | |
|---|---|
| **Search** | substring match over the opinions, hits highlighted, composing with the week filter and the target and reviewer focus |
| **Selection** | marquee or lasso on the map, yielding a readable list — this is what lets a cluster be *discussed* rather than hovered one point at a time |
| **Nearest opinions** | click a point for its eight nearest by cosine distance. The workshop's own question, asked per point: given what a person wrote, what did the model write that lands nearest to it? |
| **CSV export** | the whole corpus or just the current selection, with authorship and coordinates |

## The Embedding Atlas viewer

Alongside the map, the app serves Apple's
[Embedding Atlas](https://github.com/apple/embedding-atlas) viewer itself —
the real component, not a lookalike: a sortable table, distribution charts for
every column that cross-filter each other, SQL predicates, full-text search, and
its WebGPU embedding view with density contours and automatic cluster labels.

The analysis button currently explains why this viewer is unavailable. The
classroom map remains active at every width. Bundled worker URLs are repaired and
tested, but full WebGPU rendering and shared filter/selection behavior have not
passed the release requirements.

**It is a Mosaic application, so it queries rather than receives.** The server
keeps an in-memory DuckDB holding the corpus and answers `POST /data/query`, which
is `embedding_atlas.server`'s own endpoint with its three command types, so the
component talks to it unmodified. Every recompute reloads that relation before the
websocket broadcast goes out, so the charts and the map never disagree.

**There are two databases, and the endpoint picks one before parsing any SQL.**
The client writes the query, so no filter on the way out could hold —
`SELECT reviewer_id FROM dataset` is an ordinary Mosaic-shaped request. The
participant database therefore does not contain the column, and asking for it
returns a DuckDB binder error rather than data. The admin code travels in the
request body, never the URL, exactly as with CSV export.

**AI versus human moved from shape to filter.** `EmbeddingView` encodes one
channel, category colour, so the map's second channel has no equivalent here.
Instead the `source` chart cross-filters the whole dashboard: click `ai` and every
other chart, the table and the embedding view narrow to it. Comparing the two
clusters is a click rather than a legend.

**Rebuilding the bundle.** `web/vendor/` is committed and nothing fetches it at
class time. After changing a version:

```bash
cd frontend && npm install && npm run build   # -> ../web/vendor
```

Node is needed for that build and for nothing else. The server never runs it.

The same package also gives you the official CLI over an exported CSV, which needs
no server at all:

```bash
embedding-atlas feedback-atlas.csv --text text --x x --y y
```

## Bulk AI import (v2)

`scripts/seed_ai_opinions.py` is an **offline administrative importer**. Stop the server first; imports are durable database writes and the next server startup drains them using its single shared model worker. This script loads no model and has no online bulk-API mode.

```bash
.venv/bin/python scripts/seed_ai_opinions.py comments.csv \
  --db atlas.db --roster roster.csv --reviewer instructor --dry-run
.venv/bin/python scripts/seed_ai_opinions.py comments.csv \
  --db atlas.db --roster roster.csv --reviewer instructor --segment
```

Input is CSV or JSONL with `target_id,text[,source][,week]`. Source defaults to AI and can be overridden per row. The default mode explicitly treats each row as an already-unitized opinion; `--segment` asks the common worker to choose semantic boundaries. Both preserve the exact raw string, including quoted CSV newlines and Unicode decomposition, and neither publishes a point before its vector and coordinates exist. The limit is 20,000 raw code points per parent.

Re-running identical file bytes, row positions, effective metadata and mode is idempotent. Identical text in different rows remains separate. Changing the file or effective metadata/mode denotes a new import. Owner capabilities are generated randomly, stored only as hashes and never printed; imported originals are managed through instructor authorization.

The queue limit is 300 pending parents (`--max-pending` may lower it). A full queue returns nonzero with accepted/replayed/skipped counts; accepted rows stay saved. Drain the queue, then rerun the identical file and options to accept remaining rows. `--dry-run` validates without opening or creating the database; replay and current capacity are not checked in dry runs.

## Closing, backup and migration

Close intake in the instructor context controls, check pending/failed originals, resolve or explicitly preserve failures, export the instructor recovery CSV, then run `scripts/class.sh stop`. It checks the queue and writes a private full SQLite recovery snapshot before stopping processes. `stop --force` still creates a backup, but permits unresolved work for restart recovery; it cannot recover text that never left a browser.

```bash
.venv/bin/python scripts/backup_database.py \
  --db atlas.db --out recovery/atlas-before-upgrade.db
```

This uses SQLite's backup API, including committed WAL content and pending originals, revision history and owner credentials. The output is mode 0600 and existing destinations are refused. Keep recovery backups private. Do not copy only `atlas.db` while a WAL-mode server is active.

A v1-to-v2 migration creates a private, consistent `atlas.db.pre-v2-backup` before changing schema. If that filename exists, it creates a fresh timestamp-suffixed copy; the successful migration records its actual path in `meta.migration_backup_path`. Validate a copied database first and retain matching code. Before upgrading old loaded clients, preserve unsent v1 drafts manually and resolve in-flight submissions; old JavaScript cannot acquire the new outbox behavior retroactively.

For rollback, close intake and stop the server, retain a full current v2 backup plus recovery exports of **all** post-upgrade originals, and account for accepted new work before using an older snapshot. Restore to an unused filename with the matching old code, rather than overwriting the upgraded database:

```bash
.venv/bin/python scripts/backup_database.py \
  --db recovery/atlas-before-upgrade.db --out restored-v1.db
# Configure matching old code to use ATLAS_DB=restored-v1.db before starting it.
```

## After the semester: pseudonymous archive

```bash
.venv/bin/python scripts/deidentify.py \
  --db atlas.db --roster roster.csv --out archive/ --targets
```

The source is opened read-only and never migrated in place. The script produces `archive.db` with all parent/revision/unit lineage, `archive.csv` with published units, and `originals.csv` including pending/failed/withdrawn parents. It remaps opaque parent/unit IDs consistently, replaces reviewer identities with random codes, optionally codes targets, removes nonce/capability/request/action handles and private metadata, then compacts and verifies the copied database. SQLite retains exact original strings; CSV cells receive spreadsheet-safety escaping.

`mapping.csv` is a separate private roster-code mapping. Keep it elsewhere or destroy it. Recovery backups contain credentials; these sanitized archives do not, and must not be used as classroom recovery databases. **The archive is pseudonymous, not anonymous:** unchanged free text, timestamps and retained project details may identify people. This script cannot establish consent or remove identifying information from prose automatically.

## Tests

```bash
cd feedback-atlas
.venv/bin/python -m pytest -q
```

CPU only, no model weights, no network — the same rule as the rest of the repo.
numpy and pytest are the only hard requirements; tests needing umap, fastapi or
httpx skip rather than fail, so the suite runs on a bare interpreter.

Two of them are worth knowing about, because they check things that are usually
only hoped for:

- `test_embedder_contract.py` imports `src.embedder` **in a subprocess** and
  asserts torch is not in `sys.modules`. That single rule is what keeps this
  suite offline; once torch is loaded by any other test, an in-process check
  cannot tell who loaded it.
- `test_server_routes.py` connects a participant and an admin to one running
  server and asserts that, of a single broadcast, the admin sees `reviewer_id`
  and the participant does not. Every per-module test passed while that leak was
  live; it only exists when both channels are connected at once.
- `test_viewer_pipeline.py` submits sentences over a real websocket and then asks
  the running server for them in SQL, which is the only honest way to test the
  viewer's half: it asks for `reviewer_id` over HTTP and requires DuckDB to refuse
  because the column is absent, and it asks for `/etc/passwd` and requires the
  same. It skips without duckdb, pyarrow or embedding-atlas.

Browser lifecycle checks are opt-in: from `feedback-atlas/`, run
`.venv/bin/python -m pytest tests/browser_viewer.py -q` with Playwright and its
Chromium browser installed. They use a temporary database and a local test server
to check the disabled analysis control, classroom map continuity and live roster
updates. Full submission, recovery, correction and export commands and measured
results are in the [release report](docs/qa/feedback-atlas-release.md). Browser
scenarios use synthetic data and a fake model; semantic quality is tested separately.

## Known limits

- **Identity is unauthenticated by design.** Anyone with the link who knows a
  classmate's id can post as them. The PRD accepts a login-free flow, so this is
  a documented residual risk — but it is the reliability ceiling on admin mode's
  author view, and worth knowing before leaning on it for anything graded.
- **The map reorganises once, at 80 opinions**, when the projector switches from
  PCA to UMAP. The two produce structurally different layouts and no amount of
  alignment repairs that. `ATLAS_PCA_UMAP_THRESHOLD` is configurable; cross it
  deliberately between sessions rather than mid-presentation.
- **AI/human is self-reported.** Anyone may tag anything either way (PRD 5.2), so
  the AI-versus-human comparison is only as good as the tagging. PRD 11 flags
  this and the app does not try to detect it.
- **Keep the database on local disk.** SQLite's WAL mode is unreliable over
  NFS/SMB.
- **The viewer needs WebGPU, and `shader-f16` with it.** Embedding Atlas 0.24
  dropped the WebGL2 fallback, and its renderer asks for a device with that
  feature — so software adapters and older integrated drivers hand out an adapter
  and still cannot draw. The app checks for the feature rather than just the
  adapter, and falls back to the hand-written map; checking only for the adapter
  is how this was written first, and it fails on exactly the machines it is meant
  to catch.
- **`/data/query` runs client SQL, and that is what a Mosaic application is.**
  File and network reads are off (`enable_external_access = false`, then
  `lock_configuration = true`, which is what Apple's own server does), and the
  participant database physically lacks the reviewer columns. What is *not*
  prevented is a participant issuing DDL — Mosaic legitimately creates temporary
  tables, so the endpoint cannot be read-only. The relation is rebuilt from state
  on every recompute, so damage lasts one recompute. Set `ATLAS_DISABLE_VIEWER=1`
  if that trade is wrong for your room.

## Design notes

Apple's [Embedding Atlas](https://github.com/apple/embedding-atlas) (MIT) is used
three ways here, and only the third is cosmetic.

**Its analysis recipe, reproduced rather than called.** `src/ea_projection.py`
builds the approximate k-NN graph with umap's `nearest_neighbors` and hands it to
UMAP as `precomputed_knn`, which is the sequence and the parameters
`embedding_atlas.projection` uses. Their function is deliberately not called: it
builds a dataframe, wraps the work in a file cache keyed on the inputs, and throws
away the fitted reducer — and without the reducer every later opinion forces a
full refit, which is the one thing a map being read during a presentation cannot
afford. So the algorithm and its defaults are theirs; the call is ours. The graph
that fit produces is what fills the `neighbors` column.

Worth stating plainly because the two are easy to conflate: on the live path the
only upstream Python that executes is its Arrow writer, which `/data/query`
answers chart queries with.

**Its viewer and renderer.** Vendored into `web/vendor/` and mounted against a
server-side DuckDB. See the section above.

**Its visual language**, for the hand-written map — the slate desk, the
white/black cards, the hairline borders, the single blue accent, d3's
`category10`, the translucent tooltip, the halo'd map labels. Two departures for
Korean: opinion text at 14px/1.6 rather than 13px, and `word-break: keep-all`,
without which browsers break Korean mid-word.

The hand-written scatter in `web/atlas.js` stays, and is not redundant. It
encodes two channels where `EmbeddingView` encodes one, it runs without WebGPU,
and it animates an opinion arriving. It is what a phone gets, and what any laptop
gets when the viewer cannot draw.

No CDN, no webfont, and nothing fetched at class time. The frontend is still
plain ES modules served straight from `web/`; the bundler runs once in
`frontend/`, at development time, and its output is committed.
