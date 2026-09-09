# Real-time Feedback Atlas

<p align="center">
  <b>English</b> · <a href="README_kr.md">한국어</a>
</p>

A classroom tool for the CTP49906 *Multimodal AI* workshop. During a student
presentation, attendees open a link on their phones and write what they think.
Each opinion is embedded **locally** — no API call — projected to 2D, and pushed
to every connected screen within a second.

Colour encodes **which student the feedback is about**; shape encodes **AI or
human**. That second channel is the point: the session asks how far a model can
describe subjective experience, and this puts both kinds of comment in one
semantic space so the room can see whether they land in the same place.

Two front ends sit on that one corpus: a hand-written map built for a phone and a
projector, and Apple's Embedding Atlas viewer for the table, the linked charts and
the cross-filtering. Both read the same rows.

The specification is [`realtime-feedback-atlas-prd.md`](realtime-feedback-atlas-prd.md).

## What it is not

Not a campus login system. Each enrolled person enters only their fixed private
access code; the server resolves the matching roster identity and role. The
participant websocket accepts `{code}` and returns an 8-hour signed session
token, or accepts a later `{session_token}` resume. Existing codes stay the same
across restarts; a restart expires sessions, not the personal codes. Duplicate
codes are rejected so one code cannot identify two people.

Not anonymous, and not pretending to be. `reviewer_id` is stored on the server
and never crosses a participant connection. Admin mode, behind a fixed access
code, is the only surface that shows who wrote what.

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
kang.minsu,강민수,observer
ta.youngjun,영준,ta
```

Three roles. `student` (수강생) rows are also the **targets** — the projects
feedback can be *about*. `observer` (청강생) and `ta` (조교) write feedback and are
never targets, which is what the course actually looks like: a TA comments on
work, nobody comments on the TA.

PRD 7's table names only `student` and `observer`; `ta` is a deliberate addition,
because PRD 3 already lists 조교 as a user who may submit — and submitting needs a
roster entry. Without a role of their own a TA has to be filed as 청강생, and the
admin panel, which exists to attribute writing correctly, would then label them as
something they are not. Matching is forgiving about spaces, case and
full-width characters (a Korean IME left in 전각 mode emits `ｋｉｍ`), and about
nothing else: an edit-distance match would let one student's id resolve to
another's.

Generate the private per-person codes after the roster is final enough to share.
The JSON file contains only SHA-256 hashes and is read by the server; the CSV
contains cleartext codes and must be sent privately to each participant, one code
at a time. Do not paste the whole CSV into a public chat or slide.

```bash
python scripts/create_access_codes.py \
  --roster roster.csv \
  --hashes .run/access-codes.json \
  --out .run/access-codes.csv
```

The script writes private files with mode `0600` under `.run/` and refuses to
overwrite existing credentials. Rotate by moving the old files aside and issuing
new codes privately.

Adding someone later means editing the roster, generating a code for the new
roster state, and sending `SIGHUP` — no restart, no dropped connections:

```bash
kill -HUP $(pgrep -f 'uvicorn.*src.server')
```

### Configuration

Copy [`.env.example`](.env.example) to `.env`. `ATLAS_ADMIN_CODE` and
`ATLAS_ROSTER` have no defaults. `ATLAS_ACCESS_CODES` defaults to
`.run/access-codes.json`, and every roster id must have a hash there before the
server starts. A default admin code is a published admin code.

If the previous admin code ever crossed plain HTTP, rotate it before class:

```bash
python scripts/configure_security.py
```

The new admin code is written to `.run/admin-code.txt` with private file
permissions. Do not put its contents in documentation or slides.

## Running it

The main classroom (`/`) and practice demo (`/demo/`) are separate apps sharing
the same interface. The main classroom uses private codes and its own weeks 1–4
corpus. The demo uses one default `demo` account without a personal-code field; its comments,
sessions, queries, and exports cannot enter the main classroom. The existing 30
example opinions were assigned to the demo, with an empty main classroom.
See [the deployment runbook](SECURITY_DEPLOYMENT.md#classroom-and-demo) for data
paths and the one-time migration. For a fresh install, create `.run/classroom`
and `.run/demo` before starting the configured databases.

The classroom deployment is the fixed numeric HTTPS URL:

```text
https://143.248.249.7:8888
```

This uses the trusted IP certificate already obtained for `143.248.249.7`, stored
under `.run/tls/live/feedback-atlas-ip/`. No domain and no Cloudflare tunnel are
needed for the normal class path.

Run the HTTPS service under the user systemd unit:

```bash
systemctl --user link "$(pwd)/deploy/feedback-atlas.service"
systemctl --user link "$(pwd)/deploy/feedback-atlas-renew.service"
systemctl --user link "$(pwd)/deploy/feedback-atlas-renew.timer"
systemctl --user daemon-reload
systemctl --user enable --now feedback-atlas.service feedback-atlas-renew.timer
loginctl enable-linger "$USER"
```

`scripts/serve.sh` runs uvicorn with TLS directly, disables proxy headers, forces
`ATLAS_ALLOW_INSECURE_HTTP=0`, and uses the offline Hugging Face cache. The
renewal timer checks twice daily and restarts the service only when the
certificate file changes.

For local development and tests only, plain HTTP is allowed by setting
`ATLAS_ALLOW_INSECURE_HTTP=1` while binding to loopback:

```bash
source .venv/bin/activate
export $(grep -v '^#' .env | xargs)
export ATLAS_ALLOW_INSECURE_HTTP=1
python -m uvicorn --factory src.server:create_app \
  --host 127.0.0.1 --port 8888
```

The server entry point refuses unsafe public HTTP/proxy-trust fallbacks, so do
not bind an insecure development command to `0.0.0.0`.

Start it **five minutes early**. The first run loads the model and warms UMAP's
numba kernels; done at the bell, that cost lands on the first submission of the
class instead.

`GET /healthz` reports the model's cache key, the corpus size, the projector in
use and the connection counts. It never reports the access code.

### Getting the room connected

Share only the HTTPS IP address:

```text
https://143.248.249.7:8888
```

The page, websocket, and Embedding Atlas query endpoint are all on that origin.
The server is not restricted to a classroom IP range because no known classroom
CIDR is installed; security comes from TLS, per-person codes, signed sessions,
and the admin code rather than from source-IP filtering.

**An SSH tunnel is not a classroom path.** `ssh -L 8888:127.0.0.1:8888 …` forwards
the port to *one* machine — the laptop you typed it on. It is the right way to
preview the map yourself over SSH, and it does nothing for thirty phones.

Cloudflare is outside the recommended launch path. If you manually use an
alternative tunnel later, keep the same participant/admin credentials, HTTPS
browser origin, websocket upgrade, and origin checks.

### On the day

```bash
scripts/class.sh start     # starts the managed HTTPS service and renewal timer, then prints URL and QR
scripts/class.sh url       # prints https://143.248.249.7:8888 and refreshes .run/qr.png
scripts/class.sh status    # service status plus /healthz
scripts/class.sh stop
curl --noproxy '*' --resolve 143.248.249.7:8888:127.0.0.1 https://143.248.249.7:8888/healthz
journalctl --user -u feedback-atlas.service -f
```

Start it **five minutes early**. The first run loads the model and warms UMAP's
numba kernels; done at the bell, that cost lands on the first submission of the
class instead.

See [`SECURITY_DEPLOYMENT.md`](SECURITY_DEPLOYMENT.md) for the runbook.

## Admin mode

The **주차 일정 설정** button opens a calendar with four start-date inputs and
seven-day range previews. The main classroom assigns each new opinion from its
server receipt timestamp in **Asia/Seoul**, ignoring any client-supplied week.
Defaults are October 15–21, October 22–28, November 5–11, and November 12–18,
2026. October 29–November 4 is the midterm break. New submissions outside these
ranges are refused without discarding the browser draft; an already accepted
submission can still recover its receipt after a period ends.

Saved schedules survive restarts. Updates apply to future submissions and do
not rewrite the week on existing opinions. Concurrent admin edits require a
fresh schedule revision. The demo has separate settings and always permits
practice in week 1. The participant form has no week dropdown; its placeholder
changes with the selected AI/human source.

Open `/admin` and enter the access code. The code buys a **different websocket**,
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
every column that cross-filter each other, full-text search, and
its WebGPU embedding view with density contours and automatic cluster labels.

Toggle it with the panel button in the left rail. It turns itself on at startup on
a wide screen with a capable browser, and stays off on a phone, where the job is
to write a sentence rather than to explore a dashboard.

**It is a Mosaic application, so it queries rather than receives.** The server
keeps an in-memory DuckDB holding the corpus and answers `POST /data/query`, which
is `embedding_atlas.server`'s own endpoint with its three command types, so the
component talks to it unmodified. Every recompute reloads that relation before the
websocket broadcast goes out, so the charts and the map never disagree.

**There are two databases, and the endpoint picks one before parsing any SQL.**
The client writes the query, so no filter on the way out could hold —
`SELECT reviewer_id FROM dataset` is an ordinary Mosaic-shaped request. The
participant database therefore does not contain the column, and asking for it
returns a DuckDB binder error rather than data. Participant queries now also need
`Authorization: Bearer <session_token>` from the websocket hello. Admin queries
keep using the admin code in the request body, never the URL, exactly as with CSV
export.

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

The build applies a version- and source-hash-checked adapter for Embedding Atlas
0.24.0. It computes category IDs as read-only SQL expressions rather than altering
the shared relation. `npm run test:readonly-category` checks the adapter's build
guards. An upstream version/source change deliberately fails the build until the
adapter and real-viewer contract tests have been reviewed.

The same package also gives you the official CLI over an exported CSV, which needs
no server at all:

```bash
embedding-atlas feedback-atlas.csv --text text --x x --y y
```

## Bulk AI import (PRD 4.3)

The submit path is rate-limited to about one message a second per connection —
right for people, wrong for a script. `scripts/seed_ai_opinions.py` writes to the
database instead:

```bash
python scripts/seed_ai_opinions.py comments.csv \
  --db .run/classroom/atlas.db --roster roster.csv --reviewer instructor --dry-run
python scripts/seed_ai_opinions.py comments.csv \
  --db .run/classroom/atlas.db --roster roster.csv --reviewer instructor
kill -USR1 $(pgrep -f 'uvicorn.*src.server')   # re-read without a restart
```

Input is `target_id,text[,source][,week]` as CSV or JSONL. Rows whose target
cannot be resolved are skipped and listed, never guessed at. Use `--dry-run`
first: opinion ids are generated per row, so importing the same file twice
imports it twice.

## After the semester (PRD 6, 7)

```bash
python scripts/deidentify.py --db .run/classroom/atlas.db --roster roster.csv --out archive/
```

Writes `archive/archive.csv` with every id replaced by a random code, and
`archive/mapping.csv` holding the codes. It never touches the source database.

Keep the two apart. The mapping is the only thing that re-identifies the archive;
destroy it, or store it where the archive is not. Codes are random and shuffled,
not derived from the ids — a hash or a roster-order number is reversible by
anyone who has the class list.

**The archive is pseudonymous, not anonymous.** The opinions themselves are
unchanged, and people write things like "제가 발표에서 말했듯이". PRD 11 is right
that the real control for research use is consent obtained separately from
participation, with analysis only after grades are final. This script is the
substitution step, not the ethics.

## Tests

```bash
python -m pytest feedback-atlas
```

CPU only, no model weights, no network — the same rule as the rest of the repo.
numpy and pytest are the only hard requirements; tests needing umap, fastapi or
httpx skip rather than fail, so the suite runs on a bare interpreter.

Three of them are worth knowing about, because they check things that are usually
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
to check the WebGPU fallback, viewer choices during reconnects and delayed probes,
and roster reload notifications. The viewer is stubbed for lifecycle checks;
these tests do not verify hardware rendering.

## Known limits

- **Credentials are private classroom credentials, not campus SSO.** A person who
  receives or steals someone else's code can still post as that person. Give each
  participant only their own code, rotate codes if one is exposed, and do not use
  the admin authorship view as a grading authority.
- **The service is not IP-range limited.** No classroom CIDR is currently
  installed. Anyone who can reach `https://143.248.249.7:8888` can load the page,
  but they still need a valid participant code or admin code to use protected
  endpoints.
- **Dedicated OS-user isolation is not configured.** The service currently runs
  from this repository under the existing `june` user account.
- **Application dependencies were not changed by the security deployment.** The
  dependency advisory scan has not been rerun in this documentation pass.
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
- **`/data/query` accepts bounded, read-only queries.** File and network reads
  are off (`enable_external_access = false`, then `lock_configuration = true`),
  and the participant database physically lacks reviewer columns. Mutation and
  multi-statement SQL are refused, including requests labeled `exec`. A pinned
  viewer build adapter uses inline category expressions instead of table writes.
  If a relation refresh fails, viewer queries return 503 until both audience
  relations recover; feedback submission and the map remain available.

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
