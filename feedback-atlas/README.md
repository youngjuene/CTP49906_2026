# Real-time Feedback Atlas

<p align="center">
  <b>English</b> · <a href="README_kr.md">한국어</a>
</p>

A classroom tool for the CTP49906 *Multimodal AI* workshop. The instructor
imports feedback and students' LLM conversations on the server; anyone with the
link opens the map on a phone or the projector, with no login. Each opinion is
embedded **locally** — no API call — projected to 2D, and pushed to every
connected screen as soon as it is imported.

The home page (`/`, also `/chats.html`) is the [chat map](#chat-trajectories-chatshtml);
the opinion map is at `/map`, and admin mode at `/admin`.

Colour encodes **which student the feedback is about**; shape encodes **AI or
human**. That second channel is the point: the session asks how far a model can
describe subjective experience, and this puts both kinds of comment in one
semantic space so the room can see whether they land in the same place.

Two front ends sit on that one corpus: a hand-written map built for a phone and a
projector, and Apple's Embedding Atlas viewer for the table, the linked charts and
the cross-filtering. Both read the same rows.

The specification is [`realtime-feedback-atlas-prd.md`](realtime-feedback-atlas-prd.md).

## What it is not

Not a submission form. The participant page is anonymous and read-only: there are
no personal codes, no sessions, and nothing a browser sends can add an opinion.
Opinions arrive only through [bulk import](#bulk-ai-import-prd-43) on the server,
and conversations through [chat trajectories](#chat-trajectories-chatshtml).

Not a public record of who wrote what. `reviewer_id` is stored on the server and
never crosses a participant connection. Admin mode, behind the admin code, is the
only surface that shows it. Target names (the students whose work is discussed)
and everything on the chat map *are* visible to anyone with the link.

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
feedback can be *about*, and the people whose chats can be imported. `observer`
(청강생) and `ta` (조교) can be credited as reviewers on imported feedback and are
never targets. Matching is forgiving about spaces, case and full-width characters
(a Korean IME left in 전각 mode emits `ｋｉｍ`), and about nothing else: an
edit-distance match would let one student's id resolve to another's.

The roster is the only list of people the server holds; there are no per-person
credentials to generate. Editing it later takes a `SIGHUP` — no restart, no
dropped connections:

```bash
kill -HUP $(pgrep -f 'uvicorn.*src.server')
```

### Configuration

Copy [`.env.example`](.env.example) to `.env`. `ATLAS_ADMIN_CODE` and
`ATLAS_ROSTER` have no defaults; a default admin code is a published admin code.
`ATLAS_CHATS` (default `.run/chats/atlas.json`) is where the chat map is read
from.

If the previous admin code ever crossed plain HTTP, rotate it before class:

```bash
python scripts/configure_security.py
```

The new admin code is written to `.run/admin-code.txt` with private file
permissions. Do not put its contents in documentation or slides.

## Running it

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
numba kernels; done at the bell, that cost lands on the first import of the class
instead.

`GET /healthz` reports the model's cache key, the corpus size, the projector in
use and the connection counts. It never reports the admin code.

### Getting the room connected

Share only the HTTPS IP address:

```text
https://143.248.249.7:8888
```

The page, websocket, and Embedding Atlas query endpoint are all on that origin,
and none of them asks for a code. The server is not restricted to a classroom IP
range: anyone who has the link can read the map and the chat map. What stays
protected is authorship, behind the admin code.

**An SSH tunnel is not a classroom path.** `ssh -L 8888:127.0.0.1:8888 …` forwards
the port to *one* machine — the laptop you typed it on. It is the right way to
preview the map yourself over SSH, and it does nothing for thirty phones.

Cloudflare is outside the recommended launch path. If you manually use an
alternative tunnel later, keep the same admin code, HTTPS browser origin,
websocket upgrade, and origin checks.

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
numba kernels; done at the bell, that cost lands on the first import of the class
instead.

See [`SECURITY_DEPLOYMENT.md`](SECURITY_DEPLOYMENT.md) for the runbook.

## Admin mode

Open `/admin` and enter the admin code. The code buys a **different websocket**,
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
a wide screen with a capable browser, and stays off on a phone, where a dashboard
does not earn its space.

**It is a Mosaic application, so it queries rather than receives.** The server
keeps an in-memory DuckDB holding the corpus and answers `POST /data/query`, which
is `embedding_atlas.server`'s own endpoint with its three command types, so the
component talks to it unmodified. Every recompute reloads that relation before the
websocket broadcast goes out, so the charts and the map never disagree.

**There are two databases, and the endpoint picks one before parsing any SQL.**
The client writes the query, so no filter on the way out could hold —
`SELECT reviewer_id FROM dataset` is an ordinary Mosaic-shaped request. The
participant database therefore does not contain the column, and asking for it
returns a DuckDB binder error rather than data. Participant queries need no
credentials and are rate limited per address. Admin queries carry the admin code
in the request body, never the URL, exactly as with CSV export.

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

This is the only way opinions reach the map: `scripts/seed_ai_opinions.py` writes
to the database, and `SIGUSR1` tells the running server to pick them up. The
reviewer must be a roster id.

```bash
python scripts/seed_ai_opinions.py comments.csv \
  --db .run/classroom/atlas.db --roster roster.csv --reviewer ta.youngjun --dry-run
python scripts/seed_ai_opinions.py comments.csv \
  --db .run/classroom/atlas.db --roster roster.csv --reviewer ta.youngjun
kill -USR1 $(pgrep -f 'uvicorn.*src.server')   # re-read without a restart
```

Input is `target_id,text[,source][,week]` as CSV or JSONL. Rows whose target
cannot be resolved are skipped and listed, never guessed at. Use `--dry-run`
first: opinion ids are generated per row, so importing the same file twice
imports it twice.

## Chat trajectories (`/chats.html`)

A separate view for students' conversations with an LLM: each conversation is
split into sentences, laid out in the same embedding space, and drawn as a path
in the order it was said. It is deliberately not on the classroom map — one
conversation is ~800 sentences, the map is sized for ~1000 opinions a semester,
and a student's own prompts are not peer feedback.

1. Open the claude.ai share link, scroll to the end, and save it with
   **Ctrl+S → "Webpage, Complete"** (not "HTML Only": the page renders with
   scripts). The saved page marks who said what and when; a copy-paste does not.
2. Put it in `.run/chats/` named `<target_id>__w<week>__<nn>.html`, e.g.
   `kim.seoyeon__w1__01.html`. The target must be a student on the roster.
3. Rebuild the map (only new text is embedded; the cache is
   `.run/chats/embeddings.db`, never the classroom database):

```bash
HF_HUB_OFFLINE=1 python scripts/build_chat_atlas.py --chats .run/chats --roster roster.csv
```

No restart is needed: `/api/chats` reads `.run/chats/atlas.json` per request.
It is open to anyone with the link, like the page itself, so only import
conversations the students have agreed to show this way. To check a
conversation's split before building, `scripts/chat_to_trajectory.py` writes a
readable `<stem>.preview.md`.

Rebuilding keeps the map stable: the layout starts from the previous build and is
rotated onto it, and clusters keep their ids, colours and legend order, so adding a
chat does not spin or recolour the map (sentences still shift locally as the layout
makes room). With nothing new, nothing moves. `--fresh` lays everything out from
scratch. A second save of the same conversation (same first message at the same
time) is skipped in favour of the longer save, with a note in the build output.

What the conversion keeps and drops: student prompts (square) and model
sentences (circle) are both points; status lines such as "웹 검색됨" are kept as
markers (diamond) pinned to the next sentence rather than laid out; code blocks
(pasted Claude Code prompts), citation chips and the screen-reader heading are
dropped. Fragments under 20 characters merge into a neighbour. The week comes
from the file name, not the message times.

The view opens in **3D** (a 보기 2D/3D switch keeps the flat map one click away):
the build also fits a 3D cloud, drawn with perspective, and each step of the
conversation is a SLERP arc about the cloud's centre, so the cursor travels a
curve rather than a straight line. 3D draws the numbered turn route, the path
just left behind (the previous turn and the current one so far) in solid grey,
and a short fading blue trail behind the cursor. The view is the reader's: drag to rotate
(a flick keeps spinning briefly), Shift-drag, right-drag or two fingers to pan,
scroll or pinch to zoom toward the pointer, double-click to reset. It never
rotates on its own; with 따라가기 on and the view zoomed in, it slides to keep
the cursor on screen, and waits a few seconds after the reader moves it. The 3D cloud keeps sentence neighbourhoods slightly better than the flat
map in 3D, but any single view of it is a flattening, so the flat map remains
the more faithful still picture.

Playback is paced for reading by default: each sentence stays highlighted for
its reading time at 20 Hangul syllables a second (Latin letters, digits and
punctuation count half, minimum 0.9 s), shown by a thin bar under the sentence.
A typical sentence gets about 2 s and a 30-turn conversation about half an hour
at 1x. The speed menu scales it (0.5x-4x), and its 고정 options switch to a fixed
beat (about a second a sentence at 1x, whatever its length).

The view draws three strengths of path: the current turn sentence by sentence,
earlier sentences faintly, and a numbered **turn route** through the centre of
each turn — the conversation's overall course.

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
- `test_viewer_pipeline.py` seeds sentences into a running server and then asks
  it for them in SQL, which is the only honest way to test the
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

- **Everything except authorship is public to anyone with the link.** There is
  no participant login and no IP-range limit, so anyone who can reach
  `https://143.248.249.7:8888` can read the map, the analysis view and every
  imported conversation on `/chats.html`. Only `reviewer_id`, the admin tools and
  CSV export sit behind the admin code.
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
  relations recover; the map remains available.

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
