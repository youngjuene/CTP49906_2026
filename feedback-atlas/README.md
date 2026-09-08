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

The specification is [`realtime-feedback-atlas-prd.md`](realtime-feedback-atlas-prd.md).

## What it is not

Not a login system. There are no passwords: a landing page checks the id you type
against a roster the instructor registered in advance, and that is the whole gate.
Registration is strict — an id that is not on the list does not get in.

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
  --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips='*'
```

Start it **five minutes early**. The first run loads the model and warms UMAP's
numba kernels; done at the bell, that cost lands on the first submission of the
class instead.

`GET /healthz` reports the model's cache key, the corpus size, the projector in
use and the connection counts. It never reports the access code.

### Getting the room connected

**Try the LAN first.** Lowest latency, no quotas, no third party: share
`http://<your-ip>:8000` and a QR code. Test it from a phone on the same SSID *the
day before* — many campus and guest networks enable client isolation, which
silently blocks phone-to-laptop traffic and looks exactly like the server being
down.

**An SSH tunnel is not a classroom path.** `ssh -L 8000:127.0.0.1:8000 …` forwards
the port to *one* machine — the laptop you typed it on. It is the right way to
preview the map yourself over SSH, and it does nothing for thirty phones.

**For the room, use a Cloudflare quick tunnel.** No account, one command, and it
hands back an HTTPS URL any device can open from any network:

```bash
cloudflared tunnel --url http://localhost:8000
# → https://<random-words>.trycloudflare.com   share this, or a QR of it
```

Verified end to end on this codebase: HTTPS page load, **WebSocket connected
through the tunnel**, snapshot delivered, a submit round-tripped and acknowledged,
and no reviewer id in the participant page — all from a phone-sized client. The
URL changes on every restart, so generate it before the session and put it on a
slide, not during.

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

## Bulk AI import (PRD 4.3)

The submit path is rate-limited to about one message a second per connection —
right for people, wrong for a script. `scripts/seed_ai_opinions.py` writes to the
database instead:

```bash
python scripts/seed_ai_opinions.py comments.csv \
  --db atlas.db --roster roster.csv --reviewer instructor --dry-run
python scripts/seed_ai_opinions.py comments.csv \
  --db atlas.db --roster roster.csv --reviewer instructor
kill -USR1 $(pgrep -f 'uvicorn.*src.server')   # re-read without a restart
```

Input is `target_id,text[,source][,week]` as CSV or JSONL. Rows whose target
cannot be resolved are skipped and listed, never guessed at. Use `--dry-run`
first: opinion ids are generated per row, so importing the same file twice
imports it twice.

## After the semester (PRD 6, 7)

```bash
python scripts/deidentify.py --db atlas.db --roster roster.csv --out archive/
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

## Design notes

The visual language is Apple's
[Embedding Atlas](https://github.com/apple/embedding-atlas) (MIT) — the slate
desk, the white/black cards, the hairline borders, the single blue accent, d3's
`category10`, the translucent tooltip, the halo'd map labels. Its *code* is not
used, for three measured reasons: its `EmbeddingView` encodes only colour and
this needs colour and shape at once; v0.24 is WebGPU-only, which on a room of
assorted laptops means a blank canvas for some of them; and at a few hundred
points its density contours and automatic labels compute to zero alpha and render
nothing anyway. Two departures for Korean: opinion text at 14px/1.6 rather than
13px, and `word-break: keep-all`, without which browsers break Korean mid-word.

No npm, no bundler, no webfont. The frontend is plain ES modules served straight
from `web/`, which is deliberate for a tool whose failing network is the thing it
has to survive.
