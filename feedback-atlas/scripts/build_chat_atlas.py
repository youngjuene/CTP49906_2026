#!/usr/bin/env python
"""Lay out every saved conversation on one map, for the chat view (/chats.html).

Reads each .run/chats/<target_id>__w<week>__<nn>.html through
scripts/chat_to_trajectory.py, embeds every unit with the classroom's model, and
projects them all together, so two students' trajectories share one space and can
be compared. Writes a single JSON file the server hands to logged-in roster
members:

    uv run --python .venv/bin/python scripts/build_chat_atlas.py \\
        --chats .run/chats --roster roster.csv

Why a separate corpus rather than rows in the classroom database. One
conversation split into sentences is ~800 points; the classroom map is sized for
about a thousand opinions over the whole semester, and every point it holds moves
every other point when the layout is refitted. Chats would bury the peer feedback
the map exists to show, and a student's own prompts are not feedback at all.

Embeddings are cached in their own SQLite file (--cache), never in the live
classroom database: this script runs while the server is up, and writing into a
database another process owns is how a class loses its data. Rebuilding after a
new chat is added only embeds the new text.

The output is private: 0600, under .run/, which git ignores. It holds what
students wrote to a model.
"""

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.chat_to_trajectory import convert  # noqa: E402
from src.config import DEFAULT_MODEL, cache_key, resolve_spec  # noqa: E402
from src.projection import normalize_scale, project  # noqa: E402
from src.roster import Roster  # noqa: E402
from src.store import Store  # noqa: E402
from src.textnorm import text_hash  # noqa: E402


def _embedder(model: str, device: str, fake: bool):
    spec = resolve_spec(model)
    if fake:
        # Tests only: deterministic vectors, no model download.
        from src.embedder import HashEmbedder
        return HashEmbedder(dim=spec.dim, model_id=f"fake:{spec.model_id}"), \
            f"fake:{cache_key(spec)}"
    from src.embedder import SentenceTransformerEmbedder
    return SentenceTransformerEmbedder(spec, device=device), cache_key(spec)


def embed(texts, store: Store, embedder, key: str) -> np.ndarray:
    hashes = [text_hash(t) for t in texts]
    cached = store.get_embeddings(hashes, key)
    todo = {}
    for h, t in zip(hashes, texts):
        if h not in cached and h not in todo:
            todo[h] = t
    if todo:
        vectors = embedder.encode(list(todo.values()))
        fresh = dict(zip(todo.keys(), vectors))
        store.put_embeddings(key, vectors.shape[1], fresh)
        cached.update(fresh)
    print(f"embedded {len(todo)} new unit(s), {len(set(hashes)) - len(todo)} from cache")
    return np.stack([cached[h] for h in hashes]) if hashes else np.zeros((0, 2))


def _pin_status(rows, coords, offset: float = 0.04) -> None:
    """Place each status line just off the next prose unit of its conversation
    (the previous one when it comes last), so it sits on the path where it
    happened. Offsets fan out so stacked status lines stay individually visible."""
    def prose_in(conversation, indices):
        return next((j for j in indices if rows[j]["conversation"] == conversation
                     and rows[j]["kind"] != "status"), None)

    for i, row in enumerate(rows):
        if row["kind"] != "status":
            continue
        anchor = prose_in(row["conversation"], range(i + 1, len(rows)))
        if anchor is None:
            anchor = prose_in(row["conversation"], range(i - 1, -1, -1))
        if anchor is None:
            continue
        angle = 2.4 * row["seq"]            # golden-angle-ish spread
        coords[i] = coords[anchor] + offset * np.array([np.cos(angle), np.sin(angle)])


def _write_private(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        os.chmod(tmp, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)          # a reader never sees half a file
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chats", type=Path, default=Path(".run/chats"))
    ap.add_argument("--roster", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <chats>/atlas.json")
    ap.add_argument("--cache", type=Path, default=None,
                    help="default: <chats>/embeddings.db")
    ap.add_argument("--model", default=os.environ.get("ATLAS_EMBEDDING_MODEL", DEFAULT_MODEL))
    ap.add_argument("--device", default=os.environ.get("ATLAS_DEVICE", "cpu"))
    ap.add_argument("--min-chars", type=int, default=20)
    ap.add_argument("--max-chars", type=int, default=1000)
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fake-embedder", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    out = args.out or args.chats / "atlas.json"
    cache = args.cache or args.chats / "embeddings.db"

    roster = Roster.from_path(args.roster)
    pages = sorted(args.chats.glob("*.html"))
    if not pages:
        raise SystemExit(f"no saved .html conversations in {args.chats}")

    rows, conversations, refused = [], [], 0
    for page in pages:
        try:
            units, _ = convert(page, roster, min_chars=args.min_chars,
                               max_chars=args.max_chars)
        except (OSError, ValueError) as exc:
            print(f"  refused {exc}", file=sys.stderr)
            refused += 1
            continue
        times = [u["timestamp"] for u in units if u["timestamp"]]
        conversations.append({
            "id": page.stem, "target_id": units[0]["target_id"], "week": units[0]["week"],
            "turns": max(u["turn"] for u in units), "units": len(units),
            "started": min(times, default=None), "ended": max(times, default=None),
        })
        rows.extend(units)
        print(f"{page.name}: {len(units)} units")
    if not rows:
        raise SystemExit("nothing to lay out")

    # Status lines ("웹 검색됨", "공유 시 숨겨지는 파일") are events, not content. Laid
    # out with the sentences, nine copies of one string form an island of their
    # own and the fit stretches to reach it, squashing everything else. They are
    # left out of the layout and pinned beside the unit they precede instead.
    prose = [i for i, r in enumerate(rows) if r["kind"] != "status"]
    if not prose:
        raise SystemExit("nothing but status lines to lay out")
    embedder, key = _embedder(args.model, args.device, args.fake_embedder)
    store = Store(str(cache))
    store.migrate()
    try:
        X = embed([rows[i]["text"] for i in prose], store, embedder, key)
    finally:
        store.close()

    coords = np.zeros((len(rows), 2))
    coords[prose] = normalize_scale(project(X, n_neighbors=args.n_neighbors, seed=args.seed))
    _pin_status(rows, coords)
    points = [{**r, "id": f"{r['conversation']}#{r['seq']}",
               "x": round(float(x), 5), "y": round(float(y), 5)}
              for r, (x, y) in zip(rows, coords)]
    names = {t["id"]: t["display_name"] for t in roster.targets()}
    payload = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": key,
        "targets": [{"id": c, "display_name": names.get(c, c)}
                    for c in sorted({c["target_id"] for c in conversations})],
        "conversations": conversations,
        "points": points,
    }
    _write_private(out, json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {out}: {len(points)} points from {len(conversations)} conversation(s)")
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
