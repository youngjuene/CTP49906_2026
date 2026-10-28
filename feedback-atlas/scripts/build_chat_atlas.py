#!/usr/bin/env python
"""Lay out every saved conversation on one map, for the chat view (/chats.html).

Reads each .run/chats/<target_id>__w<week>__<nn>.html through
scripts/chat_to_trajectory.py, embeds every unit with the classroom's model, and
projects them all together, so two students' trajectories share one space and can
be compared. Sentences are also grouped into topic clusters (scripts/chat_clusters.py)
with keyword labels and colours. Writes a single JSON file the server serves at
/api/chats:

    uv run --python .venv/bin/python scripts/build_chat_atlas.py \\
        --chats .run/chats --roster roster.csv

Why a separate corpus rather than rows in the classroom database. One
conversation split into sentences is ~800 points; the classroom map is sized for
about a thousand opinions over the whole semester, and every point it holds moves
every other point when the layout is refitted. Chats would bury the peer feedback
the map exists to show, and a student's own prompts are not feedback at all.

Every unit gets two layouts: the 2D map (x, y) and a 3D cloud (x3, y3, z3) for
the 3D view, which draws each step as an arc. Both are fitted the same way.

Rebuilds are stable. When the output file already exists (and was built with
the same model), the layout starts from it: sentences already on the map start
where they were, new ones start beside their nearest neighbours in meaning, and
the result is rotated onto the old frame. Clusters start from their old centres
and keep their ids and colours. --fresh ignores the old file. A page that is a
second save of a conversation already in the folder (same first message at the
same moment) is skipped in favour of the longer save.

Embeddings are cached in their own SQLite file (--cache), never in the live
classroom database: this script runs while the server is up, and writing into a
database another process owns is how a class loses its data. Rebuilding after a
new chat is added only embeds the new text.

The output file is 0600, under .run/, which git ignores: it holds what students
wrote to a model, and the server, not the filesystem, decides who sees it.
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

from scripts.chat_clusters import assign_colours, cluster, keywords  # noqa: E402
from scripts.chat_to_trajectory import convert  # noqa: E402
from src.config import DEFAULT_MODEL, cache_key, resolve_spec  # noqa: E402
from src.projection import normalize_scale, project, stabilize  # noqa: E402
from src.roster import Roster  # noqa: E402
from src.store import Store  # noqa: E402
from src.textnorm import text_hash  # noqa: E402


def _embedder(model: str, device: str, fake: bool):
    """-> (a function that loads the embedder, cache key). Loading is deferred so
    a rebuild whose text is all cached never touches the model."""
    spec = resolve_spec(model)
    if fake:
        # Tests only: deterministic vectors, no model download.
        from src.embedder import HashEmbedder
        return (lambda: HashEmbedder(dim=spec.dim, model_id=f"fake:{spec.model_id}"),
                f"fake:{cache_key(spec)}")

    def load():
        from src.embedder import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(spec, device=device)
    return load, cache_key(spec)


def embed(texts, store: Store, load_embedder, key: str) -> np.ndarray:
    hashes = [text_hash(t) for t in texts]
    cached = store.get_embeddings(hashes, key)
    todo = {}
    for h, t in zip(hashes, texts):
        if h not in cached and h not in todo:
            todo[h] = t
    if todo:
        vectors = load_embedder().encode(list(todo.values()))
        fresh = dict(zip(todo.keys(), vectors))
        store.put_embeddings(key, vectors.shape[1], fresh)
        cached.update(fresh)
    print(f"embedded {len(todo)} new unit(s), {len(set(hashes)) - len(todo)} from cache")
    return np.stack([cached[h] for h in hashes]) if hashes else np.zeros((0, 2))


def _pin_status(rows, coords, offset: float = 0.04) -> dict[int, int]:
    """Place each status line just off the next prose unit of its conversation
    (the previous one when it comes last), so it sits on the path where it
    happened. Offsets fan out so stacked status lines stay individually visible.
    Returns {status row: the prose row it is pinned to}."""
    def prose_in(conversation, indices):
        return next((j for j in indices if rows[j]["conversation"] == conversation
                     and rows[j]["kind"] != "status"), None)

    anchors = {}
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
        anchors[i] = anchor
    return anchors


def _clusters(rows, prose, X, coords, anchors, k: int, seed: int, prev=None, unchanged=False):
    """-> (cluster id per row, cluster summaries). Status lines take the cluster
    of the sentence they are pinned to. With `prev` (the last build's clusters),
    clusters continue from their old centres and keep their ids and colours."""
    centres_from, fixed = None, {}
    if prev and len(prev["colors"]) == k:
        was = [prev["cluster"].get(f"{rows[i]['conversation']}#{rows[i]['seq']}",
                                 prev["cluster"].get(text_hash(rows[i]["text"]))) for i in prose]
        groups = [[j for j, c0 in enumerate(was) if c0 == c] for c in range(k)]
        if all(groups):
            # Means of the unit vectors, which is what cluster() fits on, so an
            # unchanged corpus starts at the converged centres and stays put.
            Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
            centres_from = np.array([Xn[g].mean(axis=0) for g in groups])
            fixed = dict(enumerate(prev["colors"]))
    if unchanged and centres_from is not None and None not in was:
        # Nothing new: keep membership exactly. A refit from the same centres can
        # still flip a sentence that sits on a boundary by rounding alone.
        labels = np.array(was)
    else:
        labels = cluster(X, k, seed=seed, centres=centres_from)
    per_row = np.full(len(rows), -1)
    per_row[prose] = labels
    for i, anchor in anchors.items():
        per_row[i] = per_row[anchor]
    words = keywords([rows[i]["text"] for i in prose], labels)
    # The label sits on the member nearest the cluster's centre on the map, so it
    # lands on the cluster's own points rather than in a gap between them.
    centres, summaries = [], []
    for c in range(int(labels.max()) + 1):
        members = np.array(prose)[labels == c]
        if not len(members):
            continue
        centre = coords[members].mean(axis=0)
        at = members[np.argmin(np.linalg.norm(coords[members] - centre, axis=1))]
        centres.append(coords[at])
        summaries.append({"id": c, "keywords": words[c],
                          "label": " · ".join(words[c]) or f"군집 {c + 1}",
                          "size": int(len(members)),
                          "x": round(float(coords[at][0]), 5), "y": round(float(coords[at][1]), 5)})
    ids = [s["id"] for s in summaries]
    colours = assign_colours(np.array(centres),
                             fixed={n: fixed[c] for n, c in enumerate(ids) if c in fixed})
    for summary, colour in zip(summaries, colours, strict=True):
        summary["color"] = colour
    return per_row, summaries


def _previous(path: Path, key: str):
    """The last build's positions and clusters, or None when there is no usable
    previous build (missing, unreadable, other model). Positions are keyed both by
    point id ("conversation#seq") and by sentence text: the id keeps a sentence
    said twice in two places, the text finds a sentence whose seq has shifted."""
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if old.get("model") != key:
        return None
    xy, xyz, members = {}, {}, {}
    for p in old.get("points", []):
        if p.get("kind") == "status":
            continue
        h = text_hash(p["text"])
        xy[p["id"]] = (p["x"], p["y"])
        xy.setdefault(h, (p["x"], p["y"]))
        if "z3" in p:
            xyz[p["id"]] = (p["x3"], p["y3"], p["z3"])
            xyz.setdefault(h, (p["x3"], p["y3"], p["z3"]))
        if "cluster" in p:
            members[p["id"]] = p["cluster"]
            members.setdefault(h, p["cluster"])
    colors = [c["color"] for c in sorted(old.get("clusters", []), key=lambda c: c["id"])]
    ids = {p["id"] for p in old.get("points", []) if p.get("kind") != "status"}
    return ({"xy": xy, "xyz": xyz, "ids": ids, "cluster": members, "colors": colors}
            if xy else None)


def _unchanged(keys, prev, space: str = "xy") -> bool:
    """True when every sentence is already on the previous map, in its place."""
    return bool(prev) and all(k in prev[space] for k in keys) \
        and prev["ids"] == {k for k in keys if "#" in k}


def _project3(X, n_neighbors: int, seed: int, init=None) -> np.ndarray:
    """The 3D cloud: the same UMAP recipe as the 2D map, one more dimension."""
    n = len(X)
    if n < 4:
        return np.pad(project(X, n_neighbors=n_neighbors, seed=seed), ((0, 0), (0, 1)))
    centred = X - X.mean(axis=0)
    pca = centred @ np.linalg.svd(centred, full_matrices=False)[2][:3].T
    if n < 80:
        return pca                     # same cut-over as src/projection.py
    try:
        import warnings

        import umap

        from src.ea_projection import knn_graph
        k = max(2, min(n_neighbors, n - 1))
        knn = knn_graph(X, n_neighbors=k, metric="cosine", random_state=seed)
        if knn is not None:
            k = int(np.asarray(knn[0]).shape[1])
        reducer = umap.UMAP(n_components=3, n_neighbors=k, min_dist=0.1, metric="cosine",
                            random_state=seed,
                            init="spectral" if init is None else np.asarray(init, dtype=np.float32),
                            **({"precomputed_knn": knn} if knn is not None else {}))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*n_jobs value .* overridden.*")
            return np.asarray(reducer.fit_transform(X), dtype=float)
    except Exception:                  # noqa: BLE001 - a worse layout beats none
        return pca


def _layout(X, keys, prev, n_neighbors: int, seed: int, dims: int = 2) -> np.ndarray:
    """Project, starting from the previous build when there is one, and rotate
    the result onto its frame so a rebuild does not spin or rescale the map."""
    space = "xy" if dims == 2 else "xyz"

    def fit(init=None):
        if dims == 2:
            return project(X, n_neighbors=n_neighbors, seed=seed, init=init)
        return _project3(X, n_neighbors, seed, init)

    if not prev or not prev[space]:
        return normalize_scale(fit())
    if _unchanged(keys, prev, space):
        # Nothing new: a refit would only jiggle the map, so nothing moves.
        return np.array([prev[space][k] for k in keys], dtype=float)
    known = [i for i, k in enumerate(keys) if k in prev[space]]
    init = None
    if known:
        start = np.zeros((len(X), dims))
        start[known] = [prev[space][keys[i]] for i in known]
        new = [i for i in range(len(X)) if i not in set(known)]
        if new:
            # A new sentence starts at the mean of its five closest known ones.
            Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
            sim = Xn[new] @ Xn[known].T
            nearest = np.argsort(-sim, axis=1)[:, :5]
            start[new] = start[np.array(known)[nearest]].mean(axis=1)
        init = start
    return stabilize(fit(init), keys, prev[space])[0]


def _same_conversation(units) -> tuple | None:
    """Two saves of one conversation share their first message and its time."""
    first = next((u for u in units if u["speaker"] == "student" and u["kind"] != "status"), None)
    return (first["timestamp"], first["text"]) if first and first["timestamp"] else None


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
    ap.add_argument("--clusters", type=int, default=8,
                    help="topic clusters to colour the map by; 0 turns them off")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore the previous build: new layout, clusters and colours")
    ap.add_argument("--fake-embedder", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    out = args.out or args.chats / "atlas.json"
    cache = args.cache or args.chats / "embeddings.db"

    roster = Roster.from_path(args.roster)
    pages = sorted(args.chats.glob("*.html"))
    if not pages:
        raise SystemExit(f"no saved .html conversations in {args.chats}")

    parsed, refused = [], 0
    for page in pages:
        try:
            units, _ = convert(page, roster, min_chars=args.min_chars,
                               max_chars=args.max_chars)
        except (OSError, ValueError) as exc:
            print(f"  refused {exc}", file=sys.stderr)
            refused += 1
            continue
        parsed.append((page, units))
    # A conversation saved twice -- often once early and again after it went on --
    # would be drawn twice. Keep the longer save.
    keep = {}
    for page, units in parsed:
        same = _same_conversation(units) or page.stem
        if same not in keep or len(units) > len(keep[same][1]):
            keep[same] = (page, units)
    kept = {page for page, _ in keep.values()}
    for page, units in parsed:
        if page not in kept:
            twin = next(p for p, u in keep.values() if (_same_conversation(u) or p.stem)
                        == (_same_conversation(units) or page.stem))
            print(f"  skipped {page.name}: the same conversation as {twin.name}, which is "
                  "the longer save", file=sys.stderr)

    rows, conversations = [], []
    for page, units in parsed:
        if page not in kept:
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
    load_embedder, key = _embedder(args.model, args.device, args.fake_embedder)
    store = Store(str(cache))
    store.migrate()
    try:
        X = embed([rows[i]["text"] for i in prose], store, load_embedder, key)
    finally:
        store.close()

    prev = None if args.fresh else _previous(out, key)
    hashes = [text_hash(rows[i]["text"]) for i in prose]
    ids = [f"{rows[i]['conversation']}#{rows[i]['seq']}" for i in prose]
    # Match a sentence to its previous self by id first (same place in the same
    # conversation), else by its text.
    keys = [i if prev and i in prev["xy"] else h for i, h in zip(ids, hashes, strict=True)]
    coords = np.zeros((len(rows), 2))
    coords[prose] = _layout(X, keys, prev, args.n_neighbors, args.seed)
    if prev:
        shared = [j for j, k in enumerate(keys) if k in prev["xy"]]
        if shared:
            moved = np.linalg.norm(coords[np.array(prose)[shared]]
                                   - np.array([prev["xy"][keys[j]] for j in shared]), axis=1)
            print(f"kept the previous layout: {len(shared)} sentence(s) already on the map moved "
                  f"a median {np.median(moved):.3f} (map radius ~1)")
    anchors = _pin_status(rows, coords)
    coords3 = np.zeros((len(rows), 3))
    coords3[prose] = _layout(X, keys, prev, args.n_neighbors, args.seed, dims=3)
    for i, anchor in anchors.items():
        a, b = 2.4 * rows[i]["seq"], 1.3 * rows[i]["seq"]
        coords3[i] = coords3[anchor] + 0.04 * np.array([np.cos(a) * np.sin(b), np.sin(a) * np.sin(b), np.cos(b)])
    per_row, clusters = (_clusters(rows, prose, X, coords, anchors, args.clusters, args.seed, prev,
                                   _unchanged(keys, prev))
                         if args.clusters > 0 else (np.full(len(rows), -1), []))
    for c in clusters:
        # The 3D label sits on the member nearest the cluster's 3D centre.
        members = [i for i in prose if per_row[i] == c["id"]]
        centre = coords3[members].mean(axis=0)
        at = coords3[members[int(np.argmin(np.linalg.norm(coords3[members] - centre, axis=1)))]]
        c.update(x3=round(float(at[0]), 5), y3=round(float(at[1]), 5), z3=round(float(at[2]), 5))
    points = [{**r, "id": f"{r['conversation']}#{r['seq']}",
               "x": round(float(x), 5), "y": round(float(y), 5),
               "x3": round(float(a), 5), "y3": round(float(b), 5), "z3": round(float(z), 5),
               **({"cluster": int(c)} if c >= 0 else {})}
              for r, (x, y), (a, b, z), c in zip(rows, coords, coords3, per_row, strict=True)]
    for c in clusters:
        print(f"  cluster {c['id'] + 1}: {c['size']:4d}  {c['color']}  {c['label']}")
    names = {t["id"]: t["display_name"] for t in roster.targets()}
    payload = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": key,
        "targets": [{"id": c, "display_name": names.get(c, c)}
                    for c in sorted({c["target_id"] for c in conversations})],
        "conversations": conversations,
        "clusters": clusters,
        "points": points,
    }
    _write_private(out, json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {out}: {len(points)} points from {len(conversations)} conversation(s)")
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
