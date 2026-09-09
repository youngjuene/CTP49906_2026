"""Embedding Atlas's neighbour graph, and the two ways this app fills it.

**What this module does and does not do.** It does not call
`embedding_atlas.projection`. That module's entry points build a dataframe, wrap
the work in a file cache keyed on the inputs, and -- the part that decides it --
throw away the fitted UMAP reducer, returning only coordinates and the graph. A
reducer that cannot be kept means every later opinion forces a full refit, which
is the one thing a map being read during a presentation cannot afford.

So the *recipe* is reproduced rather than the call: build the approximate k-NN
graph with `umap.umap_.nearest_neighbors`, then fit UMAP on it as
`precomputed_knn`. That is exactly the sequence, and exactly the parameters,
upstream uses. src/projection.py performs the second half so that it can keep the
reducer; this module performs the first half and translates the result.

Reproducing rather than importing is worth stating plainly, because the two are
easy to conflate: the algorithm and its defaults are theirs, the call is ours, and
nothing here would break if their private helpers were renamed tomorrow.

**Two sources, one metric.** A point that was placed by `transform()` since the
last fit never joined the graph, and below the floor there is no graph at all.
Those rows get an exact cosine search instead. Both report 1 - cosine, so a panel
can list neighbours from either without the numbers meaning different things.

The graph comes back with the point itself first, at distance zero, which is what
`nearest_neighbors` returns and what upstream stores verbatim. It is dropped here:
our id column is the opinion id, so a point listing itself as its own nearest
opinion is a tautology occupying the first of eight slots.
"""

from collections.abc import Sequence

import numpy as np

# The floor below which no graph is built. `nearest_neighbors` needs
# n_neighbors < n, and a UMAP fitted on a handful of points is dominated by its
# initialisation rather than by the data. The PCA cold start in src/projection.py
# owns that regime; below this the exact search answers every neighbour query.
MIN_ROWS = 15


def knn_graph(
    X: np.ndarray,
    *,
    n_neighbors: int = 15,
    metric: str = "cosine",
    random_state: int | None = 0,
):
    """The approximate k-NN graph UMAP is fitted on. Upstream's first step.

    Returned in umap's own `precomputed_knn` shape -- (indices, distances,
    rp_forest) -- so it can be handed straight to `umap.UMAP(precomputed_knn=...)`,
    which is what src/projection.py does with it.

    `random_state` is passed on. Upstream leaves it unset, which is right for a
    one-shot notebook run and wrong here: an unseeded graph lays the room's map out
    differently on every recompute, and the stabiliser downstream would then have a
    genuine reorganisation to absorb rather than a rotation.

    Returns None rather than raising when the corpus is too small or umap is
    absent. A caller that gets None fits without a precomputed graph, which is what
    umap would have built for itself anyway, and only the neighbour panel goes
    without.
    """
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    if n < MIN_ROWS:
        return None
    try:
        from umap.umap_ import nearest_neighbors
    except Exception:       # noqa: BLE001 -- absent, or a numba/llvmlite mismatch
        return None
    try:
        return nearest_neighbors(
            X,
            # Must stay below the sample count or umap truncates it and warns,
            # on every recompute of the first few minutes.
            n_neighbors=max(2, min(n_neighbors, n - 1)),
            metric=metric,
            metric_kwds=None,
            angular=False,
            random_state=random_state,
        )
    except Exception:       # noqa: BLE001 -- a degenerate graph must not stop the fit
        return None


def neighbors_from_graph(
    knn_indices, knn_distances, ids: Sequence[str], *, k: int = 8,
    drop_self: bool = True,
) -> dict[str, dict]:
    """-> {opinion_id: {"ids": [...], "distances": [...]}}, closest first.

    This is embedding-atlas's `neighbors` column contract, which its viewer reads
    directly: a dict per row whose "ids" are row ids *as given by the id column*.
    Ours is the opinion id, so the row positions the graph returns are translated
    here and nowhere else.
    """
    idx = np.asarray(knn_indices)
    dist = np.asarray(knn_distances, dtype=float)
    if len(ids) != idx.shape[0]:
        raise ValueError(
            f"neighbors_from_graph got {len(ids)} ids for {idx.shape[0]} graph "
            "rows; they are positionally paired and a mismatch would attribute "
            "one opinion's neighbours to another")

    out: dict[str, dict] = {}
    for row, oid in enumerate(ids):
        picked_ids: list[str] = []
        picked_d: list[float] = []
        for col in range(idx.shape[1]):
            j = int(idx[row, col])
            if drop_self and j == row:
                continue
            if j < 0 or j >= len(ids):
                # umap pads with -1 when the graph is short of neighbours.
                continue
            picked_ids.append(ids[j])
            d = float(dist[row, col])
            picked_d.append(round(d if np.isfinite(d) else 0.0, 6))
            if len(picked_ids) >= k:
                break
        out[oid] = {"ids": picked_ids, "distances": picked_d}
    return out


def exact_neighbors(
    vectors, ids: Sequence[str], rows: Sequence[int], *, k: int = 8
) -> dict[str, dict]:
    """Cosine neighbours for specific rows, by matmul rather than by graph.

    Exact rather than approximate, and that is an upgrade rather than a
    compromise: every embedder here returns L2-normalised vectors, so the dot
    product *is* the cosine similarity and the whole corpus is one matmul against
    an (N, dim) matrix. At a thousand rows that is microseconds. `nearest_neighbors`
    is approximate because upstream is built for millions of rows; a class is not.
    """
    V = np.asarray(vectors)
    out: dict[str, dict] = {}
    if V.ndim != 2 or V.shape[0] != len(ids):
        return out
    take = min(k + 1, V.shape[0])
    for row in rows:
        row = int(row)
        if row < 0 or row >= V.shape[0]:
            continue
        sims = V @ V[row]
        # argpartition rather than a full sort: the top k, not an order over the
        # whole corpus.
        top = np.argpartition(-sims, take - 1)[:take]
        top = top[np.argsort(-sims[top])]
        picked_ids: list[str] = []
        picked_d: list[float] = []
        for j in top:
            j = int(j)
            if j == row:
                continue
            picked_ids.append(ids[j])
            d = float(1.0 - sims[j])
            picked_d.append(round(d if np.isfinite(d) else 0.0, 6))
            if len(picked_ids) >= k:
                break
        out[ids[row]] = {"ids": picked_ids, "distances": picked_d}
    return out
