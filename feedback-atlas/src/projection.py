"""Vectors -> 2D coordinates, and the stabiliser that stops the map jumping.

PRD 7 says coordinates are recomputed over the whole corpus on every submit, and
PRD 10 files "keep existing points still" as future work. Taken literally that
means the map rearranges under a reader every time somebody types -- during a
presentation, which is the only time this tool is used. The stabiliser below is
what makes the literal reading survivable without incremental UMAP.

Only numpy is required. PCA here is centre-then-SVD, which is what
sklearn.decomposition.PCA does, in four lines; the orthogonal Procrustes fit is
another three. Writing them out rather than importing scikit-learn and scipy buys
something specific: the cold-start path cannot fail because an optional
dependency did not install, and the sign convention is ours to fix rather than
sklearn's to change between versions (it did, in 1.5). umap-learn stays optional
and is imported inside project().
"""

from collections.abc import Mapping, Sequence

import numpy as np


# Set by _umap_2d so Layout can keep the reducer that produced the last fit,
# without changing project()'s signature or making it return a tuple.
_last_reducer = None


def choose_projector(n: int, threshold: int = 80) -> str:
    """"pca" below the threshold, "umap" at or above it.

    Below ~15 points UMAP's spectral initialisation fails outright and falls back
    to random init -- noise wearing a UMAP-shaped API. Between 15 and 50 the k-NN
    graph is sparse enough that the layout is dominated by initialisation. 80 is a
    deliberately conservative crossing point.
    """
    return "umap" if n >= threshold else "pca"


def _pca_2d(X: np.ndarray) -> np.ndarray:
    """Top two principal components, with a deterministic sign convention.

    The sign fix matters more here than usual. SVD leaves each component's sign
    arbitrary, and the usual convention ("make the largest-magnitude loading
    positive") is itself data-dependent -- so as the corpus grows, the component
    that *was* largest can change and the entire plot mirrors. Fixing the sign
    only makes each fit self-consistent; stabilize() is what ties it to the
    previous fit.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    # full_matrices=False: we only ever want the first two right singular vectors.
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:2]
    for i in range(comps.shape[0]):
        j = int(np.argmax(np.abs(comps[i])))
        if comps[i, j] < 0:
            comps[i] = -comps[i]
    out = Xc @ comps.T
    if out.shape[1] == 1:  # a rank-1 corpus: every text identical but one
        out = np.hstack([out, np.zeros((out.shape[0], 1))])
    return out


def _umap_2d(X: np.ndarray, n_neighbors: int, seed: int) -> np.ndarray:
    import warnings

    import umap  # optional, and slow to import: keep it out of module scope

    n = X.shape[0]
    global _last_reducer
    reducer = umap.UMAP(
        n_components=2,
        # n_neighbors must stay below the sample count or umap truncates it and
        # warns; clamp here so the warning never reaches a classroom log.
        n_neighbors=max(2, min(n_neighbors, n - 1)),
        min_dist=0.1,
        metric="cosine",
        random_state=seed,
        init="spectral",
        verbose=False,
    )
    with warnings.catch_warnings():
        # Setting random_state makes umap single-threaded and it says so, every
        # single fit. The trade is deliberate: at a thousand points the parallel
        # speedup is worth less than a layout that is the same every time it is
        # recomputed, which is less work for the stabiliser and less movement on
        # screen. Silenced narrowly so a real umap warning still gets through to
        # the classroom terminal.
        warnings.filterwarnings(
            "ignore", message=".*n_jobs value .* overridden.*", category=UserWarning)
        out = np.asarray(reducer.fit_transform(X), dtype=float)
    _last_reducer = reducer
    return out


def project(
    X: np.ndarray, *, threshold: int = 80, n_neighbors: int = 15, seed: int = 0
) -> np.ndarray:
    """(n, dim) embeddings -> (n, 2) coordinates.

    The tiny-n cases are spelled out rather than left to the linear algebra:
    PCA on one point yields a zero-variance fit and on two points a degenerate
    one, and the RMS normalisation downstream would divide by zero. These are not
    hypothetical -- they are the first two submissions of every session.
    """
    X = np.asarray(X, dtype=float)
    if X.size and not np.isfinite(X).all():
        # One bad vector must not take down the recompute for the whole room. A
        # non-finite component makes the SVD raise (LinAlgError) and UMAP produce
        # garbage, so the row is neutralised rather than allowed to propagate:
        # that opinion lands near the origin instead of everyone's map vanishing.
        # The caller logs it; the alternative is a class with no map at all.
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    n = X.shape[0]
    if n == 0:
        return np.zeros((0, 2))
    if n == 1:
        return np.zeros((1, 2))
    if n == 2:
        # Two opinions carry one bit of geometry: same, or different. Put them on
        # a horizontal axis and let the stabiliser scale it.
        return np.array([[-1.0, 0.0], [1.0, 0.0]])
    if choose_projector(n, threshold) == "umap":
        try:
            return _umap_2d(X, n_neighbors, seed)
        except Exception:
            # umap missing or failing (numba/llvmlite mismatch, a degenerate
            # neighbour graph) must not take the class down. PCA is worse, not
            # broken; the caller logs the fallback.
            return _pca_2d(X)
    return _pca_2d(X)


def rms_radius(coords: np.ndarray) -> float:
    coords = np.asarray(coords, dtype=float)
    if coords.shape[0] == 0:
        return 0.0
    centred = coords - coords.mean(axis=0, keepdims=True)
    return float(np.sqrt((centred ** 2).sum(axis=1).mean()))


def normalize_scale(coords: np.ndarray, target_rms: float = 1.0) -> np.ndarray:
    """Centre, and scale so the cloud has a fixed RMS radius.

    Without this the plot breathes: UMAP's output extent grows with n, so the
    whole map would slowly inflate over a session even with every point in the
    same relative position.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.shape[0] == 0:
        return coords
    centred = coords - coords.mean(axis=0, keepdims=True)
    r = rms_radius(centred)
    if r < 1e-12:  # every point coincident
        return centred
    return centred * (target_rms / r)


def _similarity_fit(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, float]:
    """(R, s) minimising ||s * A @ R - B||_F over orthogonal R and scalar s > 0.

    Rotation and *uniform* scale, which is a rigid motion of the plane up to
    zoom: it preserves every ratio of distances and every angle, so it cannot
    distort the structure the projector found. That is the line this fit does not
    cross -- an affine fit could also shear and stretch, and would happily deform
    real structure to flatter its own residual.

    Uniform scale has to be in the fit rather than handled by normalising the
    whole cloud separately. Normalising the cloud makes the scale depend on every
    point in it, so adding one opinion rescales all the others by a hair -- and a
    hair is above move_epsilon, so every delta would carry every point and the
    bandwidth argument for deltas would evaporate.
    """
    U, sv, Vt = np.linalg.svd(A.T @ B)
    R = U @ Vt
    denom = float((A ** 2).sum())
    s = float(sv.sum() / denom) if denom > 1e-12 else 1.0
    return R, s


def stabilize(
    coords: np.ndarray,
    ids: Sequence[str],
    prev: Mapping[str, Sequence[float]],
    *,
    min_anchors: int = 3,
    collinearity_eps: float = 1e-6,
) -> tuple[np.ndarray, float]:
    """Rotate a fresh layout onto the previous one. -> (coords, residual_rms).

    The very first layout has nothing to align to, so it establishes the frame by
    centring and fixing its RMS radius. Every layout after that inherits position,
    orientation and scale from the shared points, which is what keeps a submitted
    opinion from nudging all the others (see _similarity_fit).

    Rotation, uniform scale and reflection; no shear, no anisotropic stretch. A
    map has no chirality so reflection is free, and we are not minimising residual
    for its own sake -- we are minimising how much the reader's eye has to re-find.

    Two bail-outs to translation-only, both real:
      * fewer than `min_anchors` shared points. With two anchors the rotation is
        exactly determined and the residual is 0 by construction -- it tells you
        nothing about the other 300 points, which is worse than not rotating.
      * near-collinear anchors. Three points on a line leave rotation about that
        line arbitrary and numerically unstable, so a tiny perturbation would
        spin the map.

    The returned residual is reported to clients as `drift`, so the frontend can
    animate a genuine reorganisation instead of teleporting through it.
    """
    coords = np.asarray(coords, dtype=float)
    if len(ids) != coords.shape[0]:
        # ids and coords are positionally paired, so a length mismatch does not
        # fail -- it silently attributes every point to the wrong opinion, and the
        # map looks entirely plausible while being wrong.
        raise ValueError(
            f"stabilize got {len(ids)} ids for {coords.shape[0]} coordinate rows; "
            "they are positionally paired and must be the same length")
    if coords.shape[0] == 0:
        return coords, 0.0

    index = [i for i, oid in enumerate(ids) if oid in prev]
    if not prev or len(index) < min_anchors:
        # No frame to inherit, so establish one: centre and fix the RMS radius.
        # Every later layout takes its scale from the anchors instead.
        return normalize_scale(coords), 0.0

    cur = coords[index]
    old = np.array([prev[ids[i]] for i in index], dtype=float)
    cur_c = cur.mean(axis=0, keepdims=True)
    old_c = old.mean(axis=0, keepdims=True)
    A, B = cur - cur_c, old - old_c

    # Degeneracy is tested on the cross-covariance A.T @ B, because that is the
    # matrix whose SVD *is* R. Testing A alone would miss the mirror case: if the
    # previous anchors are collinear and the current ones are not, R is still
    # underdetermined, and the map would spin on a rounding error.
    M = A.T @ B
    s = np.linalg.svd(M, compute_uv=False)
    if s.shape[0] < 2 or s[0] < 1e-12 or s[1] < collinearity_eps * s[0]:
        return coords - cur_c + old_c, 0.0

    R, scale = _similarity_fit(A, B)
    if not np.isfinite(scale) or not (1e-6 < scale < 1e6):
        # A pathological scale means the anchor geometry collapsed between fits.
        # Keeping the previous frame beats zooming the room's screen to infinity.
        return coords - cur_c + old_c, 0.0
    out = scale * ((coords - cur_c) @ R) + old_c
    residual = float(np.sqrt(((scale * (A @ R) - B) ** 2).sum(axis=1).mean()))
    return out, residual


class Layout:
    """A projector that keeps its fitted manifold, so new points cost a transform.

    Why this exists, measured rather than assumed. A plain refit-everything
    UMAP moves **every** point on **every** recompute -- 100% at N=100, 200, 300 and
    400 alike, where PCA falls to 0% by N=400. Procrustes cannot rescue that: the
    variation is not a rigid motion of the plane, it is a different optimisation
    landing somewhere else. The consequence is not cosmetic. Every recompute
    exceeds should_send_snapshot()'s threshold, every broadcast becomes a full
    snapshot, and the delta protocol is defeated precisely above N=80, which is
    the regime it was designed for.

    Fitting once and calling transform() for later arrivals fixes both halves at
    the same time: existing points are not recomputed at all, so they cannot move,
    and the frame stays put so the stabiliser has almost nothing left to do. A
    transform costs about 2 ms once the JIT is warm, against ~6 s for a 200-point
    refit.

    The cost is that the manifold ages -- it was fitted on the corpus as it stood --
    so the fit is renewed once the corpus has grown by `refit_growth`. That refit
    does reorganise the map, which is why it happens on a schedule that gets
    rarer as the corpus grows, rather than continuously.

    Not thread-safe, and does not need to be: exactly one recompute runs at a time.
    """

    def __init__(self, *, threshold: int = 80, n_neighbors: int = 15,
                 seed: int = 0, refit_growth: float = 0.5):
        self.threshold = threshold
        self.n_neighbors = n_neighbors
        self.seed = seed
        self.refit_growth = refit_growth
        self._reducer = None
        self._ids: list[str] = []          # every id placed so far, in row order
        self._coords: np.ndarray | None = None
        # Rows covered by the last *actual* fit. Tracked separately from _ids,
        # which grows with every transform: measuring growth against _ids would
        # move the baseline forward on each arrival, so the corpus could never
        # outgrow it and the manifold would never be renewed.
        self._fit_size = 0

    def _should_refit(self, ids: Sequence[str]) -> bool:
        if self._reducer is None or self._coords is None:
            return True
        n_placed = len(self._ids)
        if len(ids) < n_placed:
            return True
        # The fit indexes rows positionally, so it is only reusable if the corpus
        # still *starts* with the rows it was fitted on. Ordering is by
        # (timestamp, id) and normally append-only, but a clock adjustment or a
        # backfilled row would insert into the middle, and reusing the fit then
        # would attribute stored coordinates to the wrong opinions.
        if list(ids[:n_placed]) != self._ids:
            return True
        return len(ids) > self._fit_size * (1.0 + self.refit_growth)

    def project(self, X: np.ndarray, ids: Sequence[str]) -> tuple[np.ndarray, bool]:
        """-> (coords, refitted). `refitted` warns the caller the frame moved."""
        X = np.asarray(X, dtype=float)
        if len(ids) != X.shape[0]:
            raise ValueError(
                f"Layout.project got {len(ids)} ids for {X.shape[0]} rows")
        n = X.shape[0]

        if choose_projector(n, self.threshold) == "pca":
            # Below the threshold the corpus is small, a full recompute is
            # instant, and a snapshot is a few kilobytes. Keep no state.
            self._reducer, self._coords, self._ids = None, None, []
            self._fit_size = 0
            return project(X, threshold=self.threshold,
                           n_neighbors=self.n_neighbors, seed=self.seed), True

        if self._should_refit(ids):
            coords = project(X, threshold=self.threshold,
                             n_neighbors=self.n_neighbors, seed=self.seed)
            self._reducer = _last_reducer
            self._ids = list(ids)
            self._coords = coords.copy()
            self._fit_size = n
            return coords, True

        n_fitted = len(self._ids)
        if n == n_fitted:
            return self._coords.copy(), False
        try:
            fresh = np.asarray(self._reducer.transform(X[n_fitted:]), dtype=float)
        except Exception:
            # A transform can fail on a degenerate neighbour graph. Falling back
            # to a refit costs a reorganisation, which beats losing the map.
            coords = project(X, threshold=self.threshold,
                             n_neighbors=self.n_neighbors, seed=self.seed)
            self._reducer, self._ids = _last_reducer, list(ids)
            self._coords, self._fit_size = coords.copy(), n
            return coords, True

        coords = np.vstack([self._coords, fresh])
        self._ids = list(ids)
        self._coords = coords.copy()
        return coords, False


def warm_jit(dim: int, rows: int = 50, seed: int = 0) -> bool:
    """Compile umap's numba kernels now, before anyone is connected.

    The first real fit_transform otherwise spends 10-30s in the JIT -- which
    would land on the first submission of the class, mid-presentation. Returns
    False when umap is absent, which is a supported state, not an error.
    """
    try:
        import umap  # noqa: F401
    except Exception:
        return False
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(rows, dim))
    try:
        _umap_2d(X, n_neighbors=15, seed=seed)
        # transform() JITs separately from fit_transform(): the first call costs
        # ~1.5s even on a warm fit. That would land on the first submission of
        # the class, which is the whole thing warm_jit exists to prevent.
        if _last_reducer is not None:
            _last_reducer.transform(X[:1])
        return True
    except Exception:
        return False
