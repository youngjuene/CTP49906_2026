"""Topic clusters for the chat map: membership, keyword labels, and colours.

Clusters are found in the embedding space, not on the 2D map. The map is a
projection; blobs on it are partly artefacts of the fit, and density clustering
there either merges everything or leaves a quarter of the sentences unassigned.
Grouping by meaning and then drawing the groups on the map shows how well the
layout kept them together, which is the honest direction.

Colours come from matplotlib's default cycle (tab10), assigned so that clusters
sitting next to each other on the map get colours that are far apart to the eye
-- the cycle's blue and cyan, or orange and brown, never end up as neighbours.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

# matplotlib's default property cycle ("tab10"), in its own order.
TAB10 = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
         "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")
# The map already uses grey for other conversations and status lines, so grey is
# only handed out once every other colour is taken.
_GREY = "#7f7f7f"

# --- membership ------------------------------------------------------------------


def cluster(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """KMeans on unit-normalised vectors (cosine geometry). Deterministic."""
    from sklearn.cluster import KMeans

    k = max(1, min(k, len(X)))
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    labels = KMeans(k, n_init=10, random_state=seed).fit_predict(Xn)
    # Number clusters by first appearance, so cluster 0 is where the first
    # conversation starts and ids do not shuffle between identical rebuilds.
    order = {old: new for new, old in enumerate(dict.fromkeys(labels))}
    return np.array([order[c] for c in labels])


# --- keyword labels --------------------------------------------------------------

# No Korean morphological analyser is installed, so words are cut at whitespace
# and the commonest particles are stripped. Predicates are dropped by their
# endings rather than lemmatised; what is left is mostly nouns, which is what a
# topic label wants.
_JOSA = sorted((
    "은 는 이 가 을 를 의 에 에서 에게 한테 께 으로 로 와 과 랑 이랑 도 만 까지 부터 처럼 "
    "보다 이나 나 이라 라 이라는 라는 이란 란 으로서 로서 으로써 로써 에서는 에는 으로는 로는 "
    "과는 와는 이라고 라고 이에요 예요 이다 입니다 이죠 죠 들 들이 들을 들은 들의 들도").split(),
    key=len, reverse=True)
_ENDINGS = ("요", "다", "죠", "까", "네", "고", "서", "면", "지만", "는데", "니까", "어서",
            "아서", "해서", "하게", "하고", "하는", "하며", "해", "게", "던", "든", "며", "자",
            "지", "할", "한", "된", "될", "되는", "있는", "없는", "같은", "인", "을까")
_STOP = set((
    "이 그 저 것 수 등 거 더 좀 잘 또 왜 어떤 이런 그런 저런 정말 그리고 하지만 그래서 그러면 "
    "그런데 같이 때 지금 여기 거기 무엇 뭐 어떻게 이것 그것 저것 우리 제가 저는 너무 아주 매우 "
    "조금 많이 다른 모든 각 한 두 세 하나 둘 때문 경우 정도 부분 자체 사실 생각 느낌 질문 말씀 "
    "방법 이야기 얘기 가지 쪽 중 안 밖 위 아래 앞 뒤 속 거죠 건 걸 겁니다 거예요 거야 다만 가장 "
    "바로 먼저 다시 이미 아직 함께 오히려 결국 특히 보통 실제로 직접 서로 만약 혹시 역시 계속 "
    "그냥 단지 이제 이때 처음 마지막 the and for with that this from are was were not but you "
    "your can will end into its it's").split())


def tokens(text: str) -> list[str]:
    out = []
    for word in re.findall(r"[가-힣]+|[A-Za-z][A-Za-z0-9]+", text):
        if word[0].isascii():
            if len(word) >= 3 and word.lower() not in _STOP:
                out.append(word.lower())
            continue
        for josa in _JOSA:
            if word.endswith(josa) and len(word) - len(josa) >= 2:
                word = word[: -len(josa)]
                break
        if len(word) >= 2 and word not in _STOP and not word.endswith(_ENDINGS):
            out.append(word)
    return out


def keywords(texts, labels, n: int = 3, min_count: int = 3) -> list[list[str]]:
    """Class-based TF-IDF (as in BERTopic), then each word goes to the one cluster
    where it is most distinctive, so no two labels share a word."""
    k = int(max(labels)) + 1 if len(labels) else 0
    tf = [Counter() for _ in range(k)]
    for text, c in zip(texts, labels, strict=True):
        tf[c].update(tokens(text))
    total = Counter()
    for t in tf:
        total.update(t)
    avg = sum(total.values()) / max(k, 1)
    scores = []
    for t in tf:
        size = sum(t.values()) or 1
        scores.append({w: (f / size) * math.log(1 + avg / total[w])
                       for w, f in t.items() if f >= min_count})
    owner = {}
    for c, sc in enumerate(scores):
        for w, v in sc.items():
            if w not in owner or v > scores[owner[w]][w]:
                owner[w] = c
    return [sorted((w for w in sc if owner[w] == c), key=lambda w: (-sc[w], w))[:n]
            for c, sc in enumerate(scores)]


# --- colours ---------------------------------------------------------------------


def _lab(hex_colour: str) -> np.ndarray:
    """sRGB hex -> CIELAB (D65), enough to measure how different two colours look."""
    rgb = np.array([int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    rgb = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    xyz = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722],
                    [0.0193, 0.1192, 0.9505]]) @ rgb / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def assign_colours(centres: np.ndarray, palette=TAB10) -> list[str]:
    """One palette colour per cluster, with neighbours on the map kept far apart.

    Maximises sum(w_ij * dE(colour_i, colour_j)) where w_ij falls off with the
    distance between cluster centres, so the nearest pairs dominate. A greedy
    start, then swaps (including with unused colours) until nothing improves --
    deterministic, and at ten colours it settles in a few passes.
    """
    k = len(centres)
    pool = [c for c in palette if c != _GREY] if k < len(palette) else list(palette)
    if k > len(pool):
        pool = [pool[i % len(pool)] for i in range(k)]
    lab = np.array([_lab(c) for c in pool])
    dE = np.linalg.norm(lab[:, None] - lab[None], axis=-1)
    d = np.linalg.norm(centres[:, None] - centres[None], axis=-1)
    w = 1.0 / np.maximum(d, 1e-6) ** 2
    np.fill_diagonal(w, 0.0)

    def score(order):
        return float((w * dE[np.ix_(order, order)]).sum())

    # Greedy start: the most crowded cluster first, each taking the free colour
    # furthest from its already-coloured neighbours.
    chosen: dict[int, int] = {}
    for i in np.argsort(-w.sum(axis=1), kind="stable"):
        free = [c for c in range(len(pool)) if c not in chosen.values()] or list(range(len(pool)))
        chosen[int(i)] = max(free, key=lambda c: sum(w[i, j] * dE[c, cj] for j, cj in chosen.items()))
    order = [chosen[i] for i in range(k)]

    # Then swaps, including with unused colours, until nothing improves.
    improved = True
    while improved:
        improved = False
        base = score(order)
        for a in range(k):
            for c in range(len(pool)):
                trial = list(order)
                if c in trial:
                    b = trial.index(c)
                    trial[a], trial[b] = trial[b], trial[a]
                else:
                    trial[a] = c
                s = score(trial)
                if s > base + 1e-9:
                    order, base, improved = trial, s, True
    return [pool[c] for c in order]
