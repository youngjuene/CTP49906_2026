"""What changed between two layouts, so a submit costs a few hundred bytes.

The arithmetic that forces this: at 1000 points carrying ~180 bytes of Korean
text each, a full snapshot is ~250 KB. Broadcast to 30 clients that is 7.5 MB,
and a class where everyone submits once is ~225 MB -- a quarter of ngrok's free
monthly allowance in one session, over classroom wifi. So a snapshot goes out on
connect and on explicit resync, and everything after that is a delta.

`moved` carries geometry and nothing else -- {"id", "x", "y"}. That is why one
move list can serve both the participant and the admin channel: there is no
audience-dependent content in it, so it cannot be the thing that leaks. Only
`added` is built twice, once per serialiser.

**`prev_coords` must be the last coordinates actually broadcast, not the last
ones computed.** Those are different, and the difference is the whole reason
MOVE_EPSILON is safe. A point that drifts a little on every recompute, always by
less than the threshold, is never reported -- so if the baseline advances anyway,
the client's copy falls further behind on every pass and nothing ever corrects
it. Diffing against what the client was last told bounds that error at one
epsilon no matter how many recomputes go by, because the sub-threshold drift
keeps accumulating against a fixed baseline until it crosses.
"""

from collections.abc import Mapping, Sequence

# Chosen against measurement, not intuition. A full recompute re-runs PCA over the
# whole corpus, so the components themselves shift and every point moves a little;
# the movement falls off roughly as 1/N. Measured on a 3-cluster corpus, adding one
# opinion moves the existing points by a median of:
#
#       N=40   5.4e-3      N=300   7.4e-4
#       N=80   2.4e-3      N=600   4.1e-4
#      N=150   1.7e-3     N=1000   2.1e-4
#
# The coordinate space has RMS radius 1, and a map ~800px wide showing about +/-3
# RMS is ~0.0075 data-units per pixel -- so 2e-3 is roughly a quarter pixel, below
# what anyone can see. At the old 1e-4, 747 of 1000 points exceeded the threshold
# on every single recompute, which made every delta a snapshot wearing a delta's
# name and quietly cost the protocol its entire reason for existing.
#
# Below N~150 most points do exceed this, but that movement is real and visible,
# and a snapshot at that size is a few kilobytes: should_send_snapshot() takes over.
MOVE_EPSILON = 2e-3


def diff_layout(
    prev_coords: Mapping[str, Sequence[float]],
    next_coords: Mapping[str, Sequence[float]],
    *,
    move_epsilon: float = MOVE_EPSILON,
) -> tuple[list[str], list[dict]]:
    """-> (added_ids, moved). Ids in prev but not next are dropped, not reported.

    `prev_coords` is the last *broadcast* layout. See the module docstring: passing
    the last computed layout instead reintroduces unbounded client divergence.
    """
    added: list[str] = []
    moved: list[dict] = []
    for oid, xy in next_coords.items():
        before = prev_coords.get(oid)
        if before is None:
            added.append(oid)
            continue
        if abs(float(xy[0]) - float(before[0])) > move_epsilon or \
           abs(float(xy[1]) - float(before[1])) > move_epsilon:
            moved.append({"id": oid, "x": float(xy[0]), "y": float(xy[1])})
    return added, moved


def should_send_snapshot(n_moved: int, n_total: int, ratio: float = 0.6) -> bool:
    """True when the layout reorganised enough that a delta is the bigger message.

    This fires for real, not just in theory: crossing the PCA->UMAP threshold
    replaces the layout wholesale, and so does a model swap.
    """
    if n_total <= 0:
        return False
    return n_moved > ratio * n_total
