"""Incremental updates, and the epsilon that keeps them incremental. CPU only.

The defect: a full snapshot went out on every submit, the classroom wifi buckled
at twenty clients, and the cause was not the snapshot code but the *comparison* --
Procrustes and UMAP are floating point, so every point differed from its previous
position by ~1e-16 and every point was therefore "moved".

The second thing this file pins is narrower and more important: a move entry
carries geometry and nothing else. That is what lets one move list serve both the
participant and the admin channel without a second serialiser, and it is only
safe as long as nothing content-shaped can get into it.

Run:  python -m pytest feedback-atlas/tests/test_delta_math.py
  or:  python feedback-atlas/tests/test_delta_math.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> feedback-atlas/

from src.deltas import MOVE_EPSILON, diff_layout, should_send_snapshot  # noqa: E402
from src.models import Opinion  # noqa: E402
from src.payloads import participant_points  # noqa: E402
from src.protocol import delta, snapshot  # noqa: E402


def test_floating_point_jitter_is_not_a_move():
    """The defect this file is named for."""
    prev = {f"o{i}": (float(i), float(i)) for i in range(300)}
    jittered = {k: (x + 1e-16, y - 3e-17) for k, (x, y) in prev.items()}
    added, moved = diff_layout(prev, jittered)
    assert added == [] and moved == []


def test_a_real_move_is_reported():
    prev = {"a": (0.0, 0.0)}
    added, moved = diff_layout(prev, {"a": (0.0, 0.5)})
    assert added == [] and moved == [{"id": "a", "x": 0.0, "y": 0.5}]


def test_the_epsilon_boundary_is_where_it_says_it_is():
    prev = {"a": (0.0, 0.0)}
    _, under = diff_layout(prev, {"a": (0.9e-4, 0.0)}, move_epsilon=1e-4)
    _, over = diff_layout(prev, {"a": (1.1e-4, 0.0)}, move_epsilon=1e-4)
    assert under == [] and len(over) == 1


def test_a_move_entry_carries_exactly_id_x_and_y():
    """A move can never carry text or authorship, by shape.

    This is the property that makes one move list safe to broadcast to both
    channels. If a move could carry content, the admin and participant deltas
    would need separate move lists and the next person to touch this code would
    have to remember that.
    """
    prev = {"a": (0.0, 0.0), "b": (1.0, 1.0)}
    _, moved = diff_layout(prev, {"a": (5.0, 5.0), "b": (9.0, 9.0)})
    assert moved and all(set(m) == {"id", "x", "y"} for m in moved)
    assert all(isinstance(m["x"], float) and isinstance(m["y"], float) for m in moved)


def test_new_points_are_added_once_and_never_also_reported_as_moved():
    prev = {"a": (0.0, 0.0)}
    added, moved = diff_layout(prev, {"a": (0.0, 0.0), "b": (1.0, 1.0)})
    assert added == ["b"]
    assert [m["id"] for m in moved] == []


def test_a_disappeared_point_is_ignored_rather_than_crashing():
    """Opinions are never deleted today, but a layout computed from a stale
    snapshot can legitimately lack an id the previous one had."""
    added, moved = diff_layout({"a": (0.0, 0.0), "gone": (1.0, 1.0)}, {"a": (0.0, 0.0)})
    assert added == [] and moved == []


def test_revs_let_a_client_notice_it_missed_a_delta():
    """Gap detection is stateless on the server: the client compares from_rev to
    what it holds and asks for a resync. Nothing has to be remembered per socket."""
    d = delta(from_rev=41, rev=43, layout_rev=10, added=[], moved=[])
    assert d["from_rev"] == 41 and d["rev"] == 43
    assert d["from_rev"] != d["rev"]


def _drift_run(baseline_is_broadcast: bool, passes: int = 400):
    """Simulate `passes` recomputes where every point creeps by a hair each time.

    Returns the worst divergence between what the server holds and what a client
    reconstructed from the deltas it was sent.
    """
    step = MOVE_EPSILON * 0.4          # always under the reporting threshold
    truth = {f"o{i}": (0.0, 0.0) for i in range(20)}
    client = dict(truth)
    baseline = dict(truth)
    for _ in range(passes):
        truth = {k: (x + step, y) for k, (x, y) in truth.items()}
        _, moved = diff_layout(baseline, truth)
        for m in moved:
            client[m["id"]] = (m["x"], m["y"])
        # The contract: advance the baseline only for what was actually sent.
        if baseline_is_broadcast:
            for m in moved:
                baseline[m["id"]] = (m["x"], m["y"])
        else:
            baseline = dict(truth)
    return max(abs(truth[k][0] - client[k][0]) for k in truth)


def test_sub_threshold_drift_stays_bounded_when_diffed_against_what_was_sent():
    """Found by QA. The bug this guards is silent and cumulative.

    A point creeping by less than move_epsilon every recompute is never reported.
    If the comparison baseline advances to the newly computed layout anyway, the
    client falls a little further behind on every pass, forever, and nothing ever
    corrects it -- the map slowly stops matching the data with no error anywhere.
    Diffing against the last *broadcast* layout bounds the error at one epsilon.
    """
    bounded = _drift_run(baseline_is_broadcast=True)
    assert bounded <= MOVE_EPSILON, f"divergence {bounded} exceeded one epsilon"


def test_and_diverges_without_bound_when_diffed_against_the_latest_layout():
    """The other half: proves the contract is load-bearing rather than stylistic.

    If this ever stops failing to stay bounded, the epsilon has become a no-op and
    the test above is passing for the wrong reason.
    """
    unbounded = _drift_run(baseline_is_broadcast=False)
    assert unbounded > MOVE_EPSILON * 50, (
        f"divergence only reached {unbounded}; the drift scenario is no longer "
        "exercising what it was written for")


def test_the_epsilon_is_below_one_screen_pixel():
    """It has to be invisible, or the map visibly lags the data.

    Coordinates carry RMS radius 1; a map ~800px wide showing about +/-3 RMS is
    ~0.0075 data-units per pixel.
    """
    units_per_pixel = 6.0 / 800.0
    assert MOVE_EPSILON < units_per_pixel, "move_epsilon is larger than a pixel"


def test_a_mostly_reorganised_layout_is_sent_as_a_snapshot_instead():
    """Crossing the PCA to UMAP threshold, or swapping the model, replaces the
    layout wholesale -- at which point a delta is the larger message."""
    assert should_send_snapshot(70, 100) is True
    assert should_send_snapshot(20, 100) is False
    assert should_send_snapshot(0, 0) is False


def test_one_submit_costs_far_less_on_the_wire_than_a_snapshot():
    """The arithmetic that justifies the whole delta protocol, checked rather
    than asserted in a comment: ~250 KB per snapshot times 30 clients times one
    submit each is the number that does not fit down classroom wifi."""
    ops = [
        Opinion(id=f"o{i}", reviewer_id="kang.minsu", target_id="kim.seoyeon",
                text="소리가 몸 안쪽에서 나는 것처럼 들렸어요. " * 3,
                source="human", week=2, timestamp="2026-09-07T04:11:22Z")
        for i in range(400)
    ]
    coords = {op.id: (0.1, 0.2) for op in ops}
    full = snapshot(rev=400, layout_rev=9,
                    points=participant_points(ops, coords))
    incr = delta(from_rev=400, rev=401, layout_rev=10,
                 added=participant_points(ops[:1], coords), moved=[])

    def size(message):
        return len(json.dumps(message, ensure_ascii=False).encode("utf-8"))

    assert size(incr) * 50 < size(full), (
        f"delta {size(incr)}B vs snapshot {size(full)}B -- deltas are not "
        "buying what the protocol assumes they buy")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    fails = 0
    for fn in fns:
        try:
            fn()
            print("PASS", fn.__name__)
        except Exception as e:  # noqa: BLE001
            fails += 1
            print("FAIL", fn.__name__, "->", type(e).__name__, e)
    print(f"\n{len(fns) - fails} passed, {fails} failed")
    sys.exit(1 if fails else 0)
