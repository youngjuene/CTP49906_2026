"""Every frame must survive a strict JSON parser. CPU only.

The defect this file stands for is a whole-room outage from one bad number.
JSON has no NaN and no Infinity. Python's json.dumps emits them anyway, as bare
`NaN` / `Infinity`, and every browser's JSON.parse throws on both. So a single
non-finite coordinate does not corrupt one point on one screen -- it makes the
entire snapshot unparseable and blanks the map for everyone at once, mid
presentation, with nothing in the server log to suggest anything went wrong.

Python's own json.loads accepts NaN by default, so a test that just round-trips
through json would pass while the browsers all fail. Every check below therefore
parses with parse_constant, which is what makes the assertion mean what it says.

Run:  python -m pytest feedback-atlas/tests/test_wire_safety.py
  or:  python feedback-atlas/tests/test_wire_safety.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> feedback-atlas/

from src.models import new_opinion  # noqa: E402
from src.payloads import admin_point, admin_points, participant_point, participant_points  # noqa: E402
from src.protocol import ack, delta, error, hello_ok, pong, snapshot  # noqa: E402
from src.roster import Roster  # noqa: E402

ROSTER = Roster.from_csv_text(
    "id,display_name,role\nkim.seoyeon,김서연,student\nkang.minsu,강민수,student\n")
OP = new_opinion(reviewer_id="kang.minsu", target_id="kim.seoyeon",
                 text="소리가 몸 안쪽에서 나는 것처럼 들렸어요.")

NON_FINITE = (float("nan"), float("inf"), float("-inf"))


def _strict(obj):
    """json.dumps then parse the way a browser does: NaN and Infinity are errors."""
    def reject(token):
        raise AssertionError(f"frame contains {token}, which JSON.parse rejects")
    return json.loads(json.dumps(obj, ensure_ascii=False), parse_constant=reject)


def test_the_strict_parser_actually_rejects_what_python_tolerates():
    """Guards the test's own premise. Without this, every assertion below could be
    passing because the helper is toothless rather than because the code is safe.
    """
    assert json.loads("NaN") != json.loads("NaN")  # python accepts it: NaN != NaN
    try:
        _strict({"x": float("nan")})
    except AssertionError:
        pass
    else:
        raise AssertionError("the strict parser is not strict")


def test_a_non_finite_coordinate_never_reaches_a_frame():
    for bad in NON_FINITE:
        for xy in ((bad, 1.0), (1.0, bad), (bad, bad)):
            pub = participant_point(OP, xy)
            adm = admin_point(OP, xy, ROSTER)
            _strict(snapshot(rev=1, layout_rev=1, points=[pub, adm]))


def test_a_non_finite_drift_never_reaches_a_delta():
    """drift comes straight out of a least-squares residual, which is exactly the
    kind of number that goes non-finite on degenerate input."""
    for bad in NON_FINITE:
        _strict(delta(from_rev=0, rev=1, layout_rev=1, added=[], moved=[], drift=bad))


def test_every_frame_type_survives_a_strict_parse_with_korean_content():
    points = participant_points([OP], {OP.id: (0.1, -0.2)})
    admin = admin_points([OP], {OP.id: (0.1, -0.2)}, ROSTER)
    frames = [
        hello_ok(channel="participant", rev=1, layout_rev=1,
                 targets=ROSTER.targets(), role="student"),
        hello_ok(channel="admin", rev=1, layout_rev=1, targets=ROSTER.targets(),
                 roster=[{"id": "kang.minsu", "display_name": "강민수", "role": "student"}]),
        snapshot(rev=1, layout_rev=1, points=points),
        snapshot(rev=1, layout_rev=1, points=admin),
        delta(from_rev=0, rev=1, layout_rev=1, added=points,
              moved=[{"id": OP.id, "x": 0.1, "y": -0.2}]),
        ack(nonce="c-1", id=OP.id, rev=1),
        error("UNKNOWN_ID", fatal=True),
        pong(),
    ]
    for frame in frames:
        assert _strict(frame) == frame, frame["t"]


def test_coordinates_stay_floats_rather_than_becoming_strings_or_none():
    """A None or a string here parses fine and then breaks arithmetic in the
    renderer instead, which is harder to trace back."""
    pub = participant_point(OP, (float("nan"), 2.5))
    assert isinstance(pub["x"], float) and isinstance(pub["y"], float)
    assert pub["x"] == 0.0 and pub["y"] == 2.5


def test_a_point_with_no_coordinates_yet_is_still_serialisable():
    """An opinion submitted mid-recompute has no layout position. It must not be
    able to take the broadcast down while it waits for one."""
    _strict(snapshot(rev=1, layout_rev=1, points=participant_points([OP], {})))


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
