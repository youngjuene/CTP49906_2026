"""The participant channel must not be able to carry reviewer_id. CPU only.

This file stands for the defect the whole two-serialiser design exists to make
impossible: a field added to the Opinion record reaching the participant map,
because the serialiser was written as "everything except reviewer_id" and nobody
revisited it when the record grew.

So these tests do not only check that today's output lacks the field. They check
the *shape of the mechanism*: an allowlist rather than a denylist, no audience
flag on the function, and no `if admin`-style branch anywhere in the module. A
test that only inspected today's keys would keep passing right up until the day
somebody adds `reviewer_email`.

Run:  python -m pytest feedback-atlas/tests/test_payload_privacy.py
  or:  python feedback-atlas/tests/test_payload_privacy.py
"""
import ast
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> feedback-atlas/

from src import payloads, protocol  # noqa: E402
from src.models import Opinion  # noqa: E402
from src.payloads import (  # noqa: E402
    ADMIN_KEYS, PARTICIPANT_KEYS, admin_point, admin_points, participant_point,
    participant_points,
)
from src.roster import Roster  # noqa: E402

# A string that appears nowhere else, so finding it anywhere in a participant
# payload is unambiguous evidence rather than a coincidence.
SENTINEL = "REVIEWER-CANARY-8f3a1c"

# The canary is an auditor, not a student: auditors write feedback but are not
# presenting, so the id must never appear on the participant channel for any
# reason. A student id legitimately does appear -- the compose form has to list
# the projects -- which would mask the leak this file is looking for.
ROSTER = Roster.from_csv_text(
    "id,display_name,role\n"
    f"{SENTINEL},카나리아,auditor\n"
    "kim.seoyeon,김서연,student\n"
)


def _op(**kw):
    base = dict(
        id="o_abc123", reviewer_id=SENTINEL, target_id="kim.seoyeon",
        text="소리가 몸 안쪽에서 나는 것처럼 들렸어요.", source="human",
        week=2, timestamp="2026-09-07T04:11:22Z",
    )
    base.update(kw)
    return Opinion(**base)


def test_participant_point_emits_exactly_the_eight_public_keys():
    """The key set is spelled out here, not imported from PARTICIPANT_KEYS.

    Importing the constant would let a single edit widen the contract and the
    test agree with it in the same commit. Writing the eight keys literally means
    adding a ninth has to be argued for twice, in two files.
    """
    point = participant_point(_op(), (0.5, -0.25))
    assert set(point) == {
        "id", "target_id", "text", "source", "week", "timestamp", "x", "y",
    }
    assert PARTICIPANT_KEYS == set(point)  # and the constant still describes reality


def test_reviewer_id_appears_nowhere_in_any_participant_frame():
    """Serialise every message type and search the JSON text, not just the keys.

    A nested leak -- a reviewer id inside a list, or inside a value rather than
    at a key -- would pass a top-level key check. This catches it because it looks
    at the bytes that actually go on the wire.
    """
    op = _op()
    coords = {op.id: (0.1, 0.2)}
    points = participant_points([op], coords)

    frames = [
        protocol.snapshot(rev=1, layout_rev=1, points=points),
        protocol.delta(from_rev=0, rev=1, layout_rev=1, added=points, moved=[
            {"id": op.id, "x": 0.1, "y": 0.2}]),
        protocol.ack(nonce="c-1", id=op.id, rev=1),
        protocol.error("UNKNOWN_ID"),
        protocol.hello_ok(channel="participant", rev=1, layout_rev=1,
                          targets=ROSTER.targets(), role="student"),
    ]
    for frame in frames:
        blob = json.dumps(frame, ensure_ascii=False)
        assert SENTINEL not in blob, f"{frame['t']} frame leaked the reviewer id"


def test_participant_hello_ok_carries_no_display_name_at_all():
    """Not even the viewer's own real name crosses the participant channel.

    hello_ok returns `role`, never a name. With no names on the channel there is
    nothing for a curious participant to correlate points against -- which is a
    stronger property than merely omitting the author of each point.
    """
    frame = protocol.hello_ok(channel="participant", rev=0, layout_rev=0,
                              targets=[{"id": "kim.seoyeon", "display_name": "김서연"}],
                              role="student")
    assert "roster" not in frame
    assert frame["role"] == "student"
    # Target display names are present on purpose: the compose form has to name
    # the projects. Those are the presenters, not the authors of feedback.
    assert frame["targets"][0]["display_name"] == "김서연"


def test_admin_point_is_the_participant_point_plus_exactly_two_keys():
    """Same map, same coordinates -- PRD 5.6 forbids a separate dataset."""
    op = _op()
    pub = participant_point(op, (0.5, -0.25))
    adm = admin_point(op, (0.5, -0.25), ROSTER)
    assert set(adm) - set(pub) == {"reviewer_id", "reviewer_name"}
    assert set(adm) == ADMIN_KEYS
    assert (adm["x"], adm["y"]) == (pub["x"], pub["y"])
    assert all(adm[k] == pub[k] for k in pub)
    assert adm["reviewer_id"] == SENTINEL
    assert adm["reviewer_name"] == "카나리아"


def test_admin_point_falls_back_to_the_id_when_the_roster_lost_the_writer():
    """A reviewer removed from the roster mid-semester must not blank the panel.

    Returning "" here would silently un-attribute their earlier feedback, which
    is the one thing admin mode exists to do.
    """
    adm = admin_point(_op(reviewer_id="gone.missing"), (0.0, 0.0), ROSTER)
    assert adm["reviewer_name"] == "gone.missing"


def test_participant_serialiser_takes_no_audience_flag():
    """No include_reviewer=, no admin=, no mode=.

    A flag means one wrong call site leaks everything, and the wrong call site is
    usually the reconnect path that nobody exercises by hand.
    """
    for fn in (participant_point, participant_points):
        names = set(inspect.signature(fn).parameters)
        assert not (names & {"admin", "include_reviewer", "mode", "channel", "roster"}), \
            f"{fn.__name__} grew an audience parameter"
    # The admin serialiser needs the roster to resolve a display name; that is a
    # data source, not an audience switch, and it lives on the other function.
    assert "roster" in inspect.signature(admin_point).parameters


def test_payloads_module_contains_no_audience_conditional():
    """Parsed, not grepped -- the module docstring discusses `if admin:` on purpose.

    A string search over the source would trip on the prose that explains why the
    branch is absent. Walking the AST looks only at code.
    """
    tree = ast.parse(Path(payloads.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.If, ast.IfExp)):
            continue
        names = {n.id.lower() for n in ast.walk(node.test) if isinstance(n, ast.Name)}
        names |= {n.attr.lower() for n in ast.walk(node.test) if isinstance(n, ast.Attribute)}
        assert not any("admin" in n or "reviewer" in n for n in names), \
            f"payloads.py branches on audience at line {node.lineno}"


def test_participant_point_is_built_by_naming_keys_not_by_subtracting_them():
    """The allowlist has to be structural, or it is just a denylist in disguise.

    Guards against a rewrite to `{k: v for k, v in asdict(op).items() if k != ...}`,
    which passes every other test in this file today and leaks the next field
    somebody adds to Opinion.
    """
    source = inspect.getsource(participant_point)
    for banned in ("asdict", "__dict__", "vars(", ".pop(", "del "):
        assert banned not in source, \
            f"participant_point uses {banned!r}: build the dict by naming its keys"
    # Every public key appears as a literal in the function body.
    tree = ast.parse(source.lstrip())
    literals = {n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert PARTICIPANT_KEYS <= literals


def test_a_new_opinion_field_does_not_reach_participants_by_itself():
    """The regression this file is named for, simulated end to end.

    A record carrying an extra attribute is projected; the extra must not appear.
    """
    class OpinionPlus(Opinion):
        pass

    op = _op()
    grown = OpinionPlus(**{f.name: getattr(op, f.name)
                           for f in op.__dataclass_fields__.values()})
    object.__setattr__(grown, "reviewer_email", "someone@example.ac.kr")

    point = participant_point(grown, (0.0, 0.0))
    assert "reviewer_email" not in point
    assert "someone@example.ac.kr" not in json.dumps(point, ensure_ascii=False)


def test_bulk_serialisers_place_missing_coordinates_at_the_origin():
    """An opinion submitted mid-recompute has no coordinates yet.

    It must serialise rather than raise -- the alternative is one unplaced point
    taking down the whole broadcast.
    """
    op = _op()
    pub = participant_points([op], {})
    adm = admin_points([op], {}, ROSTER)
    assert (pub[0]["x"], pub[0]["y"]) == (0.0, 0.0)
    assert set(pub[0]) == PARTICIPANT_KEYS and set(adm[0]) == ADMIN_KEYS


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
