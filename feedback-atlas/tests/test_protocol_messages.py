"""Wire frames: shape, Korean text, and the parameter that is not there. CPU only.

The defect: a protocol change shipped, a student's phone had last week's page
cached, and the old client mis-parsed the new frames silently -- points appeared
in the wrong places rather than an error appearing anywhere. Hence a version in
the handshake and a fatal mismatch.

The most important test here asserts an absence. The protocol carries no week
filter in either direction, because a server that projected only the checked
weeks would refit the layout every time somebody unchecked a box, moving every
remaining point -- exactly what PRD 5.5 forbids. Filtering is the client's job and
there is deliberately no parameter through which that can be got wrong.

Run:  python -m pytest feedback-atlas/tests/test_protocol_messages.py
  or:  python feedback-atlas/tests/test_protocol_messages.py
"""
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> feedback-atlas/

from src import protocol  # noqa: E402
from src.protocol import (  # noqa: E402
    ERROR_CODES, MESSAGES, PROTOCOL_VERSION, ack, delta, error, hello_ok,
    parse_client_frame, pong, snapshot,
)

POINT = {"id": "o1", "target_id": "kim.seoyeon", "text": "좋았어요.",
         "source": "human", "week": 2, "timestamp": "2026-09-07T04:11:22.000Z",
         "x": 0.1, "y": 0.2}


def _all_server_frames():
    return [
        hello_ok(channel="participant", rev=1, layout_rev=1, targets=[], role="student"),
        snapshot(rev=1, layout_rev=1, points=[POINT]),
        delta(from_rev=0, rev=1, layout_rev=1, added=[POINT], moved=[]),
        ack(nonce="c-1", id="o1", rev=1),
        error("EMPTY_TEXT"),
        pong(),
    ]


def test_every_server_frame_is_tagged_and_json_round_trips():
    for frame in _all_server_frames():
        assert isinstance(frame.get("t"), str) and frame["t"]
        assert json.loads(json.dumps(frame, ensure_ascii=False)) == frame


def test_korean_survives_both_json_escaping_settings():
    """ensure_ascii is a library default that differs between callers. Korean
    must round-trip identically either way, or error messages arrive as \\uXXXX
    soup on somebody's phone."""
    frame = error("UNKNOWN_ID")
    for ensure_ascii in (True, False):
        assert json.loads(json.dumps(frame, ensure_ascii=ensure_ascii)) == frame
    assert "명단" in frame["message"]


def test_the_protocol_carries_no_week_filter_in_either_direction():
    """The structural guarantee behind PRD 5.5.

    Checked as an absence in the builders' signatures and in their output, so
    adding a week parameter to the wire has to break this test first.
    """
    for fn in (snapshot, delta, hello_ok):
        params = set(inspect.signature(fn).parameters)
        assert not (params & {"week", "weeks_filter", "week_filter", "only_weeks"}), \
            f"{fn.__name__} grew a week filter parameter"

    snap = snapshot(rev=1, layout_rev=1, points=[POINT])
    assert set(snap) == {"t", "rev", "layout_rev", "points"}
    d = delta(from_rev=0, rev=1, layout_rev=1, added=[], moved=[])
    assert set(d) == {"t", "from_rev", "rev", "layout_rev", "added", "moved", "drift"}

    # hello_ok's `weeks` is the compose form's vocabulary, not a filter: it is the
    # full set, always.
    assert hello_ok(channel="participant", rev=0, layout_rev=0,
                    targets=[], role="student")["weeks"] == [1, 2, 3, 4]


def test_hello_ok_tells_the_form_everything_it_needs_and_no_names():
    frame = hello_ok(channel="participant", rev=0, layout_rev=0,
                     targets=[{"id": "kim.seoyeon", "display_name": "김서연"}],
                     role="observer")
    assert frame["default_source"] == "ai"        # PRD 5.3
    assert frame["sources"] == ["ai", "human"]
    assert frame["max_text_chars"] > 0
    assert "roster" not in frame                  # admin channel only


def test_the_admin_handshake_is_the_only_one_carrying_a_roster():
    frame = hello_ok(channel="admin", rev=0, layout_rev=0, targets=[],
                     roster=[{"id": "kang.minsu", "display_name": "강민수",
                              "role": "student"}])
    assert frame["channel"] == "admin"
    assert frame["roster"][0]["display_name"] == "강민수"


def test_every_error_code_has_korean_text_and_the_set_is_closed():
    """A code with no message would reach a student as a blank red box."""
    assert set(MESSAGES) == set(ERROR_CODES)
    for code in ERROR_CODES:
        message, hint = MESSAGES[code]
        assert message and any("가" <= ch <= "힣" for ch in message), code
    try:
        error("MADE_UP_CODE")
    except KeyError:
        pass
    else:
        raise AssertionError("an unknown error code must not produce a frame")


def test_the_unknown_id_error_suspects_a_typo_before_it_blames_the_roster():
    """PRD 5.1 asks for that order specifically. Most failures really are typos,
    and sending someone to find the instructor first wastes the presentation."""
    frame = error("UNKNOWN_ID", fatal=True)
    assert "오타" in frame["hint"]
    assert frame["hint"].index("오타") < frame["hint"].index("강사")
    assert frame["fatal"] is True


def test_an_oversized_frame_is_refused_before_it_is_parsed():
    """A 10 MB frame should cost a length check, not a JSON parse."""
    assert parse_client_frame(b"[" + b"1," * 100000 + b"1]") == "TOO_LARGE"
    assert parse_client_frame('{"t":"ping"}', max_bytes=5) == "TOO_LARGE"


def test_malformed_input_yields_an_error_rather_than_an_exception():
    """One student's broken client must not be able to take the room down."""
    for bad in ("", "not json", "[]", '"a string"', '{"no_type":1}',
                '{"t":123}', '{"t":""}', b"\xff\xfe"):
        assert parse_client_frame(bad) == "MALFORMED", repr(bad)


def test_a_well_formed_frame_parses_to_its_type_and_body():
    parsed = parse_client_frame('{"t":"submit","text":"좋아요","week":2}')
    assert parsed[0] == "submit"
    assert parsed[1]["text"] == "좋아요"


def test_a_version_mismatch_is_fatal_rather_than_tolerated():
    """The cached-client defect. Failing loudly is the whole point."""
    frame = error("PROTOCOL_MISMATCH", fatal=True)
    assert frame["fatal"] is True
    assert PROTOCOL_VERSION == hello_ok(channel="participant", rev=0, layout_rev=0,
                                        targets=[])["protocol"]


def test_drift_is_rounded_so_float_noise_does_not_ride_in_every_delta():
    d = delta(from_rev=0, rev=1, layout_rev=1, added=[], moved=[],
              drift=0.00412345678901234)
    assert d["drift"] == 0.004123
    assert len(json.dumps(d["drift"])) < 12


def test_no_client_facing_builder_accepts_a_reviewer_id():
    """Authorship never travels on a client->server frame, so no builder here
    should have a place to put one."""
    for name, fn in vars(protocol).items():
        if not callable(fn) or name.startswith("_") or not hasattr(fn, "__module__"):
            continue
        if fn.__module__ != protocol.__name__:
            continue
        params = set(inspect.signature(fn).parameters)
        assert "reviewer_id" not in params, f"protocol.{name} takes a reviewer_id"


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
