"""Fan-out, and the two sets that cannot reach each other. CPU only, no server.

The defect: an `if admin:` inside the broadcaster sent admin payloads to
everyone during a reconnect storm. The fix was structural rather than careful --
two disjoint sets and no method that writes to both -- so the tests here check the
shape of the interface as much as its behaviour.

The rest of the file is about the classroom rather than the design. Thirty
sockets over a tunnel always include one that has gone away without saying so and
one that has stopped reading, and neither may cost the other twenty-nine their
update.

Run:  python -m pytest feedback-atlas/tests/test_hub_channels.py
  or:  python feedback-atlas/tests/test_hub_channels.py
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> feedback-atlas/

from src.hub import Channel, Hub  # noqa: E402


class FakeSocket:
    def __init__(self, *, fail: bool = False, delay: float = 0.0):
        self.sent: list[dict] = []
        self.fail = fail
        self.delay = delay
        self.closed = False

    async def send_json(self, message):
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("socket is gone")
        self.sent.append(message)

    async def close(self, code=1000):
        self.closed = True


def test_a_participant_broadcast_never_reaches_an_admin_socket():
    async def main():
        hub = Hub()
        p, a = FakeSocket(), FakeSocket()
        await hub.join(p, Channel.PARTICIPANT, identity="u1")
        await hub.join(a, Channel.ADMIN)

        await hub.broadcast(Channel.PARTICIPANT, {"t": "delta", "who": "participant"})
        assert a.sent == [], "an admin socket received a participant frame"

        await hub.broadcast(Channel.ADMIN, {"t": "delta", "reviewer_id": "kang.minsu"})
        assert len(p.sent) == 1 and "reviewer_id" not in p.sent[0]
    asyncio.run(main())


def test_the_hub_exposes_no_method_that_writes_to_both_channels():
    """The convenience method that would reintroduce the defect.

    broadcast_all(message) reads perfectly and is wrong: one caller passing the
    admin payload puts authorship on every screen in the room.
    """
    for name in ("broadcast_all", "send_all", "broadcast_everyone", "publish"):
        assert not hasattr(Hub, name), f"Hub.{name} defeats the channel split"
    assert "channel" in Hub.broadcast.__code__.co_varnames


def test_a_dead_socket_is_dropped_without_costing_the_others_their_update():
    async def main():
        hub = Hub()
        good = [FakeSocket() for _ in range(3)]
        for s in good:
            await hub.join(s, Channel.PARTICIPANT, identity="u")
        await hub.join(FakeSocket(fail=True), Channel.PARTICIPANT, identity="dead")

        delivered = await hub.broadcast(Channel.PARTICIPANT, {"t": "delta"})
        assert delivered == 3
        assert all(len(s.sent) == 1 for s in good)
        assert hub.counts()["participant"] == 3, "the dead socket was not dropped"
    asyncio.run(main())


def test_a_socket_that_stopped_reading_does_not_stall_the_broadcast():
    """A phone that walked out of range holds the connection open and never
    acknowledges. Awaiting sends in a loop would make everyone wait for its
    timeout; gathering with a per-send deadline does not.
    """
    async def main():
        hub = Hub(send_timeout=0.15)
        quick = FakeSocket()
        await hub.join(quick, Channel.PARTICIPANT, identity="u1")
        await hub.join(FakeSocket(delay=5.0), Channel.PARTICIPANT, identity="u2")

        started = time.monotonic()
        delivered = await hub.broadcast(Channel.PARTICIPANT, {"t": "delta"})
        elapsed = time.monotonic() - started

        assert delivered == 1 and len(quick.sent) == 1
        assert elapsed < 1.0, f"broadcast took {elapsed:.2f}s waiting on one socket"
    asyncio.run(main())


def test_leaving_twice_is_a_no_op():
    """A flaky tunnel produces both a disconnect event and a failed send for the
    same socket, and the second one must not raise inside a finally block."""
    async def main():
        hub = Hub()
        conn_id = await hub.join(FakeSocket(), Channel.PARTICIPANT, identity="u")
        await hub.leave(conn_id)
        await hub.leave(conn_id)
        await hub.leave("never-existed")
        assert hub.counts() == {"participant": 0, "admin": 0}
    asyncio.run(main())


def test_broadcasting_to_an_empty_channel_is_harmless():
    """The state of every channel before anybody arrives, and after they leave."""
    async def main():
        hub = Hub()
        assert await hub.broadcast(Channel.ADMIN, {"t": "delta"}) == 0
    asyncio.run(main())


def test_a_targeted_send_reaches_one_socket_and_reports_a_dead_one():
    """Used for ack and hello_ok, which belong to the submitter alone."""
    async def main():
        hub = Hub()
        s = FakeSocket()
        cid = await hub.join(s, Channel.PARTICIPANT, identity="u")
        assert await hub.send(cid, {"t": "ack"}) is True
        assert s.sent == [{"t": "ack"}]
        assert await hub.send("no-such-connection", {"t": "ack"}) is False
    asyncio.run(main())


def test_a_connection_remembers_its_identity_and_channel():
    """Authorship is read from here, never from the submitted frame."""
    async def main():
        hub = Hub()
        cid = await hub.join(FakeSocket(), Channel.PARTICIPANT, identity="kang.minsu")
        conn = hub.get(cid)
        assert conn.identity == "kang.minsu" and conn.channel is Channel.PARTICIPANT
        admin = hub.get(await hub.join(FakeSocket(), Channel.ADMIN))
        assert admin.identity is None, "the admin channel has no roster identity"
    asyncio.run(main())


def test_close_all_shuts_every_socket_and_empties_both_channels():
    async def main():
        hub = Hub()
        sockets = [FakeSocket() for _ in range(4)]
        for i, s in enumerate(sockets):
            await hub.join(s, Channel.ADMIN if i % 2 else Channel.PARTICIPANT,
                           identity=None if i % 2 else "u")
        await hub.close_all()
        assert all(s.closed for s in sockets)
        assert hub.counts() == {"participant": 0, "admin": 0}
    asyncio.run(main())


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
