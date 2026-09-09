"""Connection registry and fan-out. The second structural privacy boundary.

Two disjoint sets, one per channel, and deliberately **no** method that sends to
both. The caller builds two messages from two serialisers (src/payloads.py) and
calls broadcast twice. A single `broadcast_all(message)` would be more convenient
and would be the exact shape of the bug this design exists to prevent: one
caller passing the admin payload, and every participant screen showing who wrote
what.

Sends are gathered rather than awaited in turn. Thirty sockets over a classroom
tunnel include at least one that has gone away without saying so, and awaiting it
in a loop makes everybody else wait for its timeout.
"""

import asyncio
import contextlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Channel(str, Enum):
    PARTICIPANT = "participant"
    ADMIN = "admin"


@dataclass
class Connection:
    id: str
    channel: Channel
    socket: Any
    identity: str | None = None      # canonical roster id; None on the admin channel
    rev: int = 0                     # what this client last acknowledged holding
    strikes: int = 0                 # consecutive failed sends
    meta: dict = field(default_factory=dict)


class Hub:
    def __init__(self, send_timeout: float = 2.0):
        self._by_channel: dict[Channel, set[str]] = {c: set() for c in Channel}
        self._conns: dict[str, Connection] = {}
        self._seq = 0
        self._lock = asyncio.Lock()
        self.send_timeout = send_timeout

    async def join(self, socket, channel: Channel, *, identity: str | None = None) -> str:
        async with self._lock:
            self._seq += 1
            conn_id = f"c{self._seq}"
            self._conns[conn_id] = Connection(
                id=conn_id, channel=channel, socket=socket, identity=identity)
            self._by_channel[channel].add(conn_id)
        return conn_id

    async def leave(self, conn_id: str) -> None:
        """Idempotent: a flaky tunnel produces both a disconnect and a send failure
        for the same socket, and the second one must not raise."""
        async with self._lock:
            conn = self._conns.pop(conn_id, None)
            if conn is not None:
                self._by_channel[conn.channel].discard(conn_id)

    def get(self, conn_id: str) -> Connection | None:
        return self._conns.get(conn_id)

    def counts(self) -> dict[str, int]:
        return {c.value: len(ids) for c, ids in self._by_channel.items()}

    def connections(self) -> tuple[Connection, ...]:
        return tuple(self._conns.values())

    async def send(self, conn_id: str, message: dict) -> bool:
        conn = self._conns.get(conn_id)
        if conn is None:
            return False
        try:
            await asyncio.wait_for(conn.socket.send_json(message), self.send_timeout)
            conn.strikes = 0
            return True
        except Exception:       # noqa: BLE001 -- a dead socket is not exceptional here
            conn.strikes += 1
            await self.leave(conn_id)
            return False

    async def broadcast(self, channel: Channel, message: dict) -> int:
        """Send to every connection on one channel. Returns how many succeeded.

        Never raises. A closed laptop lid, a phone that walked out of range, and a
        tunnel that dropped mid-frame all look the same from here, and none of them
        is a reason for the other twenty-nine screens to miss an update.
        """
        targets = list(self._by_channel[channel])
        if not targets:
            return 0
        results = await asyncio.gather(
            *(self.send(cid, message) for cid in targets), return_exceptions=True)
        return sum(1 for r in results if r is True)

    async def close_all(self) -> None:
        for conn_id in list(self._conns):
            conn = self._conns.get(conn_id)
            if conn is not None:
                with contextlib.suppress(Exception):
                    await conn.socket.close()
            await self.leave(conn_id)
