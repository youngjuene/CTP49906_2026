"""Test-only opinion seeding, the way scripts/seed_ai_opinions.py does it.

Participants cannot write, so a test that needs opinions puts them in the store
and asks the running server to reload its corpus (what SIGUSR1 does in
production). The reload runs on the server's own event loop through the
TestClient portal, because the recompute loop's wake-up event is not
thread-safe.
"""

from __future__ import annotations

import time

from src.models import new_opinion
from src.textnorm import text_hash


def seed(client, texts, *, reviewer="writer1", target="target1", week=1,
         sources=None, wait_for_layout=False) -> list[str]:
    atlas = client.app.state.atlas
    ids = []
    for i, text in enumerate(texts):
        source = sources[i % len(sources)] if sources else "ai"
        op = new_opinion(reviewer_id=reviewer, target_id=target, text=text,
                         source=source, week=week)
        atlas.store.insert_opinion(op, text_hash(op.text))
        ids.append(op.id)
    client.portal.call(atlas.reload_corpus)
    if wait_for_layout:
        # Until every new point has coordinates, so a neighbour query or an
        # export is answered from a projected corpus.
        for _ in range(100):
            if all(i in atlas.state.server_coords for i in ids):
                break
            time.sleep(0.05)
    return ids
