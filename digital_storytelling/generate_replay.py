"""Generate the deterministic comparison classroom replay artifact."""

from __future__ import annotations

import json
from pathlib import Path

from digital_storytelling.workflow import replay_payload


OUTPUT = Path(__file__).resolve().parent / "replay" / "storytelling_replay.json"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(replay_payload(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
