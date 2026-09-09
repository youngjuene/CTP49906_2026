"""Public practice accounts with a private, independent application state."""

import hashlib
from dataclasses import replace
from pathlib import Path

from src.access import AccessControl

DEMO_ROSTER = Path(__file__).resolve().parents[1] / "demo/roster.csv"
DEMO_ADMIN_CODE = "ctp49906"
MAX_DEMO_OPINIONS = 500


def demo_accounts():
    return [{"id": "demo", "display_name": "demo", "role": "observer", "code": "demo"}]


def create_demo_atlas(main):
    from src.server import Atlas

    if main.cfg.admin_code == DEMO_ADMIN_CODE:
        raise ValueError("Classroom admin code must differ from the public demo code")
    if (main.cfg.db_path == main.cfg.demo_db_path
            or Path(main.cfg.db_path).resolve() == Path(main.cfg.demo_db_path).resolve()):
        raise ValueError("Demo and classroom must use different databases")
    cfg = replace(main.cfg, roster_path=str(DEMO_ROSTER),
                  db_path=main.cfg.demo_db_path, enable_demo=False, is_demo=True,
                  admin_code=DEMO_ADMIN_CODE, warm_umap=False,
                  debounce_s=1.0, max_debounce_s=2.0)
    demo = Atlas(cfg)
    demo.access = AccessControl({account["id"]:
        hashlib.sha256(account["code"].encode()).hexdigest()
        for account in demo_accounts()})
    return demo
