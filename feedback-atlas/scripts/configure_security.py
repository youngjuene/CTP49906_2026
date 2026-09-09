"""Rotate the previous HTTP-exposed admin code and configure private credentials."""

import os
import re
import secrets
import tempfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    secret_path = root / ".run/admin-code.txt"
    if secret_path.exists():
        raise SystemExit("Admin credential already prepared; refusing to rotate it again.")
    text = env_path.read_text()
    code = secrets.token_urlsafe(32)
    values = {
        "ATLAS_ADMIN_CODE": code,
        "ATLAS_ACCESS_CODES": ".run/access-codes.json",
        "ATLAS_ALLOW_INSECURE_HTTP": "0",
    }
    for name, value in values.items():
        pattern = rf"(?m)^(?:export\s+)?{name}=.*$"
        if re.search(pattern, text):
            text = re.sub(pattern, f"{name}={value}", text)
        else:
            text = text.rstrip() + f"\n{name}={value}\n"
    secret_path.parent.mkdir(mode=0o700, exist_ok=True)
    secret_path.parent.chmod(0o700)
    fd = os.open(secret_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as out:
        out.write(code + "\n")
    fd, temporary = tempfile.mkstemp(prefix=".env-security-", dir=root)
    try:
        with os.fdopen(fd, "w") as out:
            out.write(text)
        os.replace(temporary, env_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    for name in (".env", "roster.csv", "atlas.db", "atlas.db-wal", "atlas.db-shm"):
        path = root / name
        if path.exists():
            path.chmod(0o600)
    print("Security configured. Private admin code: .run/admin-code.txt")


if __name__ == "__main__":
    main()
