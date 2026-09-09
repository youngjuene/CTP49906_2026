"""Access-code authentication and signed session tokens.

The stored credential file deliberately contains only SHA-256 digests of
per-person random access codes. The cleartext codes are generated once by
scripts/create_access_codes.py and should be distributed privately.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType

from src.textnorm import match_key


_SHA256_HEX_LEN = 64
_SESSION_TTL_SECONDS = 8 * 60 * 60
_MAX_TOKEN_LEN = 2048


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64decode(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode((text + padding).encode("ascii"))


def _sha256_hex(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _validate_hashes(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise ValueError("access-code file must be a JSON object mapping id to SHA-256 hash")

    hashes: dict[str, str] = {}
    folded: dict[str, str] = {}
    for identity, digest in raw.items():
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("access-code ids must be non-empty strings")
        canonical = identity.strip()
        key = match_key(canonical)
        if key in folded:
            raise ValueError(
                f"access-code file has duplicate ids after normalization: "
                f"{folded[key]!r} and {canonical!r}"
            )
        if not isinstance(digest, str):
            raise ValueError(f"access-code hash for {canonical!r} must be a string")
        digest = digest.lower()
        if len(digest) != _SHA256_HEX_LEN or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"access-code hash for {canonical!r} must be SHA-256 hex")
        if digest in hashes.values():
            raise ValueError("Each participant must have a unique private access code")
        folded[key] = canonical
        hashes[canonical] = digest

    if not hashes:
        raise ValueError("access-code file has no entries")
    return hashes


class AccessControl:
    """Authenticate random access codes and issue revocable session tokens."""

    def __init__(
        self,
        hashes: Mapping[str, str],
        *,
        clock: Callable[[], float] | None = None,
        signing_key: bytes | None = None,
    ):
        self._clock = clock or time.time
        self._signing_key = signing_key or secrets.token_bytes(32)
        self._set_hashes(hashes)

    @classmethod
    def from_path(
        cls,
        path,
        *,
        clock: Callable[[], float] | None = None,
    ) -> "AccessControl":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(_validate_hashes(raw), clock=clock)

    @property
    def hashes(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self._hashes))

    def authenticate(self, identity: str, code: str) -> str | None:
        canonical = self._canonical(identity)
        owner = self.authenticate_code(code)
        return owner if owner is not None and owner == canonical else None

    def authenticate_code(self, code: str) -> str | None:
        """Resolve a participant from their fixed private code, without an ID field."""
        if not isinstance(code, str) or len(code) > 256:
            return None
        try:
            provided = _sha256_hex(code)
        except UnicodeEncodeError:
            return None
        owner = None
        # Compare every digest so the match's position does not affect lookup time.
        for identity, expected in self._hashes.items():
            if hmac.compare_digest(provided, expected):
                owner = identity
        return owner

    def issue(self, identity: str) -> str:
        canonical = self._canonical(identity)
        if canonical is None:
            raise ValueError("cannot issue a session for an unknown identity")
        now = int(self._clock())
        payload = {
            "i": canonical,
            "e": now + _SESSION_TTL_SECONDS,
            "v": self._credential_version(canonical),
            "n": secrets.token_urlsafe(16),
        }
        body = _b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        signature = self._sign(body)
        return f"{body}.{signature}"

    def submission_owner(self, identity: str, *, scope: str) -> str:
        """Opaque receipt owner, stable while the person's fixed code is unchanged.

        Session signing keys rotate on restart. Receipt ownership must survive
        that rotation without sending an identity or stored credential digest.
        The domain and environment keep this value separate from session tokens
        and from a demo using the same identity/code.
        """
        canonical = self._canonical(identity)
        if canonical is None:
            raise ValueError("unknown submission owner")
        material = f"feedback-atlas:submission-owner:v1\0{scope}\0{canonical}".encode("utf-8")
        key = bytes.fromhex(self._hashes[canonical])
        return _b64encode(hmac.new(key, material, hashlib.sha256).digest())

    def verify(self, token: str) -> str | None:
        if not isinstance(token, str) or len(token) > _MAX_TOKEN_LEN:
            return None
        try:
            body, signature = token.split(".", 1)
            if not hmac.compare_digest(self._sign(body), signature):
                return None
            payload = json.loads(_b64decode(body))
        except Exception:  # noqa: BLE001 - malformed tokens always reject.
            return None

        if not isinstance(payload, dict):
            return None
        identity = payload.get("i")
        expires = payload.get("e")
        version = payload.get("v")
        if not isinstance(identity, str) or not isinstance(expires, int):
            return None
        canonical = self._canonical(identity)
        if canonical is None or expires <= int(self._clock()):
            return None
        if not hmac.compare_digest(str(version), self._credential_version(canonical)):
            return None
        return canonical

    def reload(self, path) -> None:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self._set_hashes(_validate_hashes(raw))

    def _canonical(self, identity: str) -> str | None:
        if not isinstance(identity, str):
            return None
        return self._by_key.get(match_key(identity))

    def _set_hashes(self, hashes: Mapping[str, str]) -> None:
        self._hashes = _validate_hashes(dict(hashes))
        self._by_key = {match_key(identity): identity for identity in self._hashes}

    def _sign(self, body: str) -> str:
        return _b64encode(
            hmac.new(self._signing_key, body.encode("ascii"), hashlib.sha256).digest()
        )

    def _credential_version(self, identity: str) -> str:
        material = f"{identity}\0{self._hashes[identity]}".encode("utf-8")
        return _b64encode(hmac.new(self._signing_key, material, hashlib.sha256).digest())
