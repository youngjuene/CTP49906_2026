"""GPU-free blinded audience response surface.

The production Marimo UI can call these functions directly.  The command-line
entry point also supports a deterministic packet/response rehearsal without a
model, CUDA, network access, or treatment-only imports.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
from uuid import uuid4


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from curriculum_common.audience_packets import (  # noqa: E402
    AUDIENCE_READING_SCHEMA_VERSION,
    AudiencePacket,
    AudienceReading,
    canonical_json_bytes,
)


DISPLAY_PACKET_FIELDS = frozenset(
    {
        "exchange_artifact_id",
        "presentation_asset",
        "presentation_checksum",
        "permitted_accessibility_overlays",
        "reveal_state",
        "protocol_deviation",
        "sharing_permission_scope",
    }
)


def packet_for_display(packet: AudiencePacket) -> dict[str, Any]:
    """Return only fields approved for the separate blinded surface."""

    value = packet.to_dict()
    return {field: value[field] for field in sorted(DISPLAY_PACKET_FIELDS)}


def build_reading(
    packet: AudiencePacket,
    response: Mapping[str, Any],
    *,
    audience_pseudonym: str,
    respondent_session_pseudonym: str,
    event_index: int,
    elapsed_ms: int,
    response_id: str | None = None,
    created_at_utc: str | None = None,
) -> AudienceReading:
    """Build and validate one appendable audience reading."""

    reading = {
        "schema_version": AUDIENCE_READING_SCHEMA_VERSION,
        "response_id": response_id or f"response-{uuid4()}",
        "exchange_artifact_id": packet.exchange_artifact_id,
        "audience_pseudonym": audience_pseudonym,
        "respondent_session_pseudonym": respondent_session_pseudonym,
        "local_created_at_utc": created_at_utc
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "event_index": event_index,
        "elapsed_ms": elapsed_ms,
        "blindness_attestation": response.get("blindness_attestation"),
        "permission_scope": response.get("permission_scope"),
        "open_interpretation": response.get("open_interpretation"),
        "sound_image_relation": response.get("sound_image_relation"),
        "shared_tags": response.get("shared_tags"),
        "self_described_tags": response.get("self_described_tags"),
        "accessibility_barriers": response.get("accessibility_barriers"),
        "confidence_or_ambiguity": response.get("confidence_or_ambiguity"),
        "withdrawn": response.get("withdrawn", False),
    }
    return AudienceReading.from_mapping(reading, packet=packet)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audience-pseudonym", required=True)
    parser.add_argument("--session-pseudonym", required=True)
    parser.add_argument("--event-index", type=int, default=0)
    parser.add_argument("--elapsed-ms", type=int, default=0)
    parser.add_argument("--response-id")
    parser.add_argument("--created-at-utc")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    packet = AudiencePacket.from_mapping(_load_object(args.packet))
    reading = build_reading(
        packet,
        _load_object(args.response),
        audience_pseudonym=args.audience_pseudonym,
        respondent_session_pseudonym=args.session_pseudonym,
        event_index=args.event_index,
        elapsed_ms=args.elapsed_ms,
        response_id=args.response_id,
        created_at_utc=args.created_at_utc,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(reading.to_dict()) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
