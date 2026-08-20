"""Offline evidence review for one dLive input-fader MIDI/TCP packet.

The functions in this module only inspect caller-supplied data.  They never
open a socket, capture interface, or console connection.  An accepted report
proves that a known-good client packet observed at the selected endpoint's
ingress is byte-for-byte identical to AUTO-MIXER's planned compact NRPN
packet.  It deliberately does not prove console acceptance, physical fader
movement, or grant mutation authority.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from typing import Any, Mapping

from dlive_client import (
    FADER_PARAMETER_ID,
    build_nrpn,
    db_to_fader_value,
    validate_dlive_connection_config,
)


DLIVE_MIDI_CAPTURE_SCHEMA = "automixer.dlive_midi_capture.v1"
DLIVE_MIDI_CAPTURE_REVIEW_SCHEMA = "automixer.dlive_midi_capture_review.v1"
KNOWN_GOOD_SOURCES = frozenset(
    {
        "allen_heath_dlive_director",
        "allen_heath_dlive_surface",
        "operator_validated_midi_client",
    }
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field}_must_be_finite_number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}_must_be_finite_number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field}_must_be_finite_number")
    return parsed


def _parse_hex(value: Any) -> bytes:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("capture_bytes_hex_required")
    compact = "".join(value.split()).lower()
    if compact.startswith("0x"):
        compact = compact[2:]
    if len(compact) % 2:
        raise ValueError("capture_bytes_hex_invalid")
    try:
        decoded = bytes.fromhex(compact)
    except ValueError as exc:
        raise ValueError("capture_bytes_hex_invalid") from exc
    if not decoded:
        raise ValueError("capture_bytes_hex_required")
    return decoded


def _validate_utc_timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("capture_timestamp_utc_required")
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("capture_timestamp_utc_invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("capture_timestamp_utc_invalid")
    return value.strip()


def _contains_sensitive_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            lowered = str(key).lower()
            if any(part in lowered for part in ("password", "secret", "token")):
                return True
            if _contains_sensitive_key(nested):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_sensitive_key(item) for item in value)
    return False


def review_dlive_fader_midi_capture(
    capture: Mapping[str, Any],
    *,
    host: str,
    endpoint: str,
    port: int | None,
    tls: bool,
    midi_base_channel: int,
    user_profile: int,
    input_channel: int,
    target_db: float,
) -> dict[str, Any]:
    """Compare supplied ingress evidence with one exact planned fader packet."""

    if not isinstance(capture, Mapping):
        raise ValueError("capture_not_object")
    if _contains_sensitive_key(capture):
        raise ValueError("capture_contains_sensitive_fields")
    endpoint_config = validate_dlive_connection_config(
        host=host,
        endpoint=endpoint,
        port=port,
        tls=tls,
        midi_base_channel=midi_base_channel,
        user_profile=user_profile,
    )
    if isinstance(input_channel, bool) or not isinstance(input_channel, int):
        raise ValueError("input_channel_must_be_integer")
    if not 1 <= input_channel <= 128:
        raise ValueError("input_channel_out_of_range_1_128")
    target = _finite_number(target_db, "target_db")
    if target < -54.0 or target > 0.0:
        raise ValueError("target_db_out_of_restore_proof_range")

    if capture.get("schema") != DLIVE_MIDI_CAPTURE_SCHEMA:
        raise ValueError("capture_schema_invalid")
    capture_id = capture.get("capture_id")
    if not isinstance(capture_id, str) or not capture_id.strip():
        raise ValueError("capture_id_required")
    if capture.get("operator_owned") is not True:
        raise ValueError("capture_must_be_operator_owned")
    if capture.get("acquired_by_automixer") is not False:
        raise ValueError("capture_must_not_be_acquired_by_automixer")
    if capture.get("interface_opened_by_automixer") is not False:
        raise ValueError("capture_interface_must_not_be_opened_by_automixer")
    source_client = capture.get("source_client")
    if source_client not in KNOWN_GOOD_SOURCES:
        raise ValueError("capture_source_client_not_known_good")
    if capture.get("direction") != "client_to_selected_endpoint":
        raise ValueError("capture_direction_invalid")
    if capture.get("observation_point") != "selected_endpoint_ingress":
        raise ValueError("capture_not_observed_at_selected_endpoint_ingress")
    captured_at_utc = _validate_utc_timestamp(capture.get("captured_at_utc"))
    capture_tool = capture.get("capture_tool")
    if not isinstance(capture_tool, str) or not capture_tool.strip():
        raise ValueError("capture_tool_required")

    captured_endpoint = capture.get("endpoint")
    if not isinstance(captured_endpoint, Mapping):
        raise ValueError("capture_endpoint_missing")
    for field in (
        "host",
        "endpoint",
        "port",
        "tls",
        "midi_base_channel",
        "user_profile",
    ):
        if captured_endpoint.get(field) != endpoint_config[field]:
            raise ValueError(f"capture_endpoint_binding_mismatch:{field}")

    selector = input_channel - 1
    target_raw = db_to_fader_value(target)
    expected_packet = build_nrpn(
        midi_base_channel,
        selector,
        FADER_PARAMETER_ID,
        target_raw,
    )
    captured_packet = _parse_hex(capture.get("bytes_hex"))
    exact_wire_match = captured_packet == expected_packet
    if not exact_wire_match:
        raise ValueError("capture_packet_does_not_match_planned_compact_nrpn")

    capture_sha256 = _sha256_bytes(_canonical_json_bytes(dict(capture)))
    return {
        "schema": DLIVE_MIDI_CAPTURE_REVIEW_SCHEMA,
        "status": "accepted",
        "reason": "known_good_ingress_packet_matches_planned_compact_nrpn_exactly",
        "capture_id": capture_id.strip(),
        "capture_sha256": capture_sha256,
        "captured_at_utc": captured_at_utc,
        "capture_tool": capture_tool.strip(),
        "source_client": source_client,
        "operator_owned": True,
        "acquired_by_automixer": False,
        "interface_opened_by_automixer": False,
        "direction": "client_to_selected_endpoint",
        "observation_point": "selected_endpoint_ingress",
        "endpoint": {
            field: endpoint_config[field]
            for field in (
                "host",
                "endpoint",
                "port",
                "tls",
                "midi_base_channel",
                "user_profile",
            )
        },
        "operation": {
            "family": "input_fader",
            "input_channel": input_channel,
            "target_db": target,
            "target_fader_raw": target_raw,
            "midi_channel": midi_base_channel,
            "selector": selector,
            "parameter_id": FADER_PARAMETER_ID,
        },
        "expected_packet_hex": expected_packet.hex(),
        "captured_packet_hex": captured_packet.hex(),
        "packet_sha256": _sha256_bytes(expected_packet),
        "exact_wire_match": True,
        "wire_delivery_observed": True,
        "console_acceptance_proven": False,
        "physical_fader_movement_proven": False,
        "console_contacted_by_automixer": False,
        "midi_sent_by_automixer": False,
        "mutation_write_attempt_count": 0,
        "live_write": False,
        "execution_allowed": False,
        "live_authority_granted": False,
        "standing_live_authority": False,
    }
