"""Contract for dLive-state physical readback and WING parity artifacts.

This module is transport-free. It defines the JSON-like artifact shape a future
operator-run physical readback must produce before cached dLive state can be
treated as parity evidence for WING-oriented promotion decisions. The contract
does not query hardware, send MIDI/TCP messages, or mutate console state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
from typing import Any, Dict, Mapping, Tuple


DLIVE_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID = (
    "dlive_state_physical_readback_contract"
)
DLIVE_STATE_PHYSICAL_READBACK_SCHEMA_ID = "dlive_state_physical_readback.v1"
DLIVE_STATE_PHYSICAL_READBACK_ARTIFACT_TYPE = "dlive_state_physical_readback"
DLIVE_STATE_SURFACE_ID = "dlive_state"
DLIVE_STATE_REMEDIATION_ACTION_ID = "remediate_dlive_state"

DLIVE_STATE_REQUIRED_TOP_LEVEL_FIELDS = (
    "type",
    "schema_id",
    "surface_id",
    "remediation_action_id",
    "captured_at",
    "source",
    "console_model",
    "channel_count",
    "channels",
    "midi_state",
    "parity",
    "safety",
)
DLIVE_STATE_REQUIRED_CHANNEL_FIELDS = (
    "channel",
    "cached_channel_name",
    "fader_raw_14bit",
    "fader_db",
    "mute",
    "raw_note_velocity",
    "midi_base_channel",
    "readback_status",
    "readback_source",
    "captured_at",
    "coverage",
)
DLIVE_STATE_REQUIRED_MIDI_STATE_FIELDS = (
    "midi_base_channel",
    "raw_state_key_count",
    "raw_note_event_count",
    "raw_cc_event_count",
    "raw_pitchbend_event_count",
    "tcp_session_status",
    "midi_transport_status",
    "readback_status",
    "readback_source",
    "captured_at",
    "coverage",
)
DLIVE_STATE_REQUIRED_PARITY_FIELDS = (
    "wing_surface_equivalent",
    "mapping_status",
    "channel_identity_parity",
    "fader_scale_parity",
    "mute_state_parity",
    "midi_velocity_parity",
    "unsupported_field_policy",
    "coverage",
)
DLIVE_STATE_REQUIRED_COVERAGE_FLAGS = (
    "channel_identity",
    "fader_scale",
    "mute_state",
    "raw_midi_state",
    "midi_base_channel",
    "wing_parity_mapping",
    "cache_matches_physical_readback",
)
DLIVE_STATE_PHYSICAL_READBACK_SOURCES = (
    "physical_readback",
    "dlive_physical_readback",
)
DLIVE_STATE_VERIFIED_PHYSICAL_STATUSES = (
    "verified_from_physical_readback",
)
DLIVE_STATE_REQUIRED_FALSE_SAFETY_FLAGS = (
    "live_write",
    "mutates_console_state",
    "live_widening_allowed",
    "full_autonomous_ready",
    "midi_send_allowed",
    "tcp_send_allowed",
    "dlive_write_allowed",
    "dlive_apply_allowed",
)
DLIVE_STATE_FORBIDDEN_MUTATION_EVIDENCE_FIELDS = (
    "write_events",
    "manual_write_events",
    "apply_result",
    "rollback_result",
    "proposal_apply_result",
    "midi_send_events",
    "tcp_send_events",
    "dlive_write_events",
    "scene_recall_events",
    "mute_write_events",
    "fader_write_events",
)
MINIMUM_DLIVE_STATE_CHANNEL_COUNT = 1


@dataclass(frozen=True)
class DLiveStatePhysicalReadbackValidation:
    method_id: str
    schema_id: str
    surface_id: str
    remediation_action_id: str
    status: str
    accepted_as_physical_fixture_candidate: bool
    live_authority_granted: bool
    full_autonomous_ready: bool
    channel_count: int
    violations: Tuple[str, ...]
    evidence: Dict[str, Any]

    def __post_init__(self) -> None:
        if self.live_authority_granted:
            raise ValueError("dLive-state validation must not grant live authority")
        if self.full_autonomous_ready:
            raise ValueError(
                "dLive-state validation must not claim full autonomous readiness"
            )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["violations"] = list(self.violations)
        return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _bool_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _append_required_field_violations(payload: Mapping[str, Any], violations: list[str]) -> None:
    for key in DLIVE_STATE_REQUIRED_TOP_LEVEL_FIELDS:
        if key not in payload:
            violations.append(f"missing_top_level_field:{key}")


def _append_identity_violations(payload: Mapping[str, Any], violations: list[str]) -> None:
    if payload.get("type") != DLIVE_STATE_PHYSICAL_READBACK_ARTIFACT_TYPE:
        violations.append("artifact_type_mismatch")
    if payload.get("schema_id") != DLIVE_STATE_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("schema_id_mismatch")
    if payload.get("surface_id") != DLIVE_STATE_SURFACE_ID:
        violations.append("surface_id_mismatch")
    if payload.get("remediation_action_id") != DLIVE_STATE_REMEDIATION_ACTION_ID:
        violations.append("remediation_action_id_mismatch")
    if payload.get("source") not in DLIVE_STATE_PHYSICAL_READBACK_SOURCES:
        violations.append("source_not_physical_readback")
    if not _string(payload.get("captured_at")):
        violations.append("captured_at_missing")
    if not _string(payload.get("console_model")):
        violations.append("console_model_missing")


def _mutation_field_paths(value: Any, *, prefix: str = "") -> Tuple[str, ...]:
    if isinstance(value, Mapping):
        paths: list[str] = []
        for key, nested in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if key_text in DLIVE_STATE_FORBIDDEN_MUTATION_EVIDENCE_FIELDS:
                paths.append(path)
            paths.extend(_mutation_field_paths(nested, prefix=path))
        return tuple(paths)
    if isinstance(value, list):
        paths = []
        for index, nested in enumerate(value):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            paths.extend(_mutation_field_paths(nested, prefix=path))
        return tuple(paths)
    return ()


def _append_safety_violations(payload: Mapping[str, Any], violations: list[str]) -> None:
    safety = _mapping(payload.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("safety_read_only_not_true")
    for key in DLIVE_STATE_REQUIRED_FALSE_SAFETY_FLAGS:
        if safety.get(key) is not False:
            violations.append(f"safety_{key}_not_false")
        if _bool_true(payload.get(key)):
            violations.append(f"top_level_{key}_not_false")
    for path in _mutation_field_paths(payload):
        violations.append(f"mutation_evidence_field_present:{path}")


def _append_nested_field_violations(
    *,
    row_label: str,
    mapping: Mapping[str, Any],
    required_fields: Tuple[str, ...],
    label: str,
    violations: list[str],
) -> None:
    for key in required_fields:
        if key not in mapping:
            violations.append(f"{row_label}_{label}_missing_field:{key}")


def _append_coverage_violations(
    *,
    row_label: str,
    coverage: Mapping[str, Any],
    violations: list[str],
) -> None:
    _append_nested_field_violations(
        row_label=row_label,
        mapping=coverage,
        required_fields=DLIVE_STATE_REQUIRED_COVERAGE_FLAGS,
        label="coverage",
        violations=violations,
    )
    for key in DLIVE_STATE_REQUIRED_COVERAGE_FLAGS:
        if coverage.get(key) is not True:
            violations.append(f"{row_label}_coverage_{key}_not_true")


def _append_physical_identity_violations(
    *,
    row_label: str,
    row: Mapping[str, Any],
    violations: list[str],
) -> None:
    if row.get("readback_status") not in DLIVE_STATE_VERIFIED_PHYSICAL_STATUSES:
        violations.append(f"{row_label}_readback_status_not_physical_verified")
    if row.get("readback_source") not in DLIVE_STATE_PHYSICAL_READBACK_SOURCES:
        violations.append(f"{row_label}_readback_source_not_physical_readback")
    if not _string(row.get("captured_at")):
        violations.append(f"{row_label}_captured_at_missing")


def _append_channel_violations(payload: Mapping[str, Any], violations: list[str]) -> int:
    rows = payload.get("channels")
    if not isinstance(rows, list) or len(rows) < MINIMUM_DLIVE_STATE_CHANNEL_COUNT:
        violations.append("channels_missing")
        return 0
    if (
        not isinstance(payload.get("channel_count"), int)
        or payload.get("channel_count") != len(rows)
    ):
        violations.append("channel_count_mismatch")
    seen_channels: set[int] = set()
    for index, row_value in enumerate(rows):
        row = _mapping(row_value)
        row_label = f"channel_{index}"
        if not row:
            violations.append(f"{row_label}_not_object")
            continue
        _append_nested_field_violations(
            row_label=row_label,
            mapping=row,
            required_fields=DLIVE_STATE_REQUIRED_CHANNEL_FIELDS,
            label="field",
            violations=violations,
        )
        channel = row.get("channel")
        if not isinstance(channel, int) or channel < 1:
            violations.append(f"{row_label}_channel_invalid")
        elif channel in seen_channels:
            violations.append(f"{row_label}_channel_duplicate")
        else:
            seen_channels.add(channel)
        if not _string(row.get("cached_channel_name")):
            violations.append(f"{row_label}_cached_channel_name_missing")
        if (
            not isinstance(row.get("fader_raw_14bit"), int)
            or isinstance(row.get("fader_raw_14bit"), bool)
            or not 0 <= row.get("fader_raw_14bit", -1) <= 16383
        ):
            violations.append(f"{row_label}_fader_raw_14bit_invalid")
        if not _finite_number(row.get("fader_db")):
            violations.append(f"{row_label}_fader_db_invalid")
        if not isinstance(row.get("mute"), bool):
            violations.append(f"{row_label}_mute_not_bool")
        if (
            not isinstance(row.get("raw_note_velocity"), int)
            or isinstance(row.get("raw_note_velocity"), bool)
            or not 0 <= row.get("raw_note_velocity", -1) <= 127
        ):
            violations.append(f"{row_label}_raw_note_velocity_invalid")
        if (
            not isinstance(row.get("midi_base_channel"), int)
            or isinstance(row.get("midi_base_channel"), bool)
            or row.get("midi_base_channel") < 1
        ):
            violations.append(f"{row_label}_midi_base_channel_invalid")
        _append_physical_identity_violations(
            row_label=row_label,
            row=row,
            violations=violations,
        )
        _append_coverage_violations(
            row_label=row_label,
            coverage=_mapping(row.get("coverage")),
            violations=violations,
        )
    return len(rows)


def _append_midi_state_violations(payload: Mapping[str, Any], violations: list[str]) -> None:
    midi_state = _mapping(payload.get("midi_state"))
    if not midi_state:
        violations.append("midi_state_missing")
    _append_nested_field_violations(
        row_label="midi_state",
        mapping=midi_state,
        required_fields=DLIVE_STATE_REQUIRED_MIDI_STATE_FIELDS,
        label="field",
        violations=violations,
    )
    if (
        not isinstance(midi_state.get("midi_base_channel"), int)
        or isinstance(midi_state.get("midi_base_channel"), bool)
        or midi_state.get("midi_base_channel") < 1
    ):
        violations.append("midi_state_midi_base_channel_invalid")
    for key in (
        "raw_state_key_count",
        "raw_note_event_count",
        "raw_cc_event_count",
        "raw_pitchbend_event_count",
    ):
        if (
            not isinstance(midi_state.get(key), int)
            or isinstance(midi_state.get(key), bool)
            or midi_state.get(key) < 0
        ):
            violations.append(f"midi_state_{key}_invalid")
    if not _string(midi_state.get("tcp_session_status")):
        violations.append("midi_state_tcp_session_status_missing")
    if not _string(midi_state.get("midi_transport_status")):
        violations.append("midi_state_midi_transport_status_missing")
    _append_physical_identity_violations(
        row_label="midi_state",
        row=midi_state,
        violations=violations,
    )
    _append_coverage_violations(
        row_label="midi_state",
        coverage=_mapping(midi_state.get("coverage")),
        violations=violations,
    )


def _append_parity_violations(payload: Mapping[str, Any], violations: list[str]) -> None:
    parity = _mapping(payload.get("parity"))
    if not parity:
        violations.append("parity_missing")
    _append_nested_field_violations(
        row_label="parity",
        mapping=parity,
        required_fields=DLIVE_STATE_REQUIRED_PARITY_FIELDS,
        label="field",
        violations=violations,
    )
    for key in (
        "wing_surface_equivalent",
        "mapping_status",
        "channel_identity_parity",
        "fader_scale_parity",
        "mute_state_parity",
        "midi_velocity_parity",
        "unsupported_field_policy",
    ):
        if not _string(parity.get(key)):
            violations.append(f"parity_{key}_missing")
    _append_coverage_violations(
        row_label="parity",
        coverage=_mapping(parity.get("coverage")),
        violations=violations,
    )


def build_dlive_state_physical_readback_contract() -> Dict[str, Any]:
    """Return the non-executing artifact contract for dLive parity proof."""

    return {
        "type": "dlive_state_physical_readback_contract",
        "generated_at": _utc_now(),
        "method_id": DLIVE_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        "schema_id": DLIVE_STATE_PHYSICAL_READBACK_SCHEMA_ID,
        "status": "contract_ready_physical_fixture_missing",
        "surface_id": DLIVE_STATE_SURFACE_ID,
        "remediation_action_id": DLIVE_STATE_REMEDIATION_ACTION_ID,
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "midi_send_allowed": False,
            "tcp_send_allowed": False,
            "dlive_write_allowed": False,
            "dlive_apply_allowed": False,
        },
        "artifact_contract": {
            "type": DLIVE_STATE_PHYSICAL_READBACK_ARTIFACT_TYPE,
            "schema_id": DLIVE_STATE_PHYSICAL_READBACK_SCHEMA_ID,
            "required_top_level_fields": list(DLIVE_STATE_REQUIRED_TOP_LEVEL_FIELDS),
            "required_channel_fields": list(DLIVE_STATE_REQUIRED_CHANNEL_FIELDS),
            "required_midi_state_fields": list(DLIVE_STATE_REQUIRED_MIDI_STATE_FIELDS),
            "required_parity_fields": list(DLIVE_STATE_REQUIRED_PARITY_FIELDS),
            "required_coverage_flags": list(DLIVE_STATE_REQUIRED_COVERAGE_FLAGS),
            "required_false_safety_flags": list(DLIVE_STATE_REQUIRED_FALSE_SAFETY_FLAGS),
            "allowed_sources": list(DLIVE_STATE_PHYSICAL_READBACK_SOURCES),
            "accepted_row_statuses": list(DLIVE_STATE_VERIFIED_PHYSICAL_STATUSES),
            "forbidden_mutation_evidence_fields": list(
                DLIVE_STATE_FORBIDDEN_MUTATION_EVIDENCE_FIELDS
            ),
            "minimum_channel_count": MINIMUM_DLIVE_STATE_CHANNEL_COUNT,
        },
        "summary": {
            "physical_readback_surface_id": DLIVE_STATE_SURFACE_ID,
            "physical_readback_action_id": DLIVE_STATE_REMEDIATION_ACTION_ID,
            "artifact_required": True,
            "physical_fixture_gap_remains": True,
            "accepted_as_physical_fixture_candidate": False,
            "live_authority_granted": False,
            "promotion_families_requiring_surface": [
                "fader",
                "gain",
                "mute",
                "bus_group_matrix",
                "routing",
                "scene_preset",
                "mix_balance",
                "performance_adaptation",
            ],
            "claim": (
                "contract only; a separate operator-run dLive physical readback "
                "and WING parity artifact is still required"
            ),
        },
    }


def validate_dlive_state_physical_readback_artifact(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate a future dLive-state artifact without granting authority."""

    violations: list[str] = []
    artifact = _mapping(payload)
    _append_required_field_violations(artifact, violations)
    _append_identity_violations(artifact, violations)
    _append_safety_violations(artifact, violations)
    channel_count = _append_channel_violations(artifact, violations)
    _append_midi_state_violations(artifact, violations)
    _append_parity_violations(artifact, violations)

    status = (
        "valid_physical_readback_artifact"
        if not violations
        else "invalid_physical_readback_artifact"
    )
    validation = DLiveStatePhysicalReadbackValidation(
        method_id=DLIVE_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        schema_id=DLIVE_STATE_PHYSICAL_READBACK_SCHEMA_ID,
        surface_id=str(artifact.get("surface_id") or ""),
        remediation_action_id=str(artifact.get("remediation_action_id") or ""),
        status=status,
        accepted_as_physical_fixture_candidate=not violations,
        live_authority_granted=False,
        full_autonomous_ready=False,
        channel_count=channel_count,
        violations=tuple(sorted(set(violations))),
        evidence={
            "source": artifact.get("source"),
            "captured_at": artifact.get("captured_at"),
            "console_model": artifact.get("console_model"),
            "channel_count": artifact.get("channel_count"),
        },
    )
    return validation.to_dict()
