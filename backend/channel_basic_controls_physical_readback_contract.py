"""Contract for channel identity/basic-control physical readback artifacts.

This module is transport-free. It defines the JSON-like artifact shape a future
operator-run physical readback must produce before the
``channel_identity_and_basic_controls`` read surface can be treated as physical
fixture evidence for fader, gain, mute, and pan promotion decisions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple


CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_CONTRACT_METHOD_ID = (
    "channel_basic_controls_physical_readback_contract"
)
CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_SCHEMA_ID = (
    "channel_basic_controls_physical_readback.v1"
)
CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_ARTIFACT_TYPE = (
    "channel_basic_controls_physical_readback"
)
CHANNEL_BASIC_CONTROLS_SURFACE_ID = "channel_identity_and_basic_controls"
CHANNEL_BASIC_CONTROLS_REMEDIATION_ACTION_ID = (
    "remediate_channel_identity_and_basic_controls"
)

CHANNEL_BASIC_CONTROLS_REQUIRED_TOP_LEVEL_FIELDS = (
    "type",
    "schema_id",
    "surface_id",
    "remediation_action_id",
    "captured_at",
    "source",
    "console_model",
    "channel_count",
    "channels",
    "safety",
)
CHANNEL_BASIC_CONTROLS_REQUIRED_CHANNEL_FIELDS = (
    "channel_id",
    "channel_label",
    "status",
    "source",
    "captured_at",
    "identity",
    "controls",
    "coverage",
)
CHANNEL_BASIC_CONTROLS_REQUIRED_IDENTITY_FIELDS = (
    "source_name",
    "source_role",
    "input_source",
)
CHANNEL_BASIC_CONTROLS_REQUIRED_CONTROL_FIELDS = (
    "fader_db",
    "trim_db",
    "mute",
    "pan",
)
CHANNEL_BASIC_CONTROLS_REQUIRED_COVERAGE_FLAGS = (
    "identity",
    "basic_controls",
    "cache_matches_physical_readback",
)
CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_SOURCES = (
    "physical_readback",
    "wing_physical_readback",
    "dlive_physical_readback",
)
CHANNEL_BASIC_CONTROLS_VERIFIED_PHYSICAL_STATUSES = (
    "verified_from_physical_readback",
)
CHANNEL_BASIC_CONTROLS_REQUIRED_FALSE_SAFETY_FLAGS = (
    "live_write",
    "mutates_console_state",
    "live_widening_allowed",
    "full_autonomous_ready",
)
CHANNEL_BASIC_CONTROLS_FORBIDDEN_MUTATION_EVIDENCE_FIELDS = (
    "write_events",
    "manual_write_events",
    "apply_result",
    "rollback_result",
    "proposal_apply_result",
    "mute_write_events",
    "pan_write_events",
)


@dataclass(frozen=True)
class ChannelBasicControlsPhysicalReadbackValidation:
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
            raise ValueError("Channel readback validation must not grant live authority")
        if self.full_autonomous_ready:
            raise ValueError("Channel readback validation must not claim full autonomous readiness")

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
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _append_required_field_violations(payload: Mapping[str, Any], violations: list[str]) -> None:
    for key in CHANNEL_BASIC_CONTROLS_REQUIRED_TOP_LEVEL_FIELDS:
        if key not in payload:
            violations.append(f"missing_top_level_field:{key}")


def _append_identity_violations(payload: Mapping[str, Any], violations: list[str]) -> None:
    if payload.get("type") != CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_ARTIFACT_TYPE:
        violations.append("artifact_type_mismatch")
    if payload.get("schema_id") != CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("schema_id_mismatch")
    if payload.get("surface_id") != CHANNEL_BASIC_CONTROLS_SURFACE_ID:
        violations.append("surface_id_mismatch")
    if payload.get("remediation_action_id") != CHANNEL_BASIC_CONTROLS_REMEDIATION_ACTION_ID:
        violations.append("remediation_action_id_mismatch")
    if payload.get("source") not in CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_SOURCES:
        violations.append("source_not_physical_readback")
    if not _string(payload.get("captured_at")):
        violations.append("captured_at_missing")
    if not _string(payload.get("console_model")):
        violations.append("console_model_missing")


def _append_safety_violations(payload: Mapping[str, Any], violations: list[str]) -> None:
    safety = _mapping(payload.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("safety_read_only_not_true")
    for key in CHANNEL_BASIC_CONTROLS_REQUIRED_FALSE_SAFETY_FLAGS:
        if safety.get(key) is not False:
            violations.append(f"safety_{key}_not_false")
        if _bool_true(payload.get(key)):
            violations.append(f"top_level_{key}_not_false")
    for key in CHANNEL_BASIC_CONTROLS_FORBIDDEN_MUTATION_EVIDENCE_FIELDS:
        if key in payload:
            violations.append(f"mutation_evidence_field_present:{key}")


def _append_nested_field_violations(
    *,
    channel_index: int,
    mapping: Mapping[str, Any],
    required_fields: Tuple[str, ...],
    label: str,
    violations: list[str],
) -> None:
    for key in required_fields:
        if key not in mapping:
            violations.append(f"channel_{channel_index}_{label}_missing_field:{key}")


def _append_channel_violations(payload: Mapping[str, Any], violations: list[str]) -> int:
    channels = payload.get("channels")
    if not isinstance(channels, list) or not channels:
        violations.append("channels_missing")
        return 0

    declared_count = payload.get("channel_count")
    if not isinstance(declared_count, int) or declared_count != len(channels):
        violations.append("channel_count_mismatch")

    seen_channel_ids: set[int] = set()
    for index, channel_value in enumerate(channels):
        channel = _mapping(channel_value)
        if not channel:
            violations.append(f"channel_{index}_not_object")
            continue
        for key in CHANNEL_BASIC_CONTROLS_REQUIRED_CHANNEL_FIELDS:
            if key not in channel:
                violations.append(f"channel_{index}_missing_field:{key}")

        channel_id = channel.get("channel_id")
        if not isinstance(channel_id, int) or channel_id < 1:
            violations.append(f"channel_{index}_channel_id_invalid")
        elif channel_id in seen_channel_ids:
            violations.append(f"channel_{index}_channel_id_duplicate")
        else:
            seen_channel_ids.add(channel_id)
        if not _string(channel.get("channel_label")):
            violations.append(f"channel_{index}_channel_label_missing")
        if channel.get("status") not in CHANNEL_BASIC_CONTROLS_VERIFIED_PHYSICAL_STATUSES:
            violations.append(f"channel_{index}_status_not_physical_verified")
        if channel.get("source") not in CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_SOURCES:
            violations.append(f"channel_{index}_source_not_physical_readback")
        if not _string(channel.get("captured_at")):
            violations.append(f"channel_{index}_captured_at_missing")

        identity = _mapping(channel.get("identity"))
        controls = _mapping(channel.get("controls"))
        coverage = _mapping(channel.get("coverage"))
        _append_nested_field_violations(
            channel_index=index,
            mapping=identity,
            required_fields=CHANNEL_BASIC_CONTROLS_REQUIRED_IDENTITY_FIELDS,
            label="identity",
            violations=violations,
        )
        _append_nested_field_violations(
            channel_index=index,
            mapping=controls,
            required_fields=CHANNEL_BASIC_CONTROLS_REQUIRED_CONTROL_FIELDS,
            label="controls",
            violations=violations,
        )
        _append_nested_field_violations(
            channel_index=index,
            mapping=coverage,
            required_fields=CHANNEL_BASIC_CONTROLS_REQUIRED_COVERAGE_FLAGS,
            label="coverage",
            violations=violations,
        )
        for key in CHANNEL_BASIC_CONTROLS_REQUIRED_IDENTITY_FIELDS:
            if key in identity and not _string(identity.get(key)):
                violations.append(f"channel_{index}_identity_{key}_missing")
        for key in ("fader_db", "trim_db", "pan"):
            if key in controls and not _finite_number(controls.get(key)):
                violations.append(f"channel_{index}_controls_{key}_not_number")
        if "mute" in controls and not isinstance(controls.get("mute"), bool):
            violations.append(f"channel_{index}_controls_mute_not_bool")
        for key in CHANNEL_BASIC_CONTROLS_REQUIRED_COVERAGE_FLAGS:
            if coverage.get(key) is not True:
                violations.append(f"channel_{index}_coverage_{key}_not_true")
    return len(channels)


def build_channel_basic_controls_physical_readback_contract() -> Dict[str, Any]:
    """Return the non-executing artifact contract for channel basic-control proof."""

    return {
        "type": "channel_basic_controls_physical_readback_contract",
        "generated_at": _utc_now(),
        "method_id": CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        "schema_id": CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_SCHEMA_ID,
        "status": "contract_ready_physical_fixture_missing",
        "surface_id": CHANNEL_BASIC_CONTROLS_SURFACE_ID,
        "remediation_action_id": CHANNEL_BASIC_CONTROLS_REMEDIATION_ACTION_ID,
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
        },
        "artifact_contract": {
            "type": CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_ARTIFACT_TYPE,
            "schema_id": CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_SCHEMA_ID,
            "required_top_level_fields": list(CHANNEL_BASIC_CONTROLS_REQUIRED_TOP_LEVEL_FIELDS),
            "required_channel_fields": list(CHANNEL_BASIC_CONTROLS_REQUIRED_CHANNEL_FIELDS),
            "required_identity_fields": list(CHANNEL_BASIC_CONTROLS_REQUIRED_IDENTITY_FIELDS),
            "required_control_fields": list(CHANNEL_BASIC_CONTROLS_REQUIRED_CONTROL_FIELDS),
            "required_coverage_flags": list(CHANNEL_BASIC_CONTROLS_REQUIRED_COVERAGE_FLAGS),
            "required_false_safety_flags": list(CHANNEL_BASIC_CONTROLS_REQUIRED_FALSE_SAFETY_FLAGS),
            "allowed_sources": list(CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_SOURCES),
            "accepted_channel_statuses": list(CHANNEL_BASIC_CONTROLS_VERIFIED_PHYSICAL_STATUSES),
            "forbidden_mutation_evidence_fields": list(
                CHANNEL_BASIC_CONTROLS_FORBIDDEN_MUTATION_EVIDENCE_FIELDS
            ),
            "minimum_channel_count": 1,
        },
        "summary": {
            "physical_readback_surface_id": CHANNEL_BASIC_CONTROLS_SURFACE_ID,
            "physical_readback_action_id": CHANNEL_BASIC_CONTROLS_REMEDIATION_ACTION_ID,
            "artifact_required": True,
            "physical_fixture_gap_remains": True,
            "accepted_as_physical_fixture_candidate": False,
            "live_authority_granted": False,
            "promotion_families_requiring_surface": ["fader", "gain", "mute", "pan_phase_delay"],
            "claim": (
                "contract only; a separate operator-run channel physical readback "
                "artifact is still required"
            ),
        },
    }


def validate_channel_basic_controls_physical_readback_artifact(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate a future channel physical artifact without granting live authority."""

    violations: list[str] = []
    artifact = _mapping(payload)
    _append_required_field_violations(artifact, violations)
    _append_identity_violations(artifact, violations)
    _append_safety_violations(artifact, violations)
    channel_count = _append_channel_violations(artifact, violations)

    status = (
        "valid_physical_readback_artifact"
        if not violations
        else "invalid_physical_readback_artifact"
    )
    validation = ChannelBasicControlsPhysicalReadbackValidation(
        method_id=CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        schema_id=CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_SCHEMA_ID,
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
            "validated_at": _utc_now(),
        },
    )
    return validation.to_dict()
