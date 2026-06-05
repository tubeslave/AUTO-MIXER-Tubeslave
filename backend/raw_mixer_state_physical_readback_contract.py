"""Contract for raw mixer-state physical readback artifacts.

This module is deliberately transport-free. It defines the JSON-like artifact
shape a future operator-run physical readback must produce before the
``raw_mixer_state_cache`` remediation action can be treated as evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple


RAW_MIXER_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID = (
    "raw_mixer_state_physical_readback_contract"
)
RAW_MIXER_STATE_PHYSICAL_READBACK_SCHEMA_ID = "raw_mixer_state_physical_readback.v1"
RAW_MIXER_STATE_PHYSICAL_READBACK_ARTIFACT_TYPE = "raw_mixer_state_physical_readback"
RAW_MIXER_STATE_SURFACE_ID = "raw_mixer_state_cache"
RAW_MIXER_STATE_REMEDIATION_ACTION_ID = "remediate_raw_mixer_state_cache"

REQUIRED_TOP_LEVEL_FIELDS = (
    "type",
    "schema_id",
    "surface_id",
    "remediation_action_id",
    "captured_at",
    "source",
    "console_model",
    "address_count",
    "readback_rows",
    "safety",
)
REQUIRED_ROW_FIELDS = (
    "address",
    "value",
    "status",
    "source",
    "captured_at",
)
PHYSICAL_READBACK_SOURCES = (
    "physical_readback",
    "wing_physical_readback",
    "dlive_physical_readback",
)
VERIFIED_PHYSICAL_ROW_STATUSES = ("verified_from_physical_readback",)
REQUIRED_FALSE_SAFETY_FLAGS = (
    "live_write",
    "mutates_console_state",
    "live_widening_allowed",
    "full_autonomous_ready",
)
FORBIDDEN_MUTATION_EVIDENCE_FIELDS = (
    "write_events",
    "manual_write_events",
    "apply_result",
    "rollback_result",
    "proposal_apply_result",
)


@dataclass(frozen=True)
class RawMixerStatePhysicalReadbackValidation:
    method_id: str
    schema_id: str
    surface_id: str
    remediation_action_id: str
    status: str
    accepted_as_physical_fixture_candidate: bool
    live_authority_granted: bool
    full_autonomous_ready: bool
    row_count: int
    violations: Tuple[str, ...]
    evidence: Dict[str, Any]

    def __post_init__(self) -> None:
        if self.live_authority_granted:
            raise ValueError("Raw mixer readback validation must not grant live authority")
        if self.full_autonomous_ready:
            raise ValueError("Raw mixer readback validation must not claim full autonomous readiness")

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


def _append_required_field_violations(payload: Mapping[str, Any], violations: list[str]) -> None:
    for key in REQUIRED_TOP_LEVEL_FIELDS:
        if key not in payload:
            violations.append(f"missing_top_level_field:{key}")


def _append_identity_violations(payload: Mapping[str, Any], violations: list[str]) -> None:
    if payload.get("type") != RAW_MIXER_STATE_PHYSICAL_READBACK_ARTIFACT_TYPE:
        violations.append("artifact_type_mismatch")
    if payload.get("schema_id") != RAW_MIXER_STATE_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("schema_id_mismatch")
    if payload.get("surface_id") != RAW_MIXER_STATE_SURFACE_ID:
        violations.append("surface_id_mismatch")
    if payload.get("remediation_action_id") != RAW_MIXER_STATE_REMEDIATION_ACTION_ID:
        violations.append("remediation_action_id_mismatch")
    if payload.get("source") not in PHYSICAL_READBACK_SOURCES:
        violations.append("source_not_physical_readback")
    if not _string(payload.get("captured_at")):
        violations.append("captured_at_missing")
    if not _string(payload.get("console_model")):
        violations.append("console_model_missing")


def _append_safety_violations(payload: Mapping[str, Any], violations: list[str]) -> None:
    safety = _mapping(payload.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("safety_read_only_not_true")
    for key in REQUIRED_FALSE_SAFETY_FLAGS:
        if safety.get(key) is not False:
            violations.append(f"safety_{key}_not_false")
        if _bool_true(payload.get(key)):
            violations.append(f"top_level_{key}_not_false")
    for key in FORBIDDEN_MUTATION_EVIDENCE_FIELDS:
        if key in payload:
            violations.append(f"mutation_evidence_field_present:{key}")


def _append_row_violations(payload: Mapping[str, Any], violations: list[str]) -> int:
    rows = payload.get("readback_rows")
    if not isinstance(rows, list) or not rows:
        violations.append("readback_rows_missing")
        return 0

    declared_count = payload.get("address_count")
    if not isinstance(declared_count, int) or declared_count != len(rows):
        violations.append("address_count_mismatch")

    for index, row_value in enumerate(rows):
        row = _mapping(row_value)
        if not row:
            violations.append(f"row_{index}_not_object")
            continue
        for key in REQUIRED_ROW_FIELDS:
            if key not in row:
                violations.append(f"row_{index}_missing_field:{key}")
        address = _string(row.get("address"))
        if not address or not address.startswith("/"):
            violations.append(f"row_{index}_address_invalid")
        if row.get("value") is None:
            violations.append(f"row_{index}_value_missing")
        if row.get("status") not in VERIFIED_PHYSICAL_ROW_STATUSES:
            violations.append(f"row_{index}_status_not_physical_verified")
        if row.get("source") not in PHYSICAL_READBACK_SOURCES:
            violations.append(f"row_{index}_source_not_physical_readback")
        if not _string(row.get("captured_at")):
            violations.append(f"row_{index}_captured_at_missing")
    return len(rows)


def build_raw_mixer_state_physical_readback_contract() -> Dict[str, Any]:
    """Return the non-executing artifact contract for raw mixer-state proof."""

    return {
        "type": "raw_mixer_state_physical_readback_contract",
        "generated_at": _utc_now(),
        "method_id": RAW_MIXER_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        "schema_id": RAW_MIXER_STATE_PHYSICAL_READBACK_SCHEMA_ID,
        "status": "contract_ready_physical_fixture_missing",
        "surface_id": RAW_MIXER_STATE_SURFACE_ID,
        "remediation_action_id": RAW_MIXER_STATE_REMEDIATION_ACTION_ID,
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
            "type": RAW_MIXER_STATE_PHYSICAL_READBACK_ARTIFACT_TYPE,
            "schema_id": RAW_MIXER_STATE_PHYSICAL_READBACK_SCHEMA_ID,
            "required_top_level_fields": list(REQUIRED_TOP_LEVEL_FIELDS),
            "required_row_fields": list(REQUIRED_ROW_FIELDS),
            "required_false_safety_flags": list(REQUIRED_FALSE_SAFETY_FLAGS),
            "allowed_sources": list(PHYSICAL_READBACK_SOURCES),
            "accepted_row_statuses": list(VERIFIED_PHYSICAL_ROW_STATUSES),
            "forbidden_mutation_evidence_fields": list(FORBIDDEN_MUTATION_EVIDENCE_FIELDS),
            "minimum_row_count": 1,
        },
        "summary": {
            "next_physical_readback_surface_id": RAW_MIXER_STATE_SURFACE_ID,
            "next_physical_readback_action_id": RAW_MIXER_STATE_REMEDIATION_ACTION_ID,
            "artifact_required": True,
            "physical_fixture_gap_remains": True,
            "accepted_as_physical_fixture_candidate": False,
            "live_authority_granted": False,
            "claim": (
                "contract only; a separate operator-run physical readback artifact "
                "is still required"
            ),
        },
    }


def validate_raw_mixer_state_physical_readback_artifact(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate a future physical artifact without granting live authority."""

    violations: list[str] = []
    artifact = _mapping(payload)
    _append_required_field_violations(artifact, violations)
    _append_identity_violations(artifact, violations)
    _append_safety_violations(artifact, violations)
    row_count = _append_row_violations(artifact, violations)

    status = (
        "valid_physical_readback_artifact"
        if not violations
        else "invalid_physical_readback_artifact"
    )
    validation = RawMixerStatePhysicalReadbackValidation(
        method_id=RAW_MIXER_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        schema_id=RAW_MIXER_STATE_PHYSICAL_READBACK_SCHEMA_ID,
        surface_id=str(artifact.get("surface_id") or ""),
        remediation_action_id=str(artifact.get("remediation_action_id") or ""),
        status=status,
        accepted_as_physical_fixture_candidate=not violations,
        live_authority_granted=False,
        full_autonomous_ready=False,
        row_count=row_count,
        violations=tuple(sorted(set(violations))),
        evidence={
            "source": artifact.get("source"),
            "captured_at": artifact.get("captured_at"),
            "console_model": artifact.get("console_model"),
            "address_count": artifact.get("address_count"),
            "validated_at": _utc_now(),
        },
    )
    return validation.to_dict()
