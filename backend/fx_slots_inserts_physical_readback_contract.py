"""Contract for FX slot and insert physical readback artifacts.

This module is transport-free. It defines the JSON-like artifact shape a future
operator-run physical readback must produce before the ``fx_slots_and_inserts``
read surface can be treated as physical fixture evidence for FX, insert,
routing, and mix-balance promotion decisions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple


FX_SLOTS_INSERTS_PHYSICAL_READBACK_CONTRACT_METHOD_ID = (
    "fx_slots_inserts_physical_readback_contract"
)
FX_SLOTS_INSERTS_PHYSICAL_READBACK_SCHEMA_ID = "fx_slots_inserts_physical_readback.v1"
FX_SLOTS_INSERTS_PHYSICAL_READBACK_ARTIFACT_TYPE = "fx_slots_inserts_physical_readback"
FX_SLOTS_INSERTS_SURFACE_ID = "fx_slots_and_inserts"
FX_SLOTS_INSERTS_REMEDIATION_ACTION_ID = "remediate_fx_slots_and_inserts"

FX_SLOTS_INSERTS_REQUIRED_TOP_LEVEL_FIELDS = (
    "type",
    "schema_id",
    "surface_id",
    "remediation_action_id",
    "captured_at",
    "source",
    "console_model",
    "fx_slot_count",
    "insert_count",
    "fx_slots",
    "inserts",
    "safety",
)
FX_SLOT_REQUIRED_FIELDS = (
    "slot",
    "slot_number",
    "readback_status",
    "readback_source",
    "captured_at",
    "model",
    "on",
    "mix",
    "source",
    "mode",
    "assigned_channel",
    "assigned_position",
    "active",
    "parameters",
    "parameter_schema",
    "coverage",
)
FX_PARAMETER_REQUIRED_FIELDS = (
    "parameter_index",
    "value",
    "schema_status",
    "schema_family",
    "schema_source",
    "semantic_name",
    "unit",
    "safe_range",
)
FX_PARAMETER_SAFE_RANGE_FIELDS = (
    "min",
    "max",
)
FX_PARAMETER_SCHEMA_REQUIRED_FIELDS = (
    "status",
    "schema_family",
    "schema_source",
    "mapped_parameter_count",
    "generic_parameter_count",
    "cached_parameter_count",
    "physical_model_schema_verified",
    "live_write_authority",
)
FX_SLOT_REQUIRED_COVERAGE_FLAGS = (
    "identity",
    "assignment",
    "parameters",
    "parameter_schema",
    "cache_matches_physical_readback",
)
FX_INSERT_REQUIRED_FIELDS = (
    "target_type",
    "target",
    "position",
    "readback_status",
    "readback_source",
    "captured_at",
    "on",
    "slot",
    "status",
    "mode",
    "weight",
    "linked_fx_slot",
    "active",
    "insert_assignment_policy",
    "destructive_change_marker",
    "coverage",
)
FX_INSERT_REQUIRED_COVERAGE_FLAGS = (
    "assignment",
    "status",
    "scoped_diff_marker",
    "cache_matches_physical_readback",
)
FX_SLOTS_INSERTS_PHYSICAL_READBACK_SOURCES = (
    "physical_readback",
    "wing_physical_readback",
    "dlive_physical_readback",
)
FX_SLOTS_INSERTS_VERIFIED_PHYSICAL_STATUSES = (
    "verified_from_physical_readback",
)
FX_INSERT_TARGET_TYPES = (
    "channel",
    "bus",
    "main",
    "matrix",
)
FX_INSERT_POSITIONS = (
    "pre",
    "post",
)
FX_SLOTS_INSERTS_REQUIRED_FALSE_SAFETY_FLAGS = (
    "live_write",
    "mutates_console_state",
    "live_widening_allowed",
    "full_autonomous_ready",
)
FX_SLOTS_INSERTS_FORBIDDEN_MUTATION_EVIDENCE_FIELDS = (
    "write_events",
    "manual_write_events",
    "apply_result",
    "rollback_result",
    "proposal_apply_result",
    "fx_write_events",
    "fx_slot_write_events",
    "fx_parameter_write_events",
    "fx_send_write_events",
    "insert_write_events",
    "insert_assignment_write_events",
    "routing_write_events",
)
MINIMUM_FX_SLOT_COUNT = 1
MINIMUM_FX_PARAMETER_COUNT = 4
MINIMUM_FX_INSERT_COUNT = 1


@dataclass(frozen=True)
class FxSlotsInsertsPhysicalReadbackValidation:
    method_id: str
    schema_id: str
    surface_id: str
    remediation_action_id: str
    status: str
    accepted_as_physical_fixture_candidate: bool
    live_authority_granted: bool
    full_autonomous_ready: bool
    fx_slot_count: int
    insert_count: int
    violations: Tuple[str, ...]
    evidence: Dict[str, Any]

    def __post_init__(self) -> None:
        if self.live_authority_granted:
            raise ValueError("FX readback validation must not grant live authority")
        if self.full_autonomous_ready:
            raise ValueError(
                "FX readback validation must not claim full autonomous readiness"
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
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _bool_or_0_1(value: Any) -> bool:
    return isinstance(value, bool) or (
        isinstance(value, int) and not isinstance(value, bool) and value in (0, 1)
    )


def _append_required_field_violations(payload: Mapping[str, Any], violations: list[str]) -> None:
    for key in FX_SLOTS_INSERTS_REQUIRED_TOP_LEVEL_FIELDS:
        if key not in payload:
            violations.append(f"missing_top_level_field:{key}")


def _append_identity_violations(payload: Mapping[str, Any], violations: list[str]) -> None:
    if payload.get("type") != FX_SLOTS_INSERTS_PHYSICAL_READBACK_ARTIFACT_TYPE:
        violations.append("artifact_type_mismatch")
    if payload.get("schema_id") != FX_SLOTS_INSERTS_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("schema_id_mismatch")
    if payload.get("surface_id") != FX_SLOTS_INSERTS_SURFACE_ID:
        violations.append("surface_id_mismatch")
    if payload.get("remediation_action_id") != FX_SLOTS_INSERTS_REMEDIATION_ACTION_ID:
        violations.append("remediation_action_id_mismatch")
    if payload.get("source") not in FX_SLOTS_INSERTS_PHYSICAL_READBACK_SOURCES:
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
            if key_text in FX_SLOTS_INSERTS_FORBIDDEN_MUTATION_EVIDENCE_FIELDS:
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
    for key in FX_SLOTS_INSERTS_REQUIRED_FALSE_SAFETY_FLAGS:
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
    required_flags: Tuple[str, ...],
    violations: list[str],
) -> None:
    _append_nested_field_violations(
        row_label=row_label,
        mapping=coverage,
        required_fields=required_flags,
        label="coverage",
        violations=violations,
    )
    for key in required_flags:
        if coverage.get(key) is not True:
            violations.append(f"{row_label}_coverage_{key}_not_true")


def _append_parameter_schema_violations(
    *,
    slot_index: int,
    schema: Mapping[str, Any],
    parameter_count: int,
    violations: list[str],
) -> None:
    row_label = f"fx_slot_{slot_index}"
    _append_nested_field_violations(
        row_label=row_label,
        mapping=schema,
        required_fields=FX_PARAMETER_SCHEMA_REQUIRED_FIELDS,
        label="parameter_schema",
        violations=violations,
    )
    for key in (
        "status",
        "schema_family",
        "schema_source",
    ):
        if key in schema and not _string(schema.get(key)):
            violations.append(f"{row_label}_parameter_schema_{key}_missing")
    for key in (
        "mapped_parameter_count",
        "generic_parameter_count",
        "cached_parameter_count",
    ):
        if key in schema and not isinstance(schema.get(key), int):
            violations.append(f"{row_label}_parameter_schema_{key}_not_int")
    if schema.get("physical_model_schema_verified") is not True:
        violations.append(f"{row_label}_parameter_schema_physical_model_schema_not_verified")
    if schema.get("live_write_authority") is not False:
        violations.append(f"{row_label}_parameter_schema_live_write_authority_not_false")
    if isinstance(schema.get("generic_parameter_count"), int) and schema.get("generic_parameter_count") != 0:
        violations.append(f"{row_label}_parameter_schema_generic_parameter_count_not_zero")
    if isinstance(schema.get("cached_parameter_count"), int) and schema.get("cached_parameter_count") != parameter_count:
        violations.append(f"{row_label}_parameter_schema_cached_parameter_count_mismatch")


def _append_parameter_violations(
    *,
    slot_index: int,
    parameters_value: Any,
    violations: list[str],
) -> int:
    if not isinstance(parameters_value, list) or len(parameters_value) < MINIMUM_FX_PARAMETER_COUNT:
        violations.append(f"fx_slot_{slot_index}_parameters_missing")
        return 0
    seen_indexes: set[int] = set()
    for parameter_row_index, parameter_value in enumerate(parameters_value):
        parameter = _mapping(parameter_value)
        row_label = f"fx_slot_{slot_index}_parameter_{parameter_row_index}"
        if not parameter:
            violations.append(f"{row_label}_not_object")
            continue
        _append_nested_field_violations(
            row_label=row_label,
            mapping=parameter,
            required_fields=FX_PARAMETER_REQUIRED_FIELDS,
            label="row",
            violations=violations,
        )
        parameter_index = parameter.get("parameter_index")
        if not isinstance(parameter_index, int) or parameter_index < 1:
            violations.append(f"{row_label}_parameter_index_invalid")
        elif parameter_index in seen_indexes:
            violations.append(f"{row_label}_parameter_index_duplicate")
        else:
            seen_indexes.add(parameter_index)
        if "value" in parameter and not _finite_number(parameter.get("value")):
            violations.append(f"{row_label}_value_not_number")
        for key in (
            "schema_status",
            "schema_family",
            "schema_source",
            "semantic_name",
            "unit",
        ):
            if key in parameter and not _string(parameter.get(key)):
                violations.append(f"{row_label}_{key}_missing")
        safe_range = _mapping(parameter.get("safe_range"))
        _append_nested_field_violations(
            row_label=row_label,
            mapping=safe_range,
            required_fields=FX_PARAMETER_SAFE_RANGE_FIELDS,
            label="safe_range",
            violations=violations,
        )
        for key in FX_PARAMETER_SAFE_RANGE_FIELDS:
            value = safe_range.get(key)
            if value is not None and not _finite_number(value):
                violations.append(f"{row_label}_safe_range_{key}_not_number_or_null")
    return len(parameters_value)


def _append_fx_slot_violations(payload: Mapping[str, Any], violations: list[str]) -> int:
    rows = payload.get("fx_slots")
    if not isinstance(rows, list) or len(rows) < MINIMUM_FX_SLOT_COUNT:
        violations.append("fx_slots_missing")
        return 0
    if not isinstance(payload.get("fx_slot_count"), int) or payload.get("fx_slot_count") != len(rows):
        violations.append("fx_slot_count_mismatch")
    seen_slots: set[int] = set()
    for index, row_value in enumerate(rows):
        row = _mapping(row_value)
        row_label = f"fx_slot_{index}"
        if not row:
            violations.append(f"{row_label}_not_object")
            continue
        for key in FX_SLOT_REQUIRED_FIELDS:
            if key not in row:
                violations.append(f"{row_label}_missing_field:{key}")
        slot_number = row.get("slot_number")
        if not isinstance(slot_number, int) or slot_number < 1:
            violations.append(f"{row_label}_slot_number_invalid")
        elif slot_number in seen_slots:
            violations.append(f"{row_label}_slot_number_duplicate")
        else:
            seen_slots.add(slot_number)
        if not _string(row.get("slot")):
            violations.append(f"{row_label}_slot_missing")
        if row.get("readback_status") not in FX_SLOTS_INSERTS_VERIFIED_PHYSICAL_STATUSES:
            violations.append(f"{row_label}_readback_status_not_physical_verified")
        if row.get("readback_source") not in FX_SLOTS_INSERTS_PHYSICAL_READBACK_SOURCES:
            violations.append(f"{row_label}_readback_source_not_physical_readback")
        if not _string(row.get("captured_at")):
            violations.append(f"{row_label}_captured_at_missing")
        for key in ("model", "source", "mode", "assigned_position"):
            if key in row and not _string(row.get(key)):
                violations.append(f"{row_label}_{key}_missing")
        if "on" in row and not _bool_or_0_1(row.get("on")):
            violations.append(f"{row_label}_on_not_bool_or_0_1")
        if "mix" in row and not _finite_number(row.get("mix")):
            violations.append(f"{row_label}_mix_not_number")
        if "assigned_channel" in row and not isinstance(row.get("assigned_channel"), int):
            violations.append(f"{row_label}_assigned_channel_not_int")
        if "active" in row and not isinstance(row.get("active"), bool):
            violations.append(f"{row_label}_active_not_bool")
        parameter_count = _append_parameter_violations(
            slot_index=index,
            parameters_value=row.get("parameters"),
            violations=violations,
        )
        _append_parameter_schema_violations(
            slot_index=index,
            schema=_mapping(row.get("parameter_schema")),
            parameter_count=parameter_count,
            violations=violations,
        )
        _append_coverage_violations(
            row_label=row_label,
            coverage=_mapping(row.get("coverage")),
            required_flags=FX_SLOT_REQUIRED_COVERAGE_FLAGS,
            violations=violations,
        )
    return len(rows)


def _append_insert_violations(payload: Mapping[str, Any], violations: list[str]) -> int:
    rows = payload.get("inserts")
    if not isinstance(rows, list) or len(rows) < MINIMUM_FX_INSERT_COUNT:
        violations.append("inserts_missing")
        return 0
    if not isinstance(payload.get("insert_count"), int) or payload.get("insert_count") != len(rows):
        violations.append("insert_count_mismatch")
    seen_keys: set[Tuple[str, int, str]] = set()
    for index, row_value in enumerate(rows):
        row = _mapping(row_value)
        row_label = f"insert_{index}"
        if not row:
            violations.append(f"{row_label}_not_object")
            continue
        for key in FX_INSERT_REQUIRED_FIELDS:
            if key not in row:
                violations.append(f"{row_label}_missing_field:{key}")
        target_type = row.get("target_type")
        target = row.get("target")
        position = row.get("position")
        if target_type not in FX_INSERT_TARGET_TYPES:
            violations.append(f"{row_label}_target_type_invalid")
        if not isinstance(target, int) or target < 1:
            violations.append(f"{row_label}_target_invalid")
        if position not in FX_INSERT_POSITIONS:
            violations.append(f"{row_label}_position_invalid")
        if isinstance(target_type, str) and isinstance(target, int) and isinstance(position, str):
            key = (target_type, target, position)
            if key in seen_keys:
                violations.append(f"{row_label}_target_position_duplicate")
            else:
                seen_keys.add(key)
        if row.get("readback_status") not in FX_SLOTS_INSERTS_VERIFIED_PHYSICAL_STATUSES:
            violations.append(f"{row_label}_readback_status_not_physical_verified")
        if row.get("readback_source") not in FX_SLOTS_INSERTS_PHYSICAL_READBACK_SOURCES:
            violations.append(f"{row_label}_readback_source_not_physical_readback")
        if not _string(row.get("captured_at")):
            violations.append(f"{row_label}_captured_at_missing")
        if "on" in row and not _bool_or_0_1(row.get("on")):
            violations.append(f"{row_label}_on_not_bool_or_0_1")
        for key in (
            "slot",
            "status",
            "mode",
            "insert_assignment_policy",
            "destructive_change_marker",
        ):
            if key in row and not _string(row.get(key)):
                violations.append(f"{row_label}_{key}_missing")
        if "weight" in row and not _finite_number(row.get("weight")):
            violations.append(f"{row_label}_weight_not_number")
        linked_fx_slot = row.get("linked_fx_slot")
        if linked_fx_slot is not None and not isinstance(linked_fx_slot, str):
            violations.append(f"{row_label}_linked_fx_slot_not_string_or_null")
        if "active" in row and not isinstance(row.get("active"), bool):
            violations.append(f"{row_label}_active_not_bool")
        if row.get("active") is True and not _string(linked_fx_slot):
            violations.append(f"{row_label}_active_insert_missing_linked_fx_slot")
        _append_coverage_violations(
            row_label=row_label,
            coverage=_mapping(row.get("coverage")),
            required_flags=FX_INSERT_REQUIRED_COVERAGE_FLAGS,
            violations=violations,
        )
    return len(rows)


def build_fx_slots_inserts_physical_readback_contract() -> Dict[str, Any]:
    """Return the non-executing artifact contract for FX slot and insert proof."""

    return {
        "type": "fx_slots_inserts_physical_readback_contract",
        "generated_at": _utc_now(),
        "method_id": FX_SLOTS_INSERTS_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        "schema_id": FX_SLOTS_INSERTS_PHYSICAL_READBACK_SCHEMA_ID,
        "status": "contract_ready_physical_fixture_missing",
        "surface_id": FX_SLOTS_INSERTS_SURFACE_ID,
        "remediation_action_id": FX_SLOTS_INSERTS_REMEDIATION_ACTION_ID,
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
            "type": FX_SLOTS_INSERTS_PHYSICAL_READBACK_ARTIFACT_TYPE,
            "schema_id": FX_SLOTS_INSERTS_PHYSICAL_READBACK_SCHEMA_ID,
            "required_top_level_fields": list(FX_SLOTS_INSERTS_REQUIRED_TOP_LEVEL_FIELDS),
            "required_fx_slot_fields": list(FX_SLOT_REQUIRED_FIELDS),
            "required_parameter_fields": list(FX_PARAMETER_REQUIRED_FIELDS),
            "required_parameter_safe_range_fields": list(FX_PARAMETER_SAFE_RANGE_FIELDS),
            "required_parameter_schema_fields": list(FX_PARAMETER_SCHEMA_REQUIRED_FIELDS),
            "required_fx_slot_coverage_flags": list(FX_SLOT_REQUIRED_COVERAGE_FLAGS),
            "required_insert_fields": list(FX_INSERT_REQUIRED_FIELDS),
            "required_insert_coverage_flags": list(FX_INSERT_REQUIRED_COVERAGE_FLAGS),
            "required_false_safety_flags": list(
                FX_SLOTS_INSERTS_REQUIRED_FALSE_SAFETY_FLAGS
            ),
            "allowed_sources": list(FX_SLOTS_INSERTS_PHYSICAL_READBACK_SOURCES),
            "accepted_row_statuses": list(FX_SLOTS_INSERTS_VERIFIED_PHYSICAL_STATUSES),
            "allowed_insert_target_types": list(FX_INSERT_TARGET_TYPES),
            "allowed_insert_positions": list(FX_INSERT_POSITIONS),
            "forbidden_mutation_evidence_fields": list(
                FX_SLOTS_INSERTS_FORBIDDEN_MUTATION_EVIDENCE_FIELDS
            ),
            "minimum_fx_slot_count": MINIMUM_FX_SLOT_COUNT,
            "minimum_fx_parameter_count": MINIMUM_FX_PARAMETER_COUNT,
            "minimum_insert_count": MINIMUM_FX_INSERT_COUNT,
        },
        "summary": {
            "physical_readback_surface_id": FX_SLOTS_INSERTS_SURFACE_ID,
            "physical_readback_action_id": FX_SLOTS_INSERTS_REMEDIATION_ACTION_ID,
            "artifact_required": True,
            "physical_fixture_gap_remains": True,
            "accepted_as_physical_fixture_candidate": False,
            "live_authority_granted": False,
            "promotion_families_requiring_surface": [
                "fx_send_insert",
                "routing",
                "mix_balance",
                "performance_adaptation",
            ],
            "claim": (
                "contract only; a separate operator-run FX slot and insert "
                "physical readback artifact is still required"
            ),
        },
    }


def validate_fx_slots_inserts_physical_readback_artifact(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate a future FX slots/inserts artifact without granting authority."""

    violations: list[str] = []
    artifact = _mapping(payload)
    _append_required_field_violations(artifact, violations)
    _append_identity_violations(artifact, violations)
    _append_safety_violations(artifact, violations)
    fx_slot_count = _append_fx_slot_violations(artifact, violations)
    insert_count = _append_insert_violations(artifact, violations)

    status = (
        "valid_physical_readback_artifact"
        if not violations
        else "invalid_physical_readback_artifact"
    )
    validation = FxSlotsInsertsPhysicalReadbackValidation(
        method_id=FX_SLOTS_INSERTS_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        schema_id=FX_SLOTS_INSERTS_PHYSICAL_READBACK_SCHEMA_ID,
        surface_id=str(artifact.get("surface_id") or ""),
        remediation_action_id=str(artifact.get("remediation_action_id") or ""),
        status=status,
        accepted_as_physical_fixture_candidate=not violations,
        live_authority_granted=False,
        full_autonomous_ready=False,
        fx_slot_count=fx_slot_count,
        insert_count=insert_count,
        violations=tuple(sorted(set(violations))),
        evidence={
            "source": artifact.get("source"),
            "captured_at": artifact.get("captured_at"),
            "console_model": artifact.get("console_model"),
            "fx_slot_count": artifact.get("fx_slot_count"),
            "insert_count": artifact.get("insert_count"),
            "validated_at": _utc_now(),
        },
    )
    return validation.to_dict()
