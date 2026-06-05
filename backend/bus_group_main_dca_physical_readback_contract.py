"""Contract for bus/group/main/DCA physical readback artifacts.

This module is transport-free. It defines the JSON-like artifact shape a future
operator-run physical readback must produce before the
``bus_group_main_dca_state`` read surface can be treated as physical fixture
evidence for bus/group balance, DCA, and main-bus promotion decisions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple


BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_CONTRACT_METHOD_ID = (
    "bus_group_main_dca_physical_readback_contract"
)
BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_SCHEMA_ID = "bus_group_main_dca_physical_readback.v1"
BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_ARTIFACT_TYPE = "bus_group_main_dca_physical_readback"
BUS_GROUP_MAIN_DCA_SURFACE_ID = "bus_group_main_dca_state"
BUS_GROUP_MAIN_DCA_REMEDIATION_ACTION_ID = "remediate_bus_group_main_dca_state"

BUS_GROUP_MAIN_DCA_REQUIRED_TOP_LEVEL_FIELDS = (
    "type",
    "schema_id",
    "surface_id",
    "remediation_action_id",
    "captured_at",
    "source",
    "console_model",
    "bus_count",
    "main_count",
    "dca_count",
    "buses",
    "mains",
    "dcas",
    "safety",
)
BUS_GROUP_MAIN_DCA_REQUIRED_BUS_FIELDS = (
    "bus_id",
    "bus_label",
    "status",
    "source",
    "captured_at",
    "controls",
    "assignments",
    "metering",
    "coverage",
)
BUS_GROUP_MAIN_DCA_REQUIRED_MAIN_FIELDS = (
    "main_id",
    "main_label",
    "status",
    "source",
    "captured_at",
    "controls",
    "metering",
    "coverage",
)
BUS_GROUP_MAIN_DCA_REQUIRED_DCA_FIELDS = (
    "dca_id",
    "dca_label",
    "status",
    "source",
    "captured_at",
    "controls",
    "assignments",
    "coverage",
)
BUS_GROUP_MAIN_DCA_REQUIRED_BUS_CONTROL_FIELDS = (
    "fader_db",
    "mute",
    "pan",
    "width",
    "mono",
)
BUS_GROUP_MAIN_DCA_REQUIRED_MAIN_CONTROL_FIELDS = (
    "fader_db",
    "mute",
    "pan",
)
BUS_GROUP_MAIN_DCA_REQUIRED_DCA_CONTROL_FIELDS = (
    "fader_db",
    "mute",
    "monitor",
)
BUS_GROUP_MAIN_DCA_REQUIRED_BUS_ASSIGNMENT_FIELDS = (
    "dca_assignments",
    "main_send_on",
    "main_send_level_db",
    "main_send_pre",
)
BUS_GROUP_MAIN_DCA_REQUIRED_DCA_ASSIGNMENT_FIELDS = (
    "member_channels",
    "member_buses",
)
BUS_GROUP_MAIN_DCA_REQUIRED_METERING_FIELDS = (
    "lufs",
    "true_peak_dbtp",
    "headroom_db",
)
BUS_GROUP_MAIN_DCA_REQUIRED_BUS_COVERAGE_FLAGS = (
    "controls",
    "assignments",
    "metering",
    "cache_matches_physical_readback",
)
BUS_GROUP_MAIN_DCA_REQUIRED_MAIN_COVERAGE_FLAGS = (
    "controls",
    "metering",
    "cache_matches_physical_readback",
)
BUS_GROUP_MAIN_DCA_REQUIRED_DCA_COVERAGE_FLAGS = (
    "identity",
    "controls",
    "assignments",
    "cache_matches_physical_readback",
)
BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_SOURCES = (
    "physical_readback",
    "wing_physical_readback",
    "dlive_physical_readback",
)
BUS_GROUP_MAIN_DCA_VERIFIED_PHYSICAL_STATUSES = (
    "verified_from_physical_readback",
)
BUS_GROUP_MAIN_DCA_REQUIRED_FALSE_SAFETY_FLAGS = (
    "live_write",
    "mutates_console_state",
    "live_widening_allowed",
    "full_autonomous_ready",
)
BUS_GROUP_MAIN_DCA_FORBIDDEN_MUTATION_EVIDENCE_FIELDS = (
    "write_events",
    "manual_write_events",
    "apply_result",
    "rollback_result",
    "proposal_apply_result",
    "bus_write_events",
    "group_write_events",
    "dca_write_events",
    "main_bus_write_events",
    "matrix_write_events",
)


@dataclass(frozen=True)
class BusGroupMainDcaPhysicalReadbackValidation:
    method_id: str
    schema_id: str
    surface_id: str
    remediation_action_id: str
    status: str
    accepted_as_physical_fixture_candidate: bool
    live_authority_granted: bool
    full_autonomous_ready: bool
    bus_count: int
    main_count: int
    dca_count: int
    violations: Tuple[str, ...]
    evidence: Dict[str, Any]

    def __post_init__(self) -> None:
        if self.live_authority_granted:
            raise ValueError("Bus/group/main/DCA readback validation must not grant live authority")
        if self.full_autonomous_ready:
            raise ValueError(
                "Bus/group/main/DCA readback validation must not claim full autonomous readiness"
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


def _int_list(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, int) and not isinstance(item, bool) and item >= 1
        for item in value
    )


def _append_required_field_violations(payload: Mapping[str, Any], violations: list[str]) -> None:
    for key in BUS_GROUP_MAIN_DCA_REQUIRED_TOP_LEVEL_FIELDS:
        if key not in payload:
            violations.append(f"missing_top_level_field:{key}")


def _append_identity_violations(payload: Mapping[str, Any], violations: list[str]) -> None:
    if payload.get("type") != BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_ARTIFACT_TYPE:
        violations.append("artifact_type_mismatch")
    if payload.get("schema_id") != BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("schema_id_mismatch")
    if payload.get("surface_id") != BUS_GROUP_MAIN_DCA_SURFACE_ID:
        violations.append("surface_id_mismatch")
    if payload.get("remediation_action_id") != BUS_GROUP_MAIN_DCA_REMEDIATION_ACTION_ID:
        violations.append("remediation_action_id_mismatch")
    if payload.get("source") not in BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_SOURCES:
        violations.append("source_not_physical_readback")
    if not _string(payload.get("captured_at")):
        violations.append("captured_at_missing")
    if not _string(payload.get("console_model")):
        violations.append("console_model_missing")


def _append_safety_violations(payload: Mapping[str, Any], violations: list[str]) -> None:
    safety = _mapping(payload.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("safety_read_only_not_true")
    for key in BUS_GROUP_MAIN_DCA_REQUIRED_FALSE_SAFETY_FLAGS:
        if safety.get(key) is not False:
            violations.append(f"safety_{key}_not_false")
        if _bool_true(payload.get(key)):
            violations.append(f"top_level_{key}_not_false")
    for key in BUS_GROUP_MAIN_DCA_FORBIDDEN_MUTATION_EVIDENCE_FIELDS:
        if key in payload:
            violations.append(f"mutation_evidence_field_present:{key}")


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


def _append_control_violations(
    *,
    row_label: str,
    controls: Mapping[str, Any],
    required_fields: Tuple[str, ...],
    bool_fields: Tuple[str, ...],
    violations: list[str],
) -> None:
    _append_nested_field_violations(
        row_label=row_label,
        mapping=controls,
        required_fields=required_fields,
        label="controls",
        violations=violations,
    )
    for key in required_fields:
        if key not in controls:
            continue
        if key in bool_fields:
            if not _bool_or_0_1(controls.get(key)):
                violations.append(f"{row_label}_controls_{key}_not_bool_or_0_1")
        elif not _finite_number(controls.get(key)):
            violations.append(f"{row_label}_controls_{key}_not_number")


def _append_assignment_violations(
    *,
    row_label: str,
    assignments: Mapping[str, Any],
    required_fields: Tuple[str, ...],
    bool_fields: Tuple[str, ...],
    list_fields: Tuple[str, ...],
    violations: list[str],
) -> None:
    _append_nested_field_violations(
        row_label=row_label,
        mapping=assignments,
        required_fields=required_fields,
        label="assignments",
        violations=violations,
    )
    for key in bool_fields:
        if key in assignments and not _bool_or_0_1(assignments.get(key)):
            violations.append(f"{row_label}_assignments_{key}_not_bool_or_0_1")
    for key in list_fields:
        if key in assignments and not _int_list(assignments.get(key)):
            violations.append(f"{row_label}_assignments_{key}_not_int_list")
    for key in ("main_send_level_db",):
        if key in assignments and not _finite_number(assignments.get(key)):
            violations.append(f"{row_label}_assignments_{key}_not_number")


def _append_metering_violations(
    *,
    row_label: str,
    metering: Mapping[str, Any],
    violations: list[str],
) -> None:
    _append_nested_field_violations(
        row_label=row_label,
        mapping=metering,
        required_fields=BUS_GROUP_MAIN_DCA_REQUIRED_METERING_FIELDS,
        label="metering",
        violations=violations,
    )
    for key in BUS_GROUP_MAIN_DCA_REQUIRED_METERING_FIELDS:
        if key in metering and not _finite_number(metering.get(key)):
            violations.append(f"{row_label}_metering_{key}_not_number")


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


def _append_row_identity_violations(
    *,
    row_label: str,
    row: Mapping[str, Any],
    id_key: str,
    label_key: str,
    seen_ids: set[int],
    violations: list[str],
) -> None:
    row_id = row.get(id_key)
    if not isinstance(row_id, int) or row_id < 1:
        violations.append(f"{row_label}_{id_key}_invalid")
    elif row_id in seen_ids:
        violations.append(f"{row_label}_{id_key}_duplicate")
    else:
        seen_ids.add(row_id)
    if not _string(row.get(label_key)):
        violations.append(f"{row_label}_{label_key}_missing")
    if row.get("status") not in BUS_GROUP_MAIN_DCA_VERIFIED_PHYSICAL_STATUSES:
        violations.append(f"{row_label}_status_not_physical_verified")
    if row.get("source") not in BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_SOURCES:
        violations.append(f"{row_label}_source_not_physical_readback")
    if not _string(row.get("captured_at")):
        violations.append(f"{row_label}_captured_at_missing")


def _append_bus_violations(payload: Mapping[str, Any], violations: list[str]) -> int:
    rows = payload.get("buses")
    if not isinstance(rows, list) or not rows:
        violations.append("buses_missing")
        return 0
    if not isinstance(payload.get("bus_count"), int) or payload.get("bus_count") != len(rows):
        violations.append("bus_count_mismatch")
    seen_ids: set[int] = set()
    for index, row_value in enumerate(rows):
        row = _mapping(row_value)
        row_label = f"bus_{index}"
        if not row:
            violations.append(f"{row_label}_not_object")
            continue
        for key in BUS_GROUP_MAIN_DCA_REQUIRED_BUS_FIELDS:
            if key not in row:
                violations.append(f"{row_label}_missing_field:{key}")
        _append_row_identity_violations(
            row_label=row_label,
            row=row,
            id_key="bus_id",
            label_key="bus_label",
            seen_ids=seen_ids,
            violations=violations,
        )
        _append_control_violations(
            row_label=row_label,
            controls=_mapping(row.get("controls")),
            required_fields=BUS_GROUP_MAIN_DCA_REQUIRED_BUS_CONTROL_FIELDS,
            bool_fields=("mute", "mono"),
            violations=violations,
        )
        _append_assignment_violations(
            row_label=row_label,
            assignments=_mapping(row.get("assignments")),
            required_fields=BUS_GROUP_MAIN_DCA_REQUIRED_BUS_ASSIGNMENT_FIELDS,
            bool_fields=("main_send_on", "main_send_pre"),
            list_fields=("dca_assignments",),
            violations=violations,
        )
        _append_metering_violations(
            row_label=row_label,
            metering=_mapping(row.get("metering")),
            violations=violations,
        )
        _append_coverage_violations(
            row_label=row_label,
            coverage=_mapping(row.get("coverage")),
            required_flags=BUS_GROUP_MAIN_DCA_REQUIRED_BUS_COVERAGE_FLAGS,
            violations=violations,
        )
    return len(rows)


def _append_main_violations(payload: Mapping[str, Any], violations: list[str]) -> int:
    rows = payload.get("mains")
    if not isinstance(rows, list) or not rows:
        violations.append("mains_missing")
        return 0
    if not isinstance(payload.get("main_count"), int) or payload.get("main_count") != len(rows):
        violations.append("main_count_mismatch")
    seen_ids: set[int] = set()
    for index, row_value in enumerate(rows):
        row = _mapping(row_value)
        row_label = f"main_{index}"
        if not row:
            violations.append(f"{row_label}_not_object")
            continue
        for key in BUS_GROUP_MAIN_DCA_REQUIRED_MAIN_FIELDS:
            if key not in row:
                violations.append(f"{row_label}_missing_field:{key}")
        _append_row_identity_violations(
            row_label=row_label,
            row=row,
            id_key="main_id",
            label_key="main_label",
            seen_ids=seen_ids,
            violations=violations,
        )
        _append_control_violations(
            row_label=row_label,
            controls=_mapping(row.get("controls")),
            required_fields=BUS_GROUP_MAIN_DCA_REQUIRED_MAIN_CONTROL_FIELDS,
            bool_fields=("mute",),
            violations=violations,
        )
        _append_metering_violations(
            row_label=row_label,
            metering=_mapping(row.get("metering")),
            violations=violations,
        )
        _append_coverage_violations(
            row_label=row_label,
            coverage=_mapping(row.get("coverage")),
            required_flags=BUS_GROUP_MAIN_DCA_REQUIRED_MAIN_COVERAGE_FLAGS,
            violations=violations,
        )
    return len(rows)


def _append_dca_violations(payload: Mapping[str, Any], violations: list[str]) -> int:
    rows = payload.get("dcas")
    if not isinstance(rows, list) or not rows:
        violations.append("dcas_missing")
        return 0
    if not isinstance(payload.get("dca_count"), int) or payload.get("dca_count") != len(rows):
        violations.append("dca_count_mismatch")
    seen_ids: set[int] = set()
    for index, row_value in enumerate(rows):
        row = _mapping(row_value)
        row_label = f"dca_{index}"
        if not row:
            violations.append(f"{row_label}_not_object")
            continue
        for key in BUS_GROUP_MAIN_DCA_REQUIRED_DCA_FIELDS:
            if key not in row:
                violations.append(f"{row_label}_missing_field:{key}")
        _append_row_identity_violations(
            row_label=row_label,
            row=row,
            id_key="dca_id",
            label_key="dca_label",
            seen_ids=seen_ids,
            violations=violations,
        )
        _append_control_violations(
            row_label=row_label,
            controls=_mapping(row.get("controls")),
            required_fields=BUS_GROUP_MAIN_DCA_REQUIRED_DCA_CONTROL_FIELDS,
            bool_fields=("mute", "monitor"),
            violations=violations,
        )
        _append_assignment_violations(
            row_label=row_label,
            assignments=_mapping(row.get("assignments")),
            required_fields=BUS_GROUP_MAIN_DCA_REQUIRED_DCA_ASSIGNMENT_FIELDS,
            bool_fields=(),
            list_fields=("member_channels", "member_buses"),
            violations=violations,
        )
        _append_coverage_violations(
            row_label=row_label,
            coverage=_mapping(row.get("coverage")),
            required_flags=BUS_GROUP_MAIN_DCA_REQUIRED_DCA_COVERAGE_FLAGS,
            violations=violations,
        )
    return len(rows)


def build_bus_group_main_dca_physical_readback_contract() -> Dict[str, Any]:
    """Return the non-executing artifact contract for bus/group/main/DCA proof."""

    return {
        "type": "bus_group_main_dca_physical_readback_contract",
        "generated_at": _utc_now(),
        "method_id": BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        "schema_id": BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_SCHEMA_ID,
        "status": "contract_ready_physical_fixture_missing",
        "surface_id": BUS_GROUP_MAIN_DCA_SURFACE_ID,
        "remediation_action_id": BUS_GROUP_MAIN_DCA_REMEDIATION_ACTION_ID,
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
            "type": BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_ARTIFACT_TYPE,
            "schema_id": BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_SCHEMA_ID,
            "required_top_level_fields": list(BUS_GROUP_MAIN_DCA_REQUIRED_TOP_LEVEL_FIELDS),
            "required_bus_fields": list(BUS_GROUP_MAIN_DCA_REQUIRED_BUS_FIELDS),
            "required_main_fields": list(BUS_GROUP_MAIN_DCA_REQUIRED_MAIN_FIELDS),
            "required_dca_fields": list(BUS_GROUP_MAIN_DCA_REQUIRED_DCA_FIELDS),
            "required_bus_control_fields": list(
                BUS_GROUP_MAIN_DCA_REQUIRED_BUS_CONTROL_FIELDS
            ),
            "required_main_control_fields": list(
                BUS_GROUP_MAIN_DCA_REQUIRED_MAIN_CONTROL_FIELDS
            ),
            "required_dca_control_fields": list(
                BUS_GROUP_MAIN_DCA_REQUIRED_DCA_CONTROL_FIELDS
            ),
            "required_bus_assignment_fields": list(
                BUS_GROUP_MAIN_DCA_REQUIRED_BUS_ASSIGNMENT_FIELDS
            ),
            "required_dca_assignment_fields": list(
                BUS_GROUP_MAIN_DCA_REQUIRED_DCA_ASSIGNMENT_FIELDS
            ),
            "required_metering_fields": list(BUS_GROUP_MAIN_DCA_REQUIRED_METERING_FIELDS),
            "required_bus_coverage_flags": list(
                BUS_GROUP_MAIN_DCA_REQUIRED_BUS_COVERAGE_FLAGS
            ),
            "required_main_coverage_flags": list(
                BUS_GROUP_MAIN_DCA_REQUIRED_MAIN_COVERAGE_FLAGS
            ),
            "required_dca_coverage_flags": list(
                BUS_GROUP_MAIN_DCA_REQUIRED_DCA_COVERAGE_FLAGS
            ),
            "required_false_safety_flags": list(BUS_GROUP_MAIN_DCA_REQUIRED_FALSE_SAFETY_FLAGS),
            "allowed_sources": list(BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_SOURCES),
            "accepted_surface_statuses": list(BUS_GROUP_MAIN_DCA_VERIFIED_PHYSICAL_STATUSES),
            "forbidden_mutation_evidence_fields": list(
                BUS_GROUP_MAIN_DCA_FORBIDDEN_MUTATION_EVIDENCE_FIELDS
            ),
            "minimum_bus_count": 1,
            "minimum_main_count": 1,
            "minimum_dca_count": 1,
        },
        "summary": {
            "physical_readback_surface_id": BUS_GROUP_MAIN_DCA_SURFACE_ID,
            "physical_readback_action_id": BUS_GROUP_MAIN_DCA_REMEDIATION_ACTION_ID,
            "artifact_required": True,
            "physical_fixture_gap_remains": True,
            "accepted_as_physical_fixture_candidate": False,
            "live_authority_granted": False,
            "promotion_families_requiring_surface": [
                "bus_group_matrix",
                "dca",
                "master_bus",
                "mix_balance",
                "performance_adaptation",
            ],
            "claim": (
                "contract only; a separate operator-run bus/group/main/DCA "
                "physical readback artifact is still required"
            ),
        },
    }


def validate_bus_group_main_dca_physical_readback_artifact(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate a future bus/group/main/DCA artifact without granting authority."""

    violations: list[str] = []
    artifact = _mapping(payload)
    _append_required_field_violations(artifact, violations)
    _append_identity_violations(artifact, violations)
    _append_safety_violations(artifact, violations)
    bus_count = _append_bus_violations(artifact, violations)
    main_count = _append_main_violations(artifact, violations)
    dca_count = _append_dca_violations(artifact, violations)

    status = (
        "valid_physical_readback_artifact"
        if not violations
        else "invalid_physical_readback_artifact"
    )
    validation = BusGroupMainDcaPhysicalReadbackValidation(
        method_id=BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        schema_id=BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_SCHEMA_ID,
        surface_id=str(artifact.get("surface_id") or ""),
        remediation_action_id=str(artifact.get("remediation_action_id") or ""),
        status=status,
        accepted_as_physical_fixture_candidate=not violations,
        live_authority_granted=False,
        full_autonomous_ready=False,
        bus_count=bus_count,
        main_count=main_count,
        dca_count=dca_count,
        violations=tuple(sorted(set(violations))),
        evidence={
            "source": artifact.get("source"),
            "captured_at": artifact.get("captured_at"),
            "console_model": artifact.get("console_model"),
            "bus_count": artifact.get("bus_count"),
            "main_count": artifact.get("main_count"),
            "dca_count": artifact.get("dca_count"),
            "validated_at": _utc_now(),
        },
    )
    return validation.to_dict()
