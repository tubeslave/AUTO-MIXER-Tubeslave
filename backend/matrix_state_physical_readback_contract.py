"""Contract for matrix-state physical readback artifacts.

This module is transport-free. It defines the JSON-like artifact shape a future
operator-run physical readback must produce before the ``matrix_state`` read
surface can be treated as physical fixture evidence for matrix balance, routing,
FX send, and master-bus promotion decisions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple


MATRIX_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID = (
    "matrix_state_physical_readback_contract"
)
MATRIX_STATE_PHYSICAL_READBACK_SCHEMA_ID = "matrix_state_physical_readback.v1"
MATRIX_STATE_PHYSICAL_READBACK_ARTIFACT_TYPE = "matrix_state_physical_readback"
MATRIX_STATE_SURFACE_ID = "matrix_state"
MATRIX_STATE_REMEDIATION_ACTION_ID = "remediate_matrix_state"

MATRIX_STATE_REQUIRED_TOP_LEVEL_FIELDS = (
    "type",
    "schema_id",
    "surface_id",
    "remediation_action_id",
    "captured_at",
    "source",
    "console_model",
    "matrix_count",
    "matrices",
    "safety",
)
MATRIX_STATE_REQUIRED_MATRIX_FIELDS = (
    "matrix_id",
    "matrix_label",
    "status",
    "source",
    "captured_at",
    "controls",
    "eq",
    "dynamics",
    "inserts",
    "metering",
    "coverage",
)
MATRIX_STATE_REQUIRED_CONTROL_FIELDS = (
    "fader_db",
    "mute",
    "pan",
)
MATRIX_STATE_REQUIRED_EQ_FIELDS = (
    "enabled",
    "bands",
)
MATRIX_STATE_REQUIRED_EQ_BAND_FIELDS = (
    "band",
    "frequency_hz",
    "gain_db",
    "q",
)
MATRIX_STATE_REQUIRED_DYNAMICS_FIELDS = (
    "enabled",
    "model",
    "threshold_db",
    "ratio",
    "knee",
    "detector",
    "attack_ms",
    "release_ms",
    "makeup_db",
    "mix_pct",
    "gain_reduction_db",
)
MATRIX_STATE_REQUIRED_INSERT_FIELDS = (
    "pre_insert_on",
    "pre_insert_slot",
    "pre_insert_status",
    "post_insert_on",
    "post_insert_slot",
    "post_insert_status",
    "post_insert_mode",
    "post_insert_weight",
)
MATRIX_STATE_REQUIRED_METERING_FIELDS = (
    "lufs",
    "true_peak_dbtp",
    "headroom_db",
)
MATRIX_STATE_REQUIRED_COVERAGE_FLAGS = (
    "controls",
    "eq",
    "dynamics",
    "inserts",
    "metering",
    "cache_matches_physical_readback",
)
MATRIX_STATE_PHYSICAL_READBACK_SOURCES = (
    "physical_readback",
    "wing_physical_readback",
    "dlive_physical_readback",
)
MATRIX_STATE_VERIFIED_PHYSICAL_STATUSES = (
    "verified_from_physical_readback",
)
MATRIX_STATE_REQUIRED_FALSE_SAFETY_FLAGS = (
    "live_write",
    "mutates_console_state",
    "live_widening_allowed",
    "full_autonomous_ready",
)
MATRIX_STATE_FORBIDDEN_MUTATION_EVIDENCE_FIELDS = (
    "write_events",
    "manual_write_events",
    "apply_result",
    "rollback_result",
    "proposal_apply_result",
    "matrix_write_events",
    "matrix_send_write_events",
    "matrix_eq_write_events",
    "matrix_dynamics_write_events",
    "matrix_insert_write_events",
    "routing_write_events",
    "fx_send_write_events",
)
MINIMUM_MATRIX_EQ_BAND_COUNT = 8


@dataclass(frozen=True)
class MatrixStatePhysicalReadbackValidation:
    method_id: str
    schema_id: str
    surface_id: str
    remediation_action_id: str
    status: str
    accepted_as_physical_fixture_candidate: bool
    live_authority_granted: bool
    full_autonomous_ready: bool
    matrix_count: int
    violations: Tuple[str, ...]
    evidence: Dict[str, Any]

    def __post_init__(self) -> None:
        if self.live_authority_granted:
            raise ValueError("Matrix-state readback validation must not grant live authority")
        if self.full_autonomous_ready:
            raise ValueError(
                "Matrix-state readback validation must not claim full autonomous readiness"
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
    for key in MATRIX_STATE_REQUIRED_TOP_LEVEL_FIELDS:
        if key not in payload:
            violations.append(f"missing_top_level_field:{key}")


def _append_identity_violations(payload: Mapping[str, Any], violations: list[str]) -> None:
    if payload.get("type") != MATRIX_STATE_PHYSICAL_READBACK_ARTIFACT_TYPE:
        violations.append("artifact_type_mismatch")
    if payload.get("schema_id") != MATRIX_STATE_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("schema_id_mismatch")
    if payload.get("surface_id") != MATRIX_STATE_SURFACE_ID:
        violations.append("surface_id_mismatch")
    if payload.get("remediation_action_id") != MATRIX_STATE_REMEDIATION_ACTION_ID:
        violations.append("remediation_action_id_mismatch")
    if payload.get("source") not in MATRIX_STATE_PHYSICAL_READBACK_SOURCES:
        violations.append("source_not_physical_readback")
    if not _string(payload.get("captured_at")):
        violations.append("captured_at_missing")
    if not _string(payload.get("console_model")):
        violations.append("console_model_missing")


def _append_safety_violations(payload: Mapping[str, Any], violations: list[str]) -> None:
    safety = _mapping(payload.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("safety_read_only_not_true")
    for key in MATRIX_STATE_REQUIRED_FALSE_SAFETY_FLAGS:
        if safety.get(key) is not False:
            violations.append(f"safety_{key}_not_false")
        if _bool_true(payload.get(key)):
            violations.append(f"top_level_{key}_not_false")
    for key in MATRIX_STATE_FORBIDDEN_MUTATION_EVIDENCE_FIELDS:
        if key in payload:
            violations.append(f"mutation_evidence_field_present:{key}")


def _append_nested_field_violations(
    *,
    matrix_index: int,
    mapping: Mapping[str, Any],
    required_fields: Tuple[str, ...],
    label: str,
    violations: list[str],
) -> None:
    for key in required_fields:
        if key not in mapping:
            violations.append(f"matrix_{matrix_index}_{label}_missing_field:{key}")


def _append_numeric_field_violations(
    *,
    matrix_index: int,
    mapping: Mapping[str, Any],
    fields: Tuple[str, ...],
    label: str,
    violations: list[str],
) -> None:
    for key in fields:
        if key in mapping and not _finite_number(mapping.get(key)):
            violations.append(f"matrix_{matrix_index}_{label}_{key}_not_number")


def _append_enabled_field_violation(
    *,
    matrix_index: int,
    mapping: Mapping[str, Any],
    label: str,
    violations: list[str],
) -> None:
    if "enabled" in mapping and not _bool_or_0_1(mapping.get("enabled")):
        violations.append(f"matrix_{matrix_index}_{label}_enabled_not_bool_or_0_1")


def _append_eq_band_violations(
    *,
    matrix_index: int,
    eq: Mapping[str, Any],
    violations: list[str],
) -> None:
    bands = eq.get("bands")
    if not isinstance(bands, list) or len(bands) < MINIMUM_MATRIX_EQ_BAND_COUNT:
        violations.append(f"matrix_{matrix_index}_eq_bands_missing")
        return
    seen_bands: set[int] = set()
    for band_index, band_value in enumerate(bands):
        band = _mapping(band_value)
        row_label = f"matrix_{matrix_index}_eq_band_{band_index}"
        if not band:
            violations.append(f"{row_label}_not_object")
            continue
        for key in MATRIX_STATE_REQUIRED_EQ_BAND_FIELDS:
            if key not in band:
                violations.append(f"{row_label}_missing_field:{key}")
        band_id = band.get("band")
        if not isinstance(band_id, int) or band_id < 1:
            violations.append(f"{row_label}_band_invalid")
        elif band_id in seen_bands:
            violations.append(f"{row_label}_band_duplicate")
        else:
            seen_bands.add(band_id)
        for key in ("frequency_hz", "gain_db", "q"):
            if key in band and not _finite_number(band.get(key)):
                violations.append(f"{row_label}_{key}_not_number")


def _append_control_violations(
    *,
    matrix_index: int,
    controls: Mapping[str, Any],
    violations: list[str],
) -> None:
    _append_nested_field_violations(
        matrix_index=matrix_index,
        mapping=controls,
        required_fields=MATRIX_STATE_REQUIRED_CONTROL_FIELDS,
        label="controls",
        violations=violations,
    )
    for key in ("fader_db", "pan"):
        if key in controls and not _finite_number(controls.get(key)):
            violations.append(f"matrix_{matrix_index}_controls_{key}_not_number")
    if "mute" in controls and not _bool_or_0_1(controls.get("mute")):
        violations.append(f"matrix_{matrix_index}_controls_mute_not_bool_or_0_1")


def _append_eq_violations(
    *,
    matrix_index: int,
    eq: Mapping[str, Any],
    violations: list[str],
) -> None:
    _append_nested_field_violations(
        matrix_index=matrix_index,
        mapping=eq,
        required_fields=MATRIX_STATE_REQUIRED_EQ_FIELDS,
        label="eq",
        violations=violations,
    )
    _append_enabled_field_violation(
        matrix_index=matrix_index,
        mapping=eq,
        label="eq",
        violations=violations,
    )
    _append_eq_band_violations(matrix_index=matrix_index, eq=eq, violations=violations)


def _append_dynamics_violations(
    *,
    matrix_index: int,
    dynamics: Mapping[str, Any],
    violations: list[str],
) -> None:
    _append_nested_field_violations(
        matrix_index=matrix_index,
        mapping=dynamics,
        required_fields=MATRIX_STATE_REQUIRED_DYNAMICS_FIELDS,
        label="dynamics",
        violations=violations,
    )
    _append_enabled_field_violation(
        matrix_index=matrix_index,
        mapping=dynamics,
        label="dynamics",
        violations=violations,
    )
    for key in ("model", "detector"):
        if key in dynamics and not _string(dynamics.get(key)):
            violations.append(f"matrix_{matrix_index}_dynamics_{key}_missing")
    _append_numeric_field_violations(
        matrix_index=matrix_index,
        mapping=dynamics,
        fields=(
            "threshold_db",
            "ratio",
            "knee",
            "attack_ms",
            "release_ms",
            "makeup_db",
            "mix_pct",
            "gain_reduction_db",
        ),
        label="dynamics",
        violations=violations,
    )


def _append_insert_violations(
    *,
    matrix_index: int,
    inserts: Mapping[str, Any],
    violations: list[str],
) -> None:
    _append_nested_field_violations(
        matrix_index=matrix_index,
        mapping=inserts,
        required_fields=MATRIX_STATE_REQUIRED_INSERT_FIELDS,
        label="inserts",
        violations=violations,
    )
    for key in ("pre_insert_on", "post_insert_on"):
        if key in inserts and not _bool_or_0_1(inserts.get(key)):
            violations.append(f"matrix_{matrix_index}_inserts_{key}_not_bool_or_0_1")
    if "post_insert_weight" in inserts and not _finite_number(inserts.get("post_insert_weight")):
        violations.append(f"matrix_{matrix_index}_inserts_post_insert_weight_not_number")
    for key in (
        "pre_insert_slot",
        "pre_insert_status",
        "post_insert_slot",
        "post_insert_status",
        "post_insert_mode",
    ):
        if key in inserts and not _string(inserts.get(key)):
            violations.append(f"matrix_{matrix_index}_inserts_{key}_missing")


def _append_metering_violations(
    *,
    matrix_index: int,
    metering: Mapping[str, Any],
    violations: list[str],
) -> None:
    _append_nested_field_violations(
        matrix_index=matrix_index,
        mapping=metering,
        required_fields=MATRIX_STATE_REQUIRED_METERING_FIELDS,
        label="metering",
        violations=violations,
    )
    _append_numeric_field_violations(
        matrix_index=matrix_index,
        mapping=metering,
        fields=MATRIX_STATE_REQUIRED_METERING_FIELDS,
        label="metering",
        violations=violations,
    )


def _append_coverage_violations(
    *,
    matrix_index: int,
    coverage: Mapping[str, Any],
    violations: list[str],
) -> None:
    _append_nested_field_violations(
        matrix_index=matrix_index,
        mapping=coverage,
        required_fields=MATRIX_STATE_REQUIRED_COVERAGE_FLAGS,
        label="coverage",
        violations=violations,
    )
    for key in MATRIX_STATE_REQUIRED_COVERAGE_FLAGS:
        if coverage.get(key) is not True:
            violations.append(f"matrix_{matrix_index}_coverage_{key}_not_true")


def _append_matrix_violations(payload: Mapping[str, Any], violations: list[str]) -> int:
    rows = payload.get("matrices")
    if not isinstance(rows, list) or not rows:
        violations.append("matrices_missing")
        return 0
    if not isinstance(payload.get("matrix_count"), int) or payload.get("matrix_count") != len(rows):
        violations.append("matrix_count_mismatch")
    seen_ids: set[int] = set()
    for index, row_value in enumerate(rows):
        row = _mapping(row_value)
        if not row:
            violations.append(f"matrix_{index}_not_object")
            continue
        for key in MATRIX_STATE_REQUIRED_MATRIX_FIELDS:
            if key not in row:
                violations.append(f"matrix_{index}_missing_field:{key}")
        matrix_id = row.get("matrix_id")
        if not isinstance(matrix_id, int) or matrix_id < 1:
            violations.append(f"matrix_{index}_matrix_id_invalid")
        elif matrix_id in seen_ids:
            violations.append(f"matrix_{index}_matrix_id_duplicate")
        else:
            seen_ids.add(matrix_id)
        if not _string(row.get("matrix_label")):
            violations.append(f"matrix_{index}_matrix_label_missing")
        if row.get("status") not in MATRIX_STATE_VERIFIED_PHYSICAL_STATUSES:
            violations.append(f"matrix_{index}_status_not_physical_verified")
        if row.get("source") not in MATRIX_STATE_PHYSICAL_READBACK_SOURCES:
            violations.append(f"matrix_{index}_source_not_physical_readback")
        if not _string(row.get("captured_at")):
            violations.append(f"matrix_{index}_captured_at_missing")
        _append_control_violations(
            matrix_index=index,
            controls=_mapping(row.get("controls")),
            violations=violations,
        )
        _append_eq_violations(
            matrix_index=index,
            eq=_mapping(row.get("eq")),
            violations=violations,
        )
        _append_dynamics_violations(
            matrix_index=index,
            dynamics=_mapping(row.get("dynamics")),
            violations=violations,
        )
        _append_insert_violations(
            matrix_index=index,
            inserts=_mapping(row.get("inserts")),
            violations=violations,
        )
        _append_metering_violations(
            matrix_index=index,
            metering=_mapping(row.get("metering")),
            violations=violations,
        )
        _append_coverage_violations(
            matrix_index=index,
            coverage=_mapping(row.get("coverage")),
            violations=violations,
        )
    return len(rows)


def build_matrix_state_physical_readback_contract() -> Dict[str, Any]:
    """Return the non-executing artifact contract for matrix-state proof."""

    return {
        "type": "matrix_state_physical_readback_contract",
        "generated_at": _utc_now(),
        "method_id": MATRIX_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        "schema_id": MATRIX_STATE_PHYSICAL_READBACK_SCHEMA_ID,
        "status": "contract_ready_physical_fixture_missing",
        "surface_id": MATRIX_STATE_SURFACE_ID,
        "remediation_action_id": MATRIX_STATE_REMEDIATION_ACTION_ID,
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
            "type": MATRIX_STATE_PHYSICAL_READBACK_ARTIFACT_TYPE,
            "schema_id": MATRIX_STATE_PHYSICAL_READBACK_SCHEMA_ID,
            "required_top_level_fields": list(MATRIX_STATE_REQUIRED_TOP_LEVEL_FIELDS),
            "required_matrix_fields": list(MATRIX_STATE_REQUIRED_MATRIX_FIELDS),
            "required_control_fields": list(MATRIX_STATE_REQUIRED_CONTROL_FIELDS),
            "required_eq_fields": list(MATRIX_STATE_REQUIRED_EQ_FIELDS),
            "required_eq_band_fields": list(MATRIX_STATE_REQUIRED_EQ_BAND_FIELDS),
            "required_dynamics_fields": list(MATRIX_STATE_REQUIRED_DYNAMICS_FIELDS),
            "required_insert_fields": list(MATRIX_STATE_REQUIRED_INSERT_FIELDS),
            "required_metering_fields": list(MATRIX_STATE_REQUIRED_METERING_FIELDS),
            "required_coverage_flags": list(MATRIX_STATE_REQUIRED_COVERAGE_FLAGS),
            "required_false_safety_flags": list(MATRIX_STATE_REQUIRED_FALSE_SAFETY_FLAGS),
            "allowed_sources": list(MATRIX_STATE_PHYSICAL_READBACK_SOURCES),
            "accepted_matrix_statuses": list(MATRIX_STATE_VERIFIED_PHYSICAL_STATUSES),
            "forbidden_mutation_evidence_fields": list(
                MATRIX_STATE_FORBIDDEN_MUTATION_EVIDENCE_FIELDS
            ),
            "minimum_matrix_count": 1,
            "minimum_eq_band_count": MINIMUM_MATRIX_EQ_BAND_COUNT,
        },
        "summary": {
            "physical_readback_surface_id": MATRIX_STATE_SURFACE_ID,
            "physical_readback_action_id": MATRIX_STATE_REMEDIATION_ACTION_ID,
            "artifact_required": True,
            "physical_fixture_gap_remains": True,
            "accepted_as_physical_fixture_candidate": False,
            "live_authority_granted": False,
            "promotion_families_requiring_surface": [
                "bus_group_matrix",
                "fx_send_insert",
                "master_bus",
                "mix_balance",
                "routing",
                "performance_adaptation",
            ],
            "claim": (
                "contract only; a separate operator-run matrix physical "
                "readback artifact is still required"
            ),
        },
    }


def validate_matrix_state_physical_readback_artifact(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate a future matrix-state artifact without granting authority."""

    violations: list[str] = []
    artifact = _mapping(payload)
    _append_required_field_violations(artifact, violations)
    _append_identity_violations(artifact, violations)
    _append_safety_violations(artifact, violations)
    matrix_count = _append_matrix_violations(artifact, violations)

    status = (
        "valid_physical_readback_artifact"
        if not violations
        else "invalid_physical_readback_artifact"
    )
    validation = MatrixStatePhysicalReadbackValidation(
        method_id=MATRIX_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        schema_id=MATRIX_STATE_PHYSICAL_READBACK_SCHEMA_ID,
        surface_id=str(artifact.get("surface_id") or ""),
        remediation_action_id=str(artifact.get("remediation_action_id") or ""),
        status=status,
        accepted_as_physical_fixture_candidate=not violations,
        live_authority_granted=False,
        full_autonomous_ready=False,
        matrix_count=matrix_count,
        violations=tuple(sorted(set(violations))),
        evidence={
            "source": artifact.get("source"),
            "captured_at": artifact.get("captured_at"),
            "console_model": artifact.get("console_model"),
            "matrix_count": artifact.get("matrix_count"),
            "validated_at": _utc_now(),
        },
    )
    return validation.to_dict()
