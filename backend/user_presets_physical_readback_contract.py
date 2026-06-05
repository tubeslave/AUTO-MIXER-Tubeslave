"""Contract for user-preset physical readback artifacts.

This module is transport-free. It defines the JSON-like artifact shape a future
operator-run physical readback must produce before the ``user_presets`` read
surface can be treated as scene/preset promotion evidence. The contract does
not load, store, or apply presets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple


USER_PRESETS_PHYSICAL_READBACK_CONTRACT_METHOD_ID = (
    "user_presets_physical_readback_contract"
)
USER_PRESETS_PHYSICAL_READBACK_SCHEMA_ID = "user_presets_physical_readback.v1"
USER_PRESETS_PHYSICAL_READBACK_ARTIFACT_TYPE = "user_presets_physical_readback"
USER_PRESETS_SURFACE_ID = "user_presets"
USER_PRESETS_REMEDIATION_ACTION_ID = "remediate_user_presets"

USER_PRESETS_REQUIRED_TOP_LEVEL_FIELDS = (
    "type",
    "schema_id",
    "surface_id",
    "remediation_action_id",
    "captured_at",
    "source",
    "console_model",
    "preset_count",
    "presets",
    "safety",
)
USER_PRESETS_REQUIRED_PRESET_FIELDS = (
    "preset_index",
    "preset_name",
    "preset_scope",
    "target_type",
    "target_label",
    "readback_status",
    "readback_source",
    "captured_at",
    "preset_recall_policy",
    "preset_store_policy",
    "destructive_recall_marker",
    "destructive_store_marker",
    "preset_content_fingerprint",
    "scoped_diff_marker",
    "rollback_reference",
    "coverage",
)
USER_PRESETS_REQUIRED_COVERAGE_FLAGS = (
    "catalog_identity",
    "scope",
    "target",
    "preset_content_fingerprint",
    "scoped_diff_marker",
    "rollback_reference",
    "cache_matches_physical_readback",
)
USER_PRESETS_PHYSICAL_READBACK_SOURCES = (
    "physical_readback",
    "wing_physical_readback",
    "dlive_physical_readback",
)
USER_PRESETS_VERIFIED_PHYSICAL_STATUSES = (
    "verified_from_physical_readback",
)
USER_PRESETS_REQUIRED_FALSE_SAFETY_FLAGS = (
    "live_write",
    "mutates_console_state",
    "live_widening_allowed",
    "full_autonomous_ready",
    "preset_recall_allowed",
    "preset_store_allowed",
)
USER_PRESETS_FORBIDDEN_MUTATION_EVIDENCE_FIELDS = (
    "write_events",
    "manual_write_events",
    "apply_result",
    "rollback_result",
    "proposal_apply_result",
    "preset_recall_events",
    "preset_store_events",
    "preset_load_events",
    "preset_save_events",
    "library_write_events",
    "scene_recall_events",
)
MINIMUM_USER_PRESET_COUNT = 1


@dataclass(frozen=True)
class UserPresetsPhysicalReadbackValidation:
    method_id: str
    schema_id: str
    surface_id: str
    remediation_action_id: str
    status: str
    accepted_as_physical_fixture_candidate: bool
    live_authority_granted: bool
    full_autonomous_ready: bool
    preset_count: int
    violations: Tuple[str, ...]
    evidence: Dict[str, Any]

    def __post_init__(self) -> None:
        if self.live_authority_granted:
            raise ValueError("User-preset validation must not grant live authority")
        if self.full_autonomous_ready:
            raise ValueError(
                "User-preset validation must not claim full autonomous readiness"
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


def _append_required_field_violations(payload: Mapping[str, Any], violations: list[str]) -> None:
    for key in USER_PRESETS_REQUIRED_TOP_LEVEL_FIELDS:
        if key not in payload:
            violations.append(f"missing_top_level_field:{key}")


def _append_identity_violations(payload: Mapping[str, Any], violations: list[str]) -> None:
    if payload.get("type") != USER_PRESETS_PHYSICAL_READBACK_ARTIFACT_TYPE:
        violations.append("artifact_type_mismatch")
    if payload.get("schema_id") != USER_PRESETS_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("schema_id_mismatch")
    if payload.get("surface_id") != USER_PRESETS_SURFACE_ID:
        violations.append("surface_id_mismatch")
    if payload.get("remediation_action_id") != USER_PRESETS_REMEDIATION_ACTION_ID:
        violations.append("remediation_action_id_mismatch")
    if payload.get("source") not in USER_PRESETS_PHYSICAL_READBACK_SOURCES:
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
            if key_text in USER_PRESETS_FORBIDDEN_MUTATION_EVIDENCE_FIELDS:
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
    for key in USER_PRESETS_REQUIRED_FALSE_SAFETY_FLAGS:
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
        required_fields=USER_PRESETS_REQUIRED_COVERAGE_FLAGS,
        label="coverage",
        violations=violations,
    )
    for key in USER_PRESETS_REQUIRED_COVERAGE_FLAGS:
        if coverage.get(key) is not True:
            violations.append(f"{row_label}_coverage_{key}_not_true")


def _append_physical_identity_violations(
    *,
    row_label: str,
    row: Mapping[str, Any],
    violations: list[str],
) -> None:
    if row.get("readback_status") not in USER_PRESETS_VERIFIED_PHYSICAL_STATUSES:
        violations.append(f"{row_label}_readback_status_not_physical_verified")
    if row.get("readback_source") not in USER_PRESETS_PHYSICAL_READBACK_SOURCES:
        violations.append(f"{row_label}_readback_source_not_physical_readback")
    if not _string(row.get("captured_at")):
        violations.append(f"{row_label}_captured_at_missing")


def _append_preset_violations(payload: Mapping[str, Any], violations: list[str]) -> int:
    rows = payload.get("presets")
    if not isinstance(rows, list) or len(rows) < MINIMUM_USER_PRESET_COUNT:
        violations.append("presets_missing")
        return 0
    if (
        not isinstance(payload.get("preset_count"), int)
        or payload.get("preset_count") != len(rows)
    ):
        violations.append("preset_count_mismatch")
    seen_indexes: set[int] = set()
    for index, row_value in enumerate(rows):
        row = _mapping(row_value)
        row_label = f"preset_{index}"
        if not row:
            violations.append(f"{row_label}_not_object")
            continue
        _append_nested_field_violations(
            row_label=row_label,
            mapping=row,
            required_fields=USER_PRESETS_REQUIRED_PRESET_FIELDS,
            label="field",
            violations=violations,
        )
        preset_index = row.get("preset_index")
        if not isinstance(preset_index, int) or preset_index < 1:
            violations.append(f"{row_label}_preset_index_invalid")
        elif preset_index in seen_indexes:
            violations.append(f"{row_label}_preset_index_duplicate")
        else:
            seen_indexes.add(preset_index)
        for key in ("preset_name", "preset_scope", "target_type", "target_label"):
            if not _string(row.get(key)):
                violations.append(f"{row_label}_{key}_missing")
        if row.get("preset_recall_policy") != "blocked_manual_only":
            violations.append(f"{row_label}_preset_recall_policy_not_blocked")
        if row.get("preset_store_policy") != "blocked_manual_only":
            violations.append(f"{row_label}_preset_store_policy_not_blocked")
        for key in (
            "destructive_recall_marker",
            "destructive_store_marker",
            "preset_content_fingerprint",
            "scoped_diff_marker",
            "rollback_reference",
        ):
            if not _string(row.get(key)):
                violations.append(f"{row_label}_{key}_missing")
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


def build_user_presets_physical_readback_contract() -> Dict[str, Any]:
    """Return the non-executing artifact contract for user-preset proof."""

    return {
        "type": "user_presets_physical_readback_contract",
        "generated_at": _utc_now(),
        "method_id": USER_PRESETS_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        "schema_id": USER_PRESETS_PHYSICAL_READBACK_SCHEMA_ID,
        "status": "contract_ready_physical_fixture_missing",
        "surface_id": USER_PRESETS_SURFACE_ID,
        "remediation_action_id": USER_PRESETS_REMEDIATION_ACTION_ID,
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "preset_recall_allowed": False,
            "preset_store_allowed": False,
        },
        "artifact_contract": {
            "type": USER_PRESETS_PHYSICAL_READBACK_ARTIFACT_TYPE,
            "schema_id": USER_PRESETS_PHYSICAL_READBACK_SCHEMA_ID,
            "required_top_level_fields": list(USER_PRESETS_REQUIRED_TOP_LEVEL_FIELDS),
            "required_preset_fields": list(USER_PRESETS_REQUIRED_PRESET_FIELDS),
            "required_coverage_flags": list(USER_PRESETS_REQUIRED_COVERAGE_FLAGS),
            "required_false_safety_flags": list(
                USER_PRESETS_REQUIRED_FALSE_SAFETY_FLAGS
            ),
            "allowed_sources": list(USER_PRESETS_PHYSICAL_READBACK_SOURCES),
            "accepted_row_statuses": list(USER_PRESETS_VERIFIED_PHYSICAL_STATUSES),
            "forbidden_mutation_evidence_fields": list(
                USER_PRESETS_FORBIDDEN_MUTATION_EVIDENCE_FIELDS
            ),
            "minimum_preset_count": MINIMUM_USER_PRESET_COUNT,
        },
        "summary": {
            "physical_readback_surface_id": USER_PRESETS_SURFACE_ID,
            "physical_readback_action_id": USER_PRESETS_REMEDIATION_ACTION_ID,
            "artifact_required": True,
            "physical_fixture_gap_remains": True,
            "accepted_as_physical_fixture_candidate": False,
            "live_authority_granted": False,
            "promotion_families_requiring_surface": [
                "scene_preset",
                "routing",
                "mix_balance",
                "performance_adaptation",
            ],
            "claim": (
                "contract only; a separate operator-run user-preset physical "
                "readback artifact is still required"
            ),
        },
    }


def validate_user_presets_physical_readback_artifact(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate a future user-preset artifact without granting authority."""

    violations: list[str] = []
    artifact = _mapping(payload)
    _append_required_field_violations(artifact, violations)
    _append_identity_violations(artifact, violations)
    _append_safety_violations(artifact, violations)
    preset_count = _append_preset_violations(artifact, violations)

    status = (
        "valid_physical_readback_artifact"
        if not violations
        else "invalid_physical_readback_artifact"
    )
    validation = UserPresetsPhysicalReadbackValidation(
        method_id=USER_PRESETS_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        schema_id=USER_PRESETS_PHYSICAL_READBACK_SCHEMA_ID,
        surface_id=str(artifact.get("surface_id") or ""),
        remediation_action_id=str(artifact.get("remediation_action_id") or ""),
        status=status,
        accepted_as_physical_fixture_candidate=not violations,
        live_authority_granted=False,
        full_autonomous_ready=False,
        preset_count=preset_count,
        violations=tuple(sorted(set(violations))),
        evidence={
            "source": artifact.get("source"),
            "captured_at": artifact.get("captured_at"),
            "console_model": artifact.get("console_model"),
            "preset_count": artifact.get("preset_count"),
        },
    )
    return validation.to_dict()
