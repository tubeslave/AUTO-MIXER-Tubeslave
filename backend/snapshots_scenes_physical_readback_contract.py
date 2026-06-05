"""Contract for snapshots/scenes physical readback artifacts.

This module is transport-free. It defines the JSON-like artifact shape a future
operator-run physical readback must produce before the ``snapshots_scenes`` read
surface can be treated as scene/preset promotion evidence. The contract does
not load, save, or recall scenes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple


SNAPSHOTS_SCENES_PHYSICAL_READBACK_CONTRACT_METHOD_ID = (
    "snapshots_scenes_physical_readback_contract"
)
SNAPSHOTS_SCENES_PHYSICAL_READBACK_SCHEMA_ID = "snapshots_scenes_physical_readback.v1"
SNAPSHOTS_SCENES_PHYSICAL_READBACK_ARTIFACT_TYPE = "snapshots_scenes_physical_readback"
SNAPSHOTS_SCENES_SURFACE_ID = "snapshots_scenes"
SNAPSHOTS_SCENES_REMEDIATION_ACTION_ID = "remediate_snapshots_scenes"

SNAPSHOTS_SCENES_REQUIRED_TOP_LEVEL_FIELDS = (
    "type",
    "schema_id",
    "surface_id",
    "remediation_action_id",
    "captured_at",
    "source",
    "console_model",
    "snapshot_count",
    "active_scene",
    "snapshots",
    "safety",
)
SNAPSHOTS_SCENES_REQUIRED_ACTIVE_SCENE_FIELDS = (
    "active_snapshot_index",
    "active_snapshot_name",
    "current_show",
    "readback_status",
    "readback_source",
    "captured_at",
    "coverage",
)
SNAPSHOTS_SCENES_REQUIRED_SNAPSHOT_FIELDS = (
    "snapshot_index",
    "snapshot_name",
    "readback_status",
    "readback_source",
    "captured_at",
    "active",
    "scene_load_policy",
    "scene_save_policy",
    "destructive_change_marker",
    "destructive_persistence_marker",
    "scene_content_fingerprint",
    "scoped_diff_marker",
    "rollback_reference",
    "coverage",
)
SNAPSHOTS_SCENES_REQUIRED_COVERAGE_FLAGS = (
    "catalog_identity",
    "active_scene",
    "scene_content_fingerprint",
    "scoped_diff_marker",
    "rollback_reference",
    "cache_matches_physical_readback",
)
SNAPSHOTS_SCENES_PHYSICAL_READBACK_SOURCES = (
    "physical_readback",
    "wing_physical_readback",
    "dlive_physical_readback",
)
SNAPSHOTS_SCENES_VERIFIED_PHYSICAL_STATUSES = (
    "verified_from_physical_readback",
)
SNAPSHOTS_SCENES_REQUIRED_FALSE_SAFETY_FLAGS = (
    "live_write",
    "mutates_console_state",
    "live_widening_allowed",
    "full_autonomous_ready",
    "scene_load_allowed",
    "scene_save_allowed",
)
SNAPSHOTS_SCENES_FORBIDDEN_MUTATION_EVIDENCE_FIELDS = (
    "write_events",
    "manual_write_events",
    "apply_result",
    "rollback_result",
    "proposal_apply_result",
    "scene_load_events",
    "scene_save_events",
    "scene_recall_events",
    "snapshot_load_events",
    "snapshot_save_events",
    "snapshot_recall_events",
    "show_load_events",
)
MINIMUM_SNAPSHOT_SCENE_COUNT = 1


@dataclass(frozen=True)
class SnapshotsScenesPhysicalReadbackValidation:
    method_id: str
    schema_id: str
    surface_id: str
    remediation_action_id: str
    status: str
    accepted_as_physical_fixture_candidate: bool
    live_authority_granted: bool
    full_autonomous_ready: bool
    snapshot_count: int
    violations: Tuple[str, ...]
    evidence: Dict[str, Any]

    def __post_init__(self) -> None:
        if self.live_authority_granted:
            raise ValueError("Snapshots/scenes validation must not grant live authority")
        if self.full_autonomous_ready:
            raise ValueError(
                "Snapshots/scenes validation must not claim full autonomous readiness"
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
    for key in SNAPSHOTS_SCENES_REQUIRED_TOP_LEVEL_FIELDS:
        if key not in payload:
            violations.append(f"missing_top_level_field:{key}")


def _append_identity_violations(payload: Mapping[str, Any], violations: list[str]) -> None:
    if payload.get("type") != SNAPSHOTS_SCENES_PHYSICAL_READBACK_ARTIFACT_TYPE:
        violations.append("artifact_type_mismatch")
    if payload.get("schema_id") != SNAPSHOTS_SCENES_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("schema_id_mismatch")
    if payload.get("surface_id") != SNAPSHOTS_SCENES_SURFACE_ID:
        violations.append("surface_id_mismatch")
    if payload.get("remediation_action_id") != SNAPSHOTS_SCENES_REMEDIATION_ACTION_ID:
        violations.append("remediation_action_id_mismatch")
    if payload.get("source") not in SNAPSHOTS_SCENES_PHYSICAL_READBACK_SOURCES:
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
            if key_text in SNAPSHOTS_SCENES_FORBIDDEN_MUTATION_EVIDENCE_FIELDS:
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
    for key in SNAPSHOTS_SCENES_REQUIRED_FALSE_SAFETY_FLAGS:
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
        required_fields=SNAPSHOTS_SCENES_REQUIRED_COVERAGE_FLAGS,
        label="coverage",
        violations=violations,
    )
    for key in SNAPSHOTS_SCENES_REQUIRED_COVERAGE_FLAGS:
        if coverage.get(key) is not True:
            violations.append(f"{row_label}_coverage_{key}_not_true")


def _append_physical_identity_violations(
    *,
    row_label: str,
    row: Mapping[str, Any],
    violations: list[str],
) -> None:
    if row.get("readback_status") not in SNAPSHOTS_SCENES_VERIFIED_PHYSICAL_STATUSES:
        violations.append(f"{row_label}_readback_status_not_physical_verified")
    if row.get("readback_source") not in SNAPSHOTS_SCENES_PHYSICAL_READBACK_SOURCES:
        violations.append(f"{row_label}_readback_source_not_physical_readback")
    if not _string(row.get("captured_at")):
        violations.append(f"{row_label}_captured_at_missing")


def _append_active_scene_violations(payload: Mapping[str, Any], violations: list[str]) -> None:
    active_scene = _mapping(payload.get("active_scene"))
    if not active_scene:
        violations.append("active_scene_missing")
        return
    _append_nested_field_violations(
        row_label="active_scene",
        mapping=active_scene,
        required_fields=SNAPSHOTS_SCENES_REQUIRED_ACTIVE_SCENE_FIELDS,
        label="field",
        violations=violations,
    )
    index = active_scene.get("active_snapshot_index")
    if index is not None and (not isinstance(index, int) or index < 1):
        violations.append("active_scene_active_snapshot_index_invalid")
    if active_scene.get("active_snapshot_name") is not None and not _string(
        active_scene.get("active_snapshot_name")
    ):
        violations.append("active_scene_active_snapshot_name_invalid")
    if active_scene.get("current_show") is not None and not _string(
        active_scene.get("current_show")
    ):
        violations.append("active_scene_current_show_invalid")
    _append_physical_identity_violations(
        row_label="active_scene",
        row=active_scene,
        violations=violations,
    )
    _append_coverage_violations(
        row_label="active_scene",
        coverage=_mapping(active_scene.get("coverage")),
        violations=violations,
    )


def _append_snapshot_violations(payload: Mapping[str, Any], violations: list[str]) -> int:
    rows = payload.get("snapshots")
    if not isinstance(rows, list) or len(rows) < MINIMUM_SNAPSHOT_SCENE_COUNT:
        violations.append("snapshots_missing")
        return 0
    if (
        not isinstance(payload.get("snapshot_count"), int)
        or payload.get("snapshot_count") != len(rows)
    ):
        violations.append("snapshot_count_mismatch")
    seen_indexes: set[int] = set()
    for index, row_value in enumerate(rows):
        row = _mapping(row_value)
        row_label = f"snapshot_{index}"
        if not row:
            violations.append(f"{row_label}_not_object")
            continue
        _append_nested_field_violations(
            row_label=row_label,
            mapping=row,
            required_fields=SNAPSHOTS_SCENES_REQUIRED_SNAPSHOT_FIELDS,
            label="field",
            violations=violations,
        )
        snapshot_index = row.get("snapshot_index")
        if not isinstance(snapshot_index, int) or snapshot_index < 1:
            violations.append(f"{row_label}_snapshot_index_invalid")
        elif snapshot_index in seen_indexes:
            violations.append(f"{row_label}_snapshot_index_duplicate")
        else:
            seen_indexes.add(snapshot_index)
        if not _string(row.get("snapshot_name")):
            violations.append(f"{row_label}_snapshot_name_missing")
        if not isinstance(row.get("active"), bool):
            violations.append(f"{row_label}_active_not_bool")
        if row.get("scene_load_policy") != "blocked_manual_only":
            violations.append(f"{row_label}_scene_load_policy_not_blocked")
        if row.get("scene_save_policy") != "blocked_manual_only":
            violations.append(f"{row_label}_scene_save_policy_not_blocked")
        for key in (
            "destructive_change_marker",
            "destructive_persistence_marker",
            "scene_content_fingerprint",
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


def build_snapshots_scenes_physical_readback_contract() -> Dict[str, Any]:
    """Return the non-executing artifact contract for scene/snapshot proof."""

    return {
        "type": "snapshots_scenes_physical_readback_contract",
        "generated_at": _utc_now(),
        "method_id": SNAPSHOTS_SCENES_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        "schema_id": SNAPSHOTS_SCENES_PHYSICAL_READBACK_SCHEMA_ID,
        "status": "contract_ready_physical_fixture_missing",
        "surface_id": SNAPSHOTS_SCENES_SURFACE_ID,
        "remediation_action_id": SNAPSHOTS_SCENES_REMEDIATION_ACTION_ID,
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "scene_load_allowed": False,
            "scene_save_allowed": False,
        },
        "artifact_contract": {
            "type": SNAPSHOTS_SCENES_PHYSICAL_READBACK_ARTIFACT_TYPE,
            "schema_id": SNAPSHOTS_SCENES_PHYSICAL_READBACK_SCHEMA_ID,
            "required_top_level_fields": list(SNAPSHOTS_SCENES_REQUIRED_TOP_LEVEL_FIELDS),
            "required_active_scene_fields": list(
                SNAPSHOTS_SCENES_REQUIRED_ACTIVE_SCENE_FIELDS
            ),
            "required_snapshot_fields": list(SNAPSHOTS_SCENES_REQUIRED_SNAPSHOT_FIELDS),
            "required_coverage_flags": list(SNAPSHOTS_SCENES_REQUIRED_COVERAGE_FLAGS),
            "required_false_safety_flags": list(
                SNAPSHOTS_SCENES_REQUIRED_FALSE_SAFETY_FLAGS
            ),
            "allowed_sources": list(SNAPSHOTS_SCENES_PHYSICAL_READBACK_SOURCES),
            "accepted_row_statuses": list(SNAPSHOTS_SCENES_VERIFIED_PHYSICAL_STATUSES),
            "forbidden_mutation_evidence_fields": list(
                SNAPSHOTS_SCENES_FORBIDDEN_MUTATION_EVIDENCE_FIELDS
            ),
            "minimum_snapshot_count": MINIMUM_SNAPSHOT_SCENE_COUNT,
        },
        "summary": {
            "physical_readback_surface_id": SNAPSHOTS_SCENES_SURFACE_ID,
            "physical_readback_action_id": SNAPSHOTS_SCENES_REMEDIATION_ACTION_ID,
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
                "contract only; a separate operator-run scene/snapshot physical "
                "readback artifact is still required"
            ),
        },
    }


def validate_snapshots_scenes_physical_readback_artifact(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate a future scene/snapshot artifact without granting authority."""

    violations: list[str] = []
    artifact = _mapping(payload)
    _append_required_field_violations(artifact, violations)
    _append_identity_violations(artifact, violations)
    _append_safety_violations(artifact, violations)
    _append_active_scene_violations(artifact, violations)
    snapshot_count = _append_snapshot_violations(artifact, violations)

    status = (
        "valid_physical_readback_artifact"
        if not violations
        else "invalid_physical_readback_artifact"
    )
    validation = SnapshotsScenesPhysicalReadbackValidation(
        method_id=SNAPSHOTS_SCENES_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        schema_id=SNAPSHOTS_SCENES_PHYSICAL_READBACK_SCHEMA_ID,
        surface_id=str(artifact.get("surface_id") or ""),
        remediation_action_id=str(artifact.get("remediation_action_id") or ""),
        status=status,
        accepted_as_physical_fixture_candidate=not violations,
        live_authority_granted=False,
        full_autonomous_ready=False,
        snapshot_count=snapshot_count,
        violations=tuple(sorted(set(violations))),
        evidence={
            "source": artifact.get("source"),
            "captured_at": artifact.get("captured_at"),
            "console_model": artifact.get("console_model"),
            "snapshot_count": artifact.get("snapshot_count"),
        },
    )
    return validation.to_dict()
