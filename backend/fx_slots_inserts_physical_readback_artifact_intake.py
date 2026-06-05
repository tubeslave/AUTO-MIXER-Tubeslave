"""Read-only intake for FX slots/inserts physical readback artifacts.

This layer inventories operator-captured JSON artifacts and validates them
against the FX slots/inserts physical readback contract. A valid artifact is
only a manual-review candidate; this module never closes fixture gaps or grants
live authority.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

try:
    from fx_slots_inserts_physical_readback_contract import (
        FX_SLOTS_INSERTS_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        FX_SLOTS_INSERTS_PHYSICAL_READBACK_SCHEMA_ID,
        FX_SLOTS_INSERTS_REMEDIATION_ACTION_ID,
        FX_SLOTS_INSERTS_SURFACE_ID,
        validate_fx_slots_inserts_physical_readback_artifact,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.fx_slots_inserts_physical_readback_contract import (
        FX_SLOTS_INSERTS_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        FX_SLOTS_INSERTS_PHYSICAL_READBACK_SCHEMA_ID,
        FX_SLOTS_INSERTS_REMEDIATION_ACTION_ID,
        FX_SLOTS_INSERTS_SURFACE_ID,
        validate_fx_slots_inserts_physical_readback_artifact,
    )


FX_SLOTS_INSERTS_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID = (
    "fx_slots_inserts_physical_readback_artifact_intake"
)
DEFAULT_FX_SLOTS_INSERTS_PHYSICAL_READBACK_ARTIFACT_ROOTS = (
    "artifacts/read_surface_physical_readback/fx_slots_and_inserts",
    "artifacts/fx_slots_inserts_physical_readback",
)
FX_SLOTS_INSERTS_INTAKE_STATUSES = {
    "operator_artifact_missing",
    "valid_operator_artifact_available",
    "invalid_operator_artifact_present",
}


@dataclass(frozen=True)
class FxSlotsInsertsPhysicalReadbackArtifactRecord:
    source_id: str
    status: str
    accepted_as_physical_fixture_candidate: bool
    fx_slot_count: int
    insert_count: int
    captured_at: str | None
    console_model: str | None
    source: str | None
    violations: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status not in {
            "valid_physical_readback_artifact",
            "invalid_physical_readback_artifact",
            "artifact_load_error",
        }:
            raise ValueError(f"Unknown FX slots/inserts artifact status: {self.status}")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["violations"] = list(self.violations)
        return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _safe_int(value: Any) -> int:
    return value if isinstance(value, int) and value >= 0 else 0


def _string_tuple(value: Any) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if isinstance(item, str) and item)


def _artifact_roots(artifact_roots: Iterable[str | Path] | None) -> Tuple[Path, ...]:
    roots = (
        tuple(artifact_roots)
        if artifact_roots is not None
        else DEFAULT_FX_SLOTS_INSERTS_PHYSICAL_READBACK_ARTIFACT_ROOTS
    )
    return tuple(Path(root) for root in roots)


def _discover_artifact_paths(artifact_roots: Iterable[str | Path] | None) -> Tuple[Path, ...]:
    paths: list[Path] = []
    for root in _artifact_roots(artifact_roots):
        if root.is_file() and root.suffix.lower() == ".json":
            paths.append(root)
        if not root.is_dir():
            continue
        paths.extend(path for path in root.rglob("*.json") if path.is_file())
    return tuple(sorted(dict.fromkeys(paths), key=lambda path: str(path)))


def _load_json_artifact(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("artifact JSON root must be an object")
    return payload


def _record_from_validation(
    source_id: str,
    validation: Mapping[str, Any],
) -> FxSlotsInsertsPhysicalReadbackArtifactRecord:
    evidence = _mapping(validation.get("evidence"))
    return FxSlotsInsertsPhysicalReadbackArtifactRecord(
        source_id=source_id,
        status=str(validation.get("status") or "invalid_physical_readback_artifact"),
        accepted_as_physical_fixture_candidate=(
            validation.get("accepted_as_physical_fixture_candidate") is True
        ),
        fx_slot_count=_safe_int(validation.get("fx_slot_count")),
        insert_count=_safe_int(validation.get("insert_count")),
        captured_at=_string(evidence.get("captured_at")),
        console_model=_string(evidence.get("console_model")),
        source=_string(evidence.get("source")),
        violations=_string_tuple(validation.get("violations")),
    )


def _record_from_load_error(
    source_id: str,
    exc: Exception,
) -> FxSlotsInsertsPhysicalReadbackArtifactRecord:
    return FxSlotsInsertsPhysicalReadbackArtifactRecord(
        source_id=source_id,
        status="artifact_load_error",
        accepted_as_physical_fixture_candidate=False,
        fx_slot_count=0,
        insert_count=0,
        captured_at=None,
        console_model=None,
        source=None,
        violations=(f"artifact_load_error:{type(exc).__name__}",),
    )


def _artifact_records(
    *,
    artifact_paths: Iterable[str | Path] | None,
    artifact_payloads: Iterable[Mapping[str, Any]] | None,
    artifact_roots: Iterable[str | Path] | None,
) -> Tuple[FxSlotsInsertsPhysicalReadbackArtifactRecord, ...]:
    records: list[FxSlotsInsertsPhysicalReadbackArtifactRecord] = []
    paths = (
        tuple(Path(path) for path in artifact_paths)
        if artifact_paths is not None
        else _discover_artifact_paths(artifact_roots)
    )
    for path in paths:
        try:
            validation = validate_fx_slots_inserts_physical_readback_artifact(
                _load_json_artifact(path)
            )
            records.append(_record_from_validation(str(path), validation))
        except Exception as exc:  # pragma: no cover - defensive artifact parser
            records.append(_record_from_load_error(str(path), exc))

    if artifact_payloads is not None:
        for index, payload in enumerate(artifact_payloads):
            validation = validate_fx_slots_inserts_physical_readback_artifact(payload)
            records.append(_record_from_validation(f"inline_payload_{index}", validation))

    return tuple(records)


def build_fx_slots_inserts_physical_readback_artifact_intake(
    *,
    artifact_paths: Iterable[str | Path] | None = None,
    artifact_payloads: Iterable[Mapping[str, Any]] | None = None,
    artifact_roots: Iterable[str | Path] | None = None,
) -> Dict[str, Any]:
    """Build a read-only intake snapshot for FX slots/inserts readbacks."""

    records = _artifact_records(
        artifact_paths=artifact_paths,
        artifact_payloads=artifact_payloads,
        artifact_roots=artifact_roots,
    )
    valid_records = tuple(row for row in records if row.accepted_as_physical_fixture_candidate)
    invalid_records = tuple(row for row in records if not row.accepted_as_physical_fixture_candidate)
    if invalid_records:
        status = "invalid_operator_artifact_present"
        blocked_by = ("operator_must_replace_invalid_fx_slots_inserts_physical_readback_artifact",)
    elif valid_records:
        status = "valid_operator_artifact_available"
        blocked_by = (
            "operator_must_review_valid_fx_slots_inserts_candidate_before_fixture_closure",
        )
    else:
        status = "operator_artifact_missing"
        blocked_by = ("operator_must_capture_fx_slots_inserts_physical_readback_artifact",)

    all_violations = tuple(
        dict.fromkeys(violation for row in records for violation in row.violations)
    )
    latest_valid = valid_records[-1] if valid_records else None
    valid_fx_slot_count = sum(row.fx_slot_count for row in valid_records)
    valid_insert_count = sum(row.insert_count for row in valid_records)
    return {
        "type": "fx_slots_inserts_physical_readback_artifact_intake",
        "generated_at": _utc_now(),
        "method_id": FX_SLOTS_INSERTS_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID,
        "required_contract_method_id": FX_SLOTS_INSERTS_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        "schema_id": FX_SLOTS_INSERTS_PHYSICAL_READBACK_SCHEMA_ID,
        "status": status,
        "surface_id": FX_SLOTS_INSERTS_SURFACE_ID,
        "remediation_action_id": FX_SLOTS_INSERTS_REMEDIATION_ACTION_ID,
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_intake": False,
            "fixture_closure_allowed_by_intake": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "artifact_search_roots": [str(root) for root in _artifact_roots(artifact_roots)],
            "artifact_count": len(records),
            "valid_candidate_count": len(valid_records),
            "invalid_candidate_count": len(invalid_records),
            "valid_fx_slot_count": valid_fx_slot_count,
            "valid_insert_count": valid_insert_count,
            "accepted_as_physical_fixture_candidate": bool(valid_records)
            and not invalid_records,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "fixture_closure_allowed_by_intake": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "promotion_families_requiring_surface": [
                "fx_send_insert",
                "routing",
                "mix_balance",
                "performance_adaptation",
            ],
            "latest_valid_candidate_source_id": latest_valid.source_id if latest_valid else None,
            "latest_valid_candidate_captured_at": latest_valid.captured_at if latest_valid else None,
            "next_global_step": (
                "Operator reviews the valid FX slots/inserts physical readback "
                "candidate before any fixture gap can be closed."
                if valid_records
                else "Operator captures FX slots/inserts physical readback "
                "evidence matching the schema contract."
            ),
            "blocked_by": list(blocked_by),
            "violations": list(all_violations),
        },
        "artifacts": [record.to_dict() for record in records],
    }
