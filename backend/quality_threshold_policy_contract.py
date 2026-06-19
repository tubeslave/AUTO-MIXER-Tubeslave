"""Fail-closed contract for quality threshold policy evidence.

This contract checks whether candidate quality thresholds are complete enough
for owner review, and whether they have explicit owner approval before replay
or integration promotion can use them. It never grants live authority and it
never treats threshold-policy evidence as hardware proof.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

from backend.offline_replay_evaluation_contract import REQUIRED_QUALITY_AXES


SCHEMA_VERSION = "automixer.quality_threshold_policy_contract.v1"

REQUIRED_THRESHOLD_AXES = REQUIRED_QUALITY_AXES
REQUIRED_POLICY_CHECKS = (
    "baseline_dataset_bound",
    "reference_mix_bound",
    "replay_manifest_bound",
    "fault_manifest_bound",
    "integration_bundle_bound",
    "owner_review_required",
    "no_live_transport",
)
REQUIRED_THRESHOLD_FIELDS = (
    "metric_name",
    "direction",
    "candidate_value",
    "evidence_artifact",
    "rationale",
    "rollback_policy",
)
ALLOWED_DIRECTIONS = ("min", "max", "range", "target_range")
REQUIRED_APPROVAL_FIELDS = ("approved_by", "approved_decision_id", "approved_at")


def _as_bool(value: Any) -> bool:
    return bool(value)


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _dedupe(items: Iterable[str]) -> List[str]:
    return sorted(dict.fromkeys(items))


def _nonempty_string(value: Any) -> bool:
    return bool(str(value or "").strip())


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _valid_range(value: Any) -> bool:
    if isinstance(value, Mapping):
        lower = value.get("min")
        upper = value.get("max")
        return _numeric(lower) and _numeric(upper) and lower <= upper
    if isinstance(value, list) and len(value) == 2:
        lower, upper = value
        return _numeric(lower) and _numeric(upper) and lower <= upper
    return False


def _valid_candidate_value(direction: str, value: Any) -> bool:
    if direction in {"min", "max"}:
        return _numeric(value)
    if direction in {"range", "target_range"}:
        return _valid_range(value)
    return False


def _passed(value: Any) -> bool:
    if value is True:
        return True
    if not isinstance(value, Mapping):
        return False
    status = str(value.get("status") or value.get("result") or "").strip().lower()
    return status in {"passed", "approved", "complete"}


def _policy_approved(policy: Mapping[str, Any]) -> bool:
    status = str(policy.get("status") or "").strip().lower()
    has_required_fields = all(_nonempty_string(policy.get(key)) for key in REQUIRED_APPROVAL_FIELDS)
    return status in {"approved", "owner_approved"} and has_required_fields


def _field_present(record: Mapping[str, Any], field: str, direction: str) -> bool:
    if field == "candidate_value":
        return _valid_candidate_value(direction, record.get(field))
    if field == "direction":
        return _nonempty_string(record.get(field))
    return _nonempty_string(record.get(field))


def _threshold_axis_result(record: Mapping[str, Any]) -> Dict[str, Any]:
    axis = str(record.get("axis") or record.get("name") or "").strip().lower()
    direction = str(record.get("direction") or "").strip().lower()
    axis_label = axis or "missing_axis"

    missing_fields = [
        field
        for field in REQUIRED_THRESHOLD_FIELDS
        if not _field_present(record, field, direction)
    ]
    blockers = [
        f"{axis_label}:missing_threshold_field:{field}" for field in missing_fields
    ]

    invalid_direction = bool(direction) and direction not in ALLOWED_DIRECTIONS
    if invalid_direction:
        blockers.append(f"{axis_label}:invalid_threshold_direction:{direction}")

    invalid_candidate_value = (
        direction in ALLOWED_DIRECTIONS
        and not _valid_candidate_value(direction, record.get("candidate_value"))
    )
    if invalid_candidate_value:
        blockers.append(f"{axis_label}:invalid_candidate_value")

    return {
        "axis": axis,
        "passed": bool(axis)
        and not missing_fields
        and not invalid_direction
        and not invalid_candidate_value,
        "direction": direction,
        "missing_fields": missing_fields,
        "invalid_direction": invalid_direction,
        "invalid_candidate_value": invalid_candidate_value,
        "blockers": blockers,
    }


def evaluate_quality_threshold_policy_contract(
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    """Evaluate threshold-policy evidence for owner review and promotion use."""

    manifest = _as_mapping(manifest)
    evidence_blockers: List[str] = []

    offline_only = _as_bool(manifest.get("offline_only"))
    live_write = _as_bool(manifest.get("live_write"))
    contacts_wing = _as_bool(manifest.get("contacts_wing"))
    contacts_backend = _as_bool(manifest.get("contacts_backend"))
    hardware_evidence = _as_bool(manifest.get("hardware_evidence"))
    claimed_readiness_closure = _as_bool(manifest.get("readiness_closure_allowed"))
    claimed_live_authority = _as_bool(manifest.get("live_authority_granted"))

    if not offline_only:
        evidence_blockers.append("offline_only_required")
    if live_write:
        evidence_blockers.append("quality_threshold_policy_forbids_live_write")
    if contacts_wing:
        evidence_blockers.append("quality_threshold_policy_forbids_wing_contact")
    if contacts_backend:
        evidence_blockers.append("quality_threshold_policy_forbids_backend_contact")
    if hardware_evidence:
        evidence_blockers.append("threshold_policy_evidence_must_not_be_labeled_hardware")
    if claimed_readiness_closure:
        evidence_blockers.append("threshold_policy_evidence_cannot_close_readiness")
    if claimed_live_authority:
        evidence_blockers.append("threshold_policy_evidence_cannot_grant_live_authority")

    policy_checks = _as_mapping(manifest.get("policy_checks"))
    missing_policy_checks = [
        item for item in REQUIRED_POLICY_CHECKS if not _passed(policy_checks.get(item))
    ]
    evidence_blockers.extend(
        f"missing_policy_check:{item}" for item in missing_policy_checks
    )

    threshold_records = _as_list(manifest.get("thresholds"))
    axis_results = [
        _threshold_axis_result(_as_mapping(record)) for record in threshold_records
    ]
    covered_axis_set = {
        item["axis"] for item in axis_results if item["axis"] in REQUIRED_THRESHOLD_AXES
    }
    covered_axes = [
        axis for axis in REQUIRED_THRESHOLD_AXES if axis in covered_axis_set
    ]
    missing_axes = [
        axis for axis in REQUIRED_THRESHOLD_AXES if axis not in covered_axis_set
    ]
    evidence_blockers.extend(f"missing_threshold_axis:{axis}" for axis in missing_axes)

    for result in axis_results:
        if result["axis"] not in REQUIRED_THRESHOLD_AXES:
            evidence_blockers.append(
                f"unexpected_or_missing_threshold_axis:{result['axis'] or 'missing'}"
            )
        evidence_blockers.extend(result["blockers"])

    threshold_policy = _as_mapping(manifest.get("threshold_policy"))
    threshold_policy_approved = _policy_approved(threshold_policy)
    approval_blockers: List[str] = []
    if not threshold_policy_approved:
        approval_blockers.append("owner_approved_quality_threshold_policy_required")

    review_blocker_list = _dedupe(evidence_blockers)
    blocker_list = _dedupe(evidence_blockers + approval_blockers)
    ready_for_owner_review = not review_blocker_list
    ready_for_threshold_policy_use = not blocker_list

    if ready_for_threshold_policy_use:
        status = "ready_for_threshold_policy_use"
    elif ready_for_owner_review:
        status = "ready_for_owner_review"
    else:
        status = "blocked"

    return {
        "schema": SCHEMA_VERSION,
        "policy_id": str(manifest.get("policy_id") or ""),
        "status": status,
        "ready_for_owner_review": ready_for_owner_review,
        "ready_for_threshold_policy_use": ready_for_threshold_policy_use,
        "threshold_policy_approved": threshold_policy_approved,
        "offline_only": offline_only,
        "hardware_evidence": hardware_evidence,
        "live_write": live_write,
        "authority": {
            "live_authority_granted": False,
            "readiness_closure_allowed": False,
            "contacts_wing": contacts_wing,
            "contacts_backend": contacts_backend,
        },
        "summary": {
            "required_threshold_axes": list(REQUIRED_THRESHOLD_AXES),
            "covered_threshold_axes": covered_axes,
            "missing_threshold_axes": missing_axes,
            "required_policy_checks": list(REQUIRED_POLICY_CHECKS),
            "missing_policy_checks": missing_policy_checks,
            "required_threshold_fields": list(REQUIRED_THRESHOLD_FIELDS),
            "allowed_directions": list(ALLOWED_DIRECTIONS),
            "required_approval_fields": list(REQUIRED_APPROVAL_FIELDS),
            "threshold_policy_status": str(threshold_policy.get("status") or ""),
            "threshold_policy_approved": threshold_policy_approved,
        },
        "axis_results": axis_results,
        "review_blockers": review_blocker_list,
        "approval_blockers": approval_blockers,
        "blockers": blocker_list,
    }
