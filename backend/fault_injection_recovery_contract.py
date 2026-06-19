"""Fail-closed contract for simulator fault-injection recovery evidence.

This contract evaluates no-hardware fault bundles for recovery, rollback,
emergency-stop and audit readiness. It is intentionally offline-only: a passing
result can support simulator-level promotion work, but cannot grant live
authority, close readiness or label simulator evidence as hardware proof.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Set


SCHEMA_VERSION = "automixer.fault_injection_recovery_contract.v1"

REQUIRED_FAULT_SCENARIOS = (
    "transport_timeout",
    "connection_loss",
    "stale_state",
    "emergency_stop",
    "safety_kernel_reject",
    "rollback_required",
)
REQUIRED_EVALUATION_TYPES = ("simulator", "fault_injection", "recovery")
REQUIRED_RECOVERY_OUTCOMES = (
    "fault_detected",
    "fail_closed",
    "unsafe_action_blocked",
    "safety_kernel_enforced",
    "rollback_verified",
    "emergency_stop_verified",
    "operator_visible",
    "audit_log_written",
    "no_live_write",
)
REQUIRED_AUDIT_FIELDS = (
    "fault_id",
    "trigger",
    "detected_at",
    "decision_id",
    "operator_message",
    "rollback_reference",
)


def _as_bool(value: Any) -> bool:
    return bool(value)


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _dedupe(items: Iterable[str]) -> List[str]:
    return sorted(dict.fromkeys(items))


def _policy_approved(policy: Mapping[str, Any]) -> bool:
    status = str(policy.get("status") or "").strip().lower()
    decision_id = str(policy.get("approved_decision_id") or "").strip()
    owner = str(policy.get("approved_by") or "").strip()
    return status in {"approved", "owner_approved"} and bool(decision_id) and bool(owner)


def _fault_result(record: Mapping[str, Any]) -> Dict[str, Any]:
    scenario = str(record.get("fault_scenario") or record.get("scenario") or "").strip().lower()
    evaluation_types = {
        str(item).strip().lower()
        for item in _as_list(record.get("evaluation_types"))
        if str(item).strip()
    }
    recovery_outcomes = _as_mapping(record.get("recovery_outcomes"))
    audit = _as_mapping(record.get("audit"))

    missing_evaluation_types = [
        item for item in REQUIRED_EVALUATION_TYPES if item not in evaluation_types
    ]
    missing_recovery_outcomes = [
        item for item in REQUIRED_RECOVERY_OUTCOMES if recovery_outcomes.get(item) is not True
    ]
    missing_audit_fields = [
        item for item in REQUIRED_AUDIT_FIELDS if not str(audit.get(item) or "").strip()
    ]
    scenario_label = scenario or "missing_scenario"
    blockers = [
        f"{scenario_label}:missing_evaluation_type:{item}"
        for item in missing_evaluation_types
    ]
    blockers.extend(
        f"{scenario_label}:missing_recovery_outcome:{item}"
        for item in missing_recovery_outcomes
    )
    blockers.extend(
        f"{scenario_label}:missing_audit_field:{item}" for item in missing_audit_fields
    )

    return {
        "fault_scenario": scenario,
        "passed": bool(scenario)
        and not missing_evaluation_types
        and not missing_recovery_outcomes
        and not missing_audit_fields,
        "evaluation_types": sorted(evaluation_types),
        "missing_evaluation_types": missing_evaluation_types,
        "missing_recovery_outcomes": missing_recovery_outcomes,
        "missing_audit_fields": missing_audit_fields,
        "blockers": blockers,
    }


def evaluate_fault_injection_recovery_contract(
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    """Evaluate a manifest-like payload for simulator fault recovery promotion."""

    manifest = _as_mapping(manifest)
    blockers: List[str] = []

    offline_only = _as_bool(manifest.get("offline_only"))
    live_write = _as_bool(manifest.get("live_write"))
    contacts_wing = _as_bool(manifest.get("contacts_wing"))
    contacts_backend = _as_bool(manifest.get("contacts_backend"))
    hardware_evidence = _as_bool(manifest.get("hardware_evidence"))
    claimed_readiness_closure = _as_bool(manifest.get("readiness_closure_allowed"))
    claimed_live_authority = _as_bool(manifest.get("live_authority_granted"))

    if not offline_only:
        blockers.append("offline_only_required")
    if live_write:
        blockers.append("fault_recovery_contract_forbids_live_write")
    if contacts_wing:
        blockers.append("fault_recovery_contract_forbids_wing_contact")
    if contacts_backend:
        blockers.append("fault_recovery_contract_forbids_backend_contact")
    if hardware_evidence:
        blockers.append("simulator_fault_evidence_must_not_be_labeled_hardware")
    if claimed_readiness_closure:
        blockers.append("simulator_fault_evidence_cannot_close_readiness")
    if claimed_live_authority:
        blockers.append("simulator_fault_evidence_cannot_grant_live_authority")

    recovery_policy = _as_mapping(manifest.get("recovery_policy"))
    recovery_policy_approved = _policy_approved(recovery_policy)
    if not recovery_policy_approved:
        blockers.append("approved_recovery_policy_required")

    fault_records = _as_list(manifest.get("fault_scenarios"))
    fault_results = [_fault_result(_as_mapping(record)) for record in fault_records]
    covered_fault_set: Set[str] = {
        item["fault_scenario"]
        for item in fault_results
        if item["fault_scenario"] in REQUIRED_FAULT_SCENARIOS
    }
    covered_fault_scenarios = [
        scenario for scenario in REQUIRED_FAULT_SCENARIOS if scenario in covered_fault_set
    ]
    missing_fault_scenarios = [
        scenario for scenario in REQUIRED_FAULT_SCENARIOS if scenario not in covered_fault_set
    ]
    blockers.extend(
        f"missing_required_fault_scenario:{scenario}" for scenario in missing_fault_scenarios
    )

    for result in fault_results:
        if result["fault_scenario"] not in REQUIRED_FAULT_SCENARIOS:
            blockers.append(
                f"unexpected_or_missing_fault_scenario:{result['fault_scenario'] or 'missing'}"
            )
        blockers.extend(result["blockers"])

    blocker_list = _dedupe(blockers)
    ready_for_fault_recovery_promotion = not blocker_list

    return {
        "schema": SCHEMA_VERSION,
        "manifest_id": str(manifest.get("manifest_id") or ""),
        "status": "ready_for_fault_recovery_promotion"
        if ready_for_fault_recovery_promotion
        else "blocked",
        "ready_for_fault_recovery_promotion": ready_for_fault_recovery_promotion,
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
            "required_fault_scenarios": list(REQUIRED_FAULT_SCENARIOS),
            "covered_fault_scenarios": covered_fault_scenarios,
            "missing_fault_scenarios": missing_fault_scenarios,
            "required_evaluation_types": list(REQUIRED_EVALUATION_TYPES),
            "required_recovery_outcomes": list(REQUIRED_RECOVERY_OUTCOMES),
            "required_audit_fields": list(REQUIRED_AUDIT_FIELDS),
            "recovery_policy_status": str(recovery_policy.get("status") or ""),
            "recovery_policy_approved": recovery_policy_approved,
        },
        "fault_results": fault_results,
        "blockers": blocker_list,
    }
