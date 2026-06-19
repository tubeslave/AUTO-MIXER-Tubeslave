"""Fail-closed contract for offline integration evaluation bundles.

This contract composes the replay and simulator fault-recovery contracts into
one integration-level gate. A passing bundle can support offline integration
promotion work, but it cannot grant live authority, close readiness or label
offline/simulator evidence as hardware proof.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

from backend.fault_injection_recovery_contract import (
    evaluate_fault_injection_recovery_contract,
)
from backend.offline_replay_evaluation_contract import (
    evaluate_offline_replay_evaluation_contract,
)


SCHEMA_VERSION = "automixer.evaluation_integration_bundle_contract.v1"

REQUIRED_INTEGRATION_CHECKS = (
    "analysis_trace_bound",
    "offline_replay_result_bound",
    "fault_recovery_result_bound",
    "safety_kernel_trace_bound",
    "operator_report_bound",
    "reproducibility_checked",
    "cross_scenario_consistency_checked",
    "no_live_transport",
)


def _as_bool(value: Any) -> bool:
    return bool(value)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _dedupe(items: Iterable[str]) -> List[str]:
    return sorted(dict.fromkeys(items))


def _passed(value: Any) -> bool:
    if value is True:
        return True
    if not isinstance(value, Mapping):
        return False
    status = str(value.get("status") or value.get("result") or "").strip().lower()
    return status in {"passed", "approved", "complete"}


def _policy_approved(policy: Mapping[str, Any]) -> bool:
    status = str(policy.get("status") or "").strip().lower()
    decision_id = str(policy.get("approved_decision_id") or "").strip()
    owner = str(policy.get("approved_by") or "").strip()
    return status in {"approved", "owner_approved"} and bool(decision_id) and bool(owner)


def _prefixed(prefix: str, blockers: Iterable[str]) -> List[str]:
    return [f"{prefix}:{blocker}" for blocker in blockers]


def evaluate_integration_bundle_contract(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Evaluate a manifest-like payload for offline integration promotion."""

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
        blockers.append("integration_bundle_forbids_live_write")
    if contacts_wing:
        blockers.append("integration_bundle_forbids_wing_contact")
    if contacts_backend:
        blockers.append("integration_bundle_forbids_backend_contact")
    if hardware_evidence:
        blockers.append("integration_bundle_evidence_must_not_be_labeled_hardware")
    if claimed_readiness_closure:
        blockers.append("integration_bundle_evidence_cannot_close_readiness")
    if claimed_live_authority:
        blockers.append("integration_bundle_evidence_cannot_grant_live_authority")

    integration_policy = _as_mapping(manifest.get("integration_policy"))
    integration_policy_approved = _policy_approved(integration_policy)
    if not integration_policy_approved:
        blockers.append("approved_integration_policy_required")

    integration_checks = _as_mapping(manifest.get("integration_checks"))
    missing_integration_checks = [
        item for item in REQUIRED_INTEGRATION_CHECKS if not _passed(integration_checks.get(item))
    ]
    blockers.extend(
        f"missing_integration_check:{item}" for item in missing_integration_checks
    )

    replay_result = evaluate_offline_replay_evaluation_contract(
        _as_mapping(manifest.get("offline_replay_manifest"))
    )
    fault_result = evaluate_fault_injection_recovery_contract(
        _as_mapping(manifest.get("fault_recovery_manifest"))
    )

    if replay_result["ready_for_replay_promotion"] is not True:
        blockers.append("offline_replay_contract_not_ready")
        blockers.extend(_prefixed("offline_replay", replay_result["blockers"]))
    if fault_result["ready_for_fault_recovery_promotion"] is not True:
        blockers.append("fault_recovery_contract_not_ready")
        blockers.extend(_prefixed("fault_recovery", fault_result["blockers"]))

    blocker_list = _dedupe(blockers)
    ready_for_integration_promotion = not blocker_list

    return {
        "schema": SCHEMA_VERSION,
        "manifest_id": str(manifest.get("manifest_id") or ""),
        "status": "ready_for_integration_evaluation_promotion"
        if ready_for_integration_promotion
        else "blocked",
        "ready_for_integration_evaluation_promotion": ready_for_integration_promotion,
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
            "offline_replay_ready": replay_result["ready_for_replay_promotion"],
            "fault_recovery_ready": fault_result["ready_for_fault_recovery_promotion"],
            "required_integration_checks": list(REQUIRED_INTEGRATION_CHECKS),
            "missing_integration_checks": missing_integration_checks,
            "integration_policy_status": str(integration_policy.get("status") or ""),
            "integration_policy_approved": integration_policy_approved,
        },
        "child_results": {
            "offline_replay": replay_result,
            "fault_recovery": fault_result,
        },
        "blockers": blocker_list,
    }
