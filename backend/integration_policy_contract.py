"""Fail-closed contract for integration policy evidence.

This contract validates integration-policy candidates before offline integration
promotion work can use them. It is policy evidence only: passing results cannot
grant live authority, close readiness or replace replay/fault/hardware proof.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping


SCHEMA_VERSION = "automixer.integration_policy_contract.v1"

REQUIRED_INTEGRATION_CONTROLS = (
    "analysis_trace_policy",
    "offline_replay_binding_policy",
    "fault_recovery_binding_policy",
    "safety_kernel_trace_policy",
    "operator_report_policy",
    "reproducibility_policy",
    "cross_scenario_consistency_policy",
    "no_live_transport",
)
REQUIRED_INTEGRATION_EVIDENCE = (
    "analysis_trace_bound",
    "offline_replay_manifest_bound",
    "fault_recovery_manifest_bound",
    "quality_threshold_policy_bound",
    "recovery_policy_bound",
    "integration_checks_bound",
    "operator_report_bound",
    "reproducibility_artifact_bound",
    "cross_scenario_consistency_artifact_bound",
    "human_supervision_required",
)
REQUIRED_APPROVAL_FIELDS = ("approved_by", "approved_decision_id", "approved_at")


def _as_bool(value: Any) -> bool:
    return bool(value)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _dedupe(items: Iterable[str]) -> List[str]:
    return sorted(dict.fromkeys(items))


def _nonempty_string(value: Any) -> bool:
    return bool(str(value or "").strip())


def _passed(value: Any) -> bool:
    if value is True:
        return True
    if not isinstance(value, Mapping):
        return False
    status = str(value.get("status") or value.get("result") or "").strip().lower()
    return status in {"passed", "approved", "complete"}


def _policy_approved(policy: Mapping[str, Any]) -> bool:
    status = str(policy.get("status") or "").strip().lower()
    has_required_fields = all(
        _nonempty_string(policy.get(key)) for key in REQUIRED_APPROVAL_FIELDS
    )
    return status in {"approved", "owner_approved"} and has_required_fields


def evaluate_integration_policy_contract(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Evaluate integration-policy evidence for owner review and offline use."""

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
        evidence_blockers.append("integration_policy_forbids_live_write")
    if contacts_wing:
        evidence_blockers.append("integration_policy_forbids_wing_contact")
    if contacts_backend:
        evidence_blockers.append("integration_policy_forbids_backend_contact")
    if hardware_evidence:
        evidence_blockers.append("integration_policy_evidence_must_not_be_labeled_hardware")
    if claimed_readiness_closure:
        evidence_blockers.append("integration_policy_evidence_cannot_close_readiness")
    if claimed_live_authority:
        evidence_blockers.append("integration_policy_evidence_cannot_grant_live_authority")

    integration_controls = _as_mapping(manifest.get("integration_controls"))
    missing_integration_controls = [
        item
        for item in REQUIRED_INTEGRATION_CONTROLS
        if not _passed(integration_controls.get(item))
    ]
    evidence_blockers.extend(
        f"missing_integration_control:{item}" for item in missing_integration_controls
    )

    integration_evidence = _as_mapping(manifest.get("integration_evidence"))
    missing_integration_evidence = [
        item
        for item in REQUIRED_INTEGRATION_EVIDENCE
        if not _passed(integration_evidence.get(item))
    ]
    evidence_blockers.extend(
        f"missing_integration_evidence:{item}" for item in missing_integration_evidence
    )

    integration_policy = _as_mapping(manifest.get("integration_policy"))
    integration_policy_approved = _policy_approved(integration_policy)
    approval_blockers: List[str] = []
    if not integration_policy_approved:
        approval_blockers.append("owner_approved_integration_policy_required")

    review_blocker_list = _dedupe(evidence_blockers)
    blocker_list = _dedupe(evidence_blockers + approval_blockers)
    ready_for_owner_review = not review_blocker_list
    ready_for_integration_policy_use = not blocker_list

    if ready_for_integration_policy_use:
        status = "ready_for_integration_policy_use"
    elif ready_for_owner_review:
        status = "ready_for_owner_review"
    else:
        status = "blocked"

    return {
        "schema": SCHEMA_VERSION,
        "policy_id": str(manifest.get("policy_id") or ""),
        "status": status,
        "ready_for_owner_review": ready_for_owner_review,
        "ready_for_integration_policy_use": ready_for_integration_policy_use,
        "integration_policy_approved": integration_policy_approved,
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
            "required_integration_controls": list(REQUIRED_INTEGRATION_CONTROLS),
            "missing_integration_controls": missing_integration_controls,
            "required_integration_evidence": list(REQUIRED_INTEGRATION_EVIDENCE),
            "missing_integration_evidence": missing_integration_evidence,
            "required_approval_fields": list(REQUIRED_APPROVAL_FIELDS),
            "integration_policy_status": str(integration_policy.get("status") or ""),
            "integration_policy_approved": integration_policy_approved,
        },
        "review_blockers": review_blocker_list,
        "approval_blockers": approval_blockers,
        "blockers": blocker_list,
    }
