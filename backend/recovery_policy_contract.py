"""Fail-closed contract for recovery policy evidence.

This contract validates recovery-policy candidates before they can be used by
offline fault-recovery promotion work. It is policy evidence only: passing
results cannot grant live authority, close readiness or replace supervised
hardware validation.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping


SCHEMA_VERSION = "automixer.recovery_policy_contract.v1"

REQUIRED_RECOVERY_CONTROLS = (
    "fault_detection_policy",
    "fail_closed_policy",
    "unsafe_action_block_policy",
    "safety_kernel_enforcement_policy",
    "rollback_policy",
    "emergency_stop_policy",
    "operator_visibility_policy",
    "audit_logging_policy",
    "no_live_transport",
)
REQUIRED_RECOVERY_EVIDENCE = (
    "fault_scenario_matrix_bound",
    "rollback_reference_bound",
    "operator_message_template_bound",
    "audit_schema_bound",
    "stale_state_timeout_bound",
    "connection_loss_timeout_bound",
    "emergency_stop_procedure_bound",
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


def evaluate_recovery_policy_contract(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Evaluate recovery-policy evidence for owner review and offline use."""

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
        evidence_blockers.append("recovery_policy_forbids_live_write")
    if contacts_wing:
        evidence_blockers.append("recovery_policy_forbids_wing_contact")
    if contacts_backend:
        evidence_blockers.append("recovery_policy_forbids_backend_contact")
    if hardware_evidence:
        evidence_blockers.append("recovery_policy_evidence_must_not_be_labeled_hardware")
    if claimed_readiness_closure:
        evidence_blockers.append("recovery_policy_evidence_cannot_close_readiness")
    if claimed_live_authority:
        evidence_blockers.append("recovery_policy_evidence_cannot_grant_live_authority")

    recovery_controls = _as_mapping(manifest.get("recovery_controls"))
    missing_recovery_controls = [
        item for item in REQUIRED_RECOVERY_CONTROLS if not _passed(recovery_controls.get(item))
    ]
    evidence_blockers.extend(
        f"missing_recovery_control:{item}" for item in missing_recovery_controls
    )

    recovery_evidence = _as_mapping(manifest.get("recovery_evidence"))
    missing_recovery_evidence = [
        item for item in REQUIRED_RECOVERY_EVIDENCE if not _passed(recovery_evidence.get(item))
    ]
    evidence_blockers.extend(
        f"missing_recovery_evidence:{item}" for item in missing_recovery_evidence
    )

    recovery_policy = _as_mapping(manifest.get("recovery_policy"))
    recovery_policy_approved = _policy_approved(recovery_policy)
    approval_blockers: List[str] = []
    if not recovery_policy_approved:
        approval_blockers.append("owner_approved_recovery_policy_required")

    review_blocker_list = _dedupe(evidence_blockers)
    blocker_list = _dedupe(evidence_blockers + approval_blockers)
    ready_for_owner_review = not review_blocker_list
    ready_for_recovery_policy_use = not blocker_list

    if ready_for_recovery_policy_use:
        status = "ready_for_recovery_policy_use"
    elif ready_for_owner_review:
        status = "ready_for_owner_review"
    else:
        status = "blocked"

    return {
        "schema": SCHEMA_VERSION,
        "policy_id": str(manifest.get("policy_id") or ""),
        "status": status,
        "ready_for_owner_review": ready_for_owner_review,
        "ready_for_recovery_policy_use": ready_for_recovery_policy_use,
        "recovery_policy_approved": recovery_policy_approved,
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
            "required_recovery_controls": list(REQUIRED_RECOVERY_CONTROLS),
            "missing_recovery_controls": missing_recovery_controls,
            "required_recovery_evidence": list(REQUIRED_RECOVERY_EVIDENCE),
            "missing_recovery_evidence": missing_recovery_evidence,
            "required_approval_fields": list(REQUIRED_APPROVAL_FIELDS),
            "recovery_policy_status": str(recovery_policy.get("status") or ""),
            "recovery_policy_approved": recovery_policy_approved,
        },
        "review_blockers": review_blocker_list,
        "approval_blockers": approval_blockers,
        "blockers": blocker_list,
    }
