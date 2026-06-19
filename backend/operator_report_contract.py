"""Fail-closed contract for operator-facing integration reports.

This contract validates offline reports that summarize integration evidence for
an operator or reviewer. Passing reports can support offline integration
promotion work, but they cannot grant live authority, close readiness or label
offline evidence as hardware proof.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping


SCHEMA_VERSION = "automixer.operator_report_contract.v1"

REQUIRED_REPORT_SECTIONS = (
    "current_status",
    "authority_denial",
    "evidence_summary",
    "child_contract_results",
    "blockers",
    "next_operator_action",
    "rollback_plan",
    "audit_trail",
)
REQUIRED_REPORT_BINDINGS = (
    "integration_bundle_artifact",
    "offline_replay_result",
    "fault_recovery_result",
    "integration_policy_result",
    "safety_kernel_trace_result",
    "no_live_transport_result",
)
REQUIRED_AUTHORITY_DENIAL_FLAGS = (
    "live_authority_granted",
    "readiness_closure_allowed",
    "live_write",
    "hardware_evidence",
    "contacts_wing",
    "contacts_backend",
)
REQUIRED_AUDIT_FIELDS = (
    "report_id",
    "generated_at",
    "schema_version",
    "source_manifest_id",
    "review_required",
)


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


def _passed(value: Any) -> bool:
    if value is True:
        return True
    if not isinstance(value, Mapping):
        return False
    status = str(value.get("status") or value.get("result") or "").strip().lower()
    return status in {"passed", "approved", "complete"}


def _evidence_bound(value: Any) -> bool:
    if not isinstance(value, Mapping) or not _passed(value):
        return False
    return any(
        _nonempty_string(value.get(key))
        for key in ("artifact", "artifact_path", "report_path", "sha256")
    )


def _explicit_false(value: Any) -> bool:
    return value is False


def evaluate_operator_report_contract(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Evaluate an operator-facing offline report manifest."""

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
        blockers.append("operator_report_forbids_live_write")
    if contacts_wing:
        blockers.append("operator_report_forbids_wing_contact")
    if contacts_backend:
        blockers.append("operator_report_forbids_backend_contact")
    if hardware_evidence:
        blockers.append("operator_report_must_not_be_labeled_hardware")
    if claimed_readiness_closure:
        blockers.append("operator_report_cannot_close_readiness")
    if claimed_live_authority:
        blockers.append("operator_report_cannot_grant_live_authority")

    report_sections = _as_mapping(manifest.get("report_sections"))
    missing_report_sections = [
        item for item in REQUIRED_REPORT_SECTIONS if not _passed(report_sections.get(item))
    ]
    blockers.extend(f"missing_report_section:{item}" for item in missing_report_sections)

    report_bindings = _as_mapping(manifest.get("report_bindings"))
    missing_report_bindings = [
        item
        for item in REQUIRED_REPORT_BINDINGS
        if not _evidence_bound(report_bindings.get(item))
    ]
    blockers.extend(f"missing_report_binding:{item}" for item in missing_report_bindings)

    authority_denial = _as_mapping(manifest.get("authority_denial"))
    missing_authority_denial_flags = [
        item
        for item in REQUIRED_AUTHORITY_DENIAL_FLAGS
        if item not in authority_denial
    ]
    non_denied_authority_flags = [
        item
        for item in REQUIRED_AUTHORITY_DENIAL_FLAGS
        if item in authority_denial and not _explicit_false(authority_denial.get(item))
    ]
    blockers.extend(
        f"missing_authority_denial_flag:{item}"
        for item in missing_authority_denial_flags
    )
    blockers.extend(
        f"authority_not_denied:{item}" for item in non_denied_authority_flags
    )

    visible_blockers = [
        str(item).strip() for item in _as_list(manifest.get("visible_blockers"))
    ]
    if not any(visible_blockers):
        blockers.append("visible_blockers_required")

    next_operator_action = str(manifest.get("next_operator_action") or "").strip()
    if not next_operator_action:
        blockers.append("next_operator_action_required")

    audit = _as_mapping(manifest.get("audit"))
    missing_audit_fields = [
        item for item in REQUIRED_AUDIT_FIELDS if not _nonempty_string(audit.get(item))
    ]
    blockers.extend(f"missing_audit_field:{item}" for item in missing_audit_fields)

    blocker_list = _dedupe(blockers)
    ready_for_operator_report_use = not blocker_list

    return {
        "schema": SCHEMA_VERSION,
        "report_id": str(manifest.get("report_id") or audit.get("report_id") or ""),
        "status": "ready_for_operator_report_use"
        if ready_for_operator_report_use
        else "blocked",
        "ready_for_operator_report_use": ready_for_operator_report_use,
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
            "required_report_sections": list(REQUIRED_REPORT_SECTIONS),
            "missing_report_sections": missing_report_sections,
            "required_report_bindings": list(REQUIRED_REPORT_BINDINGS),
            "missing_report_bindings": missing_report_bindings,
            "required_authority_denial_flags": list(REQUIRED_AUTHORITY_DENIAL_FLAGS),
            "missing_authority_denial_flags": missing_authority_denial_flags,
            "non_denied_authority_flags": non_denied_authority_flags,
            "visible_blocker_count": len([item for item in visible_blockers if item]),
            "next_operator_action_present": bool(next_operator_action),
            "missing_audit_fields": missing_audit_fields,
        },
        "blockers": blocker_list,
    }
