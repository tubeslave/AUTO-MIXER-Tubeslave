"""Fail-closed contract for deterministic safety-kernel trace evidence.

This contract validates offline safety-kernel traces before they can support
integration promotion work. A passing trace proves only that deterministic
safety checks were exercised in offline evidence; it cannot grant live
authority, close readiness or replace real hardware proof.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Set


SCHEMA_VERSION = "automixer.safety_kernel_trace_contract.v1"

REQUIRED_TRACE_SCENARIOS = (
    "supervised_fader_gain_allowed",
    "unsupported_live_write_rejected",
    "model_or_llm_bypass_rejected",
    "stale_state_rejected",
    "missing_readback_or_rollback_rejected",
    "headroom_or_feedback_risk_rejected",
)
REQUIRED_GUARD_CHECKS = (
    "mode_permission_checked",
    "live_write_allowlist_checked",
    "fader_ceiling_checked",
    "true_peak_headroom_checked",
    "feedback_safety_checked",
    "stale_state_checked",
    "readback_required",
    "rollback_required",
    "rate_limit_checked",
    "audit_log_required",
    "human_supervision_required",
    "model_bypass_blocked",
)
REQUIRED_AUDIT_FIELDS = (
    "decision_id",
    "checked_at",
    "candidate_action",
    "safety_decision",
    "reason",
    "policy_version",
    "audit_reference",
)
ALLOWED_DECISIONS = {"allow", "bounded_allow"}
REJECT_DECISIONS = {"reject", "blocked", "deny"}


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


def _expected_decision_for_scenario(scenario: str) -> Set[str]:
    if scenario == "supervised_fader_gain_allowed":
        return ALLOWED_DECISIONS
    return REJECT_DECISIONS


def _trace_result(record: Mapping[str, Any]) -> Dict[str, Any]:
    scenario = str(record.get("scenario") or record.get("name") or "").strip().lower()
    guard_checks = _as_mapping(record.get("guard_checks"))
    audit = _as_mapping(record.get("audit"))
    safety_decision = str(
        record.get("safety_decision") or audit.get("safety_decision") or ""
    ).strip().lower()

    missing_guard_checks = [
        item for item in REQUIRED_GUARD_CHECKS if guard_checks.get(item) is not True
    ]
    missing_audit_fields = [
        item for item in REQUIRED_AUDIT_FIELDS if not _nonempty_string(audit.get(item))
    ]
    expected_decisions = _expected_decision_for_scenario(scenario)
    decision_matches = safety_decision in expected_decisions

    scenario_label = scenario or "missing_scenario"
    blockers = [
        f"{scenario_label}:missing_guard_check:{item}"
        for item in missing_guard_checks
    ]
    blockers.extend(
        f"{scenario_label}:missing_audit_field:{item}"
        for item in missing_audit_fields
    )
    if scenario not in REQUIRED_TRACE_SCENARIOS:
        blockers.append(f"unexpected_or_missing_scenario:{scenario_label}")
    if not decision_matches:
        blockers.append(f"{scenario_label}:unexpected_safety_decision")

    return {
        "scenario": scenario,
        "passed": bool(scenario)
        and scenario in REQUIRED_TRACE_SCENARIOS
        and not missing_guard_checks
        and not missing_audit_fields
        and decision_matches,
        "safety_decision": safety_decision,
        "expected_decisions": sorted(expected_decisions),
        "missing_guard_checks": missing_guard_checks,
        "missing_audit_fields": missing_audit_fields,
        "blockers": blockers,
    }


def evaluate_safety_kernel_trace_contract(
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    """Evaluate offline safety-kernel trace evidence for integration use."""

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
        blockers.append("safety_kernel_trace_forbids_live_write")
    if contacts_wing:
        blockers.append("safety_kernel_trace_forbids_wing_contact")
    if contacts_backend:
        blockers.append("safety_kernel_trace_forbids_backend_contact")
    if hardware_evidence:
        blockers.append("safety_kernel_trace_must_not_be_labeled_hardware")
    if claimed_readiness_closure:
        blockers.append("safety_kernel_trace_cannot_close_readiness")
    if claimed_live_authority:
        blockers.append("safety_kernel_trace_cannot_grant_live_authority")

    trace_records = _as_list(manifest.get("trace_records"))
    trace_results = [_trace_result(_as_mapping(record)) for record in trace_records]
    covered_trace_set: Set[str] = {
        item["scenario"]
        for item in trace_results
        if item["scenario"] in REQUIRED_TRACE_SCENARIOS
    }
    covered_trace_scenarios = [
        scenario for scenario in REQUIRED_TRACE_SCENARIOS if scenario in covered_trace_set
    ]
    missing_trace_scenarios = [
        scenario for scenario in REQUIRED_TRACE_SCENARIOS if scenario not in covered_trace_set
    ]
    blockers.extend(
        f"missing_required_trace_scenario:{scenario}"
        for scenario in missing_trace_scenarios
    )
    for result in trace_results:
        blockers.extend(result["blockers"])

    blocker_list = _dedupe(blockers)
    ready_for_safety_kernel_trace_use = not blocker_list

    return {
        "schema": SCHEMA_VERSION,
        "trace_id": str(manifest.get("trace_id") or ""),
        "status": "ready_for_safety_kernel_trace_use"
        if ready_for_safety_kernel_trace_use
        else "blocked",
        "ready_for_safety_kernel_trace_use": ready_for_safety_kernel_trace_use,
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
            "required_trace_scenarios": list(REQUIRED_TRACE_SCENARIOS),
            "covered_trace_scenarios": covered_trace_scenarios,
            "missing_trace_scenarios": missing_trace_scenarios,
            "required_guard_checks": list(REQUIRED_GUARD_CHECKS),
            "required_audit_fields": list(REQUIRED_AUDIT_FIELDS),
        },
        "trace_results": trace_results,
        "blockers": blocker_list,
    }
