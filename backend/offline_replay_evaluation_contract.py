"""Fail-closed contract for offline/replay evaluation evidence.

The contract answers a narrow question: is a no-hardware replay bundle complete
enough to be considered for replay-level capability promotion? It never grants
live authority and it never labels offline evidence as hardware proof.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Set


SCHEMA_VERSION = "automixer.offline_replay_evaluation_contract.v1"

REQUIRED_SCENARIOS = ("soundcheck", "studio", "concert")
REQUIRED_EVALUATION_TYPES = ("offline", "replay", "fault_injection")
REQUIRED_SAFETY_CHECKS = (
    "headroom_checked",
    "feedback_checked",
    "rollback_plan_checked",
    "source_identity_checked",
    "no_live_write",
)
REQUIRED_QUALITY_AXES = (
    "objective_audio_metrics",
    "reference_mix_comparison",
    "repeatability_stability",
    "blind_listening_tests",
    "expert_assessment",
    "clipping_overload_statistics",
    "masking_intelligibility",
    "control_latency",
    "parameter_oscillation_rate",
    "recovery_failure_behavior",
)


def _as_bool(value: Any) -> bool:
    return bool(value)


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


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


def _threshold_policy_approved(threshold_policy: Mapping[str, Any]) -> bool:
    status = str(threshold_policy.get("status") or "").strip().lower()
    decision_id = str(threshold_policy.get("approved_decision_id") or "").strip()
    owner = str(threshold_policy.get("approved_by") or "").strip()
    return status in {"approved", "owner_approved"} and bool(decision_id) and bool(owner)


def _scenario_result(record: Mapping[str, Any]) -> Dict[str, Any]:
    scenario = str(record.get("scenario") or record.get("name") or "").strip().lower()
    evaluation_types = {
        str(item).strip().lower()
        for item in _as_list(record.get("evaluation_types"))
        if str(item).strip()
    }
    safety_checks = _as_mapping(record.get("safety_checks"))
    quality_axes = _as_mapping(record.get("quality_axes"))

    missing_evaluation_types = [
        item for item in REQUIRED_EVALUATION_TYPES if item not in evaluation_types
    ]
    missing_safety_checks = [
        item for item in REQUIRED_SAFETY_CHECKS if safety_checks.get(item) is not True
    ]
    missing_quality_axes = [
        item for item in REQUIRED_QUALITY_AXES if not _passed(quality_axes.get(item))
    ]
    blockers = [
        f"{scenario or 'missing_scenario'}:missing_evaluation_type:{item}"
        for item in missing_evaluation_types
    ]
    blockers.extend(
        f"{scenario or 'missing_scenario'}:missing_safety_check:{item}"
        for item in missing_safety_checks
    )
    blockers.extend(
        f"{scenario or 'missing_scenario'}:missing_quality_axis:{item}"
        for item in missing_quality_axes
    )

    return {
        "scenario": scenario,
        "passed": bool(scenario)
        and not missing_evaluation_types
        and not missing_safety_checks
        and not missing_quality_axes,
        "evaluation_types": sorted(evaluation_types),
        "missing_evaluation_types": missing_evaluation_types,
        "missing_safety_checks": missing_safety_checks,
        "missing_quality_axes": missing_quality_axes,
        "blockers": blockers,
    }


def evaluate_offline_replay_evaluation_contract(
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    """Evaluate a manifest-like payload for offline replay promotion.

    A passing result is still offline-only evidence. It can support replay-level
    promotion work, but cannot close live readiness or Full Autonomous gates.
    """

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
        blockers.append("offline_replay_contract_forbids_live_write")
    if contacts_wing:
        blockers.append("offline_replay_contract_forbids_wing_contact")
    if contacts_backend:
        blockers.append("offline_replay_contract_forbids_backend_contact")
    if hardware_evidence:
        blockers.append("offline_replay_evidence_must_not_be_labeled_hardware")
    if claimed_readiness_closure:
        blockers.append("offline_replay_evidence_cannot_close_readiness")
    if claimed_live_authority:
        blockers.append("offline_replay_evidence_cannot_grant_live_authority")

    threshold_policy = _as_mapping(manifest.get("threshold_policy"))
    thresholds_approved = _threshold_policy_approved(threshold_policy)
    if not thresholds_approved:
        blockers.append("approved_quality_threshold_policy_required")

    scenario_records = _as_list(manifest.get("scenarios"))
    scenario_results = [_scenario_result(_as_mapping(record)) for record in scenario_records]
    covered_scenario_set: Set[str] = {
        item["scenario"]
        for item in scenario_results
        if item["scenario"] in REQUIRED_SCENARIOS
    }
    covered_scenarios = [
        scenario for scenario in REQUIRED_SCENARIOS if scenario in covered_scenario_set
    ]
    missing_scenarios = [
        scenario for scenario in REQUIRED_SCENARIOS if scenario not in covered_scenario_set
    ]
    blockers.extend(f"missing_required_scenario:{scenario}" for scenario in missing_scenarios)

    for result in scenario_results:
        if result["scenario"] not in REQUIRED_SCENARIOS:
            blockers.append(f"unexpected_or_missing_scenario:{result['scenario'] or 'missing'}")
        blockers.extend(result["blockers"])

    blocker_list = _dedupe(blockers)
    ready_for_replay_promotion = not blocker_list

    return {
        "schema": SCHEMA_VERSION,
        "manifest_id": str(manifest.get("manifest_id") or ""),
        "status": "ready_for_replay_promotion"
        if ready_for_replay_promotion
        else "blocked",
        "ready_for_replay_promotion": ready_for_replay_promotion,
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
            "required_scenarios": list(REQUIRED_SCENARIOS),
            "covered_scenarios": covered_scenarios,
            "missing_scenarios": missing_scenarios,
            "required_evaluation_types": list(REQUIRED_EVALUATION_TYPES),
            "required_safety_checks": list(REQUIRED_SAFETY_CHECKS),
            "required_quality_axes": list(REQUIRED_QUALITY_AXES),
            "threshold_policy_status": str(threshold_policy.get("status") or ""),
            "thresholds_approved": thresholds_approved,
        },
        "scenario_results": scenario_results,
        "blockers": blocker_list,
    }
