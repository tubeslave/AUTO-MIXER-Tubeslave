"""Read-only contract for source-identity memory trust promotion.

The contract validates manual promotion-plan candidates, fresh replay context,
and rollback evidence before surfacing operator-review-ready promotion records.
It does not update source identity trust, downstream proposal priority,
classifier models, runtime memory, queues, or live mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Tuple

try:
    from source_identity_calibration_memory_trust_promotion_plan import (
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_PLAN_METHOD_ID,
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_PLAN_STATUSES,
        build_source_identity_calibration_memory_trust_promotion_plan,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.source_identity_calibration_memory_trust_promotion_plan import (
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_PLAN_METHOD_ID,
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_PLAN_STATUSES,
        build_source_identity_calibration_memory_trust_promotion_plan,
    )


SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_CONTRACT_METHOD_ID = (
    "source_identity_calibration_memory_trust_promotion_contract"
)
SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_CONTRACT_STATUSES = {
    "memory_trust_promotion_contract_not_ready",
    "memory_trust_promotion_contract_replacement_required",
    "manual_memory_trust_promotion_contract_ready",
}
REQUIRED_PROMOTION_PLAN_TYPE = "source_identity_calibration_memory_trust_promotion_plan"
ALLOWED_ROLLBACK_STRATEGIES = {
    "manual_revert_to_previous_trust_state",
    "append_only_revert_record_required",
}
REQUIRED_PROMOTION_CANDIDATE_FIELDS = (
    "promotion_candidate_id",
    "acceptance_id",
    "review_candidate_id",
    "memory_record_id",
    "source_candidate_id",
    "source_type",
    "recommended_trust_state",
    "reviewed_by",
    "reviewed_at",
)
REQUIRED_PLAN_FALSE_SAFETY_FLAGS = (
    "live_write",
    "mutates_console_state",
    "queries_console",
    "execution_allowed_by_plan",
    "memory_trust_promotion_allowed_by_plan",
    "queue_mutation_allowed_by_plan",
    "model_mutation_allowed_by_plan",
    "runtime_memory_mutation_allowed_by_plan",
    "downstream_priority_change_allowed_by_plan",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_PLAN_FALSE_SUMMARY_FLAGS = (
    "execution_allowed_by_plan",
    "memory_trust_promotion_allowed_by_plan",
    "queue_mutation_allowed_by_plan",
    "model_mutation_allowed_by_plan",
    "runtime_memory_mutation_allowed_by_plan",
    "downstream_priority_change_allowed_by_plan",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_ROLLBACK_FALSE_FLAGS = (
    "live_write",
    "mutates_console_state",
    "queries_console",
    "queue_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "downstream_priority_mutation",
    "live_authority_granted",
    "full_autonomous_ready",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _record_list(value: Any) -> Tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _string(value: Any) -> str:
    return value if isinstance(value, str) and value else ""


def _string_list(value: Any) -> Tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def _summary_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _is_utc_timestamp(value: str) -> bool:
    if not value:
        return False
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return False
    if parsed.tzinfo is None:
        return False
    return parsed.utcoffset() == timezone.utc.utcoffset(parsed)


def _candidate_id(candidate: Mapping[str, Any], index: int) -> str:
    explicit = candidate.get("promotion_candidate_id")
    if isinstance(explicit, str) and explicit:
        return explicit
    acceptance_id = candidate.get("acceptance_id")
    return f"{acceptance_id if isinstance(acceptance_id, str) and acceptance_id else 'unknown_memory_trust_promotion'}:{index}"


def _plan_envelope_violations(plan: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if plan.get("type") != REQUIRED_PROMOTION_PLAN_TYPE:
        violations.append("memory_trust_promotion_plan_type_mismatch")
    generated_at = plan.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("memory_trust_promotion_plan_generated_at_invalid")
    if (
        plan.get("method_id")
        != SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_PLAN_METHOD_ID
    ):
        violations.append("memory_trust_promotion_plan_method_id_mismatch")
    if (
        plan.get("status")
        not in SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_PLAN_STATUSES
    ):
        violations.append("memory_trust_promotion_plan_status_unknown")
    safety = _mapping(plan.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("memory_trust_promotion_plan_read_only_not_true")
    if safety.get("operator_required") is not True:
        violations.append("memory_trust_promotion_plan_operator_required_not_true")
    for flag in REQUIRED_PLAN_FALSE_SAFETY_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"memory_trust_promotion_plan_{flag}_not_false")
    return tuple(violations)


def _plan_status_violations(
    plan: Mapping[str, Any],
    *,
    candidates: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    status = plan.get("status")
    if status == "memory_trust_promotion_replacement_required":
        return ("memory_trust_promotion_plan_status_replacement_required",)
    if status == "memory_trust_promotion_not_ready" and candidates:
        return ("memory_trust_promotion_plan_status_not_ready_with_candidates",)
    if status == "manual_memory_trust_promotion_candidate_available" and not candidates:
        return ("memory_trust_promotion_plan_status_available_without_candidates",)
    return ()


def _candidate_violations(candidate: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    for field in REQUIRED_PROMOTION_CANDIDATE_FIELDS:
        if not isinstance(candidate.get(field), str) or not candidate.get(field):
            violations.append(f"memory_trust_promotion_contract_candidate_{field}_missing")
    reviewed_at = candidate.get("reviewed_at")
    if isinstance(reviewed_at, str) and reviewed_at and not _is_utc_timestamp(reviewed_at):
        violations.append("memory_trust_promotion_contract_candidate_reviewed_at_invalid")
    if candidate.get("manual_memory_trust_promotion_required") is not True:
        violations.append(
            "memory_trust_promotion_contract_candidate_manual_promotion_not_true"
        )
    false_flags = (
        "memory_trust_promotion_allowed_by_plan",
        "downstream_priority_change_allowed_by_plan",
        "queue_mutation_allowed_by_plan",
        "model_mutation_allowed_by_plan",
        "runtime_memory_mutation_allowed_by_plan",
        "live_authority_granted",
        "full_autonomous_ready",
    )
    for flag in false_flags:
        if candidate.get(flag) is not False:
            violations.append(f"memory_trust_promotion_contract_candidate_{flag}_not_false")
    return tuple(violations)


def _candidate_replacement_ids(
    candidates: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    return tuple(
        _candidate_id(candidate, index)
        for index, candidate in enumerate(candidates)
        if _candidate_violations(candidate)
    )


def _plan_summary_count_violations(
    summary: Mapping[str, Any],
    *,
    candidates: Tuple[Mapping[str, Any], ...],
    invalid_record_ids: Tuple[str, ...],
    unreviewed_candidate_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    expected_counts = {
        "promotion_candidate_count": len(candidates),
        "invalid_record_count": len(invalid_record_ids),
        "unreviewed_candidate_count": len(unreviewed_candidate_ids),
    }
    return tuple(
        f"memory_trust_promotion_plan_summary_{key}_mismatch"
        for key, expected in expected_counts.items()
        if _summary_int(summary.get(key)) != expected
    )


def _plan_summary_identity_violations(
    summary: Mapping[str, Any],
    *,
    candidates: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    expected_ids = tuple(
        candidate["promotion_candidate_id"]
        for candidate in candidates
        if isinstance(candidate.get("promotion_candidate_id"), str)
        and candidate.get("promotion_candidate_id")
    )
    expected_source_types = tuple(
        candidate["source_type"]
        for candidate in candidates
        if isinstance(candidate.get("source_type"), str) and candidate.get("source_type")
    )
    violations = []
    if _string_list(summary.get("promotion_candidate_ids")) != expected_ids:
        violations.append("memory_trust_promotion_plan_summary_candidate_ids_mismatch")
    if _string_list(summary.get("accepted_source_types")) != expected_source_types:
        violations.append("memory_trust_promotion_plan_summary_source_types_mismatch")
    return tuple(violations)


def _plan_summary_safety_violations(summary: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if summary.get("operator_review_required") is not True:
        violations.append("memory_trust_promotion_plan_summary_operator_review_not_true")
    for flag in REQUIRED_PLAN_FALSE_SUMMARY_FLAGS:
        if summary.get(flag) is not False:
            violations.append(f"memory_trust_promotion_plan_summary_{flag}_not_false")
    return tuple(violations)


def _freshness_status(freshness: Mapping[str, Any]) -> str:
    return str(freshness.get("status") or "missing").strip().lower() or "missing"


def _freshness_violations(freshness: Mapping[str, Any]) -> Tuple[str, ...]:
    if not freshness:
        return ()
    violations = []
    if freshness.get("read_only") is not True:
        violations.append("memory_trust_promotion_contract_freshness_read_only_not_true")
    if freshness.get("live_write") is not False:
        violations.append("memory_trust_promotion_contract_freshness_live_write_not_false")
    if freshness.get("queries_console") is not False:
        violations.append(
            "memory_trust_promotion_contract_freshness_queries_console_not_false"
        )
    return tuple(violations)


def _freshness_blockers(freshness: Mapping[str, Any]) -> Tuple[str, ...]:
    if not freshness:
        return ("fresh_replay_context_required_before_promotion_contract",)
    if _freshness_status(freshness) != "fresh":
        return ("fresh_replay_context_required_before_promotion_contract",)
    return ()


def _rollback_status(rollback_evidence: Mapping[str, Any]) -> str:
    return str(rollback_evidence.get("status") or "missing").strip().lower() or "missing"


def _rollback_violations(rollback_evidence: Mapping[str, Any]) -> Tuple[str, ...]:
    if not rollback_evidence:
        return ()
    violations = []
    strategy = _string(rollback_evidence.get("rollback_strategy"))
    reviewed_at = _string(rollback_evidence.get("reviewed_at"))
    for flag in REQUIRED_ROLLBACK_FALSE_FLAGS:
        if rollback_evidence.get(flag) is not False:
            violations.append(
                f"memory_trust_promotion_contract_rollback_{flag}_not_false"
            )
    if rollback_evidence.get("read_only") is not True:
        violations.append("memory_trust_promotion_contract_rollback_read_only_not_true")
    if strategy not in ALLOWED_ROLLBACK_STRATEGIES:
        violations.append("memory_trust_promotion_contract_rollback_strategy_invalid")
    if not _string(rollback_evidence.get("previous_trust_state")):
        violations.append("memory_trust_promotion_contract_rollback_previous_state_missing")
    if not _string(rollback_evidence.get("target_store")):
        violations.append("memory_trust_promotion_contract_rollback_target_store_missing")
    if not _string(rollback_evidence.get("reviewed_by")):
        violations.append("memory_trust_promotion_contract_rollback_reviewer_missing")
    if not reviewed_at or not _is_utc_timestamp(reviewed_at):
        violations.append("memory_trust_promotion_contract_rollback_reviewed_at_invalid")
    return tuple(violations)


def _rollback_blockers(rollback_evidence: Mapping[str, Any]) -> Tuple[str, ...]:
    if not rollback_evidence:
        return ("rollback_evidence_required_before_promotion_contract",)
    if _rollback_status(rollback_evidence) != "ready":
        return ("rollback_evidence_required_before_promotion_contract",)
    return ()


def _contract_record(
    candidate: Mapping[str, Any],
    *,
    rollback_evidence: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "promotion_contract_record_id": (
            "source_identity_memory_trust_promotion_contract:"
            f"{candidate.get('promotion_candidate_id')}"
        ),
        "promotion_candidate_id": candidate.get("promotion_candidate_id"),
        "acceptance_id": candidate.get("acceptance_id"),
        "review_candidate_id": candidate.get("review_candidate_id"),
        "memory_record_id": candidate.get("memory_record_id"),
        "source_candidate_id": candidate.get("source_candidate_id"),
        "source_type": candidate.get("source_type"),
        "recommended_trust_state": candidate.get("recommended_trust_state"),
        "previous_trust_state": rollback_evidence.get("previous_trust_state"),
        "rollback_strategy": rollback_evidence.get("rollback_strategy"),
        "rollback_target_store": rollback_evidence.get("target_store"),
        "reviewed_by": candidate.get("reviewed_by"),
        "reviewed_at": candidate.get("reviewed_at"),
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "operator_review_required": True,
        "rollback_evidence_required": True,
        "memory_trust_promotion_ready_for_operator_review": True,
        "promotion_executor_required": True,
        "execution_allowed_by_contract": False,
        "memory_trust_promotion_allowed_by_contract": False,
        "downstream_priority_change_allowed_by_contract": False,
        "queue_mutation_allowed_by_contract": False,
        "model_mutation_allowed_by_contract": False,
        "runtime_memory_mutation_allowed_by_contract": False,
        "live_widening_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
    }


def build_source_identity_calibration_memory_trust_promotion_contract(
    *,
    promotion_plan: Mapping[str, Any] | None = None,
    current_state_cache_freshness: Mapping[str, Any] | None = None,
    rollback_evidence: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a read-only promotion contract for accepted memory-trust records."""

    plan = _mapping(promotion_plan) or build_source_identity_calibration_memory_trust_promotion_plan()
    summary = _mapping(plan.get("summary"))
    candidates = _record_list(plan.get("promotion_candidates"))
    plan_invalid_record_ids = _string_list(summary.get("invalid_record_ids"))
    unreviewed_candidate_ids = _string_list(summary.get("unreviewed_candidate_ids"))
    freshness = _mapping(current_state_cache_freshness)
    rollback = _mapping(rollback_evidence)
    invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *plan_invalid_record_ids,
                *_plan_envelope_violations(plan),
                *_plan_status_violations(plan, candidates=candidates),
                *_candidate_replacement_ids(candidates),
                *_plan_summary_count_violations(
                    summary,
                    candidates=candidates,
                    invalid_record_ids=plan_invalid_record_ids,
                    unreviewed_candidate_ids=unreviewed_candidate_ids,
                ),
                *_plan_summary_identity_violations(summary, candidates=candidates),
                *_plan_summary_safety_violations(summary),
                *_freshness_violations(freshness),
                *_rollback_violations(rollback),
            )
        )
    )
    freshness_blockers = _freshness_blockers(freshness)
    rollback_blockers = _rollback_blockers(rollback)
    ready_for_records = (
        not invalid_record_ids
        and candidates
        and not unreviewed_candidate_ids
        and not freshness_blockers
        and not rollback_blockers
    )
    contract_records = tuple(
        _contract_record(candidate, rollback_evidence=rollback)
        for candidate in candidates
        if ready_for_records
    )

    if invalid_record_ids:
        status = "memory_trust_promotion_contract_replacement_required"
        blocked_by = (
            "operator_must_replace_invalid_memory_trust_promotion_contract_evidence",
        )
    elif unreviewed_candidate_ids:
        status = "memory_trust_promotion_contract_not_ready"
        blocked_by = (
            "operator_must_review_all_memory_trust_candidates_before_contract",
        )
    elif not candidates:
        status = "memory_trust_promotion_contract_not_ready"
        blocked_by = ("promotion_plan_candidate_required_before_contract",)
    elif freshness_blockers:
        status = "memory_trust_promotion_contract_not_ready"
        blocked_by = freshness_blockers
    elif rollback_blockers:
        status = "memory_trust_promotion_contract_not_ready"
        blocked_by = rollback_blockers
    else:
        status = "manual_memory_trust_promotion_contract_ready"
        blocked_by = (
            "promotion_executor_still_required_before_downstream_priority_change",
        )

    return {
        "type": "source_identity_calibration_memory_trust_promotion_contract",
        "generated_at": _utc_now(),
        "method_id": (
            SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_CONTRACT_METHOD_ID
        ),
        "source_method_id": (
            SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_PLAN_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "operator_required": True,
            "rollback_evidence_required": True,
            "execution_allowed_by_contract": False,
            "memory_trust_promotion_allowed_by_contract": False,
            "queue_mutation_allowed_by_contract": False,
            "model_mutation_allowed_by_contract": False,
            "runtime_memory_mutation_allowed_by_contract": False,
            "downstream_priority_change_allowed_by_contract": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "promotion_plan_status": plan.get("status") or "missing",
            "current_state_cache_freshness_status": _freshness_status(freshness),
            "rollback_evidence_status": _rollback_status(rollback),
            "rollback_strategy": rollback.get("rollback_strategy") or "missing",
            "promotion_candidate_count": len(candidates),
            "contract_record_count": len(contract_records),
            "invalid_record_count": len(invalid_record_ids),
            "unreviewed_candidate_count": len(unreviewed_candidate_ids),
            "contract_record_ids": [
                str(record["promotion_contract_record_id"])
                for record in contract_records
                if isinstance(record.get("promotion_contract_record_id"), str)
            ],
            "source_types": [
                str(record["source_type"])
                for record in contract_records
                if isinstance(record.get("source_type"), str)
            ],
            "operator_review_required": True,
            "rollback_evidence_required": True,
            "execution_allowed_by_contract": False,
            "memory_trust_promotion_allowed_by_contract": False,
            "queue_mutation_allowed_by_contract": False,
            "model_mutation_allowed_by_contract": False,
            "runtime_memory_mutation_allowed_by_contract": False,
            "downstream_priority_change_allowed_by_contract": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "invalid_record_ids": list(invalid_record_ids),
            "unreviewed_candidate_ids": list(unreviewed_candidate_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Operator replaces invalid promotion contract evidence."
                if invalid_record_ids
                else (
                    "Fresh replay context is required before promotion contract review."
                    if freshness_blockers
                    else (
                        "Rollback evidence is required before promotion contract review."
                        if rollback_blockers
                        else (
                            "Promotion executor remains separate before downstream priority changes."
                            if contract_records
                            else "Manual promotion candidates are required before contract review."
                        )
                    )
                )
            ),
        },
        "contract_records": list(contract_records),
    }
