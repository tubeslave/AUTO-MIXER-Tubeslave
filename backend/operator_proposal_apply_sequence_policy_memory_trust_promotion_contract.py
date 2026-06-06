"""Read-only contract for sequence-policy memory trust promotion.

The contract validates manual promotion-plan candidates, fresh sequence-policy
context, and rollback evidence before surfacing operator-review-ready promotion
records. It does not execute proposals, mutate queues, policy stores, models,
runtime memory, append-only memory, or live mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_plan import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_PLAN_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_PLAN_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_plan,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_plan import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_PLAN_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_PLAN_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_plan,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_CONTRACT_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_contract"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_CONTRACT_STATUSES = {
    "sequence_memory_trust_promotion_contract_not_ready",
    "sequence_memory_trust_promotion_contract_replacement_required",
    "manual_sequence_memory_trust_promotion_contract_ready",
}
REQUIRED_PROMOTION_PLAN_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_plan"
)
ALLOWED_ROLLBACK_STRATEGIES = {
    "manual_revert_to_previous_sequence_policy_state",
    "append_only_revert_record_required",
}
REQUIRED_PROMOTION_CANDIDATE_FIELDS = (
    "promotion_candidate_id",
    "acceptance_id",
    "review_candidate_id",
    "memory_record_id",
    "proposal_id",
    "learning_scope",
    "value_type",
    "idempotency_key",
    "reviewed_by",
    "reviewed_at",
)
REQUIRED_PLAN_FALSE_SAFETY_FLAGS = (
    "live_write",
    "apply_attempted",
    "mutates_console_state",
    "queries_console",
    "execution_allowed_by_plan",
    "sequence_policy_promotion_allowed_by_plan",
    "queue_mutation_allowed_by_plan",
    "policy_mutation_allowed_by_plan",
    "model_mutation_allowed_by_plan",
    "runtime_memory_mutation_allowed_by_plan",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_PLAN_FALSE_SUMMARY_FLAGS = (
    "execution_allowed_by_plan",
    "sequence_policy_promotion_allowed_by_plan",
    "queue_mutation_allowed_by_plan",
    "policy_mutation_allowed_by_plan",
    "model_mutation_allowed_by_plan",
    "runtime_memory_mutation_allowed_by_plan",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_FRESHNESS_FALSE_FLAGS = (
    "live_write",
    "apply_attempted",
    "queries_console",
    "queue_mutation",
    "policy_mutation",
)
REQUIRED_ROLLBACK_FALSE_FLAGS = (
    "live_write",
    "apply_attempted",
    "mutates_console_state",
    "queries_console",
    "queue_mutation",
    "policy_mutation",
    "model_mutation",
    "runtime_memory_mutation",
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
    return f"{acceptance_id if isinstance(acceptance_id, str) and acceptance_id else 'unknown_sequence_memory_trust_promotion'}:{index}"


def _plan_envelope_violations(plan: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if plan.get("type") != REQUIRED_PROMOTION_PLAN_TYPE:
        violations.append("sequence_memory_trust_promotion_plan_type_mismatch")
    generated_at = plan.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("sequence_memory_trust_promotion_plan_generated_at_invalid")
    if (
        plan.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_PLAN_METHOD_ID
    ):
        violations.append("sequence_memory_trust_promotion_plan_method_id_mismatch")
    if (
        plan.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_PLAN_STATUSES
    ):
        violations.append("sequence_memory_trust_promotion_plan_status_unknown")
    safety = _mapping(plan.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("sequence_memory_trust_promotion_plan_read_only_not_true")
    if safety.get("operator_required") is not True:
        violations.append("sequence_memory_trust_promotion_plan_operator_required_not_true")
    for flag in REQUIRED_PLAN_FALSE_SAFETY_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"sequence_memory_trust_promotion_plan_{flag}_not_false")
    return tuple(violations)


def _plan_status_violations(
    plan: Mapping[str, Any],
    *,
    candidates: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    status = plan.get("status")
    if status == "sequence_memory_trust_promotion_plan_replacement_required":
        return ("sequence_memory_trust_promotion_plan_status_replacement_required",)
    if status == "sequence_memory_trust_promotion_plan_not_ready" and candidates:
        return ("sequence_memory_trust_promotion_plan_status_not_ready_with_candidates",)
    if status == "sequence_memory_trust_promotion_plan_candidate_available" and not candidates:
        return ("sequence_memory_trust_promotion_plan_status_available_without_candidates",)
    return ()


def _candidate_violations(candidate: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    for field in REQUIRED_PROMOTION_CANDIDATE_FIELDS:
        if not isinstance(candidate.get(field), str) or not candidate.get(field):
            violations.append(
                f"sequence_memory_trust_promotion_contract_candidate_{field}_missing"
            )
    reviewed_at = candidate.get("reviewed_at")
    if isinstance(reviewed_at, str) and reviewed_at and not _is_utc_timestamp(reviewed_at):
        violations.append(
            "sequence_memory_trust_promotion_contract_candidate_reviewed_at_invalid"
        )
    if candidate.get("manual_sequence_policy_promotion_required") is not True:
        violations.append(
            "sequence_memory_trust_promotion_contract_candidate_manual_promotion_not_true"
        )
    false_flags = (
        "sequence_policy_promotion_allowed_by_plan",
        "execution_allowed_by_plan",
        "queue_mutation_allowed_by_plan",
        "policy_mutation_allowed_by_plan",
        "model_mutation_allowed_by_plan",
        "runtime_memory_mutation_allowed_by_plan",
        "live_authority_granted",
        "full_autonomous_ready",
    )
    for flag in false_flags:
        if candidate.get(flag) is not False:
            violations.append(
                f"sequence_memory_trust_promotion_contract_candidate_{flag}_not_false"
            )
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
        f"sequence_memory_trust_promotion_plan_summary_{key}_mismatch"
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
    expected_proposal_ids = tuple(
        candidate["proposal_id"]
        for candidate in candidates
        if isinstance(candidate.get("proposal_id"), str) and candidate.get("proposal_id")
    )
    violations = []
    if _string_list(summary.get("promotion_candidate_ids")) != expected_ids:
        violations.append(
            "sequence_memory_trust_promotion_plan_summary_candidate_ids_mismatch"
        )
    if _string_list(summary.get("accepted_proposal_ids")) != expected_proposal_ids:
        violations.append(
            "sequence_memory_trust_promotion_plan_summary_proposal_ids_mismatch"
        )
    return tuple(violations)


def _plan_summary_safety_violations(summary: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if summary.get("operator_review_required") is not True:
        violations.append(
            "sequence_memory_trust_promotion_plan_summary_operator_review_not_true"
        )
    for flag in REQUIRED_PLAN_FALSE_SUMMARY_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_memory_trust_promotion_plan_summary_{flag}_not_false"
            )
    return tuple(violations)


def _sequence_context_status(sequence_context: Mapping[str, Any]) -> str:
    return str(sequence_context.get("status") or "missing").strip().lower() or "missing"


def _sequence_context_violations(sequence_context: Mapping[str, Any]) -> Tuple[str, ...]:
    if not sequence_context:
        return ()
    violations = []
    if sequence_context.get("read_only") is not True:
        violations.append(
            "sequence_memory_trust_promotion_contract_context_read_only_not_true"
        )
    for flag in REQUIRED_FRESHNESS_FALSE_FLAGS:
        if sequence_context.get(flag) is not False:
            violations.append(
                f"sequence_memory_trust_promotion_contract_context_{flag}_not_false"
            )
    return tuple(violations)


def _sequence_context_blockers(sequence_context: Mapping[str, Any]) -> Tuple[str, ...]:
    if not sequence_context:
        return ("fresh_sequence_policy_context_required_before_promotion_contract",)
    if _sequence_context_status(sequence_context) != "fresh":
        return ("fresh_sequence_policy_context_required_before_promotion_contract",)
    return ()


def _rollback_status(rollback_evidence: Mapping[str, Any]) -> str:
    return str(rollback_evidence.get("status") or "missing").strip().lower() or "missing"


def _rollback_violations(rollback_evidence: Mapping[str, Any]) -> Tuple[str, ...]:
    if not rollback_evidence:
        return ()
    violations = []
    strategy = _string(rollback_evidence.get("rollback_strategy"))
    reviewed_at = _string(rollback_evidence.get("reviewed_at"))
    if rollback_evidence.get("read_only") is not True:
        violations.append(
            "sequence_memory_trust_promotion_contract_rollback_read_only_not_true"
        )
    for flag in REQUIRED_ROLLBACK_FALSE_FLAGS:
        if rollback_evidence.get(flag) is not False:
            violations.append(
                f"sequence_memory_trust_promotion_contract_rollback_{flag}_not_false"
            )
    if strategy not in ALLOWED_ROLLBACK_STRATEGIES:
        violations.append(
            "sequence_memory_trust_promotion_contract_rollback_strategy_invalid"
        )
    if not _string(rollback_evidence.get("previous_policy_state")):
        violations.append(
            "sequence_memory_trust_promotion_contract_rollback_previous_state_missing"
        )
    if not _string(rollback_evidence.get("target_policy_store")):
        violations.append(
            "sequence_memory_trust_promotion_contract_rollback_target_store_missing"
        )
    if not _string(rollback_evidence.get("reviewed_by")):
        violations.append(
            "sequence_memory_trust_promotion_contract_rollback_reviewer_missing"
        )
    if not reviewed_at or not _is_utc_timestamp(reviewed_at):
        violations.append(
            "sequence_memory_trust_promotion_contract_rollback_reviewed_at_invalid"
        )
    return tuple(violations)


def _rollback_blockers(rollback_evidence: Mapping[str, Any]) -> Tuple[str, ...]:
    if not rollback_evidence:
        return ("rollback_evidence_required_before_sequence_promotion_contract",)
    if _rollback_status(rollback_evidence) != "ready":
        return ("rollback_evidence_required_before_sequence_promotion_contract",)
    return ()


def _contract_record(
    candidate: Mapping[str, Any],
    *,
    rollback_evidence: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "promotion_contract_record_id": (
            "operator_sequence_memory_trust_promotion_contract:"
            f"{candidate.get('promotion_candidate_id')}"
        ),
        "promotion_candidate_id": candidate.get("promotion_candidate_id"),
        "acceptance_id": candidate.get("acceptance_id"),
        "review_candidate_id": candidate.get("review_candidate_id"),
        "memory_record_id": candidate.get("memory_record_id"),
        "proposal_id": candidate.get("proposal_id"),
        "learning_scope": candidate.get("learning_scope"),
        "value_type": candidate.get("value_type"),
        "idempotency_key": candidate.get("idempotency_key"),
        "previous_policy_state": rollback_evidence.get("previous_policy_state"),
        "rollback_strategy": rollback_evidence.get("rollback_strategy"),
        "rollback_target_policy_store": rollback_evidence.get("target_policy_store"),
        "reviewed_by": candidate.get("reviewed_by"),
        "reviewed_at": candidate.get("reviewed_at"),
        "read_only": True,
        "live_write": False,
        "apply_attempted": False,
        "mutates_console_state": False,
        "queries_console": False,
        "operator_review_required": True,
        "rollback_evidence_required": True,
        "sequence_policy_promotion_ready_for_operator_review": True,
        "promotion_executor_required": True,
        "execution_allowed_by_contract": False,
        "sequence_policy_promotion_allowed_by_contract": False,
        "queue_mutation_allowed_by_contract": False,
        "policy_mutation_allowed_by_contract": False,
        "model_mutation_allowed_by_contract": False,
        "runtime_memory_mutation_allowed_by_contract": False,
        "live_widening_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
    }


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_contract(
    *,
    promotion_plan: Mapping[str, Any] | None = None,
    current_sequence_policy_context: Mapping[str, Any] | None = None,
    rollback_evidence: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a read-only promotion contract for accepted sequence memory."""

    plan = (
        _mapping(promotion_plan)
        or build_operator_proposal_apply_sequence_policy_memory_trust_promotion_plan()
    )
    summary = _mapping(plan.get("summary"))
    candidates = _record_list(plan.get("promotion_candidates"))
    plan_invalid_record_ids = _string_list(summary.get("invalid_record_ids"))
    unreviewed_candidate_ids = _string_list(summary.get("unreviewed_candidate_ids"))
    sequence_context = _mapping(current_sequence_policy_context)
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
                *_sequence_context_violations(sequence_context),
                *_rollback_violations(rollback),
            )
        )
    )
    context_blockers = _sequence_context_blockers(sequence_context)
    rollback_blockers = _rollback_blockers(rollback)
    ready_for_records = (
        not invalid_record_ids
        and candidates
        and not unreviewed_candidate_ids
        and not context_blockers
        and not rollback_blockers
    )
    contract_records = tuple(
        _contract_record(candidate, rollback_evidence=rollback)
        for candidate in candidates
        if ready_for_records
    )

    if invalid_record_ids:
        status = "sequence_memory_trust_promotion_contract_replacement_required"
        blocked_by = (
            "operator_must_replace_invalid_sequence_memory_trust_promotion_contract_evidence",
        )
    elif unreviewed_candidate_ids:
        status = "sequence_memory_trust_promotion_contract_not_ready"
        blocked_by = (
            "operator_must_review_all_sequence_memory_trust_candidates_before_contract",
        )
    elif not candidates:
        status = "sequence_memory_trust_promotion_contract_not_ready"
        blocked_by = ("sequence_promotion_plan_candidate_required_before_contract",)
    elif context_blockers:
        status = "sequence_memory_trust_promotion_contract_not_ready"
        blocked_by = context_blockers
    elif rollback_blockers:
        status = "sequence_memory_trust_promotion_contract_not_ready"
        blocked_by = rollback_blockers
    else:
        status = "manual_sequence_memory_trust_promotion_contract_ready"
        blocked_by = (
            "promotion_executor_still_required_before_policy_or_execution_change",
        )

    return {
        "type": "operator_proposal_apply_sequence_policy_memory_trust_promotion_contract",
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_CONTRACT_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_PLAN_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "live_write": False,
            "apply_attempted": False,
            "mutates_console_state": False,
            "queries_console": False,
            "operator_required": True,
            "rollback_evidence_required": True,
            "execution_allowed_by_contract": False,
            "sequence_policy_promotion_allowed_by_contract": False,
            "queue_mutation_allowed_by_contract": False,
            "policy_mutation_allowed_by_contract": False,
            "model_mutation_allowed_by_contract": False,
            "runtime_memory_mutation_allowed_by_contract": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "promotion_plan_status": plan.get("status") or "missing",
            "current_sequence_policy_context_status": _sequence_context_status(
                sequence_context
            ),
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
            "proposal_ids": [
                str(record["proposal_id"])
                for record in contract_records
                if isinstance(record.get("proposal_id"), str)
            ],
            "operator_review_required": True,
            "rollback_evidence_required": True,
            "execution_allowed_by_contract": False,
            "sequence_policy_promotion_allowed_by_contract": False,
            "queue_mutation_allowed_by_contract": False,
            "policy_mutation_allowed_by_contract": False,
            "model_mutation_allowed_by_contract": False,
            "runtime_memory_mutation_allowed_by_contract": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "invalid_record_ids": list(invalid_record_ids),
            "unreviewed_candidate_ids": list(unreviewed_candidate_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Operator replaces invalid sequence promotion contract evidence."
                if invalid_record_ids
                else (
                    "Fresh sequence-policy context is required before contract review."
                    if context_blockers
                    else (
                        "Rollback evidence is required before sequence contract review."
                        if rollback_blockers
                        else (
                            "Promotion executor remains separate before policy or execution changes."
                            if contract_records
                            else "Manual sequence promotion candidates are required before contract review."
                        )
                    )
                )
            ),
        },
        "contract_records": list(contract_records),
    }
