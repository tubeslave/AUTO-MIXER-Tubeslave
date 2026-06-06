"""Dry-run executor contract for sequence-policy memory trust promotion.

The dry run converts accepted sequence-policy promotion execution candidates
into deterministic apply specifications for a future executor. It does not
promote sequence policy, execute proposals, mutate queues, policy stores,
models, runtime memory, append-only memory, or touch live mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTION_PLAN_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTION_PLAN_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTION_PLAN_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTION_PLAN_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTOR_DRY_RUN_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTOR_DRY_RUN_STATUSES = {
    "sequence_memory_trust_promotion_executor_dry_run_not_ready",
    "sequence_memory_trust_promotion_executor_dry_run_replacement_required",
    "sequence_memory_trust_promotion_executor_dry_run_ready",
    "sequence_memory_trust_promotion_executor_apply_blocked",
}
REQUIRED_EXECUTION_PLAN_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan"
)
TARGET_POLICY_STORE = "operator_sequence_policy_trust_gate"
TARGET_POLICY_STATE = "trusted_for_sequence_policy"
REQUIRED_EXECUTION_CANDIDATE_FIELDS = (
    "promotion_execution_candidate_id",
    "promotion_acceptance_id",
    "promotion_contract_record_id",
    "promotion_candidate_id",
    "proposal_id",
    "learning_scope",
    "value_type",
    "idempotency_key",
    "previous_policy_state",
    "rollback_strategy",
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
REQUIRED_CANDIDATE_FALSE_FLAGS = (
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
    explicit = _string(candidate.get("promotion_execution_candidate_id"))
    if explicit:
        return explicit
    acceptance_id = _string(candidate.get("promotion_acceptance_id"))
    return f"{acceptance_id if acceptance_id else 'unknown_sequence_execution_candidate'}:{index}"


def _plan_envelope_violations(plan: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if plan.get("type") != REQUIRED_EXECUTION_PLAN_TYPE:
        violations.append("sequence_memory_trust_promotion_execution_plan_type_mismatch")
    generated_at = plan.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append(
            "sequence_memory_trust_promotion_execution_plan_generated_at_invalid"
        )
    if (
        plan.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTION_PLAN_METHOD_ID
    ):
        violations.append("sequence_memory_trust_promotion_execution_plan_method_id_mismatch")
    if (
        plan.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTION_PLAN_STATUSES
    ):
        violations.append("sequence_memory_trust_promotion_execution_plan_status_unknown")
    safety = _mapping(plan.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("sequence_memory_trust_promotion_execution_plan_read_only_not_true")
    if safety.get("operator_required") is not True:
        violations.append(
            "sequence_memory_trust_promotion_execution_plan_operator_required_not_true"
        )
    for flag in REQUIRED_PLAN_FALSE_SAFETY_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"sequence_memory_trust_promotion_execution_plan_{flag}_not_false"
            )
    return tuple(violations)


def _plan_status_violations(
    plan: Mapping[str, Any],
    *,
    candidates: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    status = plan.get("status")
    if status == "sequence_memory_trust_promotion_execution_plan_replacement_required":
        return ("sequence_memory_trust_promotion_execution_plan_status_replacement_required",)
    if status == "sequence_memory_trust_promotion_execution_plan_not_ready" and candidates:
        return ("sequence_memory_trust_promotion_execution_plan_status_not_ready_with_candidates",)
    if (
        status == "manual_sequence_memory_trust_promotion_execution_candidate_available"
        and not candidates
    ):
        return (
            "sequence_memory_trust_promotion_execution_plan_status_available_without_candidates",
        )
    return ()


def _candidate_violations(candidate: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    for field in REQUIRED_EXECUTION_CANDIDATE_FIELDS:
        if not _string(candidate.get(field)):
            violations.append(
                f"sequence_memory_trust_promotion_executor_candidate_{field}_missing"
            )
    reviewed_at = _string(candidate.get("reviewed_at"))
    if reviewed_at and not _is_utc_timestamp(reviewed_at):
        violations.append(
            "sequence_memory_trust_promotion_executor_candidate_reviewed_at_invalid"
        )
    if candidate.get("read_only") is not True:
        violations.append(
            "sequence_memory_trust_promotion_executor_candidate_read_only_not_true"
        )
    if candidate.get("operator_review_required") is not True:
        violations.append(
            "sequence_memory_trust_promotion_executor_candidate_operator_review_not_true"
        )
    if candidate.get("separate_executor_required") is not True:
        violations.append(
            "sequence_memory_trust_promotion_executor_candidate_separate_executor_not_true"
        )
    if candidate.get("execution_plan_ready_for_operator_review") is not True:
        violations.append(
            "sequence_memory_trust_promotion_executor_candidate_ready_for_review_not_true"
        )
    if candidate.get("manual_sequence_policy_promotion_execution_required") is not True:
        violations.append(
            "sequence_memory_trust_promotion_executor_candidate_manual_execution_not_true"
        )
    for flag in REQUIRED_CANDIDATE_FALSE_FLAGS:
        if candidate.get(flag) is not False:
            violations.append(
                f"sequence_memory_trust_promotion_executor_candidate_{flag}_not_false"
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


def _ambiguous_policy_candidate_ids(
    candidates: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    candidate_ids_by_policy_key: Dict[str, list[str]] = {}
    for index, candidate in enumerate(candidates):
        if _candidate_violations(candidate):
            continue
        idempotency_key = _string(candidate.get("idempotency_key"))
        if not idempotency_key:
            continue
        candidate_ids_by_policy_key.setdefault(idempotency_key, []).append(
            _candidate_id(candidate, index)
        )
    return tuple(
        candidate_id
        for candidate_ids in candidate_ids_by_policy_key.values()
        if len(candidate_ids) > 1
        for candidate_id in candidate_ids
    )


def _plan_summary_count_violations(
    summary: Mapping[str, Any],
    *,
    candidates: Tuple[Mapping[str, Any], ...],
    invalid_record_ids: Tuple[str, ...],
    unreviewed_contract_record_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    expected_counts = {
        "execution_candidate_count": len(candidates),
        "invalid_record_count": len(invalid_record_ids),
        "unreviewed_contract_record_count": len(unreviewed_contract_record_ids),
    }
    return tuple(
        f"sequence_memory_trust_promotion_execution_plan_summary_{key}_mismatch"
        for key, expected in expected_counts.items()
        if _summary_int(summary.get(key)) != expected
    )


def _plan_summary_identity_violations(
    summary: Mapping[str, Any],
    *,
    candidates: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    expected_candidate_ids = tuple(
        candidate["promotion_execution_candidate_id"]
        for candidate in candidates
        if _string(candidate.get("promotion_execution_candidate_id"))
    )
    expected_proposal_ids = tuple(
        candidate["proposal_id"]
        for candidate in candidates
        if _string(candidate.get("proposal_id"))
    )
    violations = []
    if _string_list(summary.get("execution_candidate_ids")) != expected_candidate_ids:
        violations.append(
            "sequence_memory_trust_promotion_execution_plan_summary_candidate_ids_mismatch"
        )
    if _string_list(summary.get("accepted_proposal_ids")) != expected_proposal_ids:
        violations.append(
            "sequence_memory_trust_promotion_execution_plan_summary_proposal_ids_mismatch"
        )
    return tuple(violations)


def _plan_summary_safety_violations(summary: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if summary.get("operator_review_required") is not True:
        violations.append(
            "sequence_memory_trust_promotion_execution_plan_summary_operator_review_not_true"
        )
    for flag in REQUIRED_PLAN_FALSE_SUMMARY_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_memory_trust_promotion_execution_plan_summary_{flag}_not_false"
            )
    return tuple(violations)


def _target_policy_path(candidate: Mapping[str, Any]) -> str:
    proposal_id = candidate.get("proposal_id") or "unknown_proposal"
    idempotency_key = candidate.get("idempotency_key") or "unknown_idempotency"
    return f"{TARGET_POLICY_STORE}/{proposal_id}/{idempotency_key}/trust_state"


def _dry_run_idempotency_key(candidate: Mapping[str, Any]) -> str:
    return ":".join(
        (
            "operator_sequence_policy_memory_trust_promotion",
            str(candidate.get("proposal_id") or "unknown_proposal"),
            str(candidate.get("idempotency_key") or "unknown_idempotency"),
            str(candidate.get("promotion_acceptance_id") or "unknown_acceptance"),
        )
    )


def _dry_run_record(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    previous = _string(candidate.get("previous_policy_state"))
    return {
        "promotion_executor_dry_run_id": (
            "operator_sequence_memory_trust_promotion_executor_dry_run:"
            f"{candidate.get('promotion_execution_candidate_id')}"
        ),
        "promotion_execution_candidate_id": candidate.get(
            "promotion_execution_candidate_id"
        ),
        "promotion_acceptance_id": candidate.get("promotion_acceptance_id"),
        "promotion_contract_record_id": candidate.get("promotion_contract_record_id"),
        "promotion_candidate_id": candidate.get("promotion_candidate_id"),
        "proposal_id": candidate.get("proposal_id"),
        "learning_scope": candidate.get("learning_scope"),
        "value_type": candidate.get("value_type"),
        "source_idempotency_key": candidate.get("idempotency_key"),
        "target_policy_store": TARGET_POLICY_STORE,
        "target_policy_path": _target_policy_path(candidate),
        "operation": "set_operator_sequence_policy_trust_state",
        "previous_policy_state": previous,
        "target_policy_state": TARGET_POLICY_STATE,
        "rollback_strategy": candidate.get("rollback_strategy"),
        "idempotency_key": _dry_run_idempotency_key(candidate),
        "would_change_policy_state": previous != TARGET_POLICY_STATE,
        "dry_run": True,
        "read_only": True,
        "live_write": False,
        "apply_attempted": False,
        "mutates_console_state": False,
        "queries_console": False,
        "requires_operator_ack_before_apply": True,
        "requires_executor_contract_before_apply": True,
        "execution_allowed_by_dry_run": False,
        "sequence_policy_promotion_applied": False,
        "sequence_policy_promotion_allowed_by_dry_run": False,
        "queue_mutation_allowed_by_dry_run": False,
        "policy_mutation_allowed_by_dry_run": False,
        "model_mutation_allowed_by_dry_run": False,
        "runtime_memory_mutation_allowed_by_dry_run": False,
        "live_widening_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
    }


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run(
    *,
    promotion_execution_plan: Mapping[str, Any] | None = None,
    apply: bool = False,
    operator_ack: bool = False,
) -> Dict[str, Any]:
    """Build deterministic dry-run records for a future sequence-policy executor."""

    plan = (
        _mapping(promotion_execution_plan)
        or build_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan()
    )
    summary = _mapping(plan.get("summary"))
    candidates = _record_list(plan.get("execution_candidates"))
    plan_invalid_record_ids = _string_list(summary.get("invalid_record_ids"))
    unreviewed_contract_record_ids = _string_list(
        summary.get("unreviewed_contract_record_ids")
    )
    invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *plan_invalid_record_ids,
                *_plan_envelope_violations(plan),
                *_plan_status_violations(plan, candidates=candidates),
                *_candidate_replacement_ids(candidates),
                *_ambiguous_policy_candidate_ids(candidates),
                *_plan_summary_count_violations(
                    summary,
                    candidates=candidates,
                    invalid_record_ids=plan_invalid_record_ids,
                    unreviewed_contract_record_ids=unreviewed_contract_record_ids,
                ),
                *_plan_summary_identity_violations(summary, candidates=candidates),
                *_plan_summary_safety_violations(summary),
            )
        )
    )
    source_candidates = () if invalid_record_ids else candidates
    dry_run_records = tuple(_dry_run_record(candidate) for candidate in source_candidates)

    if invalid_record_ids:
        status = "sequence_memory_trust_promotion_executor_dry_run_replacement_required"
        blocked_by = (
            "operator_must_replace_invalid_sequence_execution_plan_before_executor_dry_run",
        )
    elif unreviewed_contract_record_ids:
        status = "sequence_memory_trust_promotion_executor_dry_run_not_ready"
        blocked_by = (
            "operator_must_review_all_sequence_promotion_contract_records_before_executor_dry_run",
        )
    elif apply:
        status = "sequence_memory_trust_promotion_executor_apply_blocked"
        blocked_by = (
            "dry_run_does_not_apply_sequence_policy_promotion",
            "separate_executor_contract_required_before_apply",
        )
    elif dry_run_records:
        status = "sequence_memory_trust_promotion_executor_dry_run_ready"
        blocked_by = (
            "separate_executor_contract_required_before_apply",
        )
    else:
        status = "sequence_memory_trust_promotion_executor_dry_run_not_ready"
        blocked_by = (
            "execution_candidate_required_before_sequence_executor_dry_run",
        )

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTOR_DRY_RUN_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_EXECUTION_PLAN_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "dry_run": True,
            "apply_requested": bool(apply),
            "operator_ack": bool(operator_ack),
            "live_write": False,
            "apply_attempted": False,
            "mutates_console_state": False,
            "queries_console": False,
            "operator_required": True,
            "execution_allowed_by_dry_run": False,
            "sequence_policy_promotion_applied": False,
            "sequence_policy_promotion_allowed_by_dry_run": False,
            "queue_mutation_allowed_by_dry_run": False,
            "policy_mutation_allowed_by_dry_run": False,
            "model_mutation_allowed_by_dry_run": False,
            "runtime_memory_mutation_allowed_by_dry_run": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "promotion_execution_plan_status": plan.get("status") or "missing",
            "execution_candidate_count": len(candidates),
            "dry_run_record_count": len(dry_run_records),
            "would_change_policy_state_count": sum(
                1 for record in dry_run_records if record["would_change_policy_state"]
            ),
            "invalid_record_count": len(invalid_record_ids),
            "unreviewed_contract_record_count": len(unreviewed_contract_record_ids),
            "target_policy_store": TARGET_POLICY_STORE,
            "target_policy_paths": [
                record["target_policy_path"] for record in dry_run_records
            ],
            "idempotency_keys": [
                record["idempotency_key"] for record in dry_run_records
            ],
            "accepted_proposal_ids": [
                str(record["proposal_id"])
                for record in dry_run_records
                if isinstance(record.get("proposal_id"), str)
            ],
            "operator_review_required": True,
            "dry_run": True,
            "apply_requested": bool(apply),
            "operator_ack": bool(operator_ack),
            "execution_allowed_by_dry_run": False,
            "sequence_policy_promotion_applied": False,
            "sequence_policy_promotion_allowed_by_dry_run": False,
            "queue_mutation_allowed_by_dry_run": False,
            "policy_mutation_allowed_by_dry_run": False,
            "model_mutation_allowed_by_dry_run": False,
            "runtime_memory_mutation_allowed_by_dry_run": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "invalid_record_ids": list(invalid_record_ids),
            "unreviewed_contract_record_ids": list(unreviewed_contract_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Operator replaces invalid sequence execution plan records before dry-run."
                if invalid_record_ids
                else (
                    "Operator reviews all sequence promotion contract records before dry-run."
                    if unreviewed_contract_record_ids
                    else (
                        "Dry-run apply specs are ready; a separate executor contract "
                        "is required before sequence policy promotion is applied."
                    )
                )
            ),
        },
        "dry_run_records": list(dry_run_records),
    }
