"""Read-only consumer for sequence-policy queue-import memory trust reviews.

This module verifies that a memory trust-review result matches queue-import
memory-index provenance before a caller may prepare operator acceptance ledger
records. It does not accept candidates, invoke ledger handlers, mutate policy,
runtime memory, models, queues, memory files, or mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_review import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_REVIEW_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_REVIEW_STATUSES,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_INDEX_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_INDEX_RESULT_CONSUMER_STATUSES,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_review import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_REVIEW_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_REVIEW_STATUSES,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_INDEX_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_INDEX_RESULT_CONSUMER_STATUSES,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_STATUSES = {
    "sequence_memory_trust_promotion_queue_import_memory_trust_review_result_consumer_not_ready",
    "sequence_memory_trust_promotion_queue_import_memory_trust_review_result_consumer_replacement_required",
    "sequence_memory_trust_promotion_queue_import_memory_trust_review_result_consumer_blocked",
    "sequence_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_request_available",
}
REQUIRED_INDEX_RESULT_CONSUMER_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer"
)
REQUIRED_INDEX_RESULT_CONSUMER_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_memory_trust_review_request_available"
)
REQUIRED_MEMORY_TRUST_REVIEW_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_review"
)
REQUIRED_MEMORY_TRUST_REVIEW_AVAILABLE_STATUS = (
    "sequence_memory_trust_review_candidate_available"
)
REQUIRED_TARGET_MEMORY_PATH = ".ai/memory/project.md"
REQUIRED_LEARNING_SCOPE = "supervised_fader_gain_sequence_execution"
ALLOWED_VALUE_TYPES = {"fader", "gain"}
VERIFIED_READBACK_STATUSES = {"verified_from_cache", "verified_from_physical_readback"}
REQUIRED_CONSUMER_FALSE_FLAGS = (
    "memory_trust_review_handler_invoked",
    "memory_trust_review_created",
    "memory_write_allowed_by_consumer",
    "execution_allowed_by_consumer",
    "live_write",
    "apply_attempted",
    "queue_mutation",
    "proposal_status_mutation",
    "idempotency_reserved",
    "manual_console_write_called",
    "policy_mutation",
    "memory_mutation",
    "memory_file_mutation",
    "runtime_memory_mutation",
    "model_mutation",
    "live_authority_granted",
    "full_autonomous_ready",
    "can_apply_live",
)
REQUIRED_REVIEW_FALSE_SAFETY_FLAGS = (
    "live_write",
    "apply_attempted",
    "mutates_console_state",
    "queries_console",
    "execution_allowed_by_review",
    "queue_mutation",
    "policy_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_REVIEW_FALSE_SUMMARY_FLAGS = (
    "execution_allowed_by_review",
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


def _consumer_envelope_violations(consumer: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if consumer.get("type") != REQUIRED_INDEX_RESULT_CONSUMER_TYPE:
        violations.append("sequence_policy_memory_trust_review_result_consumer_index_consumer_type_mismatch")
    if (
        consumer.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_INDEX_RESULT_CONSUMER_METHOD_ID
    ):
        violations.append(
            "sequence_policy_memory_trust_review_result_consumer_index_consumer_method_id_mismatch"
        )
    if (
        consumer.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_INDEX_RESULT_CONSUMER_STATUSES
    ):
        violations.append(
            "sequence_policy_memory_trust_review_result_consumer_index_consumer_status_unknown"
        )
    safety = _mapping(consumer.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("sequence_policy_memory_trust_review_result_consumer_index_consumer_read_only_not_true")
    if safety.get("memory_index_result_consumer_only") is not True:
        violations.append("sequence_policy_memory_trust_review_result_consumer_index_consumer_only_not_true")
    if safety.get("memory_trust_review_request_template_only") is not True:
        violations.append("sequence_policy_memory_trust_review_result_consumer_index_consumer_template_only_not_true")
    if safety.get("source_memory_index_available") is not True:
        violations.append("sequence_policy_memory_trust_review_result_consumer_index_consumer_source_index_not_true")
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"sequence_policy_memory_trust_review_result_consumer_index_consumer_{flag}_not_false"
            )
    return tuple(violations)


def _consumer_summary_violations(
    summary: Mapping[str, Any],
    *,
    expected_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    violations = []
    expected_counts = {
        "expected_memory_index_record_count": len(expected_ids),
        "memory_index_record_count": len(expected_ids),
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(
                f"sequence_policy_memory_trust_review_result_consumer_index_consumer_{key}_mismatch"
            )
    if summary.get("memory_trust_review_request_available") is not bool(expected_ids):
        violations.append("sequence_policy_memory_trust_review_result_consumer_index_consumer_request_available_mismatch")
    if summary.get("source_memory_index_available") is not True:
        violations.append("sequence_policy_memory_trust_review_result_consumer_index_consumer_summary_source_index_not_true")
    if summary.get("fresh_sequence_review_context_status") != "fresh":
        violations.append("sequence_policy_memory_trust_review_result_consumer_index_consumer_freshness_not_fresh")
    if _string_list(summary.get("memory_index_record_proposal_ids")) != expected_ids:
        violations.append("sequence_policy_memory_trust_review_result_consumer_index_consumer_proposal_ids_mismatch")
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_policy_memory_trust_review_result_consumer_index_consumer_summary_{flag}_not_false"
            )
    return tuple(violations)


def _consumer_request_violations(request: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if request.get("handler") != "review_operator_proposal_apply_sequence_policy_memory_trust":
        violations.append("sequence_policy_memory_trust_review_result_consumer_request_handler_mismatch")
    if request.get("read_only_template") is not True:
        violations.append("sequence_policy_memory_trust_review_result_consumer_request_read_only_template_not_true")
    if request.get("source_memory_index_available") is not True:
        violations.append("sequence_policy_memory_trust_review_result_consumer_request_source_index_not_true")
    for flag in (
        "live_write",
        "apply_attempted",
        "queue_mutation",
        "proposal_status_mutation",
        "idempotency_reserved",
        "manual_console_write_called",
        "memory_mutation",
        "memory_file_mutation",
        "execution_allowed_by_consumer",
        "policy_mutation",
        "runtime_memory_mutation",
        "model_mutation",
        "live_authority_granted",
        "full_autonomous_ready",
        "can_apply_live",
    ):
        if request.get(flag) is not False:
            violations.append(
                f"sequence_policy_memory_trust_review_result_consumer_request_{flag}_not_false"
            )
    if _mapping(request.get("memory_index")).get("status") != "append_only_sequence_memory_index_available":
        violations.append("sequence_policy_memory_trust_review_result_consumer_request_memory_index_not_available")
    if _mapping(request.get("current_sequence_review_freshness")).get("status") != "fresh":
        violations.append("sequence_policy_memory_trust_review_result_consumer_request_freshness_not_fresh")
    return tuple(violations)


def _review_envelope_violations(review: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if review.get("type") != REQUIRED_MEMORY_TRUST_REVIEW_TYPE:
        violations.append("sequence_policy_memory_trust_review_result_consumer_review_type_mismatch")
    if review.get("method_id") != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_REVIEW_METHOD_ID:
        violations.append("sequence_policy_memory_trust_review_result_consumer_review_method_id_mismatch")
    if review.get("source_method_id") != "operator_proposal_apply_sequence_policy_memory_index":
        violations.append("sequence_policy_memory_trust_review_result_consumer_review_source_method_mismatch")
    if review.get("status") not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_REVIEW_STATUSES:
        violations.append("sequence_policy_memory_trust_review_result_consumer_review_status_unknown")
    safety = _mapping(review.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("sequence_policy_memory_trust_review_result_consumer_review_read_only_not_true")
    if safety.get("operator_required") is not True:
        violations.append("sequence_policy_memory_trust_review_result_consumer_review_operator_required_not_true")
    for flag in REQUIRED_REVIEW_FALSE_SAFETY_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"sequence_policy_memory_trust_review_result_consumer_review_safety_{flag}_not_false"
            )
    return tuple(violations)


def _review_summary_violations(
    summary: Mapping[str, Any],
    *,
    candidates: Tuple[Mapping[str, Any], ...],
    expected_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    violations = []
    expected_counts = {
        "memory_record_count": len(expected_ids),
        "trust_review_candidate_count": len(candidates),
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(
                f"sequence_policy_memory_trust_review_result_consumer_review_summary_{key}_mismatch"
            )
    if summary.get("memory_index_status") != "append_only_sequence_memory_index_available":
        violations.append("sequence_policy_memory_trust_review_result_consumer_review_summary_index_not_available")
    if summary.get("current_sequence_review_freshness_status") != "fresh":
        violations.append("sequence_policy_memory_trust_review_result_consumer_review_summary_freshness_not_fresh")
    if summary.get("operator_review_required") is not True:
        violations.append("sequence_policy_memory_trust_review_result_consumer_review_summary_operator_review_not_true")
    if tuple(summary.get("invalid_record_ids") or ()) != ():
        violations.append("sequence_policy_memory_trust_review_result_consumer_review_summary_invalid_ids_not_empty")
    for flag in REQUIRED_REVIEW_FALSE_SUMMARY_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_policy_memory_trust_review_result_consumer_review_summary_{flag}_not_false"
            )
    return tuple(violations)


def _candidate_violations(
    candidate: Mapping[str, Any],
    *,
    expected_spec: Mapping[str, Any],
    index: int,
) -> Tuple[str, ...]:
    candidate_id = _string(candidate.get("review_candidate_id")) or f"trust_review_candidate:{index}"
    violations = []
    expected_pairs = {
        "proposal_id": expected_spec.get("proposal_id"),
        "draft_id": expected_spec.get("draft_id"),
        "memory_record_id": expected_spec.get("memory_record_id"),
        "learning_scope": REQUIRED_LEARNING_SCOPE,
    }
    for field, expected in expected_pairs.items():
        if candidate.get(field) != expected:
            violations.append(f"{candidate_id}:{field}_mismatch")
    value_type = str(candidate.get("value_type") or "").strip().lower()
    if value_type not in ALLOWED_VALUE_TYPES:
        violations.append(f"{candidate_id}:value_type_not_allowed")
    channel = candidate.get("channel")
    if isinstance(channel, bool) or not isinstance(channel, int) or channel < 1:
        violations.append(f"{candidate_id}:channel_invalid")
    effective_value = candidate.get("effective_value")
    if isinstance(effective_value, bool) or not isinstance(effective_value, (int, float)):
        violations.append(f"{candidate_id}:effective_value_invalid")
    elif value_type == "fader" and float(effective_value) > 0.0:
        violations.append(f"{candidate_id}:fader_ceiling_exceeded")
    if candidate.get("readback_status") not in VERIFIED_READBACK_STATUSES:
        violations.append(f"{candidate_id}:readback_not_verified")
    if candidate.get("proof_status") != "verified":
        violations.append(f"{candidate_id}:proof_not_verified")
    if candidate.get("current_sequence_review_freshness_status") != "fresh":
        violations.append(f"{candidate_id}:freshness_not_fresh")
    if candidate.get("operator_review_required") is not True:
        violations.append(f"{candidate_id}:operator_review_required_not_true")
    if candidate.get("sequence_policy_review_state") != "operator_review_required":
        violations.append(f"{candidate_id}:review_state_invalid")
    for flag in (
        "execution_allowed_by_review",
        "queue_mutation_allowed_by_review",
        "policy_mutation_allowed_by_review",
        "model_mutation_allowed_by_review",
        "runtime_memory_mutation_allowed_by_review",
        "live_authority_granted",
        "full_autonomous_ready",
    ):
        if candidate.get(flag) is not False:
            violations.append(f"{candidate_id}:{flag}_not_false")
    return tuple(violations)


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer(
    *,
    memory_index_result_consumer: Mapping[str, Any] | None = None,
    memory_trust_review_result: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Verify trust-review evidence and emit an acceptance-ledger request."""

    consumer = _mapping(memory_index_result_consumer)
    review = _mapping(memory_trust_review_result)
    consumer_summary = _mapping(consumer.get("summary"))
    expected_ids = _string_list(consumer_summary.get("memory_index_record_proposal_ids"))
    specs = _record_list(consumer.get("memory_trust_review_request_specs"))
    candidates = _record_list(review.get("trust_review_candidates"))
    candidate_violations = tuple(
        violation
        for index, (candidate, spec) in enumerate(zip(candidates, specs))
        for violation in _candidate_violations(
            candidate,
            expected_spec=spec,
            index=index,
        )
    )
    raw_invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_consumer_envelope_violations(consumer),
                *_consumer_summary_violations(consumer_summary, expected_ids=expected_ids),
                *(
                    ("sequence_policy_memory_trust_review_result_consumer_index_consumer_status_not_available",)
                    if consumer.get("status") != REQUIRED_INDEX_RESULT_CONSUMER_AVAILABLE_STATUS
                    else ()
                ),
                *_consumer_request_violations(_mapping(consumer.get("memory_trust_review_request"))),
                *_review_envelope_violations(review),
                *_review_summary_violations(
                    _mapping(review.get("summary")),
                    candidates=candidates,
                    expected_ids=expected_ids,
                ),
                *(
                    ("sequence_policy_memory_trust_review_result_consumer_review_status_not_available",)
                    if review.get("status") != REQUIRED_MEMORY_TRUST_REVIEW_AVAILABLE_STATUS
                    else ()
                ),
                *(
                    ("sequence_policy_memory_trust_review_result_consumer_candidate_count_mismatch",)
                    if len(candidates) != len(expected_ids)
                    else ()
                ),
                *(
                    ("sequence_policy_memory_trust_review_result_consumer_spec_count_mismatch",)
                    if len(specs) != len(expected_ids)
                    else ()
                ),
                *candidate_violations,
            )
        )
    )
    invalid_record_ids = (
        ()
        if not consumer or not review
        else raw_invalid_record_ids
    )

    if not consumer or not review:
        status = "sequence_memory_trust_promotion_queue_import_memory_trust_review_result_consumer_not_ready"
        blocked_by = ("memory_index_result_consumer_and_memory_trust_review_result_required",)
    elif invalid_record_ids:
        status = "sequence_memory_trust_promotion_queue_import_memory_trust_review_result_consumer_replacement_required"
        blocked_by = ("operator_must_replace_invalid_sequence_memory_trust_review_evidence",)
    elif review.get("status") != REQUIRED_MEMORY_TRUST_REVIEW_AVAILABLE_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_memory_trust_review_result_consumer_blocked"
        blocked_by = ("sequence_memory_trust_review_candidates_must_be_available_before_acceptance",)
    elif not candidates:
        status = "sequence_memory_trust_promotion_queue_import_memory_trust_review_result_consumer_not_ready"
        blocked_by = ("sequence_memory_trust_review_candidates_required",)
    else:
        status = "sequence_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_request_available"
        blocked_by = ("operator_may_record_sequence_memory_trust_acceptance_decisions",)

    request_available = (
        status
        == "sequence_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_request_available"
    )
    acceptance_templates = tuple(
        {
            "acceptance_record_template_id": (
                f"{OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_METHOD_ID}:{candidate.get('review_candidate_id')}"
            ),
            "review_candidate_id": candidate.get("review_candidate_id"),
            "memory_record_id": candidate.get("memory_record_id"),
            "proposal_id": candidate.get("proposal_id"),
            "learning_scope": candidate.get("learning_scope"),
            "value_type": candidate.get("value_type"),
            "idempotency_key": candidate.get("idempotency_key"),
            "decision_options": ["accepted", "rejected"],
            "operator_must_choose_decision": True,
            "reviewed_by_required": True,
            "reviewed_at_utc_required": True,
            "execution_allowed_by_acceptance": False,
            "sequence_policy_promotion_allowed_by_acceptance": False,
            "queue_mutation_allowed_by_acceptance": False,
            "policy_mutation_allowed_by_acceptance": False,
            "model_mutation_allowed_by_acceptance": False,
            "runtime_memory_mutation_allowed_by_acceptance": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_import_to_queue": False,
            "can_apply_live": False,
        }
        for candidate in (candidates if request_available else ())
    )
    acceptance_ledger_request: Dict[str, Any] = {
        "handler": "record_operator_proposal_apply_sequence_policy_memory_trust_acceptance",
        "read_only_template": True,
        "memory_trust_review_result_consumer_only": True,
        "acceptance_ledger_request_template_only": True,
        "acceptance_ledger_handler_invoked": False,
        "acceptance_ledger_created": False,
        "operator_must_supply_acceptance_records": True,
        "acceptance_records": [],
        "acceptance_record_templates": list(acceptance_templates),
        "memory_write_allowed_by_consumer": False,
        "execution_allowed_by_consumer": False,
        "live_write": False,
        "apply_attempted": False,
        "queue_mutation": False,
        "proposal_status_mutation": False,
        "idempotency_reserved": False,
        "manual_console_write_called": False,
        "policy_mutation": False,
        "memory_mutation": False,
        "memory_file_mutation": False,
        "runtime_memory_mutation": False,
        "model_mutation": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "can_apply_live": False,
    }
    if request_available:
        acceptance_ledger_request["memory_trust_review"] = review

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_TRUST_REVIEW_RESULT_CONSUMER_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_INDEX_RESULT_CONSUMER_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "memory_trust_review_result_consumer_only": True,
            "acceptance_ledger_request_template_only": True,
            "requires_memory_index_result_consumer": True,
            "requires_memory_trust_review_result": True,
            "acceptance_ledger_handler_invoked": False,
            "acceptance_ledger_created": False,
            "operator_acceptance_required": True,
            "memory_write_allowed_by_consumer": False,
            "execution_allowed_by_consumer": False,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "policy_mutation": False,
            "memory_mutation": False,
            "memory_file_mutation": False,
            "runtime_memory_mutation": False,
            "model_mutation": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_apply_live": False,
        },
        "summary": {
            "expected_trust_review_candidate_count": len(expected_ids),
            "trust_review_candidate_count": len(candidates),
            "invalid_record_count": len(invalid_record_ids),
            "memory_trust_review_result_consumer_only": True,
            "acceptance_ledger_request_template_only": True,
            "acceptance_ledger_request_available": request_available,
            "acceptance_ledger_handler_invoked": False,
            "acceptance_ledger_created": False,
            "operator_acceptance_required": True,
            "memory_write_allowed_by_consumer": False,
            "execution_allowed_by_consumer": False,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "policy_mutation": False,
            "memory_mutation": False,
            "memory_file_mutation": False,
            "runtime_memory_mutation": False,
            "model_mutation": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_apply_live": False,
            "trust_review_candidate_proposal_ids": list(expected_ids),
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Provide memory-index consumer and memory trust-review result."
                if not consumer or not review
                else (
                    "Replace invalid memory trust-review evidence before acceptance."
                    if invalid_record_ids
                    else (
                        "Run read-only memory trust-review before acceptance."
                        if review.get("status") != REQUIRED_MEMORY_TRUST_REVIEW_AVAILABLE_STATUS
                        else "Operator may record accept/reject decisions in the read-only acceptance ledger; this consumer does not decide."
                    )
                )
            ),
        },
        "acceptance_ledger_request": acceptance_ledger_request,
        "acceptance_record_templates": list(acceptance_templates),
    }
