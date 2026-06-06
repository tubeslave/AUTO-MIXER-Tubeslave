"""Read-only consumer for promotion memory-writer dry-run results.

This verifier connects the promotion memory-draft result consumer to an
explicit append-only memory-writer apply request template. It verifies
promotion provenance, delegates dry-run validation to the existing queue-import
memory-writer dry-run result consumer, and does not invoke writer handlers,
write memory files, reserve idempotency keys, or touch live mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_WRITER_DRY_RUN_RESULT_CONSUMER_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_draft_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_DRAFT_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_DRAFT_RESULT_CONSUMER_STATUSES,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_WRITER_DRY_RUN_RESULT_CONSUMER_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_draft_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_DRAFT_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_DRAFT_RESULT_CONSUMER_STATUSES,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_WRITER_DRY_RUN_RESULT_CONSUMER_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_dry_run_result_consumer"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_WRITER_DRY_RUN_RESULT_CONSUMER_STATUSES = {
    "sequence_memory_trust_promotion_queue_import_promotion_memory_writer_dry_run_result_consumer_not_ready",
    "sequence_memory_trust_promotion_queue_import_promotion_memory_writer_dry_run_result_consumer_replacement_required",
    "sequence_memory_trust_promotion_queue_import_promotion_memory_writer_dry_run_result_consumer_blocked",
    "sequence_memory_trust_promotion_queue_import_promotion_memory_writer_apply_request_available",
}
REQUIRED_PRIOR_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_draft_result_consumer"
)
REQUIRED_PRIOR_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_promotion_memory_writer_request_available"
)
REQUIRED_GENERIC_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_memory_writer_apply_request_available"
)
REQUIRED_GENERIC_REPLACEMENT_STATUS = (
    "sequence_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer_replacement_required"
)
REQUIRED_GENERIC_BLOCKED_STATUS = (
    "sequence_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer_blocked"
)
REQUIRED_PRIOR_FALSE_FLAGS = (
    "memory_writer_handler_invoked",
    "memory_file_write_applied",
    "memory_write_allowed_by_consumer",
    "execution_allowed_by_consumer",
    "live_write",
    "apply_attempted",
    "queue_mutation",
    "proposal_status_mutation",
    "proposal_order_mutation",
    "idempotency_reserved",
    "manual_console_write_called",
    "policy_mutation",
    "memory_mutation",
    "memory_file_mutation",
    "runtime_memory_mutation",
    "model_mutation",
    "sequence_policy_trust_store_mutation",
    "proposal_state_mutation",
    "live_authority_granted",
    "full_autonomous_ready",
    "can_apply_live",
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


def _prior_envelope_violations(prior: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if prior.get("type") != REQUIRED_PRIOR_TYPE:
        violations.append("promotion_memory_writer_dry_run_result_consumer_prior_type_mismatch")
    generated_at = prior.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("promotion_memory_writer_dry_run_result_consumer_prior_generated_at_invalid")
    if (
        prior.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_DRAFT_RESULT_CONSUMER_METHOD_ID
    ):
        violations.append("promotion_memory_writer_dry_run_result_consumer_prior_method_id_mismatch")
    if (
        prior.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_DRAFT_RESULT_CONSUMER_STATUSES
    ):
        violations.append("promotion_memory_writer_dry_run_result_consumer_prior_status_unknown")
    safety = _mapping(prior.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("promotion_memory_writer_dry_run_result_consumer_prior_read_only_not_true")
    if safety.get("promotion_memory_draft_result_consumer_only") is not True:
        violations.append("promotion_memory_writer_dry_run_result_consumer_prior_only_not_true")
    if safety.get("promotion_memory_writer_request_template_only") is not True:
        violations.append("promotion_memory_writer_dry_run_result_consumer_prior_template_only_not_true")
    if safety.get("operator_ack_required_before_write") is not True:
        violations.append("promotion_memory_writer_dry_run_result_consumer_prior_ack_required_not_true")
    if safety.get("apply_required_before_write") is not True:
        violations.append("promotion_memory_writer_dry_run_result_consumer_prior_apply_required_not_true")
    for flag in REQUIRED_PRIOR_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"promotion_memory_writer_dry_run_result_consumer_prior_{flag}_not_false"
            )
    return tuple(violations)


def _prior_summary_violations(
    summary: Mapping[str, Any],
    *,
    expected_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    violations = []
    if summary.get("promotion_memory_writer_request_available") is not bool(expected_ids):
        violations.append("promotion_memory_writer_dry_run_result_consumer_prior_summary_request_available_mismatch")
    expected_counts = {
        "expected_memory_draft_count": len(expected_ids),
        "memory_draft_record_count": len(expected_ids),
        "ready_memory_draft_record_count": len(expected_ids),
        "blocked_memory_draft_record_count": 0,
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(f"promotion_memory_writer_dry_run_result_consumer_prior_summary_{key}_mismatch")
    if summary.get("operator_ack_required_before_write") is not True:
        violations.append("promotion_memory_writer_dry_run_result_consumer_prior_summary_ack_required_not_true")
    if summary.get("apply_required_before_write") is not True:
        violations.append("promotion_memory_writer_dry_run_result_consumer_prior_summary_apply_required_not_true")
    if _string_list(summary.get("memory_draft_proposal_ids")) != expected_ids:
        violations.append("promotion_memory_writer_dry_run_result_consumer_prior_summary_draft_ids_mismatch")
    if _string_list(summary.get("ready_memory_draft_proposal_ids")) != expected_ids:
        violations.append("promotion_memory_writer_dry_run_result_consumer_prior_summary_ready_draft_ids_mismatch")
    for flag in REQUIRED_PRIOR_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"promotion_memory_writer_dry_run_result_consumer_prior_summary_{flag}_not_false"
            )
    return tuple(violations)


def _prior_request_violations(
    request: Mapping[str, Any],
    *,
    expected_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    violations = []
    if request.get("handler") != "write_operator_proposal_apply_sequence_policy_memory":
        violations.append("promotion_memory_writer_dry_run_result_consumer_prior_request_handler_mismatch")
    if request.get("apply") is not False:
        violations.append("promotion_memory_writer_dry_run_result_consumer_prior_request_apply_not_false")
    if request.get("operator_ack") is not False:
        violations.append("promotion_memory_writer_dry_run_result_consumer_prior_request_operator_ack_not_false")
    if request.get("read_only_template") is not True:
        violations.append("promotion_memory_writer_dry_run_result_consumer_prior_request_template_not_true")
    if request.get("promotion_memory_draft_result_consumer_only") is not True:
        violations.append("promotion_memory_writer_dry_run_result_consumer_prior_request_only_not_true")
    if request.get("promotion_memory_writer_request_template_only") is not True:
        violations.append("promotion_memory_writer_dry_run_result_consumer_prior_request_template_only_not_true")
    if request.get("operator_must_set_apply_true") is not True:
        violations.append("promotion_memory_writer_dry_run_result_consumer_prior_request_apply_requirement_missing")
    if request.get("operator_must_set_operator_ack_true") is not True:
        violations.append("promotion_memory_writer_dry_run_result_consumer_prior_request_ack_requirement_missing")
    payload = _mapping(request.get("memory_draft_payload"))
    if len(_record_list(payload.get("memory_draft_records"))) != len(expected_ids):
        violations.append("promotion_memory_writer_dry_run_result_consumer_prior_request_draft_count_mismatch")
    for flag in REQUIRED_PRIOR_FALSE_FLAGS:
        if request.get(flag) is not False:
            violations.append(
                f"promotion_memory_writer_dry_run_result_consumer_prior_request_{flag}_not_false"
            )
    return tuple(violations)


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_dry_run_result_consumer(
    *,
    promotion_memory_draft_result_consumer: Mapping[str, Any] | None = None,
    promotion_memory_writer_dry_run_result: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Verify promotion writer dry-run evidence and emit apply template."""

    prior = _mapping(promotion_memory_draft_result_consumer)
    dry_run = _mapping(promotion_memory_writer_dry_run_result)
    prior_summary = _mapping(prior.get("summary"))
    prior_request = _mapping(prior.get("promotion_memory_writer_request"))
    expected_ids = _string_list(prior_summary.get("memory_draft_proposal_ids"))
    prior_status = _string(prior.get("status"))
    generic_memory_draft_result_consumer = _mapping(
        prior.get("generic_memory_draft_result_consumer")
    )

    generic_consumer: Mapping[str, Any] = {}
    if prior and dry_run:
        generic_consumer = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer(
                memory_draft_result_consumer=generic_memory_draft_result_consumer,
                memory_writer_dry_run_result=dry_run,
            )
        )
    generic_status = _string(generic_consumer.get("status"))
    generic_summary = _mapping(generic_consumer.get("summary"))
    generic_request = _mapping(generic_consumer.get("memory_writer_apply_request"))
    generic_specs = _record_list(generic_consumer.get("memory_writer_apply_request_specs"))
    generic_invalid_record_ids = tuple(
        f"generic_memory_writer_dry_run_result_consumer:{item}"
        for item in _string_list(generic_summary.get("invalid_record_ids"))
    )

    raw_invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_string_list(prior_summary.get("invalid_record_ids")),
                *_prior_envelope_violations(prior),
                *_prior_summary_violations(
                    prior_summary,
                    expected_ids=expected_ids,
                ),
                *_prior_request_violations(
                    prior_request,
                    expected_ids=expected_ids,
                ),
                *(
                    generic_invalid_record_ids
                    if generic_status == REQUIRED_GENERIC_REPLACEMENT_STATUS
                    else ()
                ),
                *(
                    ("promotion_memory_writer_dry_run_result_consumer_generic_status_unknown",)
                    if (
                        generic_status
                        and generic_status
                        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_WRITER_DRY_RUN_RESULT_CONSUMER_STATUSES
                    )
                    else ()
                ),
            )
        )
    )
    invalid_record_ids = () if not prior or not dry_run else raw_invalid_record_ids

    if not prior or not dry_run:
        status = "sequence_memory_trust_promotion_queue_import_promotion_memory_writer_dry_run_result_consumer_not_ready"
        blocked_by = (
            "promotion_memory_draft_result_consumer_and_memory_writer_dry_run_result_required",
        )
    elif invalid_record_ids:
        status = "sequence_memory_trust_promotion_queue_import_promotion_memory_writer_dry_run_result_consumer_replacement_required"
        blocked_by = (
            "operator_must_replace_invalid_promotion_memory_writer_dry_run_evidence",
        )
    elif prior_status != REQUIRED_PRIOR_AVAILABLE_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_memory_writer_dry_run_result_consumer_blocked"
        blocked_by = (
            "promotion_memory_draft_result_consumer_must_request_memory_writer_first",
        )
    elif generic_status == REQUIRED_GENERIC_BLOCKED_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_memory_writer_dry_run_result_consumer_blocked"
        blocked_by = ("promotion_memory_writer_dry_run_must_be_ready",)
    elif generic_status != REQUIRED_GENERIC_AVAILABLE_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_memory_writer_dry_run_result_consumer_not_ready"
        blocked_by = ("ready_promotion_memory_writer_dry_run_required",)
    else:
        status = "sequence_memory_trust_promotion_queue_import_promotion_memory_writer_apply_request_available"
        blocked_by = ("operator_may_prepare_generic_memory_writer_apply_request",)

    request_available = (
        status
        == "sequence_memory_trust_promotion_queue_import_promotion_memory_writer_apply_request_available"
    )
    memory_draft_payload = (
        _mapping(generic_request.get("memory_draft_payload"))
        if request_available
        else {}
    )
    promotion_memory_writer_apply_request: Dict[str, Any] = {
        "handler": "write_operator_proposal_apply_sequence_policy_memory",
        "memory_draft_payload": dict(memory_draft_payload),
        "apply": False,
        "operator_ack": False,
        "read_only_template": True,
        "promotion_memory_writer_dry_run_result_consumer_only": True,
        "promotion_memory_writer_apply_request_template_only": True,
        "operator_must_set_apply_true": True,
        "operator_must_set_operator_ack_true": True,
        "memory_writer_handler_invoked": False,
        "memory_file_write_applied": False,
        "memory_write_allowed_by_consumer": False,
        "execution_allowed_by_consumer": False,
        "live_write": False,
        "apply_attempted": False,
        "queue_mutation": False,
        "proposal_status_mutation": False,
        "proposal_order_mutation": False,
        "idempotency_reserved": False,
        "manual_console_write_called": False,
        "policy_mutation": False,
        "memory_mutation": False,
        "memory_file_mutation": False,
        "runtime_memory_mutation": False,
        "model_mutation": False,
        "sequence_policy_trust_store_mutation": False,
        "proposal_state_mutation": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "can_apply_live": False,
    }
    if request_available:
        promotion_memory_writer_apply_request["promotion_memory_writer_dry_run_result_consumer"] = {
            "method_id": (
                OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_WRITER_DRY_RUN_RESULT_CONSUMER_METHOD_ID
            ),
            "source_prior_method_id": (
                OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_DRAFT_RESULT_CONSUMER_METHOD_ID
            ),
            "memory_draft_proposal_ids": list(expected_ids),
            "read_only": True,
        }

    memory_writer_apply_request_specs = tuple(
        {
            **dict(spec),
            "promotion_memory_writer_apply_request_spec_id": (
                f"{OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_WRITER_DRY_RUN_RESULT_CONSUMER_METHOD_ID}:{spec.get('proposal_id')}"
            ),
            "promotion_memory_writer_dry_run_result_consumer_only": True,
            "promotion_memory_writer_apply_request_template_only": True,
            "memory_writer_handler_invoked": False,
            "memory_file_write_applied": False,
            "memory_write_allowed_by_consumer": False,
            "operator_must_set_apply_true": True,
            "operator_must_set_operator_ack_true": True,
            "policy_mutation": False,
            "memory_mutation": False,
            "memory_file_mutation": False,
            "runtime_memory_mutation": False,
            "model_mutation": False,
            "sequence_policy_trust_store_mutation": False,
            "proposal_state_mutation": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_apply_live": False,
        }
        for spec in generic_specs
    ) if request_available else ()

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_dry_run_result_consumer"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_WRITER_DRY_RUN_RESULT_CONSUMER_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_DRAFT_RESULT_CONSUMER_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "promotion_memory_writer_dry_run_result_consumer_only": True,
            "promotion_memory_writer_apply_request_template_only": True,
            "requires_promotion_memory_draft_result_consumer": True,
            "requires_memory_writer_dry_run_result": True,
            "requires_operator_ack_before_write": True,
            "requires_apply_true_before_write": True,
            "memory_writer_handler_invoked": False,
            "memory_file_write_applied": False,
            "memory_write_allowed_by_consumer": False,
            "execution_allowed_by_consumer": False,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "proposal_order_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "policy_mutation": False,
            "memory_mutation": False,
            "memory_file_mutation": False,
            "runtime_memory_mutation": False,
            "model_mutation": False,
            "sequence_policy_trust_store_mutation": False,
            "proposal_state_mutation": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_apply_live": False,
        },
        "summary": {
            "expected_memory_writer_dry_run_count": len(expected_ids),
            "memory_writer_dry_run_record_count": _summary_int(
                generic_summary.get("memory_writer_dry_run_record_count")
            ) or 0,
            "dry_run_blocker_count": _summary_int(
                generic_summary.get("dry_run_blocker_count")
            ) or 0,
            "invalid_record_count": len(invalid_record_ids),
            "promotion_memory_writer_dry_run_result_consumer_only": True,
            "promotion_memory_writer_apply_request_template_only": True,
            "promotion_memory_writer_apply_request_available": request_available,
            "memory_writer_handler_invoked": False,
            "memory_file_write_applied": False,
            "memory_write_allowed_by_consumer": False,
            "operator_ack_required_before_write": True,
            "apply_required_before_write": True,
            "execution_allowed_by_consumer": False,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "proposal_order_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "policy_mutation": False,
            "memory_mutation": False,
            "memory_file_mutation": False,
            "runtime_memory_mutation": False,
            "model_mutation": False,
            "sequence_policy_trust_store_mutation": False,
            "proposal_state_mutation": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_apply_live": False,
            "memory_writer_dry_run_proposal_ids": list(
                _string_list(generic_summary.get("memory_writer_dry_run_proposal_ids"))
            ),
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Provide promotion memory-draft result consumer and memory-writer dry-run result."
                if not prior or not dry_run
                else (
                    "Replace invalid promotion memory-writer dry-run evidence before apply request."
                    if invalid_record_ids
                    else (
                        "Re-run memory writer dry-run until apply-request evidence is ready."
                        if generic_status == REQUIRED_GENERIC_BLOCKED_STATUS
                        else "Operator may prepare the generic memory writer apply request; this consumer does not run it."
                    )
                )
            ),
        },
        "promotion_memory_writer_apply_request": promotion_memory_writer_apply_request,
        "promotion_memory_writer_apply_request_specs": list(memory_writer_apply_request_specs),
        "generic_memory_writer_dry_run_result_consumer": generic_consumer if prior and dry_run else {},
    }
