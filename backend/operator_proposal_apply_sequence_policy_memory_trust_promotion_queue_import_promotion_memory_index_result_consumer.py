"""Read-only consumer for promotion memory-index results.

This verifier matches read-only sequence memory index evidence back to
promotion memory-writer apply provenance before exposing a memory-trust-review
request template. It delegates generic index validation to the queue-import
memory-index result consumer and does not invoke review handlers, mutate memory,
policy, models, queues, idempotency, or mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_INDEX_RESULT_CONSUMER_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_WRITER_APPLY_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_WRITER_APPLY_RESULT_CONSUMER_STATUSES,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_INDEX_RESULT_CONSUMER_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_WRITER_APPLY_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_WRITER_APPLY_RESULT_CONSUMER_STATUSES,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_INDEX_RESULT_CONSUMER_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_INDEX_RESULT_CONSUMER_STATUSES = {
    "sequence_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer_not_ready",
    "sequence_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer_replacement_required",
    "sequence_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer_blocked",
    "sequence_memory_trust_promotion_queue_import_promotion_memory_trust_review_request_available",
}
REQUIRED_PRIOR_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer"
)
REQUIRED_PRIOR_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_promotion_memory_index_request_available"
)
REQUIRED_GENERIC_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_memory_trust_review_request_available"
)
REQUIRED_GENERIC_REPLACEMENT_STATUS = (
    "sequence_memory_trust_promotion_queue_import_memory_index_result_consumer_replacement_required"
)
REQUIRED_GENERIC_BLOCKED_STATUS = (
    "sequence_memory_trust_promotion_queue_import_memory_index_result_consumer_blocked"
)
REQUIRED_TARGET_MEMORY_PATH = ".ai/memory/project.md"
PRIOR_FALSE_FLAGS = (
    "memory_index_handler_invoked",
    "memory_index_created",
    "memory_file_write_applied_by_consumer",
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
    "can_import_to_queue",
    "can_apply_live",
)
CONSUMER_FALSE_FLAGS = (
    "memory_trust_review_handler_invoked",
    "memory_trust_review_created",
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
        violations.append("promotion_memory_index_result_consumer_prior_type_mismatch")
    generated_at = prior.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("promotion_memory_index_result_consumer_prior_generated_at_invalid")
    if (
        prior.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_WRITER_APPLY_RESULT_CONSUMER_METHOD_ID
    ):
        violations.append("promotion_memory_index_result_consumer_prior_method_id_mismatch")
    if (
        prior.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_WRITER_APPLY_RESULT_CONSUMER_STATUSES
    ):
        violations.append("promotion_memory_index_result_consumer_prior_status_unknown")
    safety = _mapping(prior.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("promotion_memory_index_result_consumer_prior_read_only_not_true")
    if safety.get("promotion_memory_writer_apply_result_consumer_only") is not True:
        violations.append("promotion_memory_index_result_consumer_prior_only_not_true")
    if safety.get("promotion_memory_index_request_template_only") is not True:
        violations.append("promotion_memory_index_result_consumer_prior_template_only_not_true")
    if safety.get("source_memory_file_write_applied") is not True:
        violations.append("promotion_memory_index_result_consumer_prior_source_write_not_true")
    for flag in PRIOR_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"promotion_memory_index_result_consumer_prior_{flag}_not_false")
    return tuple(violations)


def _prior_summary_violations(
    summary: Mapping[str, Any],
    *,
    expected_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    violations = []
    expected_counts = {
        "expected_memory_write_result_count": len(expected_ids),
        "memory_write_result_count": len(expected_ids),
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(f"promotion_memory_index_result_consumer_prior_summary_{key}_mismatch")
    if summary.get("promotion_memory_index_request_available") is not bool(expected_ids):
        violations.append("promotion_memory_index_result_consumer_prior_summary_request_available_mismatch")
    if summary.get("source_memory_file_write_applied") is not True:
        violations.append("promotion_memory_index_result_consumer_prior_summary_source_write_not_true")
    if _string_list(summary.get("memory_write_result_proposal_ids")) != expected_ids:
        violations.append("promotion_memory_index_result_consumer_prior_summary_proposal_ids_mismatch")
    for flag in PRIOR_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"promotion_memory_index_result_consumer_prior_summary_{flag}_not_false"
            )
    return tuple(violations)


def _prior_request_violations(request: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if request.get("handler") != "index_operator_proposal_apply_sequence_policy_memory":
        violations.append("promotion_memory_index_result_consumer_prior_request_handler_mismatch")
    if request.get("target_memory_path") != REQUIRED_TARGET_MEMORY_PATH:
        violations.append("promotion_memory_index_result_consumer_prior_request_target_path_mismatch")
    if request.get("read_only_template") is not True:
        violations.append("promotion_memory_index_result_consumer_prior_request_template_not_true")
    if request.get("promotion_memory_writer_apply_result_consumer_only") is not True:
        violations.append("promotion_memory_index_result_consumer_prior_request_only_not_true")
    if request.get("promotion_memory_index_request_template_only") is not True:
        violations.append("promotion_memory_index_result_consumer_prior_request_template_only_not_true")
    if request.get("source_memory_file_write_applied") is not True:
        violations.append("promotion_memory_index_result_consumer_prior_request_source_write_not_true")
    for flag in PRIOR_FALSE_FLAGS:
        if request.get(flag) is not False:
            violations.append(
                f"promotion_memory_index_result_consumer_prior_request_{flag}_not_false"
            )
    return tuple(violations)


def _prior_spec_violations(
    specs: Tuple[Mapping[str, Any], ...],
    *,
    expected_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    violations = []
    if len(specs) != len(expected_ids):
        violations.append("promotion_memory_index_result_consumer_prior_spec_count_mismatch")
    spec_ids = tuple(_string(spec.get("proposal_id")) for spec in specs)
    if spec_ids != expected_ids:
        violations.append("promotion_memory_index_result_consumer_prior_spec_proposal_ids_mismatch")
    for index, spec in enumerate(specs):
        prefix = f"promotion_memory_index_request_spec:{index}"
        if spec.get("promotion_memory_writer_apply_result_consumer_only") is not True:
            violations.append(f"{prefix}:only_not_true")
        if spec.get("promotion_memory_index_request_template_only") is not True:
            violations.append(f"{prefix}:template_only_not_true")
        if spec.get("source_memory_file_write_applied") is not True:
            violations.append(f"{prefix}:source_memory_file_write_applied_not_true")
        for flag in PRIOR_FALSE_FLAGS:
            if spec.get(flag) is not False:
                violations.append(f"{prefix}:{flag}_not_false")
    return tuple(violations)


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer(
    *,
    promotion_memory_writer_apply_result_consumer: Mapping[str, Any] | None = None,
    promotion_memory_index_result: Mapping[str, Any] | None = None,
    current_sequence_review_freshness: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Verify promotion memory-index evidence and emit a trust-review template."""

    prior = _mapping(promotion_memory_writer_apply_result_consumer)
    index_result = _mapping(promotion_memory_index_result)
    freshness = _mapping(current_sequence_review_freshness)
    prior_summary = _mapping(prior.get("summary"))
    prior_request = _mapping(prior.get("promotion_memory_index_request"))
    prior_specs = _record_list(prior.get("promotion_memory_index_request_specs"))
    expected_ids = _string_list(prior_summary.get("memory_write_result_proposal_ids"))
    prior_status = _string(prior.get("status"))
    generic_writer_apply_consumer = _mapping(
        prior.get("generic_memory_writer_apply_result_consumer")
    )

    generic_consumer: Mapping[str, Any] = {}
    if prior and index_result:
        generic_consumer = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer(
                memory_writer_apply_result_consumer=generic_writer_apply_consumer,
                memory_index_result=index_result,
                current_sequence_review_freshness=freshness,
            )
        )
    generic_status = _string(generic_consumer.get("status"))
    generic_summary = _mapping(generic_consumer.get("summary"))
    generic_request = _mapping(generic_consumer.get("memory_trust_review_request"))
    generic_specs = _record_list(generic_consumer.get("memory_trust_review_request_specs"))
    generic_invalid_record_ids = tuple(
        f"generic_memory_index_result_consumer:{item}"
        for item in _string_list(generic_summary.get("invalid_record_ids"))
    )
    generic_index_ids = _string_list(
        generic_summary.get("memory_index_record_proposal_ids")
    )

    raw_invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_string_list(prior_summary.get("invalid_record_ids")),
                *_prior_envelope_violations(prior),
                *_prior_summary_violations(prior_summary, expected_ids=expected_ids),
                *_prior_request_violations(prior_request),
                *_prior_spec_violations(prior_specs, expected_ids=expected_ids),
                *(
                    generic_invalid_record_ids
                    if generic_status == REQUIRED_GENERIC_REPLACEMENT_STATUS
                    else ()
                ),
                *(
                    ("promotion_memory_index_result_consumer_generic_proposal_ids_mismatch",)
                    if generic_index_ids and generic_index_ids != expected_ids
                    else ()
                ),
                *(
                    ("promotion_memory_index_result_consumer_generic_status_unknown",)
                    if (
                        generic_status
                        and generic_status
                        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_MEMORY_INDEX_RESULT_CONSUMER_STATUSES
                    )
                    else ()
                ),
            )
        )
    )
    invalid_record_ids = () if not prior or not index_result else raw_invalid_record_ids

    if not prior or not index_result:
        status = "sequence_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer_not_ready"
        blocked_by = (
            "promotion_memory_writer_apply_result_consumer_and_memory_index_result_required",
        )
    elif not freshness:
        status = "sequence_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer_not_ready"
        blocked_by = (
            "fresh_sequence_review_context_required_before_promotion_trust_review_request",
        )
    elif invalid_record_ids:
        status = "sequence_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer_replacement_required"
        blocked_by = (
            "operator_must_replace_invalid_promotion_memory_index_evidence",
        )
    elif prior_status != REQUIRED_PRIOR_AVAILABLE_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer_blocked"
        blocked_by = (
            "promotion_memory_writer_apply_result_consumer_must_request_memory_index_first",
        )
    elif generic_status == REQUIRED_GENERIC_BLOCKED_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer_blocked"
        blocked_by = ("promotion_memory_index_must_be_available_before_trust_review",)
    elif generic_status != REQUIRED_GENERIC_AVAILABLE_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer_not_ready"
        blocked_by = ("ready_promotion_memory_index_result_required",)
    else:
        status = "sequence_memory_trust_promotion_queue_import_promotion_memory_trust_review_request_available"
        blocked_by = ("operator_may_run_read_only_promotion_memory_trust_review",)

    request_available = (
        status
        == "sequence_memory_trust_promotion_queue_import_promotion_memory_trust_review_request_available"
    )
    trust_review_request: Dict[str, Any] = {
        **dict(generic_request if request_available else {}),
        "handler": "review_operator_proposal_apply_sequence_policy_memory_trust",
        "read_only_template": True,
        "promotion_memory_index_result_consumer_only": True,
        "promotion_memory_trust_review_request_template_only": True,
        "memory_trust_review_handler_invoked": False,
        "memory_trust_review_created": False,
        "source_memory_index_available": request_available,
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
        "can_import_to_queue": False,
        "can_apply_live": False,
    }
    if request_available:
        trust_review_request["promotion_memory_index_result_consumer"] = {
            "method_id": (
                OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_INDEX_RESULT_CONSUMER_METHOD_ID
            ),
            "source_prior_method_id": (
                OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_WRITER_APPLY_RESULT_CONSUMER_METHOD_ID
            ),
            "memory_index_record_proposal_ids": list(expected_ids),
            "read_only": True,
        }

    review_specs = tuple(
        {
            **dict(spec),
            "promotion_memory_trust_review_request_spec_id": (
                f"{OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_INDEX_RESULT_CONSUMER_METHOD_ID}:{spec.get('proposal_id')}"
            ),
            "promotion_memory_index_result_consumer_only": True,
            "promotion_memory_trust_review_request_template_only": True,
            "memory_trust_review_handler_invoked": False,
            "memory_trust_review_created": False,
            "source_memory_index_available": True,
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
            "can_import_to_queue": False,
            "can_apply_live": False,
        }
        for spec in generic_specs
    ) if request_available else ()

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_INDEX_RESULT_CONSUMER_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_MEMORY_WRITER_APPLY_RESULT_CONSUMER_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "promotion_memory_index_result_consumer_only": True,
            "promotion_memory_trust_review_request_template_only": True,
            "requires_promotion_memory_writer_apply_result_consumer": True,
            "requires_memory_index_result": True,
            "requires_fresh_sequence_review_context": True,
            "source_memory_index_available": request_available,
            **{flag: False for flag in CONSUMER_FALSE_FLAGS},
        },
        "summary": {
            "expected_memory_index_record_count": len(expected_ids),
            "memory_index_record_count": (
                _summary_int(generic_summary.get("memory_index_record_count")) or 0
            ),
            "invalid_record_count": len(invalid_record_ids),
            "target_memory_path": REQUIRED_TARGET_MEMORY_PATH,
            "promotion_memory_index_result_consumer_only": True,
            "promotion_memory_trust_review_request_template_only": True,
            "promotion_memory_trust_review_request_available": request_available,
            "source_memory_index_available": request_available,
            "fresh_sequence_review_context_status": (
                freshness.get("status") if freshness else "missing"
            ),
            **{flag: False for flag in CONSUMER_FALSE_FLAGS},
            "memory_index_record_proposal_ids": list(
                generic_index_ids if request_available else ()
            ),
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Provide promotion writer-apply consumer, memory index result, and fresh review context."
                if not prior or not index_result or not freshness
                else (
                    "Replace invalid promotion memory-index evidence before trust review."
                    if invalid_record_ids
                    else (
                        "Build append-only sequence memory index before promotion trust review."
                        if generic_status == REQUIRED_GENERIC_BLOCKED_STATUS
                        else "Operator may run the read-only promotion memory trust review; this consumer does not run it."
                    )
                )
            ),
        },
        "promotion_memory_trust_review_request": trust_review_request,
        "promotion_memory_trust_review_request_specs": list(review_specs),
        "generic_memory_index_result_consumer": (
            generic_consumer if prior and index_result else {}
        ),
    }
