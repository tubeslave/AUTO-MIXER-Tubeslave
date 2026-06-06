"""Read-only consumer for promotion queue-importer results.

This verifier connects the queue-import promotion queue-ordering-plan result
consumer to the existing sequence-policy queue-import preflight planner. It
verifies importer results against the prior queue-ordering-plan provenance and
emits only a preflight-plan request template. It does not import queues, run
preflight, reserve idempotency, mutate proposal/policy/runtime/model/memory
state, execute sequences, or touch live mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_ORDERING_PLAN_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_ORDERING_PLAN_RESULT_CONSUMER_STATUSES,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORTER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORTER_STATUSES,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_ORDERING_PLAN_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_ORDERING_PLAN_RESULT_CONSUMER_STATUSES,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORTER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORTER_STATUSES,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_IMPORTER_RESULT_CONSUMER_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_IMPORTER_RESULT_CONSUMER_STATUSES = {
    "sequence_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer_not_ready",
    "sequence_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer_replacement_required",
    "sequence_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer_blocked",
    "sequence_memory_trust_promotion_queue_import_promotion_queue_import_preflight_plan_request_available",
}
REQUIRED_QUEUE_ORDERING_PLAN_RESULT_CONSUMER_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer"
)
REQUIRED_QUEUE_ORDERING_PLAN_RESULT_CONSUMER_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_promotion_queue_importer_request_available"
)
REQUIRED_QUEUE_IMPORTER_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer"
)
REQUIRED_QUEUE_IMPORTER_APPLIED_STATUS = (
    "sequence_memory_trust_promotion_queue_importer_applied"
)
REQUIRED_PRIOR_FALSE_FLAGS = (
    "promotion_queue_importer_handler_invoked",
    "promotion_queue_importer_created",
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
REQUIRED_IMPORTER_NEVER_TRUE_FLAGS = (
    "live_write",
    "mutates_console_state",
    "queries_console",
    "proposal_order_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "sequence_policy_trust_store_mutation",
    "proposal_state_mutation",
    "sequence_execution_allowed_by_importer",
    "sequence_policy_change_allowed_by_importer",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
    "can_import_to_queue",
    "can_apply_live",
)
REQUIRED_RESULT_FALSE_FLAGS = (
    "live_write",
    "mutates_console_state",
    "queries_console",
    "model_mutation",
    "runtime_memory_mutation",
    "sequence_policy_trust_store_mutation",
    "proposal_state_mutation",
    "sequence_execution_allowed_by_importer",
    "sequence_policy_change_allowed_by_importer",
    "live_widening_allowed",
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


def _queue_ordering_plan_from_prior_consumer(
    consumer: Mapping[str, Any],
) -> Mapping[str, Any]:
    request = _mapping(consumer.get("promotion_queue_importer_request"))
    return _mapping(request.get("queue_ordering_plan"))


def _proposal_from_import_result(result: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(result.get("operator_proposal"))


def _sequence_import_metadata(proposal: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = _mapping(proposal.get("raw"))
    return _mapping(raw.get("sequence_policy_queue_import")) or _mapping(
        proposal.get("sequence_policy_queue_import")
    )


def _prior_consumer_envelope_violations(
    consumer: Mapping[str, Any],
) -> Tuple[str, ...]:
    violations = []
    if consumer.get("type") != REQUIRED_QUEUE_ORDERING_PLAN_RESULT_CONSUMER_TYPE:
        violations.append("queue_importer_result_consumer_prior_type_mismatch")
    generated_at = consumer.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("queue_importer_result_consumer_prior_generated_at_invalid")
    if (
        consumer.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_ORDERING_PLAN_RESULT_CONSUMER_METHOD_ID
    ):
        violations.append("queue_importer_result_consumer_prior_method_id_mismatch")
    if (
        consumer.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_ORDERING_PLAN_RESULT_CONSUMER_STATUSES
    ):
        violations.append("queue_importer_result_consumer_prior_status_unknown")
    safety = _mapping(consumer.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("queue_importer_result_consumer_prior_read_only_not_true")
    if safety.get("promotion_queue_ordering_plan_result_consumer_only") is not True:
        violations.append("queue_importer_result_consumer_prior_only_not_true")
    if safety.get("promotion_queue_importer_request_template_only") is not True:
        violations.append("queue_importer_result_consumer_prior_template_only_not_true")
    for flag in REQUIRED_PRIOR_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"queue_importer_result_consumer_prior_{flag}_not_false")
    return tuple(violations)


def _prior_consumer_summary_violations(
    summary: Mapping[str, Any],
    *,
    plan_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    expected_counts = {
        "queue_ordering_plan_record_count": len(plan_records),
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(f"queue_importer_result_consumer_prior_summary_{key}_mismatch")
    if summary.get("promotion_queue_importer_request_available") is not bool(plan_records):
        violations.append("queue_importer_result_consumer_prior_summary_request_mismatch")
    if summary.get("promotion_queue_ordering_plan_result_consumer_only") is not True:
        violations.append("queue_importer_result_consumer_prior_summary_only_not_true")
    if summary.get("promotion_queue_importer_request_template_only") is not True:
        violations.append("queue_importer_result_consumer_prior_summary_template_not_true")
    for flag in REQUIRED_PRIOR_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"queue_importer_result_consumer_prior_summary_{flag}_not_false"
            )
    expected_ids = tuple(
        str(record["proposal_id"])
        for record in plan_records
        if isinstance(record.get("proposal_id"), str) and record.get("proposal_id")
    )
    if _string_list(summary.get("planned_queue_proposal_ids")) != expected_ids:
        violations.append("queue_importer_result_consumer_prior_summary_plan_ids_mismatch")
    return tuple(violations)


def _prior_consumer_request_violations(
    request: Mapping[str, Any],
    *,
    plan_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    if request.get("handler") != "import_operator_proposal_apply_sequence_policy_queue_ordering_plan":
        violations.append("queue_importer_result_consumer_prior_request_handler_mismatch")
    if request.get("read_only_template") is not True:
        violations.append("queue_importer_result_consumer_prior_request_template_not_true")
    if request.get("proposal_payloads_required_from_operator") is not True:
        violations.append("queue_importer_result_consumer_prior_request_payloads_not_required")
    if request.get("apply") is not False:
        violations.append("queue_importer_result_consumer_prior_request_apply_not_false")
    if request.get("operator_ack") is not False:
        violations.append("queue_importer_result_consumer_prior_request_operator_ack_not_false")
    for flag in REQUIRED_PRIOR_FALSE_FLAGS:
        if flag in request and request.get(flag) is not False:
            violations.append(
                f"queue_importer_result_consumer_prior_request_{flag}_not_false"
            )
    if plan_records and not _mapping(request.get("queue_ordering_plan")):
        violations.append("queue_importer_result_consumer_prior_request_plan_missing")
    return tuple(violations)


def _queue_importer_envelope_violations(
    importer: Mapping[str, Any],
) -> Tuple[str, ...]:
    violations = []
    if importer.get("type") != REQUIRED_QUEUE_IMPORTER_TYPE:
        violations.append("queue_importer_result_consumer_importer_type_mismatch")
    generated_at = importer.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("queue_importer_result_consumer_importer_generated_at_invalid")
    if (
        importer.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORTER_METHOD_ID
    ):
        violations.append("queue_importer_result_consumer_importer_method_id_mismatch")
    if (
        importer.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORTER_STATUSES
    ):
        violations.append("queue_importer_result_consumer_importer_status_unknown")
    return tuple(violations)


def _queue_importer_safety_violations(
    safety: Mapping[str, Any],
    *,
    importer_status: str,
) -> Tuple[str, ...]:
    applied = importer_status == REQUIRED_QUEUE_IMPORTER_APPLIED_STATUS
    violations = []
    if safety.get("read_only") is not (not applied):
        violations.append("queue_importer_result_consumer_importer_read_only_mismatch")
    if safety.get("operator_required") is not True:
        violations.append("queue_importer_result_consumer_importer_operator_required_not_true")
    if safety.get("sequence_consumer_required_before_execution") is not True:
        violations.append("queue_importer_result_consumer_importer_sequence_consumer_not_true")
    if safety.get("queue_import_dry_run") is not (not applied):
        violations.append("queue_importer_result_consumer_importer_dry_run_mismatch")
    if safety.get("queue_import_applied") is not applied:
        violations.append("queue_importer_result_consumer_importer_applied_mismatch")
    if safety.get("queue_mutation") is not applied:
        violations.append("queue_importer_result_consumer_importer_queue_mutation_mismatch")
    if applied and safety.get("operator_ack") is not True:
        violations.append("queue_importer_result_consumer_importer_ack_not_true")
    if applied and safety.get("apply_attempted") is not True:
        violations.append("queue_importer_result_consumer_importer_apply_attempted_not_true")
    for flag in REQUIRED_IMPORTER_NEVER_TRUE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"queue_importer_result_consumer_importer_{flag}_not_false")
    return tuple(violations)


def _queue_importer_summary_violations(
    summary: Mapping[str, Any],
    *,
    importer_status: str,
    plan_records: Tuple[Mapping[str, Any], ...],
    specs: Tuple[Mapping[str, Any], ...],
    results: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    applied = importer_status == REQUIRED_QUEUE_IMPORTER_APPLIED_STATUS
    violations = []
    expected_counts = {
        "queue_ordering_plan_record_count": len(plan_records),
        "queue_import_spec_count": len(specs),
        "queue_import_result_count": len(results),
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(f"queue_importer_result_consumer_importer_summary_{key}_mismatch")
    if summary.get("queue_import_dry_run") is not (not applied):
        violations.append("queue_importer_result_consumer_importer_summary_dry_run_mismatch")
    if summary.get("queue_import_applied") is not applied:
        violations.append("queue_importer_result_consumer_importer_summary_applied_mismatch")
    if summary.get("queue_mutation") is not applied:
        violations.append("queue_importer_result_consumer_importer_summary_queue_mutation_mismatch")
    if applied and summary.get("operator_ack") is not True:
        violations.append("queue_importer_result_consumer_importer_summary_ack_not_true")
    if applied and summary.get("apply_requested") is not True:
        violations.append("queue_importer_result_consumer_importer_summary_apply_not_true")
    for flag in REQUIRED_IMPORTER_NEVER_TRUE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"queue_importer_result_consumer_importer_summary_{flag}_not_false"
            )
    expected_ids = tuple(
        str(record["proposal_id"])
        for record in plan_records
        if isinstance(record.get("proposal_id"), str) and record.get("proposal_id")
    )
    if _string_list(summary.get("planned_import_proposal_ids")) != expected_ids:
        violations.append("queue_importer_result_consumer_importer_summary_planned_ids_mismatch")
    if applied and _string_list(summary.get("imported_proposal_ids")) != expected_ids:
        violations.append("queue_importer_result_consumer_importer_summary_imported_ids_mismatch")
    return tuple(violations)


def _spec_violations(
    spec: Mapping[str, Any],
    *,
    plan_record: Mapping[str, Any],
) -> Tuple[str, ...]:
    violations = []
    expected = {
        "queue_ordering_plan_record_id": plan_record.get("queue_ordering_plan_record_id"),
        "proposal_id": plan_record.get("proposal_id"),
        "planned_queue_index": plan_record.get("planned_queue_index"),
        "rank_source": plan_record.get("rank_source"),
    }
    for key, value in expected.items():
        if spec.get(key) != value:
            violations.append(f"queue_importer_result_consumer_spec_{key}_mismatch")
    if spec.get("operator_ack_required_before_queue_import") is not True:
        violations.append("queue_importer_result_consumer_spec_queue_ack_not_true")
    if spec.get("operator_ack_required_before_sequence_execution") is not True:
        violations.append("queue_importer_result_consumer_spec_sequence_ack_not_true")
    if spec.get("sequence_consumer_required_before_execution") is not True:
        violations.append("queue_importer_result_consumer_spec_sequence_consumer_not_true")
    if spec.get("queue_import_dry_run") is not True:
        violations.append("queue_importer_result_consumer_spec_dry_run_not_true")
    if spec.get("queue_import_applied") is not False:
        violations.append("queue_importer_result_consumer_spec_applied_not_false")
    for flag in (
        "queue_mutation",
        "proposal_order_mutation",
        "mutates_console_state",
        "queries_console",
        "model_mutation",
        "runtime_memory_mutation",
        "sequence_policy_trust_store_mutation",
        "proposal_state_mutation",
        "sequence_execution_allowed_by_importer",
        "sequence_policy_change_allowed_by_importer",
        "live_widening_allowed",
        "live_write",
        "live_authority_granted",
        "full_autonomous_ready",
        "can_import_to_queue",
        "can_apply_live",
    ):
        if spec.get(flag) is not False:
            violations.append(f"queue_importer_result_consumer_spec_{flag}_not_false")
    payload = _mapping(spec.get("operator_proposal_payload"))
    metadata = _mapping(payload.get("sequence_policy_queue_import"))
    if metadata.get("planned_queue_index") != plan_record.get("planned_queue_index"):
        violations.append("queue_importer_result_consumer_spec_payload_index_mismatch")
    if metadata.get("live_write") is not False:
        violations.append("queue_importer_result_consumer_spec_payload_live_write_not_false")
    if metadata.get("can_apply_live") is not False:
        violations.append("queue_importer_result_consumer_spec_payload_can_apply_live_not_false")
    return tuple(violations)


def _result_violations(
    result: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
) -> Tuple[str, ...]:
    violations = []
    expected = {
        "queue_import_spec_id": spec.get("queue_import_spec_id"),
        "proposal_id": spec.get("proposal_id"),
        "planned_queue_index": spec.get("planned_queue_index"),
    }
    for key, value in expected.items():
        if result.get(key) != value:
            violations.append(f"queue_importer_result_consumer_result_{key}_mismatch")
    if result.get("success") is not True:
        violations.append("queue_importer_result_consumer_result_success_not_true")
    if result.get("queue_import_applied") is not True:
        violations.append("queue_importer_result_consumer_result_applied_not_true")
    if result.get("queue_mutation") is not True:
        violations.append("queue_importer_result_consumer_result_queue_mutation_not_true")
    for flag in REQUIRED_RESULT_FALSE_FLAGS:
        if result.get(flag) is not False:
            violations.append(f"queue_importer_result_consumer_result_{flag}_not_false")
    proposal = _proposal_from_import_result(result)
    if not proposal:
        violations.append("queue_importer_result_consumer_result_operator_proposal_missing")
        return tuple(violations)
    if proposal.get("id") != spec.get("proposal_id"):
        violations.append("queue_importer_result_consumer_result_proposal_id_mismatch")
    metadata = _sequence_import_metadata(proposal)
    if not metadata:
        violations.append("queue_importer_result_consumer_result_import_metadata_missing")
    elif metadata.get("planned_queue_index") != spec.get("planned_queue_index"):
        violations.append("queue_importer_result_consumer_result_import_index_mismatch")
    if metadata and metadata.get("live_write") is not False:
        violations.append("metadata_live_write_not_false")
    if metadata and metadata.get("can_apply_live") is not False:
        violations.append("metadata_can_apply_live_not_false")
    return tuple(violations)


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer(
    *,
    promotion_queue_ordering_plan_result_consumer: Mapping[str, Any] | None = None,
    promotion_queue_importer_result: Mapping[str, Any] | None = None,
    idempotency_key_prefix: str = "sequence-policy-queue-import",
    limit: int = 20,
) -> Dict[str, Any]:
    """Verify queue-importer output and expose only a preflight-plan template."""

    prior = _mapping(promotion_queue_ordering_plan_result_consumer)
    importer = _mapping(promotion_queue_importer_result)
    prior_request = _mapping(prior.get("promotion_queue_importer_request"))
    queue_ordering_plan = _queue_ordering_plan_from_prior_consumer(prior)
    plan_records = _record_list(queue_ordering_plan.get("queue_ordering_plan_records"))
    specs = _record_list(importer.get("queue_import_specs"))
    results = _record_list(importer.get("queue_import_results"))
    plan_records_by_proposal_id = {
        str(record["proposal_id"]): record
        for record in plan_records
        if isinstance(record.get("proposal_id"), str) and record.get("proposal_id")
    }
    specs_by_proposal_id = {
        str(spec["proposal_id"]): spec
        for spec in specs
        if isinstance(spec.get("proposal_id"), str) and spec.get("proposal_id")
    }
    raw_invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_string_list(_mapping(prior.get("summary")).get("invalid_record_ids")),
                *_string_list(_mapping(importer.get("summary")).get("invalid_record_ids")),
                *_prior_consumer_envelope_violations(prior),
                *_prior_consumer_summary_violations(
                    _mapping(prior.get("summary")),
                    plan_records=plan_records,
                ),
                *_prior_consumer_request_violations(
                    prior_request,
                    plan_records=plan_records,
                ),
                *_queue_importer_envelope_violations(importer),
                *_queue_importer_safety_violations(
                    _mapping(importer.get("safety")),
                    importer_status=_string(importer.get("status")),
                ),
                *_queue_importer_summary_violations(
                    _mapping(importer.get("summary")),
                    importer_status=_string(importer.get("status")),
                    plan_records=plan_records,
                    specs=specs,
                    results=results,
                ),
                *(
                    f"{proposal_id}:queue_importer_result_consumer_spec_missing"
                    for proposal_id in plan_records_by_proposal_id
                    if proposal_id not in specs_by_proposal_id
                ),
                *(
                    f"{_string(spec.get('proposal_id'))}:"
                    f"{violation}"
                    for spec in specs
                    for violation in _spec_violations(
                        spec,
                        plan_record=plan_records_by_proposal_id.get(
                            _string(spec.get("proposal_id")),
                            {},
                        ),
                    )
                ),
                *(
                    f"{_string(result.get('proposal_id'))}:"
                    f"{violation}"
                    for result in results
                    for violation in _result_violations(
                        result,
                        spec=specs_by_proposal_id.get(
                            _string(result.get("proposal_id")),
                            {},
                        ),
                    )
                ),
            )
        )
    )
    invalid_record_ids = () if not prior or not importer else raw_invalid_record_ids

    prior_status = _string(prior.get("status"))
    importer_status = _string(importer.get("status"))
    if not prior or not importer:
        status = "sequence_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer_not_ready"
        blocked_by = ("promotion_queue_ordering_plan_result_consumer_and_queue_importer_result_required",)
    elif invalid_record_ids:
        status = "sequence_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer_replacement_required"
        blocked_by = (
            "operator_must_replace_invalid_sequence_memory_trust_promotion_queue_importer_evidence",
        )
    elif prior_status != REQUIRED_QUEUE_ORDERING_PLAN_RESULT_CONSUMER_AVAILABLE_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer_blocked"
        blocked_by = (
            "promotion_queue_ordering_plan_result_consumer_must_request_queue_importer_first",
        )
    elif importer_status != REQUIRED_QUEUE_IMPORTER_APPLIED_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer_blocked"
        blocked_by = (
            "applied_sequence_policy_queue_importer_result_required_before_preflight_plan",
        )
    elif not results:
        status = "sequence_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer_not_ready"
        blocked_by = ("applied_queue_import_results_required_before_preflight_plan",)
    else:
        status = "sequence_memory_trust_promotion_queue_import_promotion_queue_import_preflight_plan_request_available"
        blocked_by = (
            "operator_may_run_read_only_sequence_policy_queue_import_preflight_plan",
        )

    request_available = (
        status
        == "sequence_memory_trust_promotion_queue_import_promotion_queue_import_preflight_plan_request_available"
    )
    preflight_plan_request: Dict[str, Any] = {
        "handler": "plan_operator_proposal_apply_sequence_policy_queue_import_preflight",
        "read_only_template": True,
        "promotion_queue_importer_result_consumer_only": True,
        "promotion_queue_import_preflight_plan_request_template_only": True,
        "promotion_preflight_plan_request_template_only": True,
        "pending_actions_required_from_operator_queue": True,
        "pending_actions_source": "operator_proposal_queue",
        "requires_applied_queue_import_result": True,
        "idempotency_key_prefix": str(
            idempotency_key_prefix or "sequence-policy-queue-import"
        ),
        "limit": max(1, min(int(limit or 20), 50)),
        "promotion_queue_import_preflight_plan_handler_invoked": False,
        "promotion_preflight_plan_handler_invoked": False,
        "promotion_queue_import_preflight_plan_created": False,
        "promotion_preflight_plan_created": False,
        "batch_preflight_invoked": False,
        "sequence_plan_created": False,
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
        preflight_plan_request["imported_proposal_ids"] = [
            str(result["proposal_id"])
            for result in results
            if isinstance(result.get("proposal_id"), str) and result.get("proposal_id")
        ]
        preflight_plan_request["promotion_queue_importer_result"] = importer

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_IMPORTER_RESULT_CONSUMER_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_QUEUE_ORDERING_PLAN_RESULT_CONSUMER_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "promotion_queue_importer_result_consumer_only": True,
            "promotion_queue_import_preflight_plan_request_template_only": True,
            "promotion_preflight_plan_request_template_only": True,
            "requires_promotion_queue_ordering_plan_result_consumer": True,
            "requires_applied_queue_import_result": True,
            "promotion_queue_import_preflight_plan_handler_invoked": False,
            "promotion_preflight_plan_handler_invoked": False,
            "promotion_queue_import_preflight_plan_created": False,
            "promotion_preflight_plan_created": False,
            "batch_preflight_invoked": False,
            "sequence_plan_created": False,
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
        },
        "summary": {
            "queue_ordering_plan_record_count": len(plan_records),
            "expected_queue_ordering_plan_record_count": len(plan_records),
            "queue_import_spec_count": len(specs),
            "queue_import_result_count": len(results),
            "imported_proposal_count": len(
                tuple(
                    result
                    for result in results
                    if result.get("queue_import_applied") is True
                )
            ),
            "importer_dry_run_seen": importer_status
            == "sequence_memory_trust_promotion_queue_importer_dry_run_ready",
            "invalid_record_count": len(invalid_record_ids),
            "promotion_queue_importer_result_consumer_only": True,
            "promotion_queue_import_preflight_plan_request_template_only": True,
            "promotion_preflight_plan_request_template_only": True,
            "promotion_queue_import_preflight_plan_request_available": request_available,
            "promotion_preflight_plan_request_available": request_available,
            "promotion_queue_import_preflight_plan_handler_invoked": False,
            "promotion_preflight_plan_handler_invoked": False,
            "promotion_queue_import_preflight_plan_created": False,
            "promotion_preflight_plan_created": False,
            "batch_preflight_invoked": False,
            "sequence_plan_created": False,
            "requires_applied_queue_import_result": True,
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
            "planned_import_proposal_ids": [
                str(record["proposal_id"])
                for record in plan_records
                if isinstance(record.get("proposal_id"), str)
                and record.get("proposal_id")
            ],
            "imported_proposal_ids": [
                str(result["proposal_id"])
                for result in results
                if isinstance(result.get("proposal_id"), str)
                and result.get("proposal_id")
            ],
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Provide promotion queue-ordering-plan result consumer and queue-importer result."
                if not prior or not importer
                else (
                    "Replace invalid sequence-memory trust promotion queue-importer evidence before preflight planning."
                    if invalid_record_ids
                    else (
                        "Operator-acknowledged queue import is required before preflight planning; dry-run importer evidence is not enough."
                        if importer_status != REQUIRED_QUEUE_IMPORTER_APPLIED_STATUS
                        else "Run the separate read-only preflight-plan handler against the operator queue; this consumer does not invoke preflight or reserve idempotency."
                    )
                )
            ),
        },
        "promotion_queue_import_preflight_plan_request": preflight_plan_request,
        "promotion_preflight_plan_request": preflight_plan_request,
    }
