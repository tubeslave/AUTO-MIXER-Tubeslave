"""Read-only consumer for sequence-policy queue-import batch preflight results.

This module verifies that a batch preflight payload matches the read-only
sequence-policy queue-import preflight plan before a caller may pass the results
to the generic sequence-plan handler. It does not invoke handlers, reserve
idempotency keys, mutate queue or proposal state, create a sequence plan, execute
sequences, or touch live mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_PLAN_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_PLAN_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_PLAN_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_PLAN_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan,
    )
try:
    from operator_mode_policy import CURRENT_PROVEN_LIVE_WRITE_KINDS
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_mode_policy import CURRENT_PROVEN_LIVE_WRITE_KINDS

ALLOWED_VALUE_TYPES = set(CURRENT_PROVEN_LIVE_WRITE_KINDS)


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_RESULT_CONSUMER_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_RESULT_CONSUMER_STATUSES = {
    "sequence_memory_trust_promotion_queue_import_preflight_result_consumer_not_ready",
    "sequence_memory_trust_promotion_queue_import_preflight_result_consumer_replacement_required",
    "sequence_memory_trust_promotion_queue_import_preflight_result_consumer_blocked",
    "sequence_memory_trust_promotion_queue_import_sequence_plan_request_available",
}
REQUIRED_PREFLIGHT_PLAN_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan"
)
REQUIRED_BATCH_PREFLIGHT_TYPE = "operator_proposal_apply_preflight_batch"
REQUIRED_PREFLIGHT_PLAN_FALSE_FLAGS = (
    "live_write",
    "apply_attempted",
    "queue_mutation",
    "proposal_status_mutation",
    "idempotency_reserved",
    "manual_console_write_called",
    "sequence_plan_created",
    "sequence_execution_allowed_by_preflight_plan",
    "live_authority_granted",
    "full_autonomous_ready",
    "can_apply_live",
)
REQUIRED_BATCH_FALSE_FLAGS = (
    "live_write",
    "apply_attempted",
    "queue_mutation",
    "proposal_status_mutation",
    "idempotency_reserved",
    "manual_console_write_called",
)
REQUIRED_RESULT_FALSE_FLAGS = (
    "live_write",
    "apply_attempted",
    "queue_mutation",
    "proposal_status_mutation",
    "idempotency_reserved",
    "manual_console_write_called",
)
REQUIRED_PREFLIGHT_SPEC_FALSE_FLAGS = (
    "live_write",
    "apply_attempted",
    "queue_mutation",
    "proposal_status_mutation",
    "idempotency_reserved",
    "manual_console_write_called",
    "sequence_plan_created",
    "sequence_execution_allowed_by_preflight_plan",
    "live_authority_granted",
    "full_autonomous_ready",
    "can_apply_live",
)
REQUIRED_PREFLIGHT_SPEC_TRUE_FLAGS = (
    "operator_approval_required_for_preflight_ready",
    "sequence_consumer_required_before_execution",
    "batch_preflight_only",
    "read_only",
)
REQUIRED_PREFLIGHT_SPEC_IDENTITY_FIELDS = (
    "queue_ordering_plan_record_id",
    "consumer_record_id",
    "sequence_policy_gate_record_id",
    "policy_key",
    "source_idempotency_key",
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


def _context_status(payload: Mapping[str, Any]) -> str:
    return _string(payload.get("status"))


def _context_violations(proposal_id: str, payload: Mapping[str, Any]) -> Tuple[str, ...]:
    if not payload:
        return (f"{proposal_id}:current_sequence_review_freshness_missing",)
    violations = []
    if _context_status(payload) != "fresh":
        violations.append(f"{proposal_id}:current_sequence_review_freshness_not_fresh")
    if payload.get("read_only") is not True:
        violations.append(
            f"{proposal_id}:current_sequence_review_freshness_read_only_not_true"
        )
    for flag in (
        "live_write",
        "apply_attempted",
        "queries_console",
        "queue_mutation",
        "policy_mutation",
    ):
        if payload.get(flag) is not False:
            violations.append(
                f"{proposal_id}:current_sequence_review_freshness_{flag}_not_false"
            )
    return tuple(violations)


def _duplicate_values(values: Tuple[str, ...]) -> Tuple[str, ...]:
    counts: Dict[str, int] = {}
    for value in values:
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return tuple(value for value, count in counts.items() if count > 1)


def _preflight_result_id(result: Mapping[str, Any], index: int) -> str:
    proposal = _mapping(result.get("proposal"))
    for value in (
        result.get("proposal_id"),
        proposal.get("id"),
        proposal.get("proposal_id"),
    ):
        as_string = _string(value)
        if as_string:
            return as_string
    return f"preflight_result:{index}"


def _plan_envelope_violations(plan: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if plan.get("type") != REQUIRED_PREFLIGHT_PLAN_TYPE:
        violations.append("sequence_policy_preflight_result_consumer_plan_type_mismatch")
    if (
        plan.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_PLAN_METHOD_ID
    ):
        violations.append("sequence_policy_preflight_result_consumer_plan_method_id_mismatch")
    if (
        plan.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_PLAN_STATUSES
    ):
        violations.append("sequence_policy_preflight_result_consumer_plan_status_unknown")
    safety = _mapping(plan.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("sequence_policy_preflight_result_consumer_plan_read_only_not_true")
    if safety.get("batch_preflight_only") is not True:
        violations.append("sequence_policy_preflight_result_consumer_plan_only_not_true")
    if safety.get("batch_preflight_invoked") is not False:
        violations.append(
            "sequence_policy_preflight_result_consumer_plan_preflight_invoked_not_false"
        )
    if safety.get("operator_approval_required_for_preflight_ready") is not True:
        violations.append("sequence_policy_preflight_result_consumer_plan_operator_approval_not_true")
    if safety.get("sequence_consumer_required_before_execution") is not True:
        violations.append("sequence_policy_preflight_result_consumer_plan_sequence_consumer_not_true")
    for flag in REQUIRED_PREFLIGHT_PLAN_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"sequence_policy_preflight_result_consumer_plan_{flag}_not_false")
    return tuple(violations)


def _plan_status_violations(
    plan: Mapping[str, Any],
    *,
    planned_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    status = plan.get("status")
    if status == "sequence_memory_trust_promotion_queue_import_preflight_plan_replacement_required":
        return ("sequence_policy_preflight_result_consumer_plan_replacement_required",)
    if status == "sequence_memory_trust_promotion_queue_import_preflight_plan_not_ready":
        return ("sequence_policy_preflight_result_consumer_plan_not_ready",)
    if (
        status == "sequence_memory_trust_promotion_queue_import_preflight_plan_available"
        and not planned_ids
    ):
        return ("sequence_policy_preflight_result_consumer_plan_available_without_ids",)
    return ()


def _plan_summary_violations(
    summary: Mapping[str, Any],
    *,
    planned_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    violations = []
    expected_counts = {
        "preflight_candidate_count": len(planned_ids),
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(f"sequence_policy_preflight_result_consumer_plan_summary_{key}_mismatch")
    if summary.get("batch_preflight_request_available") is not bool(planned_ids):
        violations.append(
            "sequence_policy_preflight_result_consumer_plan_summary_request_available_mismatch"
        )
    if summary.get("batch_preflight_invoked") is not False:
        violations.append(
            "sequence_policy_preflight_result_consumer_plan_summary_preflight_invoked_not_false"
        )
    if summary.get("sequence_plan_created") is not False:
        violations.append(
            "sequence_policy_preflight_result_consumer_plan_summary_sequence_plan_created_not_false"
        )
    for flag in REQUIRED_PREFLIGHT_PLAN_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_policy_preflight_result_consumer_plan_summary_{flag}_not_false"
            )
    if _string_list(summary.get("preflight_candidate_proposal_ids")) != planned_ids:
        violations.append(
            "sequence_policy_preflight_result_consumer_plan_summary_candidate_ids_mismatch"
        )
    return tuple(violations)


def _batch_envelope_violations(batch: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if batch.get("type") != REQUIRED_BATCH_PREFLIGHT_TYPE:
        violations.append("sequence_policy_preflight_result_consumer_batch_type_mismatch")
    if batch.get("status") != "ok":
        violations.append("sequence_policy_preflight_result_consumer_batch_status_not_ok")
    if batch.get("success") is not True:
        violations.append("sequence_policy_preflight_result_consumer_batch_success_not_true")
    safety = _mapping(batch.get("safety"))
    if safety.get("batch_preflight_only") is not True:
        violations.append("sequence_policy_preflight_result_consumer_batch_only_not_true")
    if safety.get("live_apply_entrypoint") != "apply_operator_proposal":
        violations.append("sequence_policy_preflight_result_consumer_batch_entrypoint_mismatch")
    for flag in REQUIRED_BATCH_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"sequence_policy_preflight_result_consumer_batch_{flag}_not_false")
    return tuple(violations)


def _batch_summary_violations(
    summary: Mapping[str, Any],
    *,
    planned_ids: Tuple[str, ...],
    results: Tuple[Mapping[str, Any], ...],
    ready_ids: Tuple[str, ...],
    blocked_ids: Tuple[str, ...],
    prefix: str,
) -> Tuple[str, ...]:
    violations = []
    expected_counts = {
        "selected_count": len(planned_ids),
        "preflight_result_count": len(results),
        "ready_count": len(ready_ids),
        "blocked_count": len(blocked_ids),
        "not_found_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(f"sequence_policy_preflight_result_consumer_batch_summary_{key}_mismatch")
    if summary.get("approved") is not True:
        violations.append("sequence_policy_preflight_result_consumer_batch_summary_approved_not_true")
    if summary.get("idempotency_key_prefix") != prefix:
        violations.append(
            "sequence_policy_preflight_result_consumer_batch_summary_idempotency_prefix_mismatch"
        )
    if _string_list(summary.get("ready_proposal_ids")) != ready_ids:
        violations.append("sequence_policy_preflight_result_consumer_batch_summary_ready_ids_mismatch")
    if _string_list(summary.get("blocked_proposal_ids")) != blocked_ids:
        violations.append("sequence_policy_preflight_result_consumer_batch_summary_blocked_ids_mismatch")
    if _string_list(summary.get("not_found_ids")):
        violations.append("sequence_policy_preflight_result_consumer_batch_summary_not_found_ids_present")
    for flag in ("live_write", "apply_attempted", "queue_mutation", "idempotency_reserved"):
        if summary.get(flag) is not False:
            violations.append(f"sequence_policy_preflight_result_consumer_batch_summary_{flag}_not_false")
    return tuple(violations)


def _preflight_result_violations(
    result: Mapping[str, Any],
    *,
    proposal_id: str,
    prefix: str,
) -> Tuple[str, ...]:
    violations = []
    ready = result.get("ready_to_apply") is True
    if result.get("type") != "operator_proposal_apply_preflight":
        violations.append(f"{proposal_id}:preflight_result_type_mismatch")
    if result.get("status") != ("ready" if ready else "blocked"):
        violations.append(f"{proposal_id}:preflight_result_status_mismatch")
    if result.get("success") is not ready:
        violations.append(f"{proposal_id}:preflight_result_success_mismatch")
    safety = _mapping(result.get("safety"))
    if safety.get("preflight_only") is not True:
        violations.append(f"{proposal_id}:preflight_result_only_not_true")
    if safety.get("live_apply_entrypoint") != "apply_operator_proposal":
        violations.append(f"{proposal_id}:preflight_result_live_apply_entrypoint_mismatch")
    for flag in REQUIRED_RESULT_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"{proposal_id}:preflight_result_{flag}_not_false")
    idempotency = _mapping(result.get("idempotency"))
    if idempotency.get("provided") is not True:
        violations.append(f"{proposal_id}:idempotency_provided_not_true")
    if idempotency.get("idempotency_key") != f"{prefix}:{proposal_id}":
        violations.append(f"{proposal_id}:idempotency_key_mismatch")
    if idempotency.get("reserved") is not False:
        violations.append(f"{proposal_id}:idempotency_reserved_not_false")
    if idempotency.get("duplicate") is not False:
        violations.append(f"{proposal_id}:idempotency_duplicate_not_false")
    allowed_kinds = _string_list(safety.get("allowed_live_write_kinds"))
    if any(kind not in ALLOWED_VALUE_TYPES for kind in allowed_kinds):
        violations.append(f"{proposal_id}:allowed_live_write_kind_not_sequence_safe")
    return tuple(violations)


def _preflight_specs_violations(
    specs: Tuple[Mapping[str, Any], ...],
    *,
    planned_ids: Tuple[str, ...],
    prefix: str,
) -> Tuple[str, ...]:
    violations = []
    spec_ids = tuple(_string(spec.get("proposal_id")) for spec in specs)
    if spec_ids != planned_ids:
        violations.append("sequence_policy_preflight_result_consumer_specs_candidate_ids_mismatch")
    for index, spec in enumerate(specs):
        proposal_id = _string(spec.get("proposal_id")) or f"preflight_spec:{index}"
        record_id = _string(spec.get("preflight_request_spec_id")) or proposal_id
        if spec.get("planned_preflight_index") != index:
            violations.append(f"{record_id}:planned_preflight_index_mismatch")
        if spec.get("idempotency_key") != f"{prefix}:{proposal_id}":
            violations.append(f"{record_id}:idempotency_key_mismatch")
        if _summary_int(spec.get("planned_queue_index")) is None:
            violations.append(f"{proposal_id}:planned_queue_index_invalid")
        if spec.get("rank_source") != "sequence_policy_consumer":
            violations.append(f"{proposal_id}:rank_source_mismatch")
        for field in REQUIRED_PREFLIGHT_SPEC_IDENTITY_FIELDS:
            if not _string(spec.get(field)):
                violations.append(f"{proposal_id}:{field}_missing")
        violations.extend(
            _context_violations(
                proposal_id,
                _mapping(spec.get("current_sequence_review_freshness")),
            )
        )
        source_idempotency_key = _string(spec.get("source_idempotency_key"))
        policy_key = _string(spec.get("policy_key"))
        if (
            source_idempotency_key
            and policy_key
            and policy_key != f"{proposal_id}/{source_idempotency_key}"
        ):
            violations.append(f"{proposal_id}:policy_key_identity_mismatch")
        for flag in REQUIRED_PREFLIGHT_SPEC_TRUE_FLAGS:
            if spec.get(flag) is not True:
                violations.append(f"{record_id}:{flag}_not_true")
        for flag in REQUIRED_PREFLIGHT_SPEC_FALSE_FLAGS:
            if spec.get(flag) is not False:
                violations.append(f"{record_id}:{flag}_not_false")
    return tuple(violations)


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer(
    *,
    queue_import_preflight_plan: Mapping[str, Any] | None = None,
    batch_preflight_result: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Verify a queue-import preflight result and emit a sequence-plan template."""

    plan = _mapping(queue_import_preflight_plan)
    batch = _mapping(batch_preflight_result)
    plan_request = _mapping(plan.get("batch_preflight_request"))
    planned_ids = _string_list(plan_request.get("proposal_ids"))
    prefix = _string(plan_request.get("idempotency_key_prefix")) or _string(
        _mapping(plan.get("summary")).get("idempotency_key_prefix")
    )
    preflight_specs = _record_list(plan.get("preflight_request_specs"))
    preflight_specs_by_proposal_id = {
        _string(spec.get("proposal_id")): spec
        for spec in preflight_specs
        if _string(spec.get("proposal_id"))
    }
    results = _record_list(batch.get("preflight_results"))
    result_ids = tuple(
        _preflight_result_id(result, index) for index, result in enumerate(results)
    )
    ready_ids = tuple(
        _preflight_result_id(result, index)
        for index, result in enumerate(results)
        if result.get("ready_to_apply") is True
    )
    blocked_ids = tuple(
        _preflight_result_id(result, index)
        for index, result in enumerate(results)
        if result.get("ready_to_apply") is not True
    )
    result_violations = tuple(
        violation
        for index, result in enumerate(results)
        for violation in _preflight_result_violations(
            result,
            proposal_id=_preflight_result_id(result, index),
            prefix=prefix,
        )
    )
    invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_plan_envelope_violations(plan),
                *_plan_status_violations(plan, planned_ids=planned_ids),
                *_plan_summary_violations(_mapping(plan.get("summary")), planned_ids=planned_ids),
                *_preflight_specs_violations(
                    preflight_specs,
                    planned_ids=planned_ids,
                    prefix=prefix,
                ),
                *_batch_envelope_violations(batch),
                *_batch_summary_violations(
                    _mapping(batch.get("summary")),
                    planned_ids=planned_ids,
                    results=results,
                    ready_ids=ready_ids,
                    blocked_ids=blocked_ids,
                    prefix=prefix,
                ),
                *(
                    f"duplicate_preflight_result_id:{value}"
                    for value in _duplicate_values(result_ids)
                ),
                *(
                    ("sequence_policy_preflight_result_consumer_result_order_mismatch",)
                    if result_ids != planned_ids
                    else ()
                ),
                *result_violations,
            )
        )
    )

    if not plan or not batch:
        status = (
            "sequence_memory_trust_promotion_queue_import_preflight_result_consumer_not_ready"
        )
        blocked_by = ("queue_import_preflight_plan_and_batch_preflight_result_required",)
    elif invalid_record_ids:
        status = (
            "sequence_memory_trust_promotion_queue_import_preflight_result_consumer_replacement_required"
        )
        blocked_by = ("operator_must_replace_invalid_sequence_policy_preflight_result_evidence",)
    elif blocked_ids:
        status = "sequence_memory_trust_promotion_queue_import_preflight_result_consumer_blocked"
        blocked_by = ("all_sequence_policy_imported_preflight_results_must_be_ready",)
    elif not ready_ids:
        status = (
            "sequence_memory_trust_promotion_queue_import_preflight_result_consumer_not_ready"
        )
        blocked_by = ("ready_batch_preflight_results_required",)
    else:
        status = (
            "sequence_memory_trust_promotion_queue_import_sequence_plan_request_available"
        )
        blocked_by = ("operator_may_run_generic_sequence_plan_handler",)

    request_available = (
        status
        == "sequence_memory_trust_promotion_queue_import_sequence_plan_request_available"
    )
    selected_results = results if request_available else ()
    selected_preflight_specs = tuple(
        preflight_specs_by_proposal_id.get(proposal_id, {})
        for proposal_id in ready_ids
        if request_available
    )
    sequence_plan_request_specs = tuple(
        {
            "sequence_plan_request_spec_id": (
                f"{OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_RESULT_CONSUMER_METHOD_ID}:{proposal_id}"
            ),
            "proposal_id": proposal_id,
            "planned_sequence_index": index,
            "source_preflight_index": index,
            "source_preflight_request_spec_id": _string(
                preflight_specs_by_proposal_id.get(proposal_id, {}).get(
                    "preflight_request_spec_id"
                )
            ),
            "queue_ordering_plan_record_id": preflight_specs_by_proposal_id.get(
                proposal_id, {}
            ).get("queue_ordering_plan_record_id"),
            "consumer_record_id": preflight_specs_by_proposal_id.get(
                proposal_id, {}
            ).get("consumer_record_id"),
            "sequence_policy_gate_record_id": preflight_specs_by_proposal_id.get(
                proposal_id, {}
            ).get("sequence_policy_gate_record_id"),
            "policy_key": preflight_specs_by_proposal_id.get(proposal_id, {}).get(
                "policy_key"
            ),
            "source_idempotency_key": preflight_specs_by_proposal_id.get(
                proposal_id, {}
            ).get("source_idempotency_key"),
            "current_sequence_review_freshness": preflight_specs_by_proposal_id.get(
                proposal_id, {}
            ).get("current_sequence_review_freshness"),
            "planned_queue_index": preflight_specs_by_proposal_id.get(
                proposal_id, {}
            ).get("planned_queue_index"),
            "rank_source": preflight_specs_by_proposal_id.get(proposal_id, {}).get(
                "rank_source"
            ),
            "idempotency_key": f"{prefix}:{proposal_id}",
            "read_only": True,
            "preflight_result_consumer_only": True,
            "sequence_plan_handler": "plan_operator_proposal_apply_sequence",
            "operator_sequence_plan_step_required_before_dry_run": True,
            "sequence_plan_created": False,
            "sequence_execution_allowed_by_consumer": False,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_apply_live": False,
        }
        for index, proposal_id in enumerate(ready_ids if request_available else ())
    )

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_RESULT_CONSUMER_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_PLAN_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "preflight_result_consumer_only": True,
            "sequence_plan_request_template_only": True,
            "requires_queue_import_preflight_plan": True,
            "requires_batch_preflight_result": True,
            "requires_all_preflight_results_ready": True,
            "sequence_plan_handler_invoked": False,
            "sequence_plan_created": False,
            "operator_sequence_plan_step_required_before_dry_run": True,
            "sequence_execution_allowed_by_consumer": False,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_apply_live": False,
        },
        "summary": {
            "planned_preflight_candidate_count": len(planned_ids),
            "preflight_request_spec_count": len(preflight_specs),
            "batch_preflight_result_count": len(results),
            "ready_preflight_result_count": len(ready_ids),
            "blocked_preflight_result_count": len(blocked_ids),
            "invalid_record_count": len(invalid_record_ids),
            "idempotency_key_prefix": prefix,
            "preflight_result_consumer_only": True,
            "sequence_plan_request_available": request_available,
            "sequence_plan_handler_invoked": False,
            "sequence_plan_created": False,
            "sequence_execution_allowed_by_consumer": False,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_apply_live": False,
            "planned_preflight_candidate_proposal_ids": list(planned_ids),
            "preflight_request_spec_proposal_ids": [
                spec.get("proposal_id") for spec in preflight_specs
            ],
            "batch_preflight_result_proposal_ids": list(result_ids),
            "ready_preflight_result_proposal_ids": list(ready_ids),
            "blocked_preflight_result_proposal_ids": list(blocked_ids),
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Provide queue-import preflight plan plus approved batch preflight result."
                if not plan or not batch
                else (
                    "Replace invalid preflight-result evidence before sequence planning."
                    if invalid_record_ids
                    else (
                        "Re-run batch preflight until every sequence-policy imported candidate is ready."
                        if blocked_ids
                        else "Operator may run the generic sequence-plan handler; this consumer does not run it."
                    )
                )
            ),
        },
        "sequence_plan_request": {
            "handler": "plan_operator_proposal_apply_sequence",
            "preflight_results": list(selected_results),
            "preflight_request_specs": list(selected_preflight_specs),
            "read_only_template": True,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "sequence_plan_created": False,
            "sequence_execution_allowed_by_consumer": False,
        },
        "sequence_plan_request_specs": list(sequence_plan_request_specs),
    }
