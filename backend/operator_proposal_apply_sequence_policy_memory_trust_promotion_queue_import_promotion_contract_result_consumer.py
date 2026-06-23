"""Read-only consumer for sequence-policy memory trust promotion contracts.

This verifier connects a queue-import promotion-plan result to the existing
promotion acceptance-ledger builder. It verifies the promotion contract result
and emits only an operator acceptance-ledger request template. It does not
accept contract records, execute proposals, promote policy, mutate queues,
policy stores, runtime memory, models, append-only memory, proposal state,
idempotency, or mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_contract import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_CONTRACT_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_CONTRACT_STATUSES,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_plan import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_PLAN_METHOD_ID,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_PLAN_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_PLAN_RESULT_CONSUMER_STATUSES,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_contract import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_CONTRACT_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_CONTRACT_STATUSES,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_plan import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_PLAN_METHOD_ID,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_PLAN_RESULT_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_PLAN_RESULT_CONSUMER_STATUSES,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_CONTRACT_RESULT_CONSUMER_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_CONTRACT_RESULT_CONSUMER_STATUSES = {
    "sequence_memory_trust_promotion_queue_import_promotion_contract_result_consumer_not_ready",
    "sequence_memory_trust_promotion_queue_import_promotion_contract_result_consumer_replacement_required",
    "sequence_memory_trust_promotion_queue_import_promotion_contract_result_consumer_blocked",
    "sequence_memory_trust_promotion_queue_import_promotion_acceptance_ledger_request_available",
}
REQUIRED_PROMOTION_PLAN_RESULT_CONSUMER_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer"
)
REQUIRED_PROMOTION_PLAN_RESULT_CONSUMER_AVAILABLE_STATUS = (
    "sequence_memory_trust_promotion_queue_import_promotion_contract_request_available"
)
REQUIRED_PROMOTION_CONTRACT_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_contract"
)
REQUIRED_PROMOTION_CONTRACT_READY_STATUS = (
    "manual_sequence_memory_trust_promotion_contract_ready"
)
REQUIRED_PLAN_STATUS = "sequence_memory_trust_promotion_plan_candidate_available"
REQUIRED_CONSUMER_FALSE_FLAGS = (
    "promotion_contract_handler_invoked",
    "promotion_contract_created",
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
REQUIRED_CONTRACT_FALSE_SAFETY_FLAGS = (
    "live_write",
    "apply_attempted",
    "mutates_console_state",
    "queries_console",
    "execution_allowed_by_contract",
    "sequence_policy_promotion_allowed_by_contract",
    "queue_mutation_allowed_by_contract",
    "policy_mutation_allowed_by_contract",
    "model_mutation_allowed_by_contract",
    "runtime_memory_mutation_allowed_by_contract",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_CONTRACT_FALSE_SUMMARY_FLAGS = (
    "execution_allowed_by_contract",
    "sequence_policy_promotion_allowed_by_contract",
    "queue_mutation_allowed_by_contract",
    "policy_mutation_allowed_by_contract",
    "model_mutation_allowed_by_contract",
    "runtime_memory_mutation_allowed_by_contract",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_CONTRACT_RECORD_FALSE_FLAGS = (
    "live_write",
    "apply_attempted",
    "mutates_console_state",
    "queries_console",
    "execution_allowed_by_contract",
    "sequence_policy_promotion_allowed_by_contract",
    "queue_mutation_allowed_by_contract",
    "policy_mutation_allowed_by_contract",
    "model_mutation_allowed_by_contract",
    "runtime_memory_mutation_allowed_by_contract",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_CONTRACT_RECORD_FIELDS = (
    "promotion_contract_record_id",
    "promotion_candidate_id",
    "acceptance_id",
    "review_candidate_id",
    "memory_record_id",
    "proposal_id",
    "learning_scope",
    "value_type",
    "idempotency_key",
    "previous_policy_state",
    "rollback_strategy",
    "rollback_target_policy_store",
    "reviewed_by",
    "reviewed_at",
)
REQUIRED_TEMPLATE_FALSE_FLAGS = (
    "execution_allowed_by_acceptance",
    "sequence_policy_promotion_allowed_by_acceptance",
    "queue_mutation_allowed_by_acceptance",
    "policy_mutation_allowed_by_acceptance",
    "model_mutation_allowed_by_acceptance",
    "runtime_memory_mutation_allowed_by_acceptance",
    "live_authority_granted",
    "full_autonomous_ready",
    "can_import_to_queue",
    "can_apply_live",
)
REQUIRED_SEQUENCE_REVIEW_FRESHNESS_FALSE_FLAGS = (
    "live_write",
    "apply_attempted",
    "queries_console",
    "queue_mutation",
    "policy_mutation",
)
REQUIRED_SOURCE_ACCEPTANCE_TEMPLATE_IDENTITY_FIELDS = (
    "source_memory_trust_review_request_spec_id",
    "source_memory_index_request_spec_id",
    "source_memory_writer_apply_request_spec_id",
    "source_memory_writer_request_spec_id",
    "source_memory_draft_request_spec_id",
    "source_audit_policy_gate_request_spec_id",
    "source_audit_request_spec_id",
    "source_executor_request_spec_id",
    "source_execution_gate_request_spec_id",
    "source_dry_run_request_spec_id",
    "source_sequence_plan_request_spec_id",
    "source_preflight_request_spec_id",
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


def _current_sequence_review_freshness_violations(
    proposal_id: str,
    freshness: Mapping[str, Any],
) -> Tuple[str, ...]:
    if not freshness:
        return (f"{proposal_id}:current_sequence_review_freshness_missing",)

    violations = []
    if freshness.get("status") != "fresh":
        violations.append(f"{proposal_id}:current_sequence_review_freshness_not_fresh")
    if freshness.get("read_only") is not True:
        violations.append(f"{proposal_id}:current_sequence_review_freshness_read_only_not_true")
    for flag in REQUIRED_SEQUENCE_REVIEW_FRESHNESS_FALSE_FLAGS:
        if freshness.get(flag) is not False:
            violations.append(f"{proposal_id}:current_sequence_review_freshness_{flag}_not_false")
    return tuple(violations)


def _source_sequence_review_freshness_status(
    source_acceptance_templates: Tuple[Mapping[str, Any], ...],
) -> str:
    if not source_acceptance_templates:
        return "missing"
    invalid = False
    missing = False
    for index, template in enumerate(source_acceptance_templates):
        proposal_id = _string(template.get("proposal_id")) or f"acceptance_template:{index}"
        freshness = _mapping(template.get("current_sequence_review_freshness"))
        violations = _current_sequence_review_freshness_violations(proposal_id, freshness)
        if not freshness:
            missing = True
        elif violations:
            invalid = True
    if not missing and not invalid:
        return "fresh"
    return "invalid" if invalid else "missing"


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


def _promotion_candidates_from_request(
    consumer: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], ...]:
    request = _mapping(consumer.get("promotion_contract_request"))
    plan = _mapping(request.get("promotion_plan"))
    return _record_list(plan.get("promotion_candidates"))


def _source_acceptance_templates_from_request(
    consumer: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], ...]:
    request = _mapping(consumer.get("promotion_contract_request"))
    promotion_templates = _record_list(request.get("promotion_acceptance_record_templates"))
    return promotion_templates or _record_list(request.get("acceptance_record_templates"))


def _source_request_specs_from_request(
    consumer: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], ...]:
    request = _mapping(consumer.get("promotion_contract_request"))
    promotion_specs = _record_list(request.get("promotion_memory_trust_review_request_specs"))
    return promotion_specs or _record_list(request.get("memory_trust_review_request_specs"))


def _candidate_by_id(
    candidates: Tuple[Mapping[str, Any], ...],
) -> Dict[str, Mapping[str, Any]]:
    return {
        str(candidate["promotion_candidate_id"]): candidate
        for candidate in candidates
        if isinstance(candidate.get("promotion_candidate_id"), str)
        and candidate.get("promotion_candidate_id")
    }


def _source_acceptance_template_by_proposal_id(
    templates: Tuple[Mapping[str, Any], ...],
) -> Dict[str, Mapping[str, Any]]:
    return {
        str(template["proposal_id"]): template
        for template in templates
        if isinstance(template.get("proposal_id"), str) and template.get("proposal_id")
    }


def _record_id(record: Mapping[str, Any], index: int) -> str:
    explicit = _string(record.get("promotion_contract_record_id"))
    if explicit:
        return explicit
    candidate_id = _string(record.get("promotion_candidate_id"))
    return f"{candidate_id if candidate_id else 'unknown_sequence_promotion_contract'}:{index}"


def _consumer_envelope_violations(consumer: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if consumer.get("type") != REQUIRED_PROMOTION_PLAN_RESULT_CONSUMER_TYPE:
        violations.append(
            "sequence_policy_promotion_contract_result_consumer_plan_consumer_type_mismatch"
        )
    if (
        consumer.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_PLAN_RESULT_CONSUMER_METHOD_ID
    ):
        violations.append(
            "sequence_policy_promotion_contract_result_consumer_plan_consumer_method_id_mismatch"
        )
    if (
        consumer.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_PLAN_RESULT_CONSUMER_STATUSES
    ):
        violations.append(
            "sequence_policy_promotion_contract_result_consumer_plan_consumer_status_unknown"
        )
    if consumer.get("status") != REQUIRED_PROMOTION_PLAN_RESULT_CONSUMER_AVAILABLE_STATUS:
        violations.append(
            "sequence_policy_promotion_contract_result_consumer_plan_consumer_status_not_available"
        )
    safety = _mapping(consumer.get("safety"))
    if safety.get("read_only") is not True:
        violations.append(
            "sequence_policy_promotion_contract_result_consumer_plan_consumer_read_only_not_true"
        )
    if safety.get("promotion_plan_result_consumer_only") is not True:
        violations.append(
            "sequence_policy_promotion_contract_result_consumer_plan_consumer_only_not_true"
        )
    if safety.get("promotion_contract_request_template_only") is not True:
        violations.append(
            "sequence_policy_promotion_contract_result_consumer_plan_consumer_template_only_not_true"
        )
    if safety.get("current_sequence_policy_context_required") is not True:
        violations.append(
            "sequence_policy_promotion_contract_result_consumer_plan_consumer_context_required_not_true"
        )
    if safety.get("rollback_evidence_required") is not True:
        violations.append(
            "sequence_policy_promotion_contract_result_consumer_plan_consumer_rollback_required_not_true"
        )
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_contract_result_consumer_plan_consumer_{flag}_not_false"
            )
    return tuple(violations)


def _consumer_summary_violations(
    summary: Mapping[str, Any],
    *,
    candidates: Tuple[Mapping[str, Any], ...],
    source_acceptance_templates: Tuple[Mapping[str, Any], ...],
    uses_promotion_source_context: bool,
) -> Tuple[str, ...]:
    source_template_count_key = (
        "source_promotion_acceptance_record_template_count"
        if uses_promotion_source_context
        else "acceptance_record_template_count"
    )
    expected_counts = {
        "expected_promotion_candidate_count": len(candidates),
        "promotion_candidate_count": len(candidates),
        source_template_count_key: len(source_acceptance_templates),
        "invalid_record_count": 0,
    }
    violations = []
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(
                f"sequence_policy_promotion_contract_result_consumer_plan_consumer_summary_{key}_mismatch"
            )
    if summary.get("promotion_contract_request_available") is not True:
        violations.append(
            "sequence_policy_promotion_contract_result_consumer_plan_consumer_request_not_available"
        )
    if summary.get("current_sequence_policy_context_required") is not True:
        violations.append(
            "sequence_policy_promotion_contract_result_consumer_plan_consumer_summary_context_required_not_true"
        )
    source_sequence_review_status_key = (
        "source_sequence_review_context_status"
        if uses_promotion_source_context
        else "fresh_sequence_review_context_status"
    )
    if (
        source_acceptance_templates
        and summary.get(source_sequence_review_status_key) != "fresh"
    ):
        violations.append(
            "sequence_policy_promotion_contract_result_consumer_plan_consumer_summary_freshness_not_fresh"
        )
    if summary.get("rollback_evidence_required") is not True:
        violations.append(
            "sequence_policy_promotion_contract_result_consumer_plan_consumer_summary_rollback_required_not_true"
        )
    expected_proposal_ids = tuple(
        str(candidate["proposal_id"])
        for candidate in candidates
        if isinstance(candidate.get("proposal_id"), str) and candidate.get("proposal_id")
    )
    if _string_list(summary.get("accepted_proposal_ids")) != expected_proposal_ids:
        violations.append(
            "sequence_policy_promotion_contract_result_consumer_plan_consumer_accepted_proposal_ids_mismatch"
        )
    expected_template_proposal_ids = tuple(
        str(template["proposal_id"])
        for template in source_acceptance_templates
        if isinstance(template.get("proposal_id"), str) and template.get("proposal_id")
    )
    source_template_proposal_ids_key = (
        "source_promotion_acceptance_record_template_proposal_ids"
        if uses_promotion_source_context
        else "acceptance_record_template_proposal_ids"
    )
    source_template_proposal_ids = _string_list(
        summary.get(source_template_proposal_ids_key)
    )
    if source_template_proposal_ids != expected_template_proposal_ids:
        violations.append(
            "sequence_policy_promotion_contract_result_consumer_plan_consumer_acceptance_record_template_proposal_ids_mismatch"
        )
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_contract_result_consumer_plan_consumer_summary_{flag}_not_false"
            )
    return tuple(violations)


def _consumer_request_violations(
    request: Mapping[str, Any],
    *,
    candidates: Tuple[Mapping[str, Any], ...],
    source_acceptance_templates: Tuple[Mapping[str, Any], ...],
    source_request_specs: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    if request.get("handler") != "contract_operator_proposal_apply_sequence_policy_memory_trust_promotion":
        violations.append(
            "sequence_policy_promotion_contract_result_consumer_request_handler_mismatch"
        )
    if request.get("read_only_template") is not True:
        violations.append(
            "sequence_policy_promotion_contract_result_consumer_request_read_only_template_not_true"
        )
    if request.get("promotion_contract_request_template_only") is not True:
        violations.append(
            "sequence_policy_promotion_contract_result_consumer_request_template_only_not_true"
        )
    if request.get("operator_must_supply_current_sequence_policy_context") is not True:
        violations.append(
            "sequence_policy_promotion_contract_result_consumer_request_context_required_not_true"
        )
    if request.get("operator_must_supply_rollback_evidence") is not True:
        violations.append(
            "sequence_policy_promotion_contract_result_consumer_request_rollback_required_not_true"
        )
    if _mapping(request.get("promotion_plan")).get("status") != REQUIRED_PLAN_STATUS:
        violations.append(
            "sequence_policy_promotion_contract_result_consumer_request_promotion_plan_not_available"
        )
    source_template_proposal_ids = {
        str(template["proposal_id"])
        for template in source_acceptance_templates
        if isinstance(template.get("proposal_id"), str) and template.get("proposal_id")
    }
    candidate_proposal_ids = {
        str(candidate["proposal_id"])
        for candidate in candidates
        if isinstance(candidate.get("proposal_id"), str) and candidate.get("proposal_id")
    }
    if not candidate_proposal_ids.issubset(source_template_proposal_ids):
        violations.append(
            "sequence_policy_promotion_contract_result_consumer_request_acceptance_templates_missing_candidate"
        )
    source_template_proposal_id_list = tuple(
        str(template["proposal_id"])
        for template in source_acceptance_templates
        if isinstance(template.get("proposal_id"), str) and template.get("proposal_id")
    )
    source_spec_proposal_id_list = tuple(
        str(spec["proposal_id"])
        for spec in source_request_specs
        if isinstance(spec.get("proposal_id"), str) and spec.get("proposal_id")
    )
    if not set(source_template_proposal_id_list).issubset(
        set(source_spec_proposal_id_list)
    ):
        violations.append(
            "sequence_policy_promotion_contract_result_consumer_request_review_request_spec_proposal_ids_mismatch"
        )
    for index, spec in enumerate(source_request_specs):
        spec_id = _string(spec.get("memory_trust_review_request_spec_id")) or (
            f"memory_trust_review_request_spec:{index}"
        )
        proposal_id = _string(spec.get("proposal_id")) or spec_id
        violations.extend(
            _current_sequence_review_freshness_violations(
                proposal_id,
                _mapping(spec.get("current_sequence_review_freshness")),
            )
        )
    for flag in REQUIRED_CONSUMER_FALSE_FLAGS:
        if request.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_contract_result_consumer_request_{flag}_not_false"
            )
    return tuple(violations)


def _source_acceptance_template_violations(
    template: Mapping[str, Any],
    *,
    index: int,
) -> Tuple[str, ...]:
    template_id = (
        _string(template.get("promotion_acceptance_record_template_id"))
        or _string(template.get("acceptance_record_template_id"))
        or f"acceptance_template:{index}"
    )
    proposal_id = _string(template.get("proposal_id")) or template_id
    violations = []
    for field in ("proposal_id", "learning_scope", "value_type", "idempotency_key"):
        if not _string(template.get(field)):
            violations.append(f"{template_id}:{field}_missing")
    if not (
        _string(template.get("review_candidate_id"))
        or _string(template.get("source_review_candidate_id"))
    ):
        violations.append(f"{template_id}:review_candidate_id_missing")
    if not (
        _string(template.get("memory_record_id"))
        or _string(template.get("source_memory_record_id"))
    ):
        violations.append(f"{template_id}:memory_record_id_missing")
    for field in REQUIRED_SOURCE_ACCEPTANCE_TEMPLATE_IDENTITY_FIELDS:
        if not _string(template.get(field)):
            violations.append(f"{proposal_id}:{field}_missing")
    if template.get("source_memory_index_available") is not True:
        violations.append(f"{proposal_id}:source_memory_index_available_not_true")
    if _summary_int(template.get("planned_queue_index")) is None:
        violations.append(f"{proposal_id}:planned_queue_index_invalid")
    if template.get("rank_source") != "sequence_policy_consumer":
        violations.append(f"{proposal_id}:rank_source_mismatch")
    violations.extend(
        _current_sequence_review_freshness_violations(
            proposal_id,
            _mapping(template.get("current_sequence_review_freshness")),
        )
    )
    source_idempotency_key = _string(template.get("source_idempotency_key"))
    policy_key = _string(template.get("policy_key"))
    if (
        source_idempotency_key
        and policy_key
        and policy_key != f"{proposal_id}/{source_idempotency_key}"
    ):
        violations.append(f"{proposal_id}:policy_key_identity_mismatch")
    if template.get("decision_options") != ["accepted", "rejected"]:
        violations.append(f"{template_id}:decision_options_invalid")
    if template.get("operator_must_choose_decision") is not True:
        violations.append(f"{template_id}:operator_must_choose_decision_not_true")
    if template.get("reviewed_by_required") is not True:
        violations.append(f"{template_id}:reviewed_by_required_not_true")
    if template.get("reviewed_at_utc_required") is not True:
        violations.append(f"{template_id}:reviewed_at_utc_required_not_true")
    for flag in REQUIRED_TEMPLATE_FALSE_FLAGS:
        if template.get(flag) is not False:
            violations.append(f"{template_id}:{flag}_not_false")
    return tuple(violations)


def _source_acceptance_template_identity_violations(
    source_acceptance_templates: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = [
        violation
        for index, template in enumerate(source_acceptance_templates)
        for violation in _source_acceptance_template_violations(template, index=index)
    ]
    proposal_ids = tuple(
        str(template["proposal_id"])
        for template in source_acceptance_templates
        if isinstance(template.get("proposal_id"), str) and template.get("proposal_id")
    )
    if len(set(proposal_ids)) != len(proposal_ids):
        violations.append(
            "sequence_policy_promotion_contract_result_consumer_acceptance_template_proposal_ids_not_unique"
        )
    return tuple(violations)


def _source_template_candidate_identity_violations(
    *,
    candidates: Tuple[Mapping[str, Any], ...],
    source_acceptance_templates: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    templates_by_proposal_id = _source_acceptance_template_by_proposal_id(
        source_acceptance_templates
    )
    violations = []
    for candidate in candidates:
        proposal_id = _string(candidate.get("proposal_id"))
        template = _mapping(templates_by_proposal_id.get(proposal_id))
        if not template:
            violations.append(f"{proposal_id}:source_acceptance_template_missing")
            continue
        expected_pairs = {
            "proposal_id": candidate.get("proposal_id"),
            "learning_scope": candidate.get("learning_scope"),
            "value_type": candidate.get("value_type"),
        }
        direct_identity_available = bool(
            _string(template.get("review_candidate_id"))
            or _string(template.get("memory_record_id"))
        )
        if direct_identity_available:
            expected_pairs.update(
                {
                    "review_candidate_id": candidate.get("review_candidate_id"),
                    "memory_record_id": candidate.get("memory_record_id"),
                    "idempotency_key": candidate.get("idempotency_key"),
                }
            )
        for field, expected in expected_pairs.items():
            actual = template.get(field)
            if field == "review_candidate_id":
                actual = actual or template.get("source_review_candidate_id")
            elif field == "memory_record_id":
                actual = actual or template.get("source_memory_record_id")
            if actual != expected:
                violations.append(f"{proposal_id}:{field}_candidate_mismatch")
    return tuple(violations)


def _contract_envelope_violations(contract: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    if contract.get("type") != REQUIRED_PROMOTION_CONTRACT_TYPE:
        violations.append("sequence_policy_promotion_contract_result_consumer_contract_type_mismatch")
    generated_at = contract.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("sequence_policy_promotion_contract_result_consumer_contract_generated_at_invalid")
    if contract.get("method_id") != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_CONTRACT_METHOD_ID:
        violations.append("sequence_policy_promotion_contract_result_consumer_contract_method_id_mismatch")
    if contract.get("source_method_id") != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_PLAN_METHOD_ID:
        violations.append("sequence_policy_promotion_contract_result_consumer_contract_source_method_mismatch")
    if contract.get("status") not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_CONTRACT_STATUSES:
        violations.append("sequence_policy_promotion_contract_result_consumer_contract_status_unknown")
    safety = _mapping(contract.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("sequence_policy_promotion_contract_result_consumer_contract_read_only_not_true")
    if safety.get("operator_required") is not True:
        violations.append("sequence_policy_promotion_contract_result_consumer_contract_operator_required_not_true")
    if safety.get("rollback_evidence_required") is not True:
        violations.append("sequence_policy_promotion_contract_result_consumer_contract_rollback_required_not_true")
    for flag in REQUIRED_CONTRACT_FALSE_SAFETY_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_contract_result_consumer_contract_{flag}_not_false"
            )
    return tuple(violations)


def _contract_summary_violations(
    summary: Mapping[str, Any],
    *,
    records: Tuple[Mapping[str, Any], ...],
    candidates: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    expected_counts = {
        "promotion_candidate_count": len(candidates),
        "contract_record_count": len(records),
        "invalid_record_count": 0,
        "unreviewed_candidate_count": 0,
    }
    violations = []
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(
                f"sequence_policy_promotion_contract_result_consumer_contract_summary_{key}_mismatch"
            )
    if summary.get("promotion_plan_status") != REQUIRED_PLAN_STATUS:
        violations.append(
            "sequence_policy_promotion_contract_result_consumer_contract_summary_plan_status_mismatch"
        )
    if summary.get("current_sequence_policy_context_status") != "fresh":
        violations.append(
            "sequence_policy_promotion_contract_result_consumer_contract_summary_context_not_fresh"
        )
    if summary.get("rollback_evidence_status") != "ready":
        violations.append(
            "sequence_policy_promotion_contract_result_consumer_contract_summary_rollback_not_ready"
        )
    if summary.get("operator_review_required") is not True:
        violations.append(
            "sequence_policy_promotion_contract_result_consumer_contract_summary_operator_review_not_true"
        )
    if summary.get("rollback_evidence_required") is not True:
        violations.append(
            "sequence_policy_promotion_contract_result_consumer_contract_summary_rollback_required_not_true"
        )
    expected_record_ids = tuple(
        str(record["promotion_contract_record_id"])
        for record in records
        if isinstance(record.get("promotion_contract_record_id"), str)
        and record.get("promotion_contract_record_id")
    )
    if _string_list(summary.get("contract_record_ids")) != expected_record_ids:
        violations.append(
            "sequence_policy_promotion_contract_result_consumer_contract_summary_record_ids_mismatch"
        )
    expected_proposal_ids = tuple(
        str(record["proposal_id"])
        for record in records
        if isinstance(record.get("proposal_id"), str) and record.get("proposal_id")
    )
    if _string_list(summary.get("proposal_ids")) != expected_proposal_ids:
        violations.append(
            "sequence_policy_promotion_contract_result_consumer_contract_summary_proposal_ids_mismatch"
        )
    if _string_list(summary.get("invalid_record_ids")):
        violations.append(
            "sequence_policy_promotion_contract_result_consumer_contract_summary_invalid_record_ids_not_empty"
        )
    if _string_list(summary.get("unreviewed_candidate_ids")):
        violations.append(
            "sequence_policy_promotion_contract_result_consumer_contract_summary_unreviewed_candidate_ids_not_empty"
        )
    for flag in REQUIRED_CONTRACT_FALSE_SUMMARY_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_policy_promotion_contract_result_consumer_contract_summary_{flag}_not_false"
            )
    return tuple(violations)


def _record_violations(
    record: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    index: int,
) -> Tuple[str, ...]:
    record_id = _record_id(record, index)
    violations = []
    for field in REQUIRED_CONTRACT_RECORD_FIELDS:
        if not _string(record.get(field)):
            violations.append(f"{record_id}:{field}_missing")
    reviewed_at = _string(record.get("reviewed_at"))
    if reviewed_at and not _is_utc_timestamp(reviewed_at):
        violations.append(f"{record_id}:reviewed_at_invalid")
    expected_pairs = {
        "promotion_candidate_id": candidate.get("promotion_candidate_id"),
        "acceptance_id": candidate.get("acceptance_id"),
        "review_candidate_id": candidate.get("review_candidate_id"),
        "memory_record_id": candidate.get("memory_record_id"),
        "proposal_id": candidate.get("proposal_id"),
        "learning_scope": candidate.get("learning_scope"),
        "value_type": candidate.get("value_type"),
        "idempotency_key": candidate.get("idempotency_key"),
        "reviewed_by": candidate.get("reviewed_by"),
        "reviewed_at": candidate.get("reviewed_at"),
    }
    for field, expected in expected_pairs.items():
        if record.get(field) != expected:
            violations.append(f"{record_id}:{field}_mismatch")
    if not _string(record.get("previous_policy_state")):
        violations.append(f"{record_id}:previous_policy_state_missing")
    if not _string(record.get("rollback_strategy")):
        violations.append(f"{record_id}:rollback_strategy_missing")
    if not _string(record.get("rollback_target_policy_store")):
        violations.append(f"{record_id}:rollback_target_policy_store_missing")
    if record.get("read_only") is not True:
        violations.append(f"{record_id}:read_only_not_true")
    if record.get("operator_review_required") is not True:
        violations.append(f"{record_id}:operator_review_required_not_true")
    if record.get("rollback_evidence_required") is not True:
        violations.append(f"{record_id}:rollback_evidence_required_not_true")
    if record.get("sequence_policy_promotion_ready_for_operator_review") is not True:
        violations.append(f"{record_id}:ready_for_operator_review_not_true")
    if record.get("promotion_executor_required") is not True:
        violations.append(f"{record_id}:promotion_executor_required_not_true")
    for flag in REQUIRED_CONTRACT_RECORD_FALSE_FLAGS:
        if record.get(flag) is not False:
            violations.append(f"{record_id}:{flag}_not_false")
    return tuple(violations)


def _record_identity_violations(
    records: Tuple[Mapping[str, Any], ...],
    candidates: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    candidates_by_id = _candidate_by_id(candidates)
    violations = []
    if set(candidates_by_id) != {
        str(record["promotion_candidate_id"])
        for record in records
        if isinstance(record.get("promotion_candidate_id"), str)
        and record.get("promotion_candidate_id")
    }:
        violations.append(
            "sequence_policy_promotion_contract_result_consumer_record_candidate_ids_mismatch"
        )
    for index, record in enumerate(records):
        candidate_id = _string(record.get("promotion_candidate_id"))
        candidate = _mapping(candidates_by_id.get(candidate_id))
        if not candidate:
            violations.append(f"{_record_id(record, index)}:promotion_candidate_missing")
            continue
        violations.extend(_record_violations(record, candidate=candidate, index=index))
    return tuple(violations)


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer(
    *,
    promotion_plan_result_consumer: Mapping[str, Any] | None = None,
    promotion_contract_result: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Verify promotion-contract evidence and emit acceptance-ledger request."""

    consumer = _mapping(promotion_plan_result_consumer)
    contract = _mapping(promotion_contract_result)
    request = _mapping(consumer.get("promotion_contract_request"))
    candidates = _promotion_candidates_from_request(consumer)
    source_acceptance_templates = _source_acceptance_templates_from_request(consumer)
    source_request_specs = _source_request_specs_from_request(consumer)
    uses_promotion_source_context = bool(
        _record_list(request.get("promotion_acceptance_record_templates"))
    )
    records = _record_list(contract.get("contract_records"))
    raw_invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_consumer_envelope_violations(consumer),
                *_consumer_summary_violations(
                    _mapping(consumer.get("summary")),
                    candidates=candidates,
                    source_acceptance_templates=source_acceptance_templates,
                    uses_promotion_source_context=uses_promotion_source_context,
                ),
                *_consumer_request_violations(
                    request,
                    candidates=candidates,
                    source_acceptance_templates=source_acceptance_templates,
                    source_request_specs=source_request_specs,
                ),
                *_source_acceptance_template_identity_violations(
                    source_acceptance_templates
                ),
                *_source_template_candidate_identity_violations(
                    candidates=candidates,
                    source_acceptance_templates=source_acceptance_templates,
                ),
                *_contract_envelope_violations(contract),
                *_contract_summary_violations(
                    _mapping(contract.get("summary")),
                    records=records,
                    candidates=candidates,
                ),
                *(
                    ("sequence_policy_promotion_contract_result_consumer_contract_status_not_ready",)
                    if contract.get("status") != REQUIRED_PROMOTION_CONTRACT_READY_STATUS
                    else ()
                ),
                *(
                    ("sequence_policy_promotion_contract_result_consumer_record_count_mismatch",)
                    if len(records) != len(candidates)
                    else ()
                ),
                *_record_identity_violations(records, candidates),
            )
        )
    )
    invalid_record_ids = () if not consumer or not contract else raw_invalid_record_ids

    if not consumer or not contract:
        status = "sequence_memory_trust_promotion_queue_import_promotion_contract_result_consumer_not_ready"
        blocked_by = ("promotion_plan_result_consumer_and_promotion_contract_result_required",)
    elif invalid_record_ids:
        status = "sequence_memory_trust_promotion_queue_import_promotion_contract_result_consumer_replacement_required"
        blocked_by = ("operator_must_replace_invalid_sequence_memory_trust_promotion_contract_evidence",)
    elif contract.get("status") != REQUIRED_PROMOTION_CONTRACT_READY_STATUS:
        status = "sequence_memory_trust_promotion_queue_import_promotion_contract_result_consumer_blocked"
        blocked_by = ("manual_sequence_memory_trust_promotion_contract_must_be_ready_before_acceptance",)
    elif not records:
        status = "sequence_memory_trust_promotion_queue_import_promotion_contract_result_consumer_not_ready"
        blocked_by = ("sequence_memory_trust_promotion_contract_records_required",)
    else:
        status = "sequence_memory_trust_promotion_queue_import_promotion_acceptance_ledger_request_available"
        blocked_by = ("operator_may_record_sequence_memory_trust_promotion_acceptance_decisions",)

    request_available = (
        status
        == "sequence_memory_trust_promotion_queue_import_promotion_acceptance_ledger_request_available"
    )
    source_templates_by_proposal_id = _source_acceptance_template_by_proposal_id(
        source_acceptance_templates
    )

    def _source_identity_fields(record: Mapping[str, Any]) -> Dict[str, Any]:
        source_template = _mapping(
            source_templates_by_proposal_id.get(_string(record.get("proposal_id")))
        )
        if not source_template:
            return {}
        return {
            "source_acceptance_record_template_id": source_template.get(
                "promotion_acceptance_record_template_id"
            )
            or source_template.get("acceptance_record_template_id")
            or source_template.get(
                "source_acceptance_record_template_id"
            ),
            "source_review_candidate_id": source_template.get("review_candidate_id")
            or source_template.get("source_review_candidate_id"),
            "source_memory_record_id": source_template.get("memory_record_id")
            or source_template.get("source_memory_record_id"),
            "source_memory_index_available": source_template.get(
                "source_memory_index_available"
            ),
            "planned_queue_index": source_template.get("planned_queue_index"),
            "rank_source": source_template.get("rank_source"),
            "current_sequence_review_freshness": source_template.get(
                "current_sequence_review_freshness"
            ),
            **{
                field: source_template.get(field)
                for field in REQUIRED_SOURCE_ACCEPTANCE_TEMPLATE_IDENTITY_FIELDS
            },
        }

    acceptance_templates = tuple(
        {
            "promotion_acceptance_record_template_id": (
                f"{OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_CONTRACT_RESULT_CONSUMER_METHOD_ID}:{record.get('promotion_contract_record_id')}"
            ),
            "promotion_contract_record_id": record.get("promotion_contract_record_id"),
            "promotion_candidate_id": record.get("promotion_candidate_id"),
            "proposal_id": record.get("proposal_id"),
            "learning_scope": record.get("learning_scope"),
            "value_type": record.get("value_type"),
            "idempotency_key": record.get("idempotency_key"),
            "previous_policy_state": record.get("previous_policy_state"),
            "rollback_strategy": record.get("rollback_strategy"),
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
            **_source_identity_fields(record),
        }
        for record in (records if request_available else ())
    )
    promotion_acceptance_ledger_request: Dict[str, Any] = {
        "handler": "record_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance",
        "read_only_template": True,
        "promotion_contract_result_consumer_only": True,
        "promotion_acceptance_ledger_request_template_only": True,
        "promotion_acceptance_ledger_handler_invoked": False,
        "promotion_acceptance_ledger_created": False,
        "operator_must_supply_promotion_acceptance_records": True,
        "acceptance_records": [],
        "promotion_acceptance_record_templates": list(acceptance_templates),
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
        promotion_acceptance_ledger_request["promotion_contract"] = contract
        promotion_acceptance_ledger_request["memory_trust_review_request_specs"] = list(
            source_request_specs
        )

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_CONTRACT_RESULT_CONSUMER_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PROMOTION_PLAN_RESULT_CONSUMER_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "promotion_contract_result_consumer_only": True,
            "promotion_acceptance_ledger_request_template_only": True,
            "requires_promotion_plan_result_consumer": True,
            "requires_promotion_contract_result": True,
            "promotion_acceptance_ledger_handler_invoked": False,
            "promotion_acceptance_ledger_created": False,
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
            "expected_contract_record_count": len(candidates),
            "contract_record_count": len(records),
            "source_acceptance_record_template_count": len(source_acceptance_templates),
            "promotion_acceptance_record_template_count": len(acceptance_templates),
            "fresh_sequence_review_context_status": _source_sequence_review_freshness_status(
                source_acceptance_templates
            ),
            "invalid_record_count": len(invalid_record_ids),
            "promotion_contract_result_consumer_only": True,
            "promotion_acceptance_ledger_request_template_only": True,
            "promotion_acceptance_ledger_request_available": request_available,
            "promotion_acceptance_ledger_handler_invoked": False,
            "promotion_acceptance_ledger_created": False,
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
            "proposal_ids": [
                str(record["proposal_id"])
                for record in records
                if isinstance(record.get("proposal_id"), str) and record.get("proposal_id")
            ],
            "source_acceptance_record_template_proposal_ids": [
                str(template["proposal_id"])
                for template in source_acceptance_templates
                if isinstance(template.get("proposal_id"), str)
                and template.get("proposal_id")
            ],
            "promotion_acceptance_record_template_proposal_ids": [
                str(template["proposal_id"])
                for template in acceptance_templates
                if isinstance(template.get("proposal_id"), str)
                and template.get("proposal_id")
            ],
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Provide promotion-plan result consumer and promotion-contract result."
                if not consumer or not contract
                else (
                    "Replace invalid sequence-memory trust promotion-contract evidence before acceptance."
                    if invalid_record_ids
                    else (
                        "Run read-only promotion contract before acceptance."
                        if contract.get("status") != REQUIRED_PROMOTION_CONTRACT_READY_STATUS
                        else "Operator may record accept/reject decisions in the read-only promotion acceptance ledger; this consumer does not decide."
                    )
                )
            ),
        },
        "promotion_acceptance_ledger_request": promotion_acceptance_ledger_request,
        "promotion_acceptance_record_templates": list(acceptance_templates),
    }
