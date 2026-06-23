"""Operator-gated queue importer for sequence-policy queue plans.

The importer consumes read-only sequence-policy queue-ordering plan records and
proposal payloads, then emits dry-run import specs by default. It mutates only
an explicit caller-provided operator proposal queue, and only when invoked with
`apply=True` and `operator_ack=True`. Product Layer preview calls this as a dry
run, so normal previews remain non-mutating. The importer never touches live
mixer transport, console state, sequence-policy stores, runtime memory, models,
proposal state, or direct sequence execution.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_ORDERING_PLAN_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_ORDERING_PLAN_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_ORDERING_PLAN_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_ORDERING_PLAN_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORTER_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORTER_STATUSES = {
    "sequence_memory_trust_promotion_queue_importer_not_ready",
    "sequence_memory_trust_promotion_queue_importer_replacement_required",
    "sequence_memory_trust_promotion_queue_importer_proposal_payload_required",
    "sequence_memory_trust_promotion_queue_importer_operator_ack_required",
    "sequence_memory_trust_promotion_queue_importer_target_queue_required",
    "sequence_memory_trust_promotion_queue_importer_dry_run_ready",
    "sequence_memory_trust_promotion_queue_importer_apply_blocked",
    "sequence_memory_trust_promotion_queue_importer_applied",
}
REQUIRED_QUEUE_ORDERING_PLAN_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan"
)
REQUIRED_QUEUE_ORDERING_PLAN_FALSE_FLAGS = (
    "live_write",
    "mutates_console_state",
    "queries_console",
    "proposal_order_mutation",
    "queue_mutation",
    "queue_import_allowed_by_plan",
    "queue_reorder_allowed_by_plan",
    "model_mutation",
    "runtime_memory_mutation",
    "sequence_policy_trust_store_mutation",
    "proposal_state_mutation",
    "sequence_execution_allowed_by_plan",
    "sequence_policy_change_allowed_by_plan",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
    "can_import_to_queue",
    "can_apply_live",
)
REQUIRED_PLAN_RECORD_FALSE_FLAGS = REQUIRED_QUEUE_ORDERING_PLAN_FALSE_FLAGS
ALLOWED_RANK_SOURCES = {"sequence_policy_consumer", "original_order"}


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


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric or numeric in {float("inf"), float("-inf")}:
        return None
    return numeric


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


def _context_status(payload: Mapping[str, Any]) -> str:
    return _string(payload.get("status"))


def _context_violations(prefix: str, payload: Mapping[str, Any]) -> Tuple[str, ...]:
    if not payload:
        return (f"{prefix}_current_sequence_review_freshness_missing",)
    violations = []
    if _context_status(payload) != "fresh":
        violations.append(f"{prefix}_current_sequence_review_freshness_not_fresh")
    if payload.get("read_only") is not True:
        violations.append(f"{prefix}_current_sequence_review_freshness_read_only_not_true")
    for flag in (
        "live_write",
        "apply_attempted",
        "queries_console",
        "queue_mutation",
        "policy_mutation",
    ):
        if payload.get(flag) is not False:
            violations.append(
                f"{prefix}_current_sequence_review_freshness_{flag}_not_false"
            )
    return tuple(violations)


def _duplicate_values(values: Tuple[str, ...]) -> Tuple[str, ...]:
    counts: Dict[str, int] = {}
    for value in values:
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return tuple(value for value, count in counts.items() if count > 1)


def _plan_record_id(record: Mapping[str, Any], index: int) -> str:
    explicit = _string(record.get("queue_ordering_plan_record_id"))
    if explicit:
        return explicit
    proposal_id = _string(record.get("proposal_id"))
    return f"{proposal_id if proposal_id else 'unknown_sequence_queue_ordering_plan_record'}:{index}"


def _proposal_id(proposal: Mapping[str, Any], index: int) -> str:
    for key in ("id", "proposal_id", "action_id"):
        value = _string(proposal.get(key))
        if value:
            return value
    return f"proposal:{index}"


def _proposal_payload_records(
    proposal_payloads: Iterable[Mapping[str, Any]] | Mapping[str, Any] | None,
) -> Tuple[Mapping[str, Any], ...]:
    if proposal_payloads is None:
        return ()
    if isinstance(proposal_payloads, Mapping):
        if any(isinstance(value, Mapping) for value in proposal_payloads.values()):
            return tuple(
                value for value in proposal_payloads.values() if isinstance(value, Mapping)
            )
        return (proposal_payloads,)
    if not isinstance(proposal_payloads, (list, tuple)):
        return ()
    return tuple(item for item in proposal_payloads if isinstance(item, Mapping))


def _proposal_payload_map(
    proposal_payloads: Iterable[Mapping[str, Any]] | Mapping[str, Any] | None,
) -> Dict[str, Mapping[str, Any]]:
    records = _proposal_payload_records(proposal_payloads)
    return {
        _proposal_id(record, index): record
        for index, record in enumerate(records)
        if _proposal_id(record, index)
    }


def _proposal_payload_duplicate_ids(
    proposal_payloads: Iterable[Mapping[str, Any]] | Mapping[str, Any] | None,
) -> Tuple[str, ...]:
    records = _proposal_payload_records(proposal_payloads)
    return _duplicate_values(
        tuple(_proposal_id(record, index) for index, record in enumerate(records))
    )


def _default_operator_mode() -> Dict[str, Any]:
    return {
        "mode": "assist",
        "label": "Assisted Mixing",
        "capabilities": {
            "can_create_proposals": True,
            "can_apply_to_console": False,
            "requires_approval": True,
            "allowed_live_write_kinds": [],
        },
        "live_write_policy": "queue_import_only_no_live_write",
    }


def _queue_ordering_plan_envelope_violations(
    queue_ordering_plan: Mapping[str, Any],
) -> Tuple[str, ...]:
    violations = []
    if queue_ordering_plan.get("type") != REQUIRED_QUEUE_ORDERING_PLAN_TYPE:
        violations.append("sequence_policy_queue_importer_plan_type_mismatch")
    generated_at = queue_ordering_plan.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("sequence_policy_queue_importer_plan_generated_at_invalid")
    if (
        queue_ordering_plan.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_ORDERING_PLAN_METHOD_ID
    ):
        violations.append("sequence_policy_queue_importer_plan_method_id_mismatch")
    if (
        queue_ordering_plan.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_ORDERING_PLAN_STATUSES
    ):
        violations.append("sequence_policy_queue_importer_plan_status_unknown")
    safety = _mapping(queue_ordering_plan.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("sequence_policy_queue_importer_plan_read_only_not_true")
    if safety.get("queue_ordering_plan_only") is not True:
        violations.append("sequence_policy_queue_importer_plan_only_not_true")
    if safety.get("sequence_consumer_required_before_execution") is not True:
        violations.append(
            "sequence_policy_queue_importer_plan_sequence_consumer_required_not_true"
        )
    if safety.get("apply_attempted") is not False:
        violations.append("sequence_policy_queue_importer_plan_apply_attempted_not_false")
    for flag in REQUIRED_QUEUE_ORDERING_PLAN_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"sequence_policy_queue_importer_plan_{flag}_not_false")
    return tuple(violations)


def _queue_ordering_plan_status_violations(
    queue_ordering_plan: Mapping[str, Any],
    *,
    plan_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    status = queue_ordering_plan.get("status")
    if status == "sequence_memory_trust_promotion_queue_ordering_plan_replacement_required":
        return ("sequence_policy_queue_importer_plan_replacement_required",)
    if (
        status == "sequence_memory_trust_promotion_queue_ordering_plan_available"
        and not plan_records
    ):
        return ("sequence_policy_queue_importer_plan_available_without_records",)
    if (
        status != "sequence_memory_trust_promotion_queue_ordering_plan_available"
        and plan_records
    ):
        return ("sequence_policy_queue_importer_plan_records_without_available_status",)
    return ()


def _queue_ordering_plan_summary_violations(
    summary: Mapping[str, Any],
    *,
    plan_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    ranked_records = tuple(
        record
        for record in plan_records
        if record.get("rank_source") == "sequence_policy_consumer"
    )
    unranked_records = tuple(
        record for record in plan_records if record.get("rank_source") == "original_order"
    )
    violations = []
    expected_counts = {
        "queue_ordering_plan_record_count": len(plan_records),
        "ranked_queue_candidate_count": len(ranked_records),
        "unranked_queue_candidate_count": len(unranked_records),
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(f"sequence_policy_queue_importer_plan_summary_{key}_mismatch")
    if summary.get("queue_ordering_plan_available") is not bool(plan_records):
        violations.append("sequence_policy_queue_importer_plan_summary_available_mismatch")
    if summary.get("queue_ordering_plan_only") is not True:
        violations.append("sequence_policy_queue_importer_plan_summary_only_not_true")
    if summary.get("sequence_consumer_required_before_execution") is not True:
        violations.append(
            "sequence_policy_queue_importer_plan_summary_sequence_consumer_required_not_true"
        )
    for flag in REQUIRED_QUEUE_ORDERING_PLAN_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_policy_queue_importer_plan_summary_{flag}_not_false"
            )
    expected_plan_ids = tuple(
        record["proposal_id"] for record in plan_records if _string(record.get("proposal_id"))
    )
    if _string_list(summary.get("planned_queue_proposal_ids")) != expected_plan_ids:
        violations.append(
            "sequence_policy_queue_importer_plan_summary_proposal_ids_mismatch"
        )
    return tuple(violations)


def _plan_record_violations(record: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    required_fields = (
        "queue_ordering_plan_record_id",
        "proposal_id",
        "ordering_preview_record_id",
        "planned_queue_index",
        "rank_source",
        "recommended_sequence_policy_bucket",
    )
    for field in required_fields:
        if not _string(record.get(field)) and field != "planned_queue_index":
            violations.append(f"sequence_policy_queue_importer_plan_record_{field}_missing")
    if _summary_int(record.get("planned_queue_index")) is None:
        violations.append(
            "sequence_policy_queue_importer_plan_record_planned_queue_index_invalid"
        )
    for field in ("original_index", "recommended_index"):
        if _summary_int(record.get(field)) is None:
            violations.append(f"sequence_policy_queue_importer_plan_record_{field}_invalid")
    for field in ("base_policy_score", "adjusted_policy_score"):
        score = _safe_float(record.get(field))
        if score is None or score < 0.0 or score > 1.0:
            violations.append(f"sequence_policy_queue_importer_plan_record_{field}_invalid")
    if record.get("rank_source") not in ALLOWED_RANK_SOURCES:
        violations.append("sequence_policy_queue_importer_plan_record_rank_source_unknown")
    if record.get("rank_source") == "sequence_policy_consumer":
        for field in (
            "consumer_record_id",
            "sequence_policy_gate_record_id",
            "policy_key",
            "source_idempotency_key",
        ):
            if not _string(record.get(field)):
                violations.append(
                    f"sequence_policy_queue_importer_plan_record_{field}_missing"
                )
        proposal_id = _string(record.get("proposal_id"))
        source_idempotency_key = _string(record.get("source_idempotency_key"))
        policy_key = _string(record.get("policy_key"))
        if (
            proposal_id
            and source_idempotency_key
            and policy_key != f"{proposal_id}/{source_idempotency_key}"
        ):
            violations.append(
                "sequence_policy_queue_importer_plan_record_policy_key_identity_mismatch"
            )
        violations.extend(
            _context_violations(
                "sequence_policy_queue_importer_plan_record",
                _mapping(record.get("current_sequence_review_freshness")),
            )
        )
    if record.get("operator_review_required") is not True:
        violations.append("sequence_policy_queue_importer_plan_record_review_required_not_true")
    if record.get("operator_ack_required_before_queue_import") is not True:
        violations.append("sequence_policy_queue_importer_plan_record_queue_ack_not_true")
    if record.get("operator_ack_required_before_sequence_execution") is not True:
        violations.append("sequence_policy_queue_importer_plan_record_sequence_ack_not_true")
    if record.get("idempotency_key_required") is not True:
        violations.append("sequence_policy_queue_importer_plan_record_idempotency_not_true")
    if record.get("queue_ordering_plan_only") is not True:
        violations.append("sequence_policy_queue_importer_plan_record_plan_only_not_true")
    if record.get("sequence_consumer_required_before_execution") is not True:
        violations.append(
            "sequence_policy_queue_importer_plan_record_sequence_consumer_required_not_true"
        )
    if record.get("read_only") is not True:
        violations.append("sequence_policy_queue_importer_plan_record_read_only_not_true")
    for flag in REQUIRED_PLAN_RECORD_FALSE_FLAGS:
        if record.get(flag) is not False:
            violations.append(f"sequence_policy_queue_importer_plan_record_{flag}_not_false")
    return tuple(violations)


def _invalid_plan_record_ids(
    records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    return tuple(
        _plan_record_id(record, index)
        for index, record in enumerate(records)
        if _plan_record_violations(record)
    )


def _sequence_violations(records: Tuple[Mapping[str, Any], ...]) -> Tuple[str, ...]:
    planned_indexes = tuple(
        record.get("planned_queue_index")
        for record in records
        if _summary_int(record.get("planned_queue_index")) is not None
    )
    if planned_indexes != tuple(range(len(records))):
        return ("sequence_policy_queue_importer_planned_index_sequence_mismatch",)
    return ()


def _import_payload(
    plan_record: Mapping[str, Any],
    proposal_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    proposal_id = str(plan_record["proposal_id"])
    payload = dict(proposal_payload)
    payload["id"] = proposal_id
    payload["proposal_id"] = proposal_id
    payload["title"] = str(
        payload.get("title")
        or payload.get("description")
        or payload.get("message")
        or f"Sequence-policy queue import: {proposal_id}"
    )
    if not payload.get("kind"):
        payload["kind"] = plan_record.get("proposal_kind") or "analysis"
    if payload.get("target") is None and plan_record.get("proposal_target") is not None:
        payload["target"] = plan_record.get("proposal_target")
    payload["source"] = (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORTER_METHOD_ID
    )
    payload["reason"] = str(
        payload.get("reason") or payload.get("rationale") or "sequence_policy_queue_ordering_plan"
    )
    payload["sequence_policy_queue_import"] = {
        "queue_ordering_plan_record_id": plan_record.get("queue_ordering_plan_record_id"),
        "ordering_preview_record_id": plan_record.get("ordering_preview_record_id"),
        "consumer_record_id": plan_record.get("consumer_record_id"),
        "sequence_policy_gate_record_id": plan_record.get("sequence_policy_gate_record_id"),
        "policy_key": plan_record.get("policy_key"),
        "source_idempotency_key": plan_record.get("source_idempotency_key"),
        "current_sequence_review_freshness": plan_record.get(
            "current_sequence_review_freshness"
        ),
        "planned_queue_index": plan_record.get("planned_queue_index"),
        "rank_source": plan_record.get("rank_source"),
        "recommended_sequence_policy_bucket": plan_record.get(
            "recommended_sequence_policy_bucket"
        ),
        "operator_ack_required_before_queue_import": True,
        "operator_ack_required_before_sequence_execution": True,
        "sequence_consumer_required_before_execution": True,
        "live_write": False,
        "can_apply_live": False,
    }
    return payload


def _queue_import_specs(
    plan_records: Tuple[Mapping[str, Any], ...],
    payload_by_id: Mapping[str, Mapping[str, Any]],
) -> Tuple[Dict[str, Any], ...]:
    specs = []
    for plan_record in sorted(
        plan_records,
        key=lambda item: int(item.get("planned_queue_index") or 0),
    ):
        proposal_id = str(plan_record["proposal_id"])
        specs.append(
            {
                "queue_import_spec_id": (
                    f"{OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORTER_METHOD_ID}:{proposal_id}"
                ),
                "queue_ordering_plan_record_id": plan_record.get(
                    "queue_ordering_plan_record_id"
                ),
                "proposal_id": proposal_id,
                "planned_queue_index": plan_record.get("planned_queue_index"),
                "rank_source": plan_record.get("rank_source"),
                "recommended_sequence_policy_bucket": plan_record.get(
                    "recommended_sequence_policy_bucket"
                ),
                "source_idempotency_key": plan_record.get("source_idempotency_key"),
                "current_sequence_review_freshness": plan_record.get(
                    "current_sequence_review_freshness"
                ),
                "operator_proposal_payload": _import_payload(
                    plan_record,
                    payload_by_id[proposal_id],
                ),
                "operator_review_required": True,
                "operator_ack_required_before_queue_import": True,
                "operator_ack_required_before_sequence_execution": True,
                "idempotency_key_required": True,
                "sequence_consumer_required_before_execution": True,
                "queue_import_dry_run": True,
                "queue_import_applied": False,
                "read_only": True,
                "queue_mutation": False,
                "proposal_order_mutation": False,
                "mutates_console_state": False,
                "queries_console": False,
                "model_mutation": False,
                "runtime_memory_mutation": False,
                "sequence_policy_trust_store_mutation": False,
                "proposal_state_mutation": False,
                "sequence_execution_allowed_by_importer": False,
                "sequence_policy_change_allowed_by_importer": False,
                "live_widening_allowed": False,
                "live_write": False,
                "live_authority_granted": False,
                "full_autonomous_ready": False,
                "can_import_to_queue": False,
                "can_apply_live": False,
            }
        )
    return tuple(specs)


def _proposal_import_metadata(proposal: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = _mapping(proposal.get("raw"))
    return _mapping(raw.get("sequence_policy_queue_import")) or _mapping(
        proposal.get("sequence_policy_queue_import")
    )


def _proposal_dict(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _mapping(to_dict())
    return {}


def _existing_queue_proposal(target_queue: Any, proposal_id: str) -> Mapping[str, Any]:
    finder = getattr(target_queue, "find", None)
    if not callable(finder) or not proposal_id:
        return {}
    try:
        return _proposal_dict(finder(proposal_id=proposal_id))
    except TypeError:
        return {}


def _existing_queue_identity_violations(
    spec: Mapping[str, Any],
    *,
    target_queue: Any,
) -> Tuple[str, ...]:
    proposal = _existing_queue_proposal(target_queue, _string(spec.get("proposal_id")))
    if not proposal:
        return ()
    metadata = _proposal_import_metadata(proposal)
    if not metadata:
        return ("sequence_policy_queue_importer_existing_import_metadata_missing",)
    expected = {
        "queue_ordering_plan_record_id": spec.get("queue_ordering_plan_record_id"),
        "source_idempotency_key": spec.get("source_idempotency_key"),
        "current_sequence_review_freshness": spec.get(
            "current_sequence_review_freshness"
        ),
        "planned_queue_index": spec.get("planned_queue_index"),
    }
    return tuple(
        f"sequence_policy_queue_importer_existing_{key}_mismatch"
        for key, value in expected.items()
        if metadata.get(key) != value
    )


def _result_identity_violations(
    spec: Mapping[str, Any],
    *,
    created: Mapping[str, Any],
) -> Tuple[str, ...]:
    proposal = _mapping(created.get("proposal"))
    if created.get("success") is not True:
        return ("sequence_policy_queue_importer_queue_create_failed",)
    if not proposal:
        return ("sequence_policy_queue_importer_result_proposal_missing",)
    if proposal.get("id") != spec.get("proposal_id"):
        return ("sequence_policy_queue_importer_result_proposal_id_mismatch",)
    metadata = _proposal_import_metadata(proposal)
    if not metadata:
        return ("sequence_policy_queue_importer_result_import_metadata_missing",)
    expected = {
        "queue_ordering_plan_record_id": spec.get("queue_ordering_plan_record_id"),
        "source_idempotency_key": spec.get("source_idempotency_key"),
        "current_sequence_review_freshness": spec.get(
            "current_sequence_review_freshness"
        ),
        "planned_queue_index": spec.get("planned_queue_index"),
    }
    return tuple(
        f"sequence_policy_queue_importer_result_import_{key}_mismatch"
        for key, value in expected.items()
        if metadata.get(key) != value
    )


def _apply_import_specs(
    queue_import_specs: Tuple[Mapping[str, Any], ...],
    target_queue: Any,
    operator_mode: Mapping[str, Any],
) -> Tuple[Dict[str, Any], ...]:
    existing_violations = {
        _string(spec.get("proposal_id")): _existing_queue_identity_violations(
            spec,
            target_queue=target_queue,
        )
        for spec in queue_import_specs
    }
    if any(existing_violations.values()):
        return tuple(
            {
                "queue_import_result_id": (
                    "sequence_policy_queue_importer_result:"
                    f"{spec.get('proposal_id')}"
                ),
                "queue_import_spec_id": spec.get("queue_import_spec_id"),
                "proposal_id": spec.get("proposal_id"),
                "planned_queue_index": spec.get("planned_queue_index"),
                "status": (
                    "blocked"
                    if existing_violations.get(_string(spec.get("proposal_id")))
                    else "skipped"
                ),
                "success": False,
                "queue_mutation": False,
                "queue_import_applied": False,
                "operator_proposal": _existing_queue_proposal(
                    target_queue,
                    _string(spec.get("proposal_id")),
                ),
                "identity_violations": list(
                    existing_violations.get(_string(spec.get("proposal_id"))) or ()
                ),
                "blocked_by": list(
                    existing_violations.get(_string(spec.get("proposal_id")))
                    or (
                        "sequence_policy_queue_importer_batch_blocked_by_existing_identity_conflict",
                    )
                ),
                "live_write": False,
                "mutates_console_state": False,
                "queries_console": False,
                "model_mutation": False,
                "runtime_memory_mutation": False,
                "sequence_policy_trust_store_mutation": False,
                "proposal_state_mutation": False,
                "sequence_execution_allowed_by_importer": False,
                "sequence_policy_change_allowed_by_importer": False,
                "live_widening_allowed": False,
                "live_authority_granted": False,
                "full_autonomous_ready": False,
                "can_apply_live": False,
            }
            for spec in queue_import_specs
        )

    results = []
    for spec in queue_import_specs:
        result = target_queue.create(
            dict(spec["operator_proposal_payload"]),
            dict(operator_mode),
        )
        identity_violations = _result_identity_violations(spec, created=result)
        success = result.get("success") is True and not identity_violations
        results.append(
            {
                "queue_import_result_id": (
                    "sequence_policy_queue_importer_result:"
                    f"{spec.get('proposal_id')}"
                ),
                "queue_import_spec_id": spec.get("queue_import_spec_id"),
                "proposal_id": spec.get("proposal_id"),
                "planned_queue_index": spec.get("planned_queue_index"),
                "status": result.get("status"),
                "success": success,
                "queue_mutation": success and result.get("status") == "created",
                "queue_import_applied": success,
                "operator_proposal": result.get("proposal"),
                "identity_violations": list(identity_violations),
                "blocked_by": list(identity_violations),
                "live_write": False,
                "mutates_console_state": False,
                "queries_console": False,
                "model_mutation": False,
                "runtime_memory_mutation": False,
                "sequence_policy_trust_store_mutation": False,
                "proposal_state_mutation": False,
                "sequence_execution_allowed_by_importer": False,
                "sequence_policy_change_allowed_by_importer": False,
                "live_widening_allowed": False,
                "live_authority_granted": False,
                "full_autonomous_ready": False,
                "can_apply_live": False,
            }
        )
    return tuple(results)


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer(
    *,
    queue_ordering_plan: Mapping[str, Any] | None = None,
    proposal_payloads: Iterable[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    target_queue: Any = None,
    operator_mode: Mapping[str, Any] | None = None,
    apply: bool = False,
    operator_ack: bool = False,
) -> Dict[str, Any]:
    """Validate and optionally import sequence-policy-ordered proposals."""

    plan = (
        _mapping(queue_ordering_plan)
        or build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan()
    )
    summary = _mapping(plan.get("summary"))
    plan_records = _record_list(plan.get("queue_ordering_plan_records"))
    payload_by_id = _proposal_payload_map(proposal_payloads)
    payload_duplicates = _proposal_payload_duplicate_ids(proposal_payloads)
    missing_payload_ids = tuple(
        record["proposal_id"]
        for record in plan_records
        if _string(record.get("proposal_id"))
        and record["proposal_id"] not in payload_by_id
    )
    invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_string_list(summary.get("invalid_record_ids")),
                *_queue_ordering_plan_envelope_violations(plan),
                *_queue_ordering_plan_status_violations(plan, plan_records=plan_records),
                *_queue_ordering_plan_summary_violations(
                    summary,
                    plan_records=plan_records,
                ),
                *_invalid_plan_record_ids(plan_records),
                *_sequence_violations(plan_records),
                *_duplicate_values(
                    tuple(
                        _string(record.get("queue_ordering_plan_record_id"))
                        for record in plan_records
                    )
                ),
                *_duplicate_values(
                    tuple(_string(record.get("proposal_id")) for record in plan_records)
                ),
                *_duplicate_values(
                    tuple(str(record.get("planned_queue_index")) for record in plan_records)
                ),
                *payload_duplicates,
            )
        )
    )
    queue_import_specs = (
        ()
        if invalid_record_ids or not plan_records or missing_payload_ids
        else _queue_import_specs(plan_records, payload_by_id)
    )

    target_queue_provided = hasattr(target_queue, "create")
    effective_operator_mode = dict(operator_mode or _default_operator_mode())
    queue_import_results: Tuple[Dict[str, Any], ...] = ()
    if invalid_record_ids:
        status = "sequence_memory_trust_promotion_queue_importer_replacement_required"
        blocked_by = ("operator_must_replace_invalid_sequence_policy_queue_import_evidence",)
    elif not plan_records:
        status = "sequence_memory_trust_promotion_queue_importer_not_ready"
        blocked_by = ("sequence_policy_queue_ordering_plan_records_required",)
    elif missing_payload_ids:
        status = "sequence_memory_trust_promotion_queue_importer_proposal_payload_required"
        blocked_by = ("proposal_payloads_required_before_sequence_policy_queue_import",)
    elif apply and not operator_ack:
        status = "sequence_memory_trust_promotion_queue_importer_operator_ack_required"
        blocked_by = ("operator_ack_required_before_sequence_policy_queue_import",)
    elif apply and operator_ack and not target_queue_provided:
        status = "sequence_memory_trust_promotion_queue_importer_target_queue_required"
        blocked_by = ("target_operator_proposal_queue_required_before_sequence_policy_queue_import",)
    elif apply and operator_ack:
        queue_import_results = _apply_import_specs(
            queue_import_specs,
            target_queue,
            effective_operator_mode,
        )
        if queue_import_results and all(
            result.get("success") is True for result in queue_import_results
        ):
            status = "sequence_memory_trust_promotion_queue_importer_applied"
            blocked_by = ()
        else:
            status = "sequence_memory_trust_promotion_queue_importer_apply_blocked"
            blocked_by = (
                "operator_mode_or_queue_policy_blocked_sequence_policy_queue_import",
            )
    else:
        status = "sequence_memory_trust_promotion_queue_importer_dry_run_ready"
        blocked_by = ("operator_ack_and_apply_required_before_sequence_policy_queue_import",)

    applied = status == "sequence_memory_trust_promotion_queue_importer_applied"
    imported_ids = tuple(
        result["proposal_id"]
        for result in queue_import_results
        if result.get("queue_import_applied") is True and _string(result.get("proposal_id"))
    )
    queue_mutated = any(result.get("queue_mutation") is True for result in queue_import_results)
    queue_mutation_ids = tuple(
        result["proposal_id"]
        for result in queue_import_results
        if result.get("queue_mutation") is True and _string(result.get("proposal_id"))
    )
    idempotent_noop_ids = tuple(
        result["proposal_id"]
        for result in queue_import_results
        if result.get("status") == "exists"
        and result.get("success") is True
        and result.get("queue_mutation") is False
        and _string(result.get("proposal_id"))
    )
    blocked_import_ids = tuple(
        result["proposal_id"]
        for result in queue_import_results
        if result.get("success") is not True and _string(result.get("proposal_id"))
    )
    identity_violation_count = sum(
        len(_string_list(result.get("identity_violations")))
        for result in queue_import_results
    )

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORTER_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_ORDERING_PLAN_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": not applied,
            "live_write": False,
            "apply_attempted": bool(apply),
            "mutates_console_state": False,
            "queries_console": False,
            "operator_required": True,
            "operator_ack": bool(operator_ack),
            "queue_import_dry_run": not applied,
            "queue_import_applied": applied,
            "sequence_consumer_required_before_execution": True,
            "proposal_order_mutation": False,
            "queue_mutation": queue_mutated,
            "model_mutation": False,
            "runtime_memory_mutation": False,
            "sequence_policy_trust_store_mutation": False,
            "proposal_state_mutation": False,
            "sequence_execution_allowed_by_importer": False,
            "sequence_policy_change_allowed_by_importer": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_import_to_queue": False,
            "can_apply_live": False,
        },
        "summary": {
            "queue_ordering_plan_status": plan.get("status") or "missing",
            "queue_ordering_plan_record_count": len(plan_records),
            "proposal_payload_count": len(payload_by_id),
            "missing_proposal_payload_count": len(missing_payload_ids),
            "queue_import_spec_count": len(queue_import_specs),
            "queue_import_result_count": len(queue_import_results),
            "imported_proposal_count": len(imported_ids),
            "queue_mutation_count": len(queue_mutation_ids),
            "idempotent_import_noop_count": len(idempotent_noop_ids),
            "blocked_import_count": len(blocked_import_ids),
            "identity_violation_count": identity_violation_count,
            "invalid_record_count": len(invalid_record_ids),
            "apply_requested": bool(apply),
            "operator_ack": bool(operator_ack),
            "target_queue_provided": target_queue_provided,
            "queue_import_dry_run": not applied,
            "queue_import_applied": applied,
            "sequence_consumer_required_before_execution": True,
            "proposal_order_mutation": False,
            "queue_mutation": queue_mutated,
            "mutates_console_state": False,
            "queries_console": False,
            "model_mutation": False,
            "runtime_memory_mutation": False,
            "sequence_policy_trust_store_mutation": False,
            "proposal_state_mutation": False,
            "sequence_execution_allowed_by_importer": False,
            "sequence_policy_change_allowed_by_importer": False,
            "live_widening_allowed": False,
            "live_write": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_import_to_queue": False,
            "can_apply_live": False,
            "planned_import_proposal_ids": [
                spec["proposal_id"] for spec in queue_import_specs
            ],
            "imported_proposal_ids": list(imported_ids),
            "queue_mutation_proposal_ids": list(queue_mutation_ids),
            "idempotent_import_noop_proposal_ids": list(idempotent_noop_ids),
            "blocked_import_proposal_ids": list(blocked_import_ids),
            "missing_proposal_payload_ids": list(missing_payload_ids),
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Operator replaces invalid sequence-policy queue import evidence."
                if invalid_record_ids
                else (
                    "Sequence-policy queue-ordering plan records are required before queue import."
                    if not plan_records
                    else (
                        "Proposal payloads are required before sequence-policy queue import."
                        if missing_payload_ids
                        else (
                            "Operator acknowledgement is required before sequence-policy queue import."
                            if apply and not operator_ack
                            else (
                                "Explicit target operator proposal queue is required before sequence-policy queue import."
                                if apply and operator_ack and not target_queue_provided
                                else (
                                    "Sequence-policy queue import applied to explicit target queue."
                                    if applied
                                    else (
                                        "Sequence-policy queue import was blocked by operator mode or queue policy."
                                        if status
                                        == "sequence_memory_trust_promotion_queue_importer_apply_blocked"
                                        else "Operator may inspect this dry run; apply plus acknowledgement is required before queue mutation."
                                    )
                                )
                            )
                        )
                    )
                )
            ),
        },
        "queue_import_specs": list(queue_import_specs),
        "queue_import_results": list(queue_import_results),
    }
