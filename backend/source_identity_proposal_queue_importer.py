"""Operator-gated queue importer for source-identity ordering plans.

The importer consumes read-only queue-ordering plan records and source proposal
payloads, then emits dry-run import specs by default. It mutates only an
explicit caller-provided operator proposal queue, and only when invoked with
`apply=True` and `operator_ack=True`. Product Layer preview calls this as a
dry run, so normal previews remain non-mutating. The importer never touches
live mixer transport, console state, source identity trust stores, runtime
memory, or classifier models.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Tuple

try:
    from source_identity_proposal_queue_ordering_plan import (
        SOURCE_IDENTITY_PROPOSAL_QUEUE_ORDERING_PLAN_METHOD_ID,
        SOURCE_IDENTITY_PROPOSAL_QUEUE_ORDERING_PLAN_STATUSES,
        build_source_identity_proposal_queue_ordering_plan,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.source_identity_proposal_queue_ordering_plan import (
        SOURCE_IDENTITY_PROPOSAL_QUEUE_ORDERING_PLAN_METHOD_ID,
        SOURCE_IDENTITY_PROPOSAL_QUEUE_ORDERING_PLAN_STATUSES,
        build_source_identity_proposal_queue_ordering_plan,
    )


SOURCE_IDENTITY_PROPOSAL_QUEUE_IMPORTER_METHOD_ID = (
    "source_identity_proposal_queue_importer"
)
SOURCE_IDENTITY_PROPOSAL_QUEUE_IMPORTER_STATUSES = {
    "source_identity_proposal_queue_importer_not_ready",
    "source_identity_proposal_queue_importer_replacement_required",
    "source_identity_proposal_queue_importer_source_payload_required",
    "source_identity_proposal_queue_importer_operator_ack_required",
    "source_identity_proposal_queue_importer_target_queue_required",
    "source_identity_proposal_queue_importer_dry_run_ready",
    "source_identity_proposal_queue_importer_apply_blocked",
    "source_identity_proposal_queue_importer_applied",
}
REQUIRED_QUEUE_ORDERING_PLAN_TYPE = "source_identity_proposal_queue_ordering_plan"
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
    "source_identity_trust_store_mutation",
    "direct_downstream_priority_mutation",
    "downstream_priority_change_allowed_by_policy",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
    "can_import_to_queue",
    "can_apply_live",
)
REQUIRED_PLAN_RECORD_FALSE_FLAGS = REQUIRED_QUEUE_ORDERING_PLAN_FALSE_FLAGS
ALLOWED_RANK_SOURCES = {"source_identity_policy", "original_order"}


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
    return f"{proposal_id if proposal_id else 'unknown_queue_ordering_plan_record'}:{index}"


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
        violations.append("source_identity_proposal_queue_importer_plan_type_mismatch")
    generated_at = queue_ordering_plan.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("source_identity_proposal_queue_importer_plan_generated_at_invalid")
    if (
        queue_ordering_plan.get("method_id")
        != SOURCE_IDENTITY_PROPOSAL_QUEUE_ORDERING_PLAN_METHOD_ID
    ):
        violations.append("source_identity_proposal_queue_importer_plan_method_id_mismatch")
    if queue_ordering_plan.get("status") not in SOURCE_IDENTITY_PROPOSAL_QUEUE_ORDERING_PLAN_STATUSES:
        violations.append("source_identity_proposal_queue_importer_plan_status_unknown")
    safety = _mapping(queue_ordering_plan.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("source_identity_proposal_queue_importer_plan_read_only_not_true")
    if safety.get("queue_ordering_plan_only") is not True:
        violations.append("source_identity_proposal_queue_importer_plan_only_not_true")
    for flag in REQUIRED_QUEUE_ORDERING_PLAN_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"source_identity_proposal_queue_importer_plan_{flag}_not_false")
    return tuple(violations)


def _queue_ordering_plan_status_violations(
    queue_ordering_plan: Mapping[str, Any],
    *,
    plan_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    status = queue_ordering_plan.get("status")
    if status == "source_identity_proposal_queue_ordering_plan_replacement_required":
        return ("source_identity_proposal_queue_importer_plan_replacement_required",)
    if status == "source_identity_proposal_queue_ordering_plan_available" and not plan_records:
        return ("source_identity_proposal_queue_importer_plan_available_without_records",)
    if status != "source_identity_proposal_queue_ordering_plan_available" and plan_records:
        return ("source_identity_proposal_queue_importer_plan_records_without_available_status",)
    return ()


def _queue_ordering_plan_summary_violations(
    summary: Mapping[str, Any],
    *,
    plan_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    ranked_records = tuple(
        record for record in plan_records if record.get("rank_source") == "source_identity_policy"
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
            violations.append(
                f"source_identity_proposal_queue_importer_plan_summary_{key}_mismatch"
            )
    if summary.get("queue_ordering_plan_available") is not bool(plan_records):
        violations.append(
            "source_identity_proposal_queue_importer_plan_summary_available_mismatch"
        )
    if summary.get("queue_ordering_plan_only") is not True:
        violations.append("source_identity_proposal_queue_importer_plan_summary_only_not_true")
    for flag in REQUIRED_QUEUE_ORDERING_PLAN_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"source_identity_proposal_queue_importer_plan_summary_{flag}_not_false"
            )
    expected_plan_ids = tuple(
        record["proposal_id"] for record in plan_records if _string(record.get("proposal_id"))
    )
    if _string_list(summary.get("planned_queue_proposal_ids")) != expected_plan_ids:
        violations.append(
            "source_identity_proposal_queue_importer_plan_summary_proposal_ids_mismatch"
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
        "recommended_rank_bucket",
    )
    for field in required_fields:
        if not _string(record.get(field)) and field != "planned_queue_index":
            violations.append(f"source_identity_proposal_queue_importer_plan_record_{field}_missing")
    if _summary_int(record.get("planned_queue_index")) is None:
        violations.append("source_identity_proposal_queue_importer_plan_record_planned_queue_index_invalid")
    for field in ("original_index", "recommended_index"):
        if _summary_int(record.get(field)) is None:
            violations.append(f"source_identity_proposal_queue_importer_plan_record_{field}_invalid")
    for field in ("base_priority_score", "adjusted_priority_score"):
        score = _safe_float(record.get(field))
        if score is None or score < 0.0 or score > 1.0:
            violations.append(f"source_identity_proposal_queue_importer_plan_record_{field}_invalid")
    if record.get("rank_source") not in ALLOWED_RANK_SOURCES:
        violations.append("source_identity_proposal_queue_importer_plan_record_rank_source_unknown")
    if record.get("operator_review_required") is not True:
        violations.append("source_identity_proposal_queue_importer_plan_record_review_required_not_true")
    if record.get("operator_ack_required_before_queue_import") is not True:
        violations.append("source_identity_proposal_queue_importer_plan_record_ack_required_not_true")
    if record.get("idempotency_key_required") is not True:
        violations.append("source_identity_proposal_queue_importer_plan_record_idempotency_required_not_true")
    if record.get("queue_ordering_plan_only") is not True:
        violations.append("source_identity_proposal_queue_importer_plan_record_plan_only_not_true")
    if record.get("read_only") is not True:
        violations.append("source_identity_proposal_queue_importer_plan_record_read_only_not_true")
    for flag in REQUIRED_PLAN_RECORD_FALSE_FLAGS:
        if record.get(flag) is not False:
            violations.append(f"source_identity_proposal_queue_importer_plan_record_{flag}_not_false")
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
        return ("source_identity_proposal_queue_importer_planned_index_sequence_mismatch",)
    return ()


def _import_payload(
    plan_record: Mapping[str, Any],
    source_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    proposal_id = str(plan_record["proposal_id"])
    payload = dict(source_payload)
    payload["id"] = proposal_id
    payload["proposal_id"] = proposal_id
    payload["title"] = str(
        payload.get("title")
        or payload.get("description")
        or payload.get("message")
        or f"Source identity queue import: {proposal_id}"
    )
    if not payload.get("kind"):
        payload["kind"] = plan_record.get("proposal_kind") or "analysis"
    if payload.get("target") is None and plan_record.get("proposal_target") is not None:
        payload["target"] = plan_record.get("proposal_target")
    payload["source"] = SOURCE_IDENTITY_PROPOSAL_QUEUE_IMPORTER_METHOD_ID
    payload["reason"] = str(
        payload.get("reason")
        or payload.get("rationale")
        or "source_identity_queue_ordering_plan"
    )
    payload["source_identity_queue_import"] = {
        "queue_ordering_plan_record_id": plan_record.get("queue_ordering_plan_record_id"),
        "ordering_preview_record_id": plan_record.get("ordering_preview_record_id"),
        "policy_record_id": plan_record.get("policy_record_id"),
        "planned_queue_index": plan_record.get("planned_queue_index"),
        "rank_source": plan_record.get("rank_source"),
        "recommended_rank_bucket": plan_record.get("recommended_rank_bucket"),
        "operator_ack_required_before_queue_import": True,
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
                    f"{SOURCE_IDENTITY_PROPOSAL_QUEUE_IMPORTER_METHOD_ID}:{proposal_id}"
                ),
                "queue_ordering_plan_record_id": plan_record.get(
                    "queue_ordering_plan_record_id"
                ),
                "proposal_id": proposal_id,
                "planned_queue_index": plan_record.get("planned_queue_index"),
                "rank_source": plan_record.get("rank_source"),
                "recommended_rank_bucket": plan_record.get("recommended_rank_bucket"),
                "operator_proposal_payload": _import_payload(
                    plan_record,
                    payload_by_id[proposal_id],
                ),
                "operator_review_required": True,
                "operator_ack_required_before_queue_import": True,
                "idempotency_key_required": True,
                "queue_import_dry_run": True,
                "queue_import_applied": False,
                "read_only": True,
                "queue_mutation": False,
                "mutates_console_state": False,
                "queries_console": False,
                "model_mutation": False,
                "runtime_memory_mutation": False,
                "source_identity_trust_store_mutation": False,
                "direct_downstream_priority_mutation": False,
                "downstream_priority_change_allowed_by_policy": False,
                "live_widening_allowed": False,
                "live_write": False,
                "live_authority_granted": False,
                "full_autonomous_ready": False,
                "can_import_to_queue": False,
                "can_apply_live": False,
            }
        )
    return tuple(specs)


def _apply_import_specs(
    queue_import_specs: Tuple[Mapping[str, Any], ...],
    target_queue: Any,
    operator_mode: Mapping[str, Any],
) -> Tuple[Dict[str, Any], ...]:
    results = []
    for spec in queue_import_specs:
        result = target_queue.create(
            dict(spec["operator_proposal_payload"]),
            dict(operator_mode),
        )
        results.append(
            {
                "queue_import_result_id": (
                    "source_identity_proposal_queue_importer_result:"
                    f"{spec.get('proposal_id')}"
                ),
                "queue_import_spec_id": spec.get("queue_import_spec_id"),
                "proposal_id": spec.get("proposal_id"),
                "planned_queue_index": spec.get("planned_queue_index"),
                "status": result.get("status"),
                "success": result.get("success") is True,
                "queue_mutation": result.get("success") is True,
                "queue_import_applied": result.get("success") is True,
                "operator_proposal": result.get("proposal"),
                "live_write": False,
                "mutates_console_state": False,
                "queries_console": False,
                "model_mutation": False,
                "runtime_memory_mutation": False,
                "source_identity_trust_store_mutation": False,
                "direct_downstream_priority_mutation": False,
                "downstream_priority_change_allowed_by_policy": False,
                "live_widening_allowed": False,
                "live_authority_granted": False,
                "full_autonomous_ready": False,
                "can_apply_live": False,
            }
        )
    return tuple(results)


def build_source_identity_proposal_queue_importer(
    *,
    queue_ordering_plan: Mapping[str, Any] | None = None,
    proposal_payloads: Iterable[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    target_queue: Any = None,
    operator_mode: Mapping[str, Any] | None = None,
    apply: bool = False,
    operator_ack: bool = False,
) -> Dict[str, Any]:
    """Validate and optionally import source-identity-ordered proposals."""

    plan = _mapping(queue_ordering_plan) or build_source_identity_proposal_queue_ordering_plan()
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
        status = "source_identity_proposal_queue_importer_replacement_required"
        blocked_by = ("operator_must_replace_invalid_queue_import_evidence",)
    elif not plan_records:
        status = "source_identity_proposal_queue_importer_not_ready"
        blocked_by = ("queue_ordering_plan_records_required_before_queue_import",)
    elif missing_payload_ids:
        status = "source_identity_proposal_queue_importer_source_payload_required"
        blocked_by = ("source_proposal_payloads_required_before_queue_import",)
    elif apply and not operator_ack:
        status = "source_identity_proposal_queue_importer_operator_ack_required"
        blocked_by = ("operator_ack_required_before_source_identity_queue_import",)
    elif apply and operator_ack and not target_queue_provided:
        status = "source_identity_proposal_queue_importer_target_queue_required"
        blocked_by = ("target_operator_proposal_queue_required_before_queue_import",)
    elif apply and operator_ack:
        queue_import_results = _apply_import_specs(
            queue_import_specs,
            target_queue,
            effective_operator_mode,
        )
        if any(result.get("success") is True for result in queue_import_results):
            status = "source_identity_proposal_queue_importer_applied"
            blocked_by = ()
        else:
            status = "source_identity_proposal_queue_importer_apply_blocked"
            blocked_by = ("operator_mode_or_queue_policy_blocked_source_identity_queue_import",)
    else:
        status = "source_identity_proposal_queue_importer_dry_run_ready"
        blocked_by = ("operator_ack_and_apply_required_before_source_identity_queue_import",)

    applied = status == "source_identity_proposal_queue_importer_applied"
    imported_ids = tuple(
        result["proposal_id"]
        for result in queue_import_results
        if result.get("queue_import_applied") is True and _string(result.get("proposal_id"))
    )

    return {
        "type": "source_identity_proposal_queue_importer",
        "generated_at": _utc_now(),
        "method_id": SOURCE_IDENTITY_PROPOSAL_QUEUE_IMPORTER_METHOD_ID,
        "source_method_id": SOURCE_IDENTITY_PROPOSAL_QUEUE_ORDERING_PLAN_METHOD_ID,
        "status": status,
        "safety": {
            "read_only": not applied,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "operator_required": True,
            "operator_ack": bool(operator_ack),
            "queue_import_dry_run": not applied,
            "queue_import_applied": applied,
            "proposal_order_mutation": False,
            "queue_mutation": applied,
            "model_mutation": False,
            "runtime_memory_mutation": False,
            "source_identity_trust_store_mutation": False,
            "direct_downstream_priority_mutation": False,
            "downstream_priority_change_allowed_by_policy": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_import_to_queue": False,
            "can_apply_live": False,
        },
        "summary": {
            "queue_ordering_plan_status": plan.get("status") or "missing",
            "queue_ordering_plan_record_count": len(plan_records),
            "source_payload_count": len(payload_by_id),
            "missing_source_payload_count": len(missing_payload_ids),
            "queue_import_spec_count": len(queue_import_specs),
            "queue_import_result_count": len(queue_import_results),
            "imported_proposal_count": len(imported_ids),
            "invalid_record_count": len(invalid_record_ids),
            "apply_requested": bool(apply),
            "operator_ack": bool(operator_ack),
            "target_queue_provided": target_queue_provided,
            "queue_import_dry_run": not applied,
            "queue_import_applied": applied,
            "proposal_order_mutation": False,
            "queue_mutation": applied,
            "mutates_console_state": False,
            "queries_console": False,
            "model_mutation": False,
            "runtime_memory_mutation": False,
            "source_identity_trust_store_mutation": False,
            "direct_downstream_priority_mutation": False,
            "downstream_priority_change_allowed_by_policy": False,
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
            "missing_source_payload_ids": list(missing_payload_ids),
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Operator replaces invalid queue import evidence."
                if invalid_record_ids
                else (
                    "Queue-ordering plan records are required before queue import."
                    if not plan_records
                    else (
                        "Source proposal payloads are required before queue import."
                        if missing_payload_ids
                        else (
                            "Operator acknowledgement is required before queue import."
                            if apply and not operator_ack
                            else (
                                "Explicit target operator proposal queue is required before queue import."
                                if apply and operator_ack and not target_queue_provided
                                else (
                                    "Queue import applied to explicit target queue."
                                    if applied
                                    else (
                                        "Queue import was blocked by operator mode or queue policy."
                                        if status
                                        == "source_identity_proposal_queue_importer_apply_blocked"
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
