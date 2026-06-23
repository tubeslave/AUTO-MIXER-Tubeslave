"""Read-only ordering preview from sequence-policy consumer evidence.

This module consumes sequence-policy consumer records and proposal-preview
payloads, then emits a separate recommended-order view for a future ordering or
sequence consumer. It never mutates proposals, operator queues, sequence-policy
stores, runtime memory, models, or live mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_POLICY_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_POLICY_CONSUMER_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_POLICY_CONSUMER_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_POLICY_CONSUMER_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ORDERING_PREVIEW_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ORDERING_PREVIEW_STATUSES = {
    "sequence_memory_trust_promotion_ordering_preview_not_ready",
    "sequence_memory_trust_promotion_ordering_preview_replacement_required",
    "sequence_memory_trust_promotion_ordering_preview_available",
}
REQUIRED_POLICY_CONSUMER_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer"
)
REQUIRED_POLICY_CONSUMER_FALSE_FLAGS = (
    "live_write",
    "mutates_console_state",
    "queries_console",
    "proposal_order_mutation",
    "queue_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "sequence_policy_trust_store_mutation",
    "proposal_state_mutation",
    "sequence_execution_allowed_by_consumer",
    "sequence_policy_change_allowed_by_consumer",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
    "can_import_to_queue",
    "can_apply_live",
)
REQUIRED_POLICY_CONSUMER_RECORD_FALSE_FLAGS = REQUIRED_POLICY_CONSUMER_FALSE_FLAGS


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _record_list(value: Any) -> Tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _proposal_list(value: Any) -> Tuple[Mapping[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
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


def _proposal_id(proposal: Mapping[str, Any], index: int) -> str:
    for key in ("id", "proposal_id", "action_id"):
        value = _string(proposal.get(key))
        if value:
            return value
    return f"proposal:{index}"


def _proposal_identity_tokens(proposal: Mapping[str, Any]) -> Tuple[str, ...]:
    tokens = []
    for payload in (
        proposal,
        _mapping(proposal.get("proposed_action")),
        _mapping(proposal.get("evidence")),
        _mapping(proposal.get("raw")),
    ):
        for key in ("idempotency_key", "source_idempotency_key", "policy_key"):
            token = _string(payload.get(key))
            if token:
                tokens.append(token)
    return tuple(dict.fromkeys(tokens))


def _duplicate_values(values: Tuple[str, ...]) -> Tuple[str, ...]:
    counts: Dict[str, int] = {}
    for value in values:
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return tuple(value for value, count in counts.items() if count > 1)


def _consumer_record_id(record: Mapping[str, Any], index: int) -> str:
    explicit = _string(record.get("consumer_record_id"))
    if explicit:
        return explicit
    proposal_id = _string(record.get("proposal_id"))
    return f"{proposal_id if proposal_id else 'unknown_sequence_policy_consumer_record'}:{index}"


def _policy_consumer_envelope_violations(
    policy_consumer: Mapping[str, Any],
) -> Tuple[str, ...]:
    violations = []
    if policy_consumer.get("type") != REQUIRED_POLICY_CONSUMER_TYPE:
        violations.append("sequence_policy_ordering_preview_consumer_type_mismatch")
    generated_at = policy_consumer.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("sequence_policy_ordering_preview_consumer_generated_at_invalid")
    if (
        policy_consumer.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_POLICY_CONSUMER_METHOD_ID
    ):
        violations.append("sequence_policy_ordering_preview_consumer_method_id_mismatch")
    if (
        policy_consumer.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_POLICY_CONSUMER_STATUSES
    ):
        violations.append("sequence_policy_ordering_preview_consumer_status_unknown")
    safety = _mapping(policy_consumer.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("sequence_policy_ordering_preview_consumer_read_only_not_true")
    if safety.get("policy_evidence_only") is not True:
        violations.append(
            "sequence_policy_ordering_preview_consumer_policy_evidence_only_not_true"
        )
    if safety.get("apply_attempted") is not False:
        violations.append("sequence_policy_ordering_preview_consumer_apply_attempted_not_false")
    for flag in REQUIRED_POLICY_CONSUMER_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(
                f"sequence_policy_ordering_preview_consumer_{flag}_not_false"
            )
    return tuple(violations)


def _policy_consumer_status_violations(
    policy_consumer: Mapping[str, Any],
    *,
    consumer_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    status = policy_consumer.get("status")
    if status == "sequence_memory_trust_promotion_policy_consumer_replacement_required":
        return ("sequence_policy_ordering_preview_consumer_replacement_required",)
    if (
        status == "sequence_memory_trust_promotion_policy_consumer_records_available"
        and not consumer_records
    ):
        return ("sequence_policy_ordering_preview_consumer_available_without_records",)
    if (
        status != "sequence_memory_trust_promotion_policy_consumer_records_available"
        and consumer_records
    ):
        return ("sequence_policy_ordering_preview_consumer_records_without_available_status",)
    return ()


def _policy_consumer_summary_violations(
    summary: Mapping[str, Any],
    *,
    proposals: Tuple[Mapping[str, Any], ...],
    consumer_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    expected_counts = {
        "input_proposal_count": len(proposals),
        "consumer_record_count": len(consumer_records),
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(
                f"sequence_policy_ordering_preview_consumer_summary_{key}_mismatch"
            )
    if summary.get("policy_evidence_only") is not True:
        violations.append(
            "sequence_policy_ordering_preview_consumer_summary_policy_evidence_only_not_true"
        )
    for flag in REQUIRED_POLICY_CONSUMER_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_policy_ordering_preview_consumer_summary_{flag}_not_false"
            )
    expected_ids = tuple(
        record["proposal_id"]
        for record in consumer_records
        if _string(record.get("proposal_id"))
    )
    if _string_list(summary.get("policy_consumer_proposal_ids")) != expected_ids:
        violations.append(
            "sequence_policy_ordering_preview_consumer_summary_proposal_ids_mismatch"
        )
    return tuple(violations)


def _consumer_record_violations(
    record: Mapping[str, Any],
    *,
    proposals: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    required_fields = (
        "consumer_record_id",
        "proposal_id",
        "sequence_policy_gate_record_id",
        "policy_key",
        "source_idempotency_key",
        "recommended_sequence_policy_bucket",
    )
    for field in required_fields:
        if not _string(record.get(field)):
            violations.append(f"sequence_policy_ordering_preview_record_{field}_missing")
    original_index = _summary_int(record.get("proposal_original_index"))
    if original_index is None or original_index >= len(proposals):
        violations.append("sequence_policy_ordering_preview_record_original_index_invalid")
    else:
        proposal = proposals[original_index]
        proposal_id = _string(record.get("proposal_id"))
        source_idempotency_key = _string(record.get("source_idempotency_key"))
        policy_key = _string(record.get("policy_key"))
        if _proposal_id(proposal, original_index) != proposal_id:
            violations.append(
                "sequence_policy_ordering_preview_record_proposal_link_mismatch"
            )
        if (
            proposal_id
            and source_idempotency_key
            and policy_key != f"{proposal_id}/{source_idempotency_key}"
        ):
            violations.append(
                "sequence_policy_ordering_preview_record_policy_key_identity_mismatch"
            )
        proposal_identity_tokens = _proposal_identity_tokens(proposal)
        if proposal_identity_tokens and (
            source_idempotency_key not in proposal_identity_tokens
            and policy_key not in proposal_identity_tokens
        ):
            violations.append(
                "sequence_policy_ordering_preview_record_proposal_identity_mismatch"
            )
    for field in ("base_policy_score", "adjusted_policy_score"):
        score = _safe_float(record.get(field))
        if score is None or score < 0.0 or score > 1.0:
            violations.append(f"sequence_policy_ordering_preview_record_{field}_invalid")
    if record.get("sequence_policy_eligible") is not True:
        violations.append("sequence_policy_ordering_preview_record_eligible_not_true")
    if record.get("policy_consumer_required_before_ordering") is not True:
        violations.append("sequence_policy_ordering_preview_record_consumer_required_not_true")
    if record.get("read_only") is not True:
        violations.append("sequence_policy_ordering_preview_record_read_only_not_true")
    if record.get("policy_evidence_only") is not True:
        violations.append("sequence_policy_ordering_preview_record_evidence_only_not_true")
    violations.extend(
        _context_violations(
            "sequence_policy_ordering_preview_record",
            _mapping(record.get("current_sequence_review_freshness")),
        )
    )
    for flag in REQUIRED_POLICY_CONSUMER_RECORD_FALSE_FLAGS:
        if record.get(flag) is not False:
            violations.append(f"sequence_policy_ordering_preview_record_{flag}_not_false")
    return tuple(violations)


def _invalid_consumer_record_ids(
    records: Tuple[Mapping[str, Any], ...],
    *,
    proposals: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    return tuple(
        _consumer_record_id(record, index)
        for index, record in enumerate(records)
        if _consumer_record_violations(record, proposals=proposals)
    )


def _ordered_preview_records(
    proposals: Tuple[Mapping[str, Any], ...],
    consumer_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[Dict[str, Any], ...]:
    ranked_records = tuple(
        sorted(
            consumer_records,
            key=lambda item: (
                -float(item.get("adjusted_policy_score") or 0.0),
                int(item.get("proposal_original_index") or 0),
            ),
        )
    )
    rows = []
    consumed_proposal_ids = set()
    recommended_index = 0

    for consumer_record in ranked_records:
        original_index = int(consumer_record["proposal_original_index"])
        proposal = proposals[original_index]
        proposal_id = str(consumer_record["proposal_id"])
        consumed_proposal_ids.add(proposal_id)
        rows.append(
            {
                "ordering_preview_record_id": (
                    f"{OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ORDERING_PREVIEW_METHOD_ID}:{proposal_id}"
                ),
                "proposal_id": proposal_id,
                "proposal_method_id": (
                    consumer_record.get("proposal_method_id")
                    or proposal.get("method_id")
                    or proposal.get("source")
                ),
                "proposal_kind": (
                    consumer_record.get("proposal_kind") or proposal.get("kind")
                ),
                "proposal_target": (
                    consumer_record.get("proposal_target") or proposal.get("target")
                ),
                "original_index": original_index,
                "recommended_index": recommended_index,
                "base_policy_score": consumer_record.get("base_policy_score"),
                "adjusted_policy_score": consumer_record.get("adjusted_policy_score"),
                "rank_source": "sequence_policy_consumer",
                "consumer_record_id": consumer_record.get("consumer_record_id"),
                "sequence_policy_gate_record_id": consumer_record.get(
                    "sequence_policy_gate_record_id"
                ),
                "policy_key": consumer_record.get("policy_key"),
                "source_idempotency_key": consumer_record.get("source_idempotency_key"),
                "recommended_sequence_policy_bucket": consumer_record.get(
                    "recommended_sequence_policy_bucket"
                ),
                "current_sequence_review_freshness": consumer_record.get(
                    "current_sequence_review_freshness"
                ),
                "read_only": True,
                "ordering_preview_only": True,
                "policy_consumer_required_before_queue_mutation": True,
                "sequence_consumer_required_before_execution": True,
                "proposal_order_mutation": False,
                "mutates_console_state": False,
                "queries_console": False,
                "queue_mutation": False,
                "model_mutation": False,
                "runtime_memory_mutation": False,
                "sequence_policy_trust_store_mutation": False,
                "proposal_state_mutation": False,
                "sequence_execution_allowed_by_ordering_preview": False,
                "sequence_policy_change_allowed_by_ordering_preview": False,
                "live_widening_allowed": False,
                "live_write": False,
                "live_authority_granted": False,
                "full_autonomous_ready": False,
                "can_import_to_queue": False,
                "can_apply_live": False,
            }
        )
        recommended_index += 1

    for original_index, proposal in enumerate(proposals):
        proposal_id = _proposal_id(proposal, original_index)
        if proposal_id in consumed_proposal_ids:
            continue
        rows.append(
            {
                "ordering_preview_record_id": (
                    f"{OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ORDERING_PREVIEW_METHOD_ID}:{proposal_id}"
                ),
                "proposal_id": proposal_id,
                "proposal_method_id": proposal.get("method_id") or proposal.get("source"),
                "proposal_kind": proposal.get("kind"),
                "proposal_target": proposal.get("target"),
                "original_index": original_index,
                "recommended_index": recommended_index,
                "base_policy_score": 0.0,
                "adjusted_policy_score": 0.0,
                "rank_source": "original_order",
                "consumer_record_id": None,
                "sequence_policy_gate_record_id": None,
                "policy_key": None,
                "source_idempotency_key": None,
                "recommended_sequence_policy_bucket": "unranked_original_order",
                "read_only": True,
                "ordering_preview_only": True,
                "policy_consumer_required_before_queue_mutation": True,
                "sequence_consumer_required_before_execution": True,
                "proposal_order_mutation": False,
                "mutates_console_state": False,
                "queries_console": False,
                "queue_mutation": False,
                "model_mutation": False,
                "runtime_memory_mutation": False,
                "sequence_policy_trust_store_mutation": False,
                "proposal_state_mutation": False,
                "sequence_execution_allowed_by_ordering_preview": False,
                "sequence_policy_change_allowed_by_ordering_preview": False,
                "live_widening_allowed": False,
                "live_write": False,
                "live_authority_granted": False,
                "full_autonomous_ready": False,
                "can_import_to_queue": False,
                "can_apply_live": False,
            }
        )
        recommended_index += 1

    return tuple(rows)


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview(
    *,
    proposals: Iterable[Mapping[str, Any]] | None = None,
    policy_consumer: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a read-only ordered view from sequence-policy consumer records."""

    proposal_records = _proposal_list(proposals)
    consumer = (
        _mapping(policy_consumer)
        or build_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer(
            proposals=proposal_records,
        )
    )
    summary = _mapping(consumer.get("summary"))
    consumer_records = _record_list(consumer.get("consumer_records"))
    proposal_ids = tuple(
        _proposal_id(proposal, index)
        for index, proposal in enumerate(proposal_records)
    )
    invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_string_list(summary.get("invalid_record_ids")),
                *_policy_consumer_envelope_violations(consumer),
                *_policy_consumer_status_violations(
                    consumer,
                    consumer_records=consumer_records,
                ),
                *_policy_consumer_summary_violations(
                    summary,
                    proposals=proposal_records,
                    consumer_records=consumer_records,
                ),
                *_invalid_consumer_record_ids(
                    consumer_records,
                    proposals=proposal_records,
                ),
                *_duplicate_values(proposal_ids),
                *_duplicate_values(
                    tuple(
                        _string(record.get("consumer_record_id"))
                        for record in consumer_records
                    )
                ),
                *_duplicate_values(
                    tuple(
                        _string(record.get("proposal_id"))
                        for record in consumer_records
                    )
                ),
            )
        )
    )
    ordered_proposals = (
        ()
        if invalid_record_ids or not consumer_records
        else _ordered_preview_records(proposal_records, consumer_records)
    )
    policy_ranked_ids = tuple(
        record["proposal_id"]
        for record in consumer_records
        if _string(record.get("proposal_id"))
    )
    ordered_ids = tuple(record["proposal_id"] for record in ordered_proposals)
    unranked_ids = tuple(
        record["proposal_id"]
        for record in ordered_proposals
        if record.get("rank_source") == "original_order"
    )

    if invalid_record_ids:
        status = "sequence_memory_trust_promotion_ordering_preview_replacement_required"
        blocked_by = (
            "operator_must_replace_invalid_sequence_policy_ordering_preview_evidence",
        )
    elif not consumer_records:
        status = "sequence_memory_trust_promotion_ordering_preview_not_ready"
        blocked_by = ("valid_sequence_policy_consumer_records_required",)
    else:
        status = "sequence_memory_trust_promotion_ordering_preview_available"
        blocked_by = (
            "sequence_policy_ordering_preview_is_not_queue_execution_or_live_authority",
        )

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_ORDERING_PREVIEW_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_POLICY_CONSUMER_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "live_write": False,
            "apply_attempted": False,
            "mutates_console_state": False,
            "queries_console": False,
            "operator_required": True,
            "ordering_preview_only": True,
            "proposal_order_mutation": False,
            "queue_mutation": False,
            "model_mutation": False,
            "runtime_memory_mutation": False,
            "sequence_policy_trust_store_mutation": False,
            "proposal_state_mutation": False,
            "sequence_execution_allowed_by_ordering_preview": False,
            "sequence_policy_change_allowed_by_ordering_preview": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_import_to_queue": False,
            "can_apply_live": False,
        },
        "summary": {
            "policy_consumer_status": consumer.get("status") or "missing",
            "input_proposal_count": len(proposal_records),
            "consumer_record_count": (
                len(consumer_records) if not invalid_record_ids else 0
            ),
            "ordered_preview_count": len(ordered_proposals),
            "policy_ranked_proposal_count": (
                len(policy_ranked_ids) if not invalid_record_ids else 0
            ),
            "unranked_proposal_count": len(unranked_ids),
            "invalid_record_count": len(invalid_record_ids),
            "recommended_order_available": bool(ordered_proposals),
            "ordering_preview_only": True,
            "mutates_console_state": False,
            "queries_console": False,
            "proposal_order_mutation": False,
            "queue_mutation": False,
            "model_mutation": False,
            "runtime_memory_mutation": False,
            "sequence_policy_trust_store_mutation": False,
            "proposal_state_mutation": False,
            "sequence_execution_allowed_by_ordering_preview": False,
            "sequence_policy_change_allowed_by_ordering_preview": False,
            "live_widening_allowed": False,
            "live_write": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_import_to_queue": False,
            "can_apply_live": False,
            "recommended_ordered_proposal_ids": list(ordered_ids),
            "policy_ranked_proposal_ids": (
                [] if invalid_record_ids else list(policy_ranked_ids)
            ),
            "unranked_proposal_ids": list(unranked_ids),
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Operator replaces invalid sequence-policy ordering-preview evidence."
                if invalid_record_ids
                else (
                    "Valid sequence-policy consumer records are required before ordering preview."
                    if not consumer_records
                    else "Operator may inspect this view; a separate queue or sequence consumer is required before mutation."
                )
            ),
        },
        "ordered_proposals": list(ordered_proposals),
    }
