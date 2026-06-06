"""Read-only sequence-policy consumer from trusted sequence-memory evidence.

This module consumes sequence-policy gate records and proposal-preview payloads,
then emits read-only policy evidence for a future ordering or sequence consumer.
It does not reorder proposals in-place, import queue items, mutate policy stores,
update models, or touch live mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_SEQUENCE_POLICY_GATE_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_SEQUENCE_POLICY_GATE_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_SEQUENCE_POLICY_GATE_METHOD_ID,
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_SEQUENCE_POLICY_GATE_STATUSES,
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_POLICY_CONSUMER_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_POLICY_CONSUMER_STATUSES = {
    "sequence_memory_trust_promotion_policy_consumer_not_ready",
    "sequence_memory_trust_promotion_policy_consumer_replacement_required",
    "sequence_memory_trust_promotion_policy_consumer_records_available",
}
REQUIRED_SEQUENCE_POLICY_GATE_TYPE = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate"
)
REQUIRED_SEQUENCE_GATE_FALSE_SAFETY_FLAGS = (
    "live_write",
    "apply_attempted",
    "mutates_console_state",
    "queries_console",
    "queue_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "sequence_policy_trust_store_mutation",
    "proposal_state_mutation",
    "sequence_execution_allowed_by_gate",
    "sequence_policy_change_allowed_by_gate",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_SEQUENCE_GATE_FALSE_SUMMARY_FLAGS = (
    "queue_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "sequence_policy_trust_store_mutation",
    "proposal_state_mutation",
    "sequence_execution_allowed_by_gate",
    "sequence_policy_change_allowed_by_gate",
    "live_write",
    "live_authority_granted",
    "full_autonomous_ready",
    "can_import_to_queue",
    "can_apply_live",
)
REQUIRED_POLICY_GATE_RECORD_FALSE_FLAGS = (
    "live_write",
    "apply_attempted",
    "mutates_console_state",
    "queries_console",
    "queue_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "sequence_policy_trust_store_mutation",
    "proposal_state_mutation",
    "sequence_execution_allowed_by_gate",
    "sequence_policy_change_allowed_by_gate",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
    "can_import_to_queue",
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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if numeric != numeric or numeric in {float("inf"), float("-inf")}:
        return default
    return numeric


def _clip(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


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


def _gate_record_id(record: Mapping[str, Any], index: int) -> str:
    explicit = _string(record.get("sequence_policy_gate_record_id"))
    if explicit:
        return explicit
    policy_key = _string(record.get("policy_key"))
    return f"{policy_key if policy_key else 'unknown_sequence_policy_gate_record'}:{index}"


def _duplicate_ids(
    records: Tuple[Mapping[str, Any], ...],
    field: str,
) -> Tuple[str, ...]:
    values: Dict[str, list[str]] = {}
    for index, record in enumerate(records):
        value = _string(record.get(field))
        if not value:
            continue
        values.setdefault(value, []).append(_gate_record_id(record, index))
    return tuple(
        record_id
        for record_ids in values.values()
        if len(record_ids) > 1
        for record_id in record_ids
    )


def _sequence_gate_envelope_violations(
    sequence_policy_gate: Mapping[str, Any],
) -> Tuple[str, ...]:
    violations = []
    if sequence_policy_gate.get("type") != REQUIRED_SEQUENCE_POLICY_GATE_TYPE:
        violations.append(
            "sequence_policy_consumer_gate_type_mismatch"
        )
    generated_at = sequence_policy_gate.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("sequence_policy_consumer_gate_generated_at_invalid")
    if (
        sequence_policy_gate.get("method_id")
        != OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_SEQUENCE_POLICY_GATE_METHOD_ID
    ):
        violations.append("sequence_policy_consumer_gate_method_id_mismatch")
    if (
        sequence_policy_gate.get("status")
        not in OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_SEQUENCE_POLICY_GATE_STATUSES
    ):
        violations.append("sequence_policy_consumer_gate_status_unknown")
    safety = _mapping(sequence_policy_gate.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("sequence_policy_consumer_gate_read_only_not_true")
    if safety.get("policy_gate_only") is not True:
        violations.append("sequence_policy_consumer_gate_policy_only_not_true")
    for flag in REQUIRED_SEQUENCE_GATE_FALSE_SAFETY_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"sequence_policy_consumer_gate_{flag}_not_false")
    return tuple(violations)


def _sequence_gate_status_violations(
    sequence_policy_gate: Mapping[str, Any],
    *,
    policy_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    status = sequence_policy_gate.get("status")
    if status == "sequence_memory_trust_promotion_sequence_policy_gate_replacement_required":
        return ("sequence_policy_consumer_gate_replacement_required",)
    if (
        status == "verified_sequence_memory_trust_promotion_policy_records_available"
        and not policy_records
    ):
        return ("sequence_policy_consumer_gate_available_without_records",)
    if (
        status != "verified_sequence_memory_trust_promotion_policy_records_available"
        and policy_records
    ):
        return ("sequence_policy_consumer_gate_records_without_available_status",)
    return ()


def _sequence_gate_summary_violations(
    summary: Mapping[str, Any],
    *,
    policy_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    eligible_records = tuple(
        record for record in policy_records if record.get("sequence_policy_eligible") is True
    )
    violations = []
    expected_counts = {
        "sequence_policy_gate_record_count": len(policy_records),
        "eligible_record_count": len(eligible_records),
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(f"sequence_policy_consumer_gate_summary_{key}_mismatch")
    if summary.get("sequence_policy_eligible") is not bool(eligible_records):
        violations.append("sequence_policy_consumer_gate_summary_eligibility_mismatch")
    if summary.get("policy_gate_only") is not True:
        violations.append("sequence_policy_consumer_gate_summary_policy_only_not_true")
    for flag in REQUIRED_SEQUENCE_GATE_FALSE_SUMMARY_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"sequence_policy_consumer_gate_summary_{flag}_not_false"
            )
    expected_policy_keys = tuple(
        record["policy_key"]
        for record in eligible_records
        if _string(record.get("policy_key"))
    )
    if _string_list(summary.get("eligible_policy_keys")) != expected_policy_keys:
        violations.append(
            "sequence_policy_consumer_gate_summary_eligible_policy_keys_mismatch"
        )
    return tuple(violations)


def _policy_gate_record_violations(record: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    required_fields = (
        "sequence_policy_gate_record_id",
        "proposal_id",
        "learning_scope",
        "value_type",
        "source_idempotency_key",
        "policy_key",
        "target_policy_state",
        "sequence_policy_gate_status",
    )
    for field in required_fields:
        if not _string(record.get(field)):
            violations.append(f"sequence_policy_consumer_gate_record_{field}_missing")
    if record.get("rollback_verified") is not True:
        violations.append("sequence_policy_consumer_gate_record_rollback_verified_not_true")
    if record.get("sequence_policy_eligible") is not True:
        violations.append("sequence_policy_consumer_gate_record_eligible_not_true")
    if record.get("eligible_for_sequence_policy_consumer") is not True:
        violations.append("sequence_policy_consumer_gate_record_consumer_eligible_not_true")
    if record.get("policy_gate_only") is not True:
        violations.append("sequence_policy_consumer_gate_record_policy_only_not_true")
    if record.get("sequence_policy_consumer_required") is not True:
        violations.append("sequence_policy_consumer_gate_record_consumer_required_not_true")
    if record.get("read_only") is not True:
        violations.append("sequence_policy_consumer_gate_record_read_only_not_true")
    for flag in REQUIRED_POLICY_GATE_RECORD_FALSE_FLAGS:
        if record.get(flag) is not False:
            violations.append(f"sequence_policy_consumer_gate_record_{flag}_not_false")
    return tuple(violations)


def _invalid_policy_gate_record_ids(
    records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    return tuple(
        _gate_record_id(record, index)
        for index, record in enumerate(records)
        if _policy_gate_record_violations(record)
    )


def _proposal_id(proposal: Mapping[str, Any], index: int) -> str:
    for key in ("id", "proposal_id", "action_id"):
        value = _string(proposal.get(key))
        if value:
            return value
    return f"proposal:{index}"


def _normalize_token(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text if text else ""


def _proposal_policy_tokens(proposal: Mapping[str, Any], index: int) -> Tuple[str, ...]:
    tokens = [_proposal_id(proposal, index)]
    nested_payloads = (
        proposal,
        _mapping(proposal.get("proposed_action")),
        _mapping(proposal.get("evidence")),
        _mapping(proposal.get("raw")),
    )
    for payload in nested_payloads:
        for key in (
            "proposal_id",
            "idempotency_key",
            "source_idempotency_key",
            "policy_key",
            "learning_scope",
            "value_type",
            "action_id",
        ):
            token = _normalize_token(payload.get(key))
            if token:
                tokens.append(token)
    return tuple(dict.fromkeys(tokens))


def _severity_score(value: Any) -> float:
    severity = str(value or "").strip().lower()
    return {
        "critical": 0.42,
        "high": 0.34,
        "medium": 0.24,
        "low": 0.14,
        "info": 0.08,
    }.get(severity, 0.18)


def _explicit_priority_score(value: Any) -> float:
    if value is None:
        return 0.0
    numeric = _safe_float(value, default=0.0)
    if numeric <= 0.0:
        return 0.0
    return _clip((4.0 - numeric) / 3.0, 0.0, 1.0) * 0.24


def _confidence_score(value: Any) -> float:
    numeric = _safe_float(value, default=0.0)
    if numeric > 1.0:
        numeric = numeric / 100.0
    return _clip(numeric, 0.0, 1.0) * 0.34


def _base_policy_score(proposal: Mapping[str, Any]) -> float:
    return round(
        _clip(
            0.12
            + _severity_score(proposal.get("severity"))
            + _confidence_score(proposal.get("confidence"))
            + _explicit_priority_score(proposal.get("proposal_priority")),
            0.0,
            1.0,
        ),
        3,
    )


def _eligible_lookup_keyed(
    eligible_records: Tuple[Mapping[str, Any], ...],
) -> Dict[str, Mapping[str, Any]]:
    lookup: Dict[str, Mapping[str, Any]] = {}
    for record in eligible_records:
        for key in (
            "proposal_id",
            "source_idempotency_key",
            "policy_key",
            "learning_scope",
            "value_type",
        ):
            token = _string(record.get(key))
            if token:
                lookup.setdefault(token, record)
    return lookup


def _matching_gate_record(
    proposal: Mapping[str, Any],
    index: int,
    eligible_by_token: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    for token in _proposal_policy_tokens(proposal, index):
        record = eligible_by_token.get(token)
        if record:
            return record
    return {}


def _consumer_records(
    proposals: Tuple[Mapping[str, Any], ...],
    eligible_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[Dict[str, Any], ...]:
    eligible_by_token = _eligible_lookup_keyed(eligible_records)
    records = []
    for index, proposal in enumerate(proposals):
        gate_record = _matching_gate_record(proposal, index, eligible_by_token)
        if not gate_record:
            continue
        proposal_id = _proposal_id(proposal, index)
        base_score = _base_policy_score(proposal)
        sequence_policy_adjustment = 0.2
        adjusted_score = round(_clip(base_score + sequence_policy_adjustment, 0.0, 1.0), 3)
        records.append(
            {
                "consumer_record_id": (
                    f"operator_sequence_policy_memory_trust_promotion_policy_consumer:{proposal_id}"
                ),
                "proposal_id": proposal_id,
                "proposal_method_id": proposal.get("method_id") or proposal.get("source"),
                "proposal_kind": proposal.get("kind"),
                "proposal_target": proposal.get("target"),
                "proposal_original_index": index,
                "sequence_policy_gate_record_id": gate_record.get(
                    "sequence_policy_gate_record_id"
                ),
                "policy_key": gate_record.get("policy_key"),
                "learning_scope": gate_record.get("learning_scope"),
                "value_type": gate_record.get("value_type"),
                "source_idempotency_key": gate_record.get("source_idempotency_key"),
                "target_policy_state": gate_record.get("target_policy_state"),
                "sequence_policy_gate_status": gate_record.get(
                    "sequence_policy_gate_status"
                ),
                "base_policy_score": base_score,
                "sequence_policy_adjustment": sequence_policy_adjustment,
                "adjusted_policy_score": adjusted_score,
                "recommended_sequence_policy_bucket": "trusted_sequence_memory",
                "sequence_policy_eligible": True,
                "policy_consumer_required_before_ordering": True,
                "read_only": True,
                "policy_evidence_only": True,
                "proposal_order_mutation": False,
                "mutates_console_state": False,
                "queries_console": False,
                "queue_mutation": False,
                "model_mutation": False,
                "runtime_memory_mutation": False,
                "sequence_policy_trust_store_mutation": False,
                "proposal_state_mutation": False,
                "sequence_execution_allowed_by_consumer": False,
                "sequence_policy_change_allowed_by_consumer": False,
                "live_widening_allowed": False,
                "live_write": False,
                "live_authority_granted": False,
                "full_autonomous_ready": False,
                "can_import_to_queue": False,
                "can_apply_live": False,
            }
        )
    return tuple(
        sorted(
            records,
            key=lambda item: (
                -float(item["adjusted_policy_score"]),
                int(item["proposal_original_index"]),
            ),
        )
    )


def _unmatched_eligible_policy_keys(
    records: Tuple[Mapping[str, Any], ...],
    consumer_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    matched = {
        _string(record.get("policy_key"))
        for record in consumer_records
        if _string(record.get("policy_key"))
    }
    return tuple(
        record["policy_key"]
        for record in records
        if _string(record.get("policy_key")) and record["policy_key"] not in matched
    )


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer(
    *,
    proposals: Iterable[Mapping[str, Any]] | None = None,
    sequence_policy_gate: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build read-only policy evidence for sequence-aware proposal handling."""

    gate = (
        _mapping(sequence_policy_gate)
        or build_operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate()
    )
    summary = _mapping(gate.get("summary"))
    policy_gate_records = _record_list(gate.get("policy_records"))
    eligible_records = tuple(
        record
        for record in policy_gate_records
        if record.get("sequence_policy_eligible") is True
    )
    proposal_records = _proposal_list(proposals)
    invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_string_list(summary.get("invalid_record_ids")),
                *_sequence_gate_envelope_violations(gate),
                *_sequence_gate_status_violations(
                    gate,
                    policy_records=policy_gate_records,
                ),
                *_sequence_gate_summary_violations(
                    summary,
                    policy_records=policy_gate_records,
                ),
                *_invalid_policy_gate_record_ids(eligible_records),
                *_duplicate_ids(eligible_records, "sequence_policy_gate_record_id"),
                *_duplicate_ids(eligible_records, "policy_key"),
                *_duplicate_ids(eligible_records, "proposal_id"),
            )
        )
    )
    consumer_records = (
        ()
        if invalid_record_ids
        else _consumer_records(proposal_records, eligible_records)
    )
    unmatched_policy_keys = (
        ()
        if invalid_record_ids
        else _unmatched_eligible_policy_keys(eligible_records, consumer_records)
    )

    if invalid_record_ids:
        status = "sequence_memory_trust_promotion_policy_consumer_replacement_required"
        blocked_by = ("operator_must_replace_invalid_sequence_policy_consumer_evidence",)
    elif not eligible_records:
        status = "sequence_memory_trust_promotion_policy_consumer_not_ready"
        blocked_by = ("eligible_sequence_policy_gate_records_required",)
    elif not consumer_records:
        status = "sequence_memory_trust_promotion_policy_consumer_not_ready"
        blocked_by = ("matching_preview_proposals_required_for_sequence_policy_consumer",)
    else:
        status = "sequence_memory_trust_promotion_policy_consumer_records_available"
        blocked_by = (
            "sequence_policy_consumer_records_are_not_queue_execution_or_live_authority",
        )

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_POLICY_CONSUMER_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_SEQUENCE_POLICY_GATE_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "live_write": False,
            "apply_attempted": False,
            "mutates_console_state": False,
            "queries_console": False,
            "operator_required": True,
            "policy_evidence_only": True,
            "proposal_order_mutation": False,
            "queue_mutation": False,
            "model_mutation": False,
            "runtime_memory_mutation": False,
            "sequence_policy_trust_store_mutation": False,
            "proposal_state_mutation": False,
            "sequence_execution_allowed_by_consumer": False,
            "sequence_policy_change_allowed_by_consumer": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_import_to_queue": False,
            "can_apply_live": False,
        },
        "summary": {
            "sequence_policy_gate_status": gate.get("status") or "missing",
            "input_proposal_count": len(proposal_records),
            "sequence_policy_gate_record_count": len(policy_gate_records),
            "eligible_sequence_policy_gate_record_count": len(eligible_records),
            "consumer_record_count": len(consumer_records),
            "unmatched_eligible_policy_count": len(unmatched_policy_keys),
            "invalid_record_count": len(invalid_record_ids),
            "policy_evidence_only": True,
            "mutates_console_state": False,
            "queries_console": False,
            "proposal_order_mutation": False,
            "queue_mutation": False,
            "model_mutation": False,
            "runtime_memory_mutation": False,
            "sequence_policy_trust_store_mutation": False,
            "proposal_state_mutation": False,
            "sequence_execution_allowed_by_consumer": False,
            "sequence_policy_change_allowed_by_consumer": False,
            "live_widening_allowed": False,
            "live_write": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_import_to_queue": False,
            "can_apply_live": False,
            "policy_consumer_proposal_ids": [
                record["proposal_id"] for record in consumer_records
            ],
            "eligible_policy_keys": [
                record["policy_key"] for record in eligible_records
            ],
            "unmatched_eligible_policy_keys": list(unmatched_policy_keys),
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Operator replaces invalid sequence-policy consumer evidence."
                if invalid_record_ids
                else (
                    "Eligible sequence-policy gate records are required before policy consumption."
                    if not eligible_records
                    else (
                        "Add matching preview proposals before sequence-policy evidence can be projected."
                        if not consumer_records
                        else "Feed consumer records to a separate ordering or sequence-policy consumer before mutating order, queues, policy stores, or live state."
                    )
                )
            ),
        },
        "consumer_records": list(consumer_records),
    }
