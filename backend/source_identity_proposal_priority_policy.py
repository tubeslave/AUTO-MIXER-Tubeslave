"""Read-only proposal priority policy from trusted source-identity evidence.

This module consumes source-identity priority-gate records and proposal-preview
payloads, then emits ranking evidence for a future proposal-priority consumer.
It does not reorder proposals in-place, import queue items, mutate source
identity trust, update models, or touch live mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Tuple

try:
    from source_identity_calibration_memory_trust_promotion_priority_gate import (
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_PRIORITY_GATE_METHOD_ID,
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_PRIORITY_GATE_STATUSES,
        build_source_identity_calibration_memory_trust_promotion_priority_gate,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.source_identity_calibration_memory_trust_promotion_priority_gate import (
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_PRIORITY_GATE_METHOD_ID,
        SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_PRIORITY_GATE_STATUSES,
        build_source_identity_calibration_memory_trust_promotion_priority_gate,
    )


SOURCE_IDENTITY_PROPOSAL_PRIORITY_POLICY_METHOD_ID = (
    "source_identity_proposal_priority_policy"
)
SOURCE_IDENTITY_PROPOSAL_PRIORITY_POLICY_STATUSES = {
    "source_identity_proposal_priority_policy_not_ready",
    "source_identity_proposal_priority_policy_replacement_required",
    "source_identity_proposal_priority_policy_records_available",
}
REQUIRED_PRIORITY_GATE_TYPE = (
    "source_identity_calibration_memory_trust_promotion_priority_gate"
)
REQUIRED_PRIORITY_GATE_FALSE_SAFETY_FLAGS = (
    "live_write",
    "mutates_console_state",
    "queries_console",
    "queue_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "source_identity_trust_store_mutation",
    "direct_downstream_priority_mutation",
    "downstream_priority_change_allowed_by_gate",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
)
REQUIRED_PRIORITY_GATE_FALSE_SUMMARY_FLAGS = (
    "queue_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "source_identity_trust_store_mutation",
    "direct_downstream_priority_mutation",
    "downstream_priority_change_allowed_by_gate",
    "live_write",
    "live_authority_granted",
    "full_autonomous_ready",
    "can_import_to_queue",
    "can_apply_live",
)
REQUIRED_PRIORITY_RECORD_FALSE_FLAGS = (
    "live_write",
    "mutates_console_state",
    "queries_console",
    "queue_mutation",
    "model_mutation",
    "runtime_memory_mutation",
    "source_identity_trust_store_mutation",
    "direct_downstream_priority_mutation",
    "downstream_priority_change_allowed_by_gate",
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


def _priority_record_id(record: Mapping[str, Any], index: int) -> str:
    explicit = _string(record.get("priority_gate_record_id"))
    if explicit:
        return explicit
    source_id = _string(record.get("source_candidate_id"))
    return f"{source_id if source_id else 'unknown_priority_gate_record'}:{index}"


def _duplicate_ids(
    records: Tuple[Mapping[str, Any], ...],
    field: str,
) -> Tuple[str, ...]:
    values: Dict[str, list[str]] = {}
    for index, record in enumerate(records):
        value = _string(record.get(field))
        if not value:
            continue
        values.setdefault(value, []).append(_priority_record_id(record, index))
    return tuple(
        record_id
        for record_ids in values.values()
        if len(record_ids) > 1
        for record_id in record_ids
    )


def _priority_gate_envelope_violations(
    priority_gate: Mapping[str, Any],
) -> Tuple[str, ...]:
    violations = []
    if priority_gate.get("type") != REQUIRED_PRIORITY_GATE_TYPE:
        violations.append("source_identity_proposal_priority_policy_gate_type_mismatch")
    generated_at = priority_gate.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("source_identity_proposal_priority_policy_gate_generated_at_invalid")
    if (
        priority_gate.get("method_id")
        != SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_PRIORITY_GATE_METHOD_ID
    ):
        violations.append("source_identity_proposal_priority_policy_gate_method_id_mismatch")
    if (
        priority_gate.get("status")
        not in SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_PRIORITY_GATE_STATUSES
    ):
        violations.append("source_identity_proposal_priority_policy_gate_status_unknown")
    safety = _mapping(priority_gate.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("source_identity_proposal_priority_policy_gate_read_only_not_true")
    if safety.get("priority_gate_only") is not True:
        violations.append("source_identity_proposal_priority_policy_gate_only_not_true")
    for flag in REQUIRED_PRIORITY_GATE_FALSE_SAFETY_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"source_identity_proposal_priority_policy_gate_{flag}_not_false")
    return tuple(violations)


def _priority_gate_status_violations(
    priority_gate: Mapping[str, Any],
    *,
    priority_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    status = priority_gate.get("status")
    if status == "memory_trust_promotion_priority_gate_replacement_required":
        return ("source_identity_proposal_priority_policy_gate_replacement_required",)
    if (
        status == "verified_memory_trust_promotion_priority_records_available"
        and not priority_records
    ):
        return ("source_identity_proposal_priority_policy_gate_available_without_records",)
    if status != "verified_memory_trust_promotion_priority_records_available" and priority_records:
        return ("source_identity_proposal_priority_policy_gate_records_without_available_status",)
    return ()


def _priority_gate_summary_violations(
    summary: Mapping[str, Any],
    *,
    priority_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    eligible_records = tuple(
        record for record in priority_records if record.get("downstream_priority_eligible") is True
    )
    violations = []
    expected_counts = {
        "priority_gate_record_count": len(priority_records),
        "eligible_record_count": len(eligible_records),
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(f"source_identity_proposal_priority_policy_gate_summary_{key}_mismatch")
    if summary.get("downstream_priority_eligible") is not bool(eligible_records):
        violations.append(
            "source_identity_proposal_priority_policy_gate_summary_eligibility_mismatch"
        )
    if summary.get("policy_gate_only") is not True:
        violations.append("source_identity_proposal_priority_policy_gate_summary_policy_only_not_true")
    for flag in REQUIRED_PRIORITY_GATE_FALSE_SUMMARY_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"source_identity_proposal_priority_policy_gate_summary_{flag}_not_false"
            )
    expected_source_ids = tuple(
        record["source_candidate_id"]
        for record in eligible_records
        if _string(record.get("source_candidate_id"))
    )
    if _string_list(summary.get("eligible_source_candidate_ids")) != expected_source_ids:
        violations.append(
            "source_identity_proposal_priority_policy_gate_summary_eligible_source_ids_mismatch"
        )
    return tuple(violations)


def _priority_record_violations(record: Mapping[str, Any]) -> Tuple[str, ...]:
    violations = []
    required_fields = (
        "priority_gate_record_id",
        "source_candidate_id",
        "source_type",
        "target_trust_state",
        "source_identity_downstream_gate_status",
    )
    for field in required_fields:
        if not _string(record.get(field)):
            violations.append(f"source_identity_proposal_priority_policy_gate_record_{field}_missing")
    if record.get("rollback_verified") is not True:
        violations.append("source_identity_proposal_priority_policy_gate_record_rollback_verified_not_true")
    if record.get("downstream_priority_eligible") is not True:
        violations.append("source_identity_proposal_priority_policy_gate_record_eligible_not_true")
    if record.get("eligible_for_downstream_priority_policy") is not True:
        violations.append("source_identity_proposal_priority_policy_gate_record_policy_eligible_not_true")
    if record.get("policy_gate_only") is not True:
        violations.append("source_identity_proposal_priority_policy_gate_record_policy_only_not_true")
    if record.get("proposal_priority_consumer_required") is not True:
        violations.append(
            "source_identity_proposal_priority_policy_gate_record_consumer_required_not_true"
        )
    if record.get("read_only") is not True:
        violations.append("source_identity_proposal_priority_policy_gate_record_read_only_not_true")
    for flag in REQUIRED_PRIORITY_RECORD_FALSE_FLAGS:
        if record.get(flag) is not False:
            violations.append(
                f"source_identity_proposal_priority_policy_gate_record_{flag}_not_false"
            )
    return tuple(violations)


def _invalid_priority_record_ids(
    records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    return tuple(
        _priority_record_id(record, index)
        for index, record in enumerate(records)
        if _priority_record_violations(record)
    )


def _proposal_id(proposal: Mapping[str, Any], index: int) -> str:
    for key in ("id", "proposal_id", "action_id"):
        value = _string(proposal.get(key))
        if value:
            return value
    return f"proposal:{index}"


def _normalize_source_token(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text if text else ""


def _proposal_source_tokens(proposal: Mapping[str, Any]) -> Tuple[str, ...]:
    tokens = []
    nested_payloads = (
        proposal,
        _mapping(proposal.get("proposed_action")),
        _mapping(proposal.get("evidence")),
        _mapping(proposal.get("raw")),
    )
    for payload in nested_payloads:
        for key in (
            "source_candidate_id",
            "source_id",
            "target_source_id",
            "channel_source_id",
            "target",
            "channel_id",
        ):
            token = _normalize_source_token(payload.get(key))
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


def _base_priority_score(proposal: Mapping[str, Any]) -> float:
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


def _matching_priority_record(
    proposal: Mapping[str, Any],
    eligible_by_source: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    for token in _proposal_source_tokens(proposal):
        record = eligible_by_source.get(token)
        if record:
            return record
    return {}


def _policy_records(
    proposals: Tuple[Mapping[str, Any], ...],
    eligible_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[Dict[str, Any], ...]:
    eligible_by_source = {
        record["source_candidate_id"]: record
        for record in eligible_records
        if _string(record.get("source_candidate_id"))
    }
    records = []
    for index, proposal in enumerate(proposals):
        gate_record = _matching_priority_record(proposal, eligible_by_source)
        if not gate_record:
            continue
        proposal_id = _proposal_id(proposal, index)
        base_score = _base_priority_score(proposal)
        source_identity_adjustment = 0.22
        adjusted_score = round(_clip(base_score + source_identity_adjustment, 0.0, 1.0), 3)
        records.append(
            {
                "policy_record_id": f"source_identity_proposal_priority_policy:{proposal_id}",
                "proposal_id": proposal_id,
                "proposal_method_id": proposal.get("method_id") or proposal.get("source"),
                "proposal_kind": proposal.get("kind"),
                "proposal_target": proposal.get("target"),
                "proposal_original_index": index,
                "source_candidate_id": gate_record.get("source_candidate_id"),
                "source_type": gate_record.get("source_type"),
                "priority_gate_record_id": gate_record.get("priority_gate_record_id"),
                "source_identity_downstream_gate_status": gate_record.get(
                    "source_identity_downstream_gate_status"
                ),
                "base_priority_score": base_score,
                "source_identity_priority_adjustment": source_identity_adjustment,
                "adjusted_priority_score": adjusted_score,
                "recommended_rank_bucket": "trusted_source_identity",
                "downstream_priority_eligible": True,
                "policy_consumer_required_before_ordering": True,
                "read_only": True,
                "policy_evidence_only": True,
                "proposal_order_mutation": False,
                "mutates_console_state": False,
                "queries_console": False,
                "queue_mutation": False,
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
    return tuple(
        sorted(
            records,
            key=lambda item: (
                -float(item["adjusted_priority_score"]),
                int(item["proposal_original_index"]),
            ),
        )
    )


def _unmatched_eligible_source_ids(
    records: Tuple[Mapping[str, Any], ...],
    policy_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    matched = {
        _string(record.get("source_candidate_id"))
        for record in policy_records
        if _string(record.get("source_candidate_id"))
    }
    return tuple(
        record["source_candidate_id"]
        for record in records
        if _string(record.get("source_candidate_id"))
        and record["source_candidate_id"] not in matched
    )


def build_source_identity_proposal_priority_policy(
    *,
    proposals: Iterable[Mapping[str, Any]] | None = None,
    priority_gate: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build read-only ranking evidence for source-identity-aware proposals."""

    gate = (
        _mapping(priority_gate)
        or build_source_identity_calibration_memory_trust_promotion_priority_gate()
    )
    summary = _mapping(gate.get("summary"))
    priority_records = _record_list(gate.get("priority_records"))
    eligible_records = tuple(
        record
        for record in priority_records
        if record.get("downstream_priority_eligible") is True
    )
    proposal_records = _proposal_list(proposals)
    invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_string_list(summary.get("invalid_record_ids")),
                *_priority_gate_envelope_violations(gate),
                *_priority_gate_status_violations(
                    gate,
                    priority_records=priority_records,
                ),
                *_priority_gate_summary_violations(
                    summary,
                    priority_records=priority_records,
                ),
                *_invalid_priority_record_ids(eligible_records),
                *_duplicate_ids(eligible_records, "priority_gate_record_id"),
                *_duplicate_ids(eligible_records, "source_candidate_id"),
            )
        )
    )
    policy_records = (
        ()
        if invalid_record_ids
        else _policy_records(proposal_records, eligible_records)
    )
    unmatched_source_ids = (
        ()
        if invalid_record_ids
        else _unmatched_eligible_source_ids(eligible_records, policy_records)
    )

    if invalid_record_ids:
        status = "source_identity_proposal_priority_policy_replacement_required"
        blocked_by = ("operator_must_replace_invalid_proposal_priority_policy_evidence",)
    elif not eligible_records:
        status = "source_identity_proposal_priority_policy_not_ready"
        blocked_by = ("eligible_source_identity_priority_gate_records_required",)
    elif not policy_records:
        status = "source_identity_proposal_priority_policy_not_ready"
        blocked_by = ("matching_preview_proposals_required_for_priority_policy",)
    else:
        status = "source_identity_proposal_priority_policy_records_available"
        blocked_by = ("proposal_priority_policy_records_are_not_queue_or_live_authority",)

    return {
        "type": "source_identity_proposal_priority_policy",
        "generated_at": _utc_now(),
        "method_id": SOURCE_IDENTITY_PROPOSAL_PRIORITY_POLICY_METHOD_ID,
        "source_method_id": (
            SOURCE_IDENTITY_CALIBRATION_MEMORY_TRUST_PROMOTION_PRIORITY_GATE_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "operator_required": True,
            "policy_evidence_only": True,
            "proposal_order_mutation": False,
            "queue_mutation": False,
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
            "priority_gate_status": gate.get("status") or "missing",
            "input_proposal_count": len(proposal_records),
            "priority_gate_record_count": len(priority_records),
            "eligible_priority_gate_record_count": len(eligible_records),
            "policy_record_count": len(policy_records),
            "unmatched_eligible_source_count": len(unmatched_source_ids),
            "invalid_record_count": len(invalid_record_ids),
            "policy_evidence_only": True,
            "mutates_console_state": False,
            "queries_console": False,
            "proposal_order_mutation": False,
            "queue_mutation": False,
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
            "ranked_proposal_ids": [record["proposal_id"] for record in policy_records],
            "eligible_source_candidate_ids": [
                record["source_candidate_id"] for record in eligible_records
            ],
            "unmatched_eligible_source_candidate_ids": list(unmatched_source_ids),
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Operator replaces invalid proposal-priority policy evidence."
                if invalid_record_ids
                else (
                    "Eligible source-identity priority-gate records are required before ranking."
                    if not eligible_records
                    else (
                        "Add matching preview proposals before ranking can be projected."
                        if not policy_records
                        else "Feed ranked policy records to a separate proposal-priority consumer before mutating order or queues."
                    )
                )
            ),
        },
        "policy_records": list(policy_records),
    }
