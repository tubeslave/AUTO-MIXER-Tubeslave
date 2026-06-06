"""Read-only ordered proposal view from source-identity priority policy evidence.

This module consumes the source-identity proposal-priority policy output and
emits a separate recommended-order preview. It never mutates the original
proposal list, operator queue, source-identity trust store, runtime memory,
models, or live mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Tuple

try:
    from source_identity_proposal_priority_policy import (
        SOURCE_IDENTITY_PROPOSAL_PRIORITY_POLICY_METHOD_ID,
        SOURCE_IDENTITY_PROPOSAL_PRIORITY_POLICY_STATUSES,
        build_source_identity_proposal_priority_policy,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.source_identity_proposal_priority_policy import (
        SOURCE_IDENTITY_PROPOSAL_PRIORITY_POLICY_METHOD_ID,
        SOURCE_IDENTITY_PROPOSAL_PRIORITY_POLICY_STATUSES,
        build_source_identity_proposal_priority_policy,
    )


SOURCE_IDENTITY_PROPOSAL_ORDERING_PREVIEW_METHOD_ID = (
    "source_identity_proposal_ordering_preview"
)
SOURCE_IDENTITY_PROPOSAL_ORDERING_PREVIEW_STATUSES = {
    "source_identity_proposal_ordering_preview_not_ready",
    "source_identity_proposal_ordering_preview_replacement_required",
    "source_identity_proposal_ordering_preview_available",
}
REQUIRED_PRIORITY_POLICY_TYPE = "source_identity_proposal_priority_policy"
REQUIRED_PRIORITY_POLICY_FALSE_FLAGS = (
    "live_write",
    "mutates_console_state",
    "queries_console",
    "proposal_order_mutation",
    "queue_mutation",
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
REQUIRED_POLICY_RECORD_FALSE_FLAGS = REQUIRED_PRIORITY_POLICY_FALSE_FLAGS


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


def _proposal_id(proposal: Mapping[str, Any], index: int) -> str:
    for key in ("id", "proposal_id", "action_id"):
        value = _string(proposal.get(key))
        if value:
            return value
    return f"proposal:{index}"


def _duplicate_values(values: Tuple[str, ...]) -> Tuple[str, ...]:
    counts: Dict[str, int] = {}
    for value in values:
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return tuple(value for value, count in counts.items() if count > 1)


def _policy_record_id(record: Mapping[str, Any], index: int) -> str:
    explicit = _string(record.get("policy_record_id"))
    if explicit:
        return explicit
    proposal_id = _string(record.get("proposal_id"))
    return f"{proposal_id if proposal_id else 'unknown_policy_record'}:{index}"


def _priority_policy_envelope_violations(
    priority_policy: Mapping[str, Any],
) -> Tuple[str, ...]:
    violations = []
    if priority_policy.get("type") != REQUIRED_PRIORITY_POLICY_TYPE:
        violations.append("source_identity_proposal_ordering_preview_policy_type_mismatch")
    generated_at = priority_policy.get("generated_at")
    if not isinstance(generated_at, str) or not _is_utc_timestamp(generated_at):
        violations.append("source_identity_proposal_ordering_preview_policy_generated_at_invalid")
    if (
        priority_policy.get("method_id")
        != SOURCE_IDENTITY_PROPOSAL_PRIORITY_POLICY_METHOD_ID
    ):
        violations.append("source_identity_proposal_ordering_preview_policy_method_id_mismatch")
    if priority_policy.get("status") not in SOURCE_IDENTITY_PROPOSAL_PRIORITY_POLICY_STATUSES:
        violations.append("source_identity_proposal_ordering_preview_policy_status_unknown")
    safety = _mapping(priority_policy.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("source_identity_proposal_ordering_preview_policy_read_only_not_true")
    if safety.get("policy_evidence_only") is not True:
        violations.append("source_identity_proposal_ordering_preview_policy_evidence_only_not_true")
    for flag in REQUIRED_PRIORITY_POLICY_FALSE_FLAGS:
        if safety.get(flag) is not False:
            violations.append(f"source_identity_proposal_ordering_preview_policy_{flag}_not_false")
    return tuple(violations)


def _priority_policy_status_violations(
    priority_policy: Mapping[str, Any],
    *,
    policy_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    status = priority_policy.get("status")
    if status == "source_identity_proposal_priority_policy_replacement_required":
        return ("source_identity_proposal_ordering_preview_policy_replacement_required",)
    if status == "source_identity_proposal_priority_policy_records_available" and not policy_records:
        return ("source_identity_proposal_ordering_preview_policy_available_without_records",)
    if status != "source_identity_proposal_priority_policy_records_available" and policy_records:
        return ("source_identity_proposal_ordering_preview_policy_records_without_available_status",)
    return ()


def _priority_policy_summary_violations(
    summary: Mapping[str, Any],
    *,
    proposals: Tuple[Mapping[str, Any], ...],
    policy_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    expected_counts = {
        "input_proposal_count": len(proposals),
        "policy_record_count": len(policy_records),
        "invalid_record_count": 0,
    }
    for key, expected in expected_counts.items():
        if _summary_int(summary.get(key)) != expected:
            violations.append(
                f"source_identity_proposal_ordering_preview_policy_summary_{key}_mismatch"
            )
    if summary.get("policy_evidence_only") is not True:
        violations.append("source_identity_proposal_ordering_preview_policy_summary_evidence_only_not_true")
    for flag in REQUIRED_PRIORITY_POLICY_FALSE_FLAGS:
        if summary.get(flag) is not False:
            violations.append(
                f"source_identity_proposal_ordering_preview_policy_summary_{flag}_not_false"
            )
    expected_ranked_ids = tuple(
        record["proposal_id"]
        for record in policy_records
        if _string(record.get("proposal_id"))
    )
    if _string_list(summary.get("ranked_proposal_ids")) != expected_ranked_ids:
        violations.append(
            "source_identity_proposal_ordering_preview_policy_summary_ranked_ids_mismatch"
        )
    return tuple(violations)


def _policy_record_violations(
    record: Mapping[str, Any],
    *,
    proposals: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    violations = []
    required_fields = (
        "policy_record_id",
        "proposal_id",
        "source_candidate_id",
        "priority_gate_record_id",
        "recommended_rank_bucket",
    )
    for field in required_fields:
        if not _string(record.get(field)):
            violations.append(f"source_identity_proposal_ordering_preview_record_{field}_missing")
    original_index = _summary_int(record.get("proposal_original_index"))
    if original_index is None or original_index >= len(proposals):
        violations.append("source_identity_proposal_ordering_preview_record_original_index_invalid")
    elif _proposal_id(proposals[original_index], original_index) != record.get("proposal_id"):
        violations.append("source_identity_proposal_ordering_preview_record_proposal_link_mismatch")
    for field in ("base_priority_score", "adjusted_priority_score"):
        score = _safe_float(record.get(field))
        if score is None or score < 0.0 or score > 1.0:
            violations.append(f"source_identity_proposal_ordering_preview_record_{field}_invalid")
    if record.get("downstream_priority_eligible") is not True:
        violations.append("source_identity_proposal_ordering_preview_record_eligible_not_true")
    if record.get("policy_consumer_required_before_ordering") is not True:
        violations.append("source_identity_proposal_ordering_preview_record_consumer_required_not_true")
    if record.get("read_only") is not True:
        violations.append("source_identity_proposal_ordering_preview_record_read_only_not_true")
    if record.get("policy_evidence_only") is not True:
        violations.append("source_identity_proposal_ordering_preview_record_evidence_only_not_true")
    for flag in REQUIRED_POLICY_RECORD_FALSE_FLAGS:
        if record.get(flag) is not False:
            violations.append(f"source_identity_proposal_ordering_preview_record_{flag}_not_false")
    return tuple(violations)


def _invalid_policy_record_ids(
    records: Tuple[Mapping[str, Any], ...],
    *,
    proposals: Tuple[Mapping[str, Any], ...],
) -> Tuple[str, ...]:
    return tuple(
        _policy_record_id(record, index)
        for index, record in enumerate(records)
        if _policy_record_violations(record, proposals=proposals)
    )


def _ordered_proposal_records(
    proposals: Tuple[Mapping[str, Any], ...],
    policy_records: Tuple[Mapping[str, Any], ...],
) -> Tuple[Dict[str, Any], ...]:
    policy_by_proposal_id = {
        record["proposal_id"]: record
        for record in policy_records
        if _string(record.get("proposal_id"))
    }
    ranked_records = tuple(
        sorted(
            policy_records,
            key=lambda item: (
                -float(item.get("adjusted_priority_score") or 0.0),
                int(item.get("proposal_original_index") or 0),
            ),
        )
    )
    rows = []
    consumed_proposal_ids = set()
    recommended_index = 0

    for policy_record in ranked_records:
        original_index = int(policy_record["proposal_original_index"])
        proposal = proposals[original_index]
        proposal_id = str(policy_record["proposal_id"])
        consumed_proposal_ids.add(proposal_id)
        rows.append(
            {
                "ordering_preview_record_id": (
                    f"{SOURCE_IDENTITY_PROPOSAL_ORDERING_PREVIEW_METHOD_ID}:{proposal_id}"
                ),
                "proposal_id": proposal_id,
                "proposal_method_id": (
                    policy_record.get("proposal_method_id")
                    or proposal.get("method_id")
                    or proposal.get("source")
                ),
                "proposal_kind": policy_record.get("proposal_kind") or proposal.get("kind"),
                "proposal_target": policy_record.get("proposal_target") or proposal.get("target"),
                "source_candidate_id": policy_record.get("source_candidate_id"),
                "original_index": original_index,
                "recommended_index": recommended_index,
                "base_priority_score": policy_record.get("base_priority_score"),
                "adjusted_priority_score": policy_record.get("adjusted_priority_score"),
                "rank_source": "source_identity_policy",
                "policy_record_id": policy_record.get("policy_record_id"),
                "recommended_rank_bucket": policy_record.get("recommended_rank_bucket"),
                "read_only": True,
                "ordering_preview_only": True,
                "policy_consumer_required_before_queue_mutation": True,
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
        recommended_index += 1

    for original_index, proposal in enumerate(proposals):
        proposal_id = _proposal_id(proposal, original_index)
        if proposal_id in consumed_proposal_ids or proposal_id in policy_by_proposal_id:
            continue
        rows.append(
            {
                "ordering_preview_record_id": (
                    f"{SOURCE_IDENTITY_PROPOSAL_ORDERING_PREVIEW_METHOD_ID}:{proposal_id}"
                ),
                "proposal_id": proposal_id,
                "proposal_method_id": proposal.get("method_id") or proposal.get("source"),
                "proposal_kind": proposal.get("kind"),
                "proposal_target": proposal.get("target"),
                "source_candidate_id": None,
                "original_index": original_index,
                "recommended_index": recommended_index,
                "base_priority_score": 0.0,
                "adjusted_priority_score": 0.0,
                "rank_source": "original_order",
                "policy_record_id": None,
                "recommended_rank_bucket": "unranked_original_order",
                "read_only": True,
                "ordering_preview_only": True,
                "policy_consumer_required_before_queue_mutation": True,
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
        recommended_index += 1

    return tuple(rows)


def build_source_identity_proposal_ordering_preview(
    *,
    proposals: Iterable[Mapping[str, Any]] | None = None,
    priority_policy: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a read-only ordered view from proposal-priority policy records."""

    proposal_records = _proposal_list(proposals)
    policy = (
        _mapping(priority_policy)
        or build_source_identity_proposal_priority_policy(proposals=proposal_records)
    )
    summary = _mapping(policy.get("summary"))
    policy_records = _record_list(policy.get("policy_records"))
    proposal_ids = tuple(
        _proposal_id(proposal, index)
        for index, proposal in enumerate(proposal_records)
    )
    invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *_string_list(summary.get("invalid_record_ids")),
                *_priority_policy_envelope_violations(policy),
                *_priority_policy_status_violations(policy, policy_records=policy_records),
                *_priority_policy_summary_violations(
                    summary,
                    proposals=proposal_records,
                    policy_records=policy_records,
                ),
                *_invalid_policy_record_ids(policy_records, proposals=proposal_records),
                *_duplicate_values(proposal_ids),
                *_duplicate_values(
                    tuple(
                        _string(record.get("policy_record_id"))
                        for record in policy_records
                    )
                ),
                *_duplicate_values(
                    tuple(
                        _string(record.get("proposal_id"))
                        for record in policy_records
                    )
                ),
            )
        )
    )
    ordered_proposals = (
        ()
        if invalid_record_ids or not policy_records
        else _ordered_proposal_records(proposal_records, policy_records)
    )
    policy_ranked_ids = tuple(
        record["proposal_id"]
        for record in policy_records
        if _string(record.get("proposal_id"))
    )
    ordered_ids = tuple(record["proposal_id"] for record in ordered_proposals)
    unranked_ids = tuple(
        record["proposal_id"]
        for record in ordered_proposals
        if record.get("rank_source") == "original_order"
    )

    if invalid_record_ids:
        status = "source_identity_proposal_ordering_preview_replacement_required"
        blocked_by = ("operator_must_replace_invalid_ordering_preview_policy_evidence",)
    elif not policy_records:
        status = "source_identity_proposal_ordering_preview_not_ready"
        blocked_by = ("valid_source_identity_proposal_priority_policy_records_required",)
    else:
        status = "source_identity_proposal_ordering_preview_available"
        blocked_by = ("ordering_preview_is_not_queue_ordering_or_live_authority",)

    return {
        "type": "source_identity_proposal_ordering_preview",
        "generated_at": _utc_now(),
        "method_id": SOURCE_IDENTITY_PROPOSAL_ORDERING_PREVIEW_METHOD_ID,
        "source_method_id": SOURCE_IDENTITY_PROPOSAL_PRIORITY_POLICY_METHOD_ID,
        "status": status,
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "operator_required": True,
            "ordering_preview_only": True,
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
            "priority_policy_status": policy.get("status") or "missing",
            "input_proposal_count": len(proposal_records),
            "policy_record_count": len(policy_records) if not invalid_record_ids else 0,
            "ordered_preview_count": len(ordered_proposals),
            "policy_ranked_proposal_count": len(policy_ranked_ids) if not invalid_record_ids else 0,
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
            "source_identity_trust_store_mutation": False,
            "direct_downstream_priority_mutation": False,
            "downstream_priority_change_allowed_by_policy": False,
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
                "Operator replaces invalid ordering-preview policy evidence."
                if invalid_record_ids
                else (
                    "Valid source-identity proposal-priority policy records are required before ordering preview."
                    if not policy_records
                    else "Operator may inspect this view; a separate queue ordering consumer is required before mutation."
                )
            ),
        },
        "ordered_proposals": list(ordered_proposals),
    }
