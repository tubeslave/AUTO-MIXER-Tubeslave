"""Read-only preflight plan for sequence-policy imported queue proposals.

This module consumes pending operator-queue actions that were imported from the
sequence-policy queue importer. It emits an ordered batch-preflight request
template for the existing preflight handler. It does not run preflight, reserve
idempotency keys, mutate queue state, execute sequences, touch policy stores, or
call live mixer transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Tuple

try:
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORTER_METHOD_ID,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer import (
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORTER_METHOD_ID,
    )


OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_PLAN_METHOD_ID = (
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan"
)
OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_PLAN_STATUSES = {
    "sequence_memory_trust_promotion_queue_import_preflight_plan_not_ready",
    "sequence_memory_trust_promotion_queue_import_preflight_plan_replacement_required",
    "sequence_memory_trust_promotion_queue_import_preflight_plan_available",
}
DEFAULT_IDEMPOTENCY_KEY_PREFIX = "sequence-policy-queue-import"
ALLOWED_RANK_SOURCES = {"sequence_policy_consumer", "original_order"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _action_list(value: Any) -> Tuple[Mapping[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _string(value: Any) -> str:
    return value if isinstance(value, str) and value else ""


def _summary_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _duplicate_values(values: Tuple[str, ...]) -> Tuple[str, ...]:
    counts: Dict[str, int] = {}
    for value in values:
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return tuple(value for value, count in counts.items() if count > 1)


def _duplicate_markers(values: Tuple[str, ...], label: str) -> Tuple[str, ...]:
    return tuple(f"{label}:{value}" for value in _duplicate_values(values))


def _action_id(action: Mapping[str, Any], index: int) -> str:
    value = action.get("id") or action.get("proposal_id") or action.get("action_id")
    return str(value) if value is not None and str(value).strip() else f"proposal:{index}"


def _metadata(action: Mapping[str, Any]) -> Mapping[str, Any]:
    direct = action.get("sequence_policy_queue_import")
    if isinstance(direct, Mapping):
        return direct
    raw = action.get("raw")
    if isinstance(raw, Mapping):
        nested = raw.get("sequence_policy_queue_import")
        if isinstance(nested, Mapping):
            return nested
    return {}


def _source(action: Mapping[str, Any]) -> str:
    direct = _string(action.get("source"))
    if direct:
        return direct
    raw = action.get("raw")
    if isinstance(raw, Mapping):
        return _string(raw.get("source"))
    return ""


def _is_sequence_policy_import(action: Mapping[str, Any]) -> bool:
    return bool(_metadata(action)) or (
        _source(action)
        == OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORTER_METHOD_ID
    )


def _metadata_violations(
    action: Mapping[str, Any],
    *,
    index: int,
) -> Tuple[str, ...]:
    metadata = _metadata(action)
    proposal_id = _action_id(action, index)
    violations = []
    if not metadata:
        violations.append(f"{proposal_id}:sequence_policy_queue_import_metadata_missing")
        return tuple(violations)
    if _summary_int(metadata.get("planned_queue_index")) is None:
        violations.append(f"{proposal_id}:planned_queue_index_invalid")
    if metadata.get("rank_source") not in ALLOWED_RANK_SOURCES:
        violations.append(f"{proposal_id}:rank_source_unknown")
    if metadata.get("operator_ack_required_before_queue_import") is not True:
        violations.append(f"{proposal_id}:queue_import_ack_required_not_true")
    if metadata.get("operator_ack_required_before_sequence_execution") is not True:
        violations.append(f"{proposal_id}:sequence_execution_ack_required_not_true")
    if metadata.get("sequence_consumer_required_before_execution") is not True:
        violations.append(f"{proposal_id}:sequence_consumer_required_not_true")
    if metadata.get("live_write") is not False:
        violations.append(f"{proposal_id}:live_write_not_false")
    if metadata.get("can_apply_live") is not False:
        violations.append(f"{proposal_id}:can_apply_live_not_false")
    if _source(action) not in {
        "",
        OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORTER_METHOD_ID,
    }:
        violations.append(f"{proposal_id}:source_mismatch")
    return tuple(violations)


def _ordered_imported_actions(
    actions: Tuple[Mapping[str, Any], ...],
) -> Tuple[Mapping[str, Any], ...]:
    imported = tuple(action for action in actions if _is_sequence_policy_import(action))
    return tuple(
        sorted(
            imported,
            key=lambda action: int(_metadata(action).get("planned_queue_index") or 0),
        )
    )


def build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan(
    *,
    pending_actions: Iterable[Mapping[str, Any]] | None = None,
    idempotency_key_prefix: str = DEFAULT_IDEMPOTENCY_KEY_PREFIX,
    limit: int = 20,
) -> Dict[str, Any]:
    """Build a read-only batch-preflight request from sequence-policy imports."""

    actions = _action_list(pending_actions)
    imported_actions = _ordered_imported_actions(actions)
    try:
        bounded_limit = int(limit or 20)
    except (TypeError, ValueError):
        bounded_limit = 20
    bounded_limit = max(1, min(bounded_limit, 50))
    invalid_record_ids = tuple(
        dict.fromkeys(
            (
                *(
                    violation
                    for index, action in enumerate(imported_actions)
                    for violation in _metadata_violations(action, index=index)
                ),
                *_duplicate_markers(
                    tuple(
                        _action_id(action, index)
                        for index, action in enumerate(imported_actions)
                    ),
                    "duplicate_proposal_id",
                ),
                *_duplicate_markers(
                    tuple(
                        str(_metadata(action).get("planned_queue_index"))
                        for action in imported_actions
                    ),
                    "duplicate_planned_queue_index",
                ),
            )
        )
    )
    candidate_actions = tuple(
        action
        for action in imported_actions
        if action.get("can_apply") is True and action.get("status", "pending") == "pending"
    )
    analysis_only_actions = tuple(
        action for action in imported_actions if action.get("can_apply") is not True
    )
    non_pending_actions = tuple(
        action
        for action in imported_actions
        if action.get("can_apply") is True and action.get("status", "pending") != "pending"
    )
    selected_actions = candidate_actions[:bounded_limit] if not invalid_record_ids else ()
    selected_ids = tuple(
        _action_id(action, index) for index, action in enumerate(selected_actions)
    )
    all_imported_ids = tuple(
        _action_id(action, index) for index, action in enumerate(imported_actions)
    )
    analysis_only_ids = tuple(
        _action_id(action, index) for index, action in enumerate(analysis_only_actions)
    )
    non_pending_ids = tuple(
        _action_id(action, index) for index, action in enumerate(non_pending_actions)
    )
    prefix = str(idempotency_key_prefix or DEFAULT_IDEMPOTENCY_KEY_PREFIX).strip()
    if not prefix:
        prefix = DEFAULT_IDEMPOTENCY_KEY_PREFIX

    preflight_specs = tuple(
        {
            "preflight_request_spec_id": (
                f"{OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_PLAN_METHOD_ID}:{proposal_id}"
            ),
            "proposal_id": proposal_id,
            "planned_preflight_index": index,
            "planned_queue_index": _metadata(action).get("planned_queue_index"),
            "rank_source": _metadata(action).get("rank_source"),
            "idempotency_key": f"{prefix}:{proposal_id}",
            "operator_approval_required_for_preflight_ready": True,
            "sequence_consumer_required_before_execution": True,
            "batch_preflight_only": True,
            "read_only": True,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "sequence_plan_created": False,
            "sequence_execution_allowed_by_preflight_plan": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_apply_live": False,
        }
        for index, (proposal_id, action) in enumerate(zip(selected_ids, selected_actions))
    )

    if invalid_record_ids:
        status = (
            "sequence_memory_trust_promotion_queue_import_preflight_plan_replacement_required"
        )
        blocked_by = ("operator_must_replace_invalid_sequence_policy_preflight_plan_evidence",)
    elif not imported_actions:
        status = "sequence_memory_trust_promotion_queue_import_preflight_plan_not_ready"
        blocked_by = ("sequence_policy_imported_pending_proposals_required",)
    else:
        status = "sequence_memory_trust_promotion_queue_import_preflight_plan_available"
        blocked_by = ("operator_approval_required_before_batch_preflight",)

    return {
        "type": (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan"
        ),
        "generated_at": _utc_now(),
        "method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORT_PREFLIGHT_PLAN_METHOD_ID
        ),
        "source_method_id": (
            OPERATOR_PROPOSAL_APPLY_SEQUENCE_POLICY_MEMORY_TRUST_PROMOTION_QUEUE_IMPORTER_METHOD_ID
        ),
        "status": status,
        "safety": {
            "read_only": True,
            "batch_preflight_only": True,
            "batch_preflight_invoked": False,
            "operator_approval_required_for_preflight_ready": True,
            "sequence_consumer_required_before_execution": True,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "sequence_plan_created": False,
            "sequence_execution_allowed_by_preflight_plan": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_apply_live": False,
        },
        "summary": {
            "input_pending_action_count": len(actions),
            "sequence_policy_imported_pending_count": len(imported_actions),
            "preflight_candidate_count": len(selected_actions),
            "analysis_only_count": len(analysis_only_actions),
            "non_pending_count": len(non_pending_actions),
            "invalid_record_count": len(invalid_record_ids),
            "limit": bounded_limit,
            "idempotency_key_prefix": prefix,
            "operator_approval_required_for_preflight_ready": True,
            "sequence_consumer_required_before_execution": True,
            "batch_preflight_invoked": False,
            "batch_preflight_request_available": bool(selected_actions) and not invalid_record_ids,
            "sequence_plan_created": False,
            "sequence_execution_allowed_by_preflight_plan": False,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "proposal_status_mutation": False,
            "idempotency_reserved": False,
            "manual_console_write_called": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "can_apply_live": False,
            "imported_proposal_ids": list(all_imported_ids),
            "preflight_candidate_proposal_ids": list(selected_ids),
            "analysis_only_proposal_ids": list(analysis_only_ids),
            "non_pending_proposal_ids": list(non_pending_ids),
            "invalid_record_ids": list(invalid_record_ids),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Operator replaces invalid sequence-policy preflight-plan evidence."
                if invalid_record_ids
                else (
                    "Import sequence-policy queue records before batch preflight planning."
                    if not imported_actions
                    else "Operator may run the existing batch preflight handler with explicit approval; this plan does not run it."
                )
            ),
        },
        "batch_preflight_request": {
            "handler": "preflight_operator_proposal_apply_batch",
            "proposal_ids": list(selected_ids),
            "idempotency_key_prefix": prefix,
            "limit": bounded_limit,
            "approved": False,
            "operator_must_set_approved_true": True,
            "read_only_template": True,
            "live_write": False,
            "apply_attempted": False,
            "queue_mutation": False,
            "idempotency_reserved": False,
        },
        "preflight_request_specs": list(preflight_specs),
    }
