"""Proposal-only adapters for processing methods.

Adapters in this module translate existing analysis artifacts into the Product
Layer preview contract. They must remain write-free: no mixer clients, no OSC
clients, and no live executor imports.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

try:
    from processing_registry import (
        LIVE_ALLOWED_WRITE_KINDS,
        get_pipeline_profile,
        get_processing_method,
        iter_processing_methods,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.processing_registry import (
        LIVE_ALLOWED_WRITE_KINDS,
        get_pipeline_profile,
        get_processing_method,
        iter_processing_methods,
    )


MAIN_PIPELINE_ID = "main_live_assist"
PROPOSAL_ONLY_BLOCKED_DOMAINS = {"eq", "dynamics", "gate", "pan", "phase", "fx"}
BLOCKED_METHOD_STATUSES = {"quarantined", "placeholder", "deprecated"}
ALLOWED_REQUESTED_CHANGE_TYPES = set(LIVE_ALLOWED_WRITE_KINDS)


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _risk_from_severity(severity: Any) -> str:
    normalized = str(severity or "").strip().lower()
    if normalized in {"critical", "high"}:
        return "high"
    if normalized == "low":
        return "low"
    return "medium"


def _normalize_requested_change(change: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(change, dict):
        return None
    value_type = str(change.get("value_type") or "").strip().lower()
    channel = _coerce_int(change.get("channel"))
    value = _coerce_float(change.get("value"))
    if value_type not in ALLOWED_REQUESTED_CHANGE_TYPES or channel is None or value is None:
        return None

    normalized = dict(change)
    normalized["value_type"] = value_type
    normalized["channel"] = channel
    normalized["value"] = round(value, 1)
    return normalized


def _normalize_requested_changes(changes: Any) -> List[Dict[str, Any]]:
    if not isinstance(changes, list):
        return []
    normalized = []
    for change in changes:
        item = _normalize_requested_change(change)
        if item:
            normalized.append(item)
    return normalized


def _method_contract(method_id: str) -> Dict[str, Any]:
    method = get_processing_method(method_id)
    if method is None:
        return {
            "method_id": method_id,
            "domain": "evaluation",
            "status": "placeholder",
            "method_label": method_id,
        }
    return {
        "method_id": method.id,
        "domain": method.domain,
        "status": method.status,
        "method_label": method.label,
    }


def _proposal_evidence(proposal: Dict[str, Any], requested_change: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = proposal.get("raw") if isinstance(proposal.get("raw"), dict) else {}
    return {
        "source": proposal.get("source"),
        "severity": proposal.get("severity"),
        "target": proposal.get("target"),
        "current_value": (requested_change or {}).get("current_value"),
        "proposed_value": (requested_change or {}).get("value"),
        "rollback_value": (requested_change or {}).get("rollback_value")
        or (requested_change or {}).get("current_value"),
        "readback_required": bool((requested_change or {}).get("readback_required", True))
        if requested_change
        else False,
        "raw": raw,
    }


def _proposal_adapter_output(
    proposal: Dict[str, Any],
    *,
    adapter_id: str,
    method_id: str,
    pipeline_id: str = MAIN_PIPELINE_ID,
) -> Dict[str, Any]:
    contract = _method_contract(method_id)
    requested_change = _normalize_requested_change(proposal.get("requested_change"))
    requested_changes = _normalize_requested_changes(proposal.get("requested_changes"))
    has_applyable_change = bool(requested_change or requested_changes)
    blockers: List[str] = []
    if not has_applyable_change:
        blockers.append("missing_applyable_fader_gain_requested_change")

    output = {
        "id": f"{pipeline_id}:{adapter_id}:{proposal.get('id') or proposal.get('title')}",
        "adapter_id": adapter_id,
        "pipeline_id": pipeline_id,
        **contract,
        "title": str(proposal.get("title") or "Processing proposal"),
        "reason": str(proposal.get("reason") or "Analyzer produced a proposal-only recommendation."),
        "confidence": proposal.get("confidence"),
        "risk": "blocked" if blockers else _risk_from_severity(proposal.get("severity")),
        "target": proposal.get("target") or proposal.get("channel") or proposal.get("channels"),
        "queue_action": "preview_only_not_queued",
        "can_import_to_queue": has_applyable_change,
        "blockers": blockers,
        "evidence": _proposal_evidence(proposal, requested_change),
        "source_proposal_id": proposal.get("id"),
    }
    if requested_change:
        output["requested_change"] = requested_change
    if requested_changes:
        output["requested_changes"] = requested_changes
    return output


def adapt_operator_analysis_gain_fader_proposals(
    proposals: Iterable[Dict[str, Any]],
    *,
    pipeline_id: str = MAIN_PIPELINE_ID,
) -> List[Dict[str, Any]]:
    """Adapt AutoSoundcheck/operator analysis proposals for preview only."""

    return [
        _proposal_adapter_output(
            proposal,
            adapter_id="operator_analysis_gain_fader",
            method_id="auto_soundcheck_engine",
            pipeline_id=pipeline_id,
        )
        for proposal in proposals
        if isinstance(proposal, dict)
    ]


def adapt_safe_gain_suggestions(
    proposals: Iterable[Dict[str, Any]],
    *,
    pipeline_id: str = MAIN_PIPELINE_ID,
) -> List[Dict[str, Any]]:
    """Adapt SafeGain proposal candidates without importing them into the queue."""

    return [
        _proposal_adapter_output(
            proposal,
            adapter_id="safe_gain_suggestions",
            method_id="lufs_gain_staging",
            pipeline_id=pipeline_id,
        )
        for proposal in proposals
        if isinstance(proposal, dict)
    ]


def build_registry_blockers(*, pipeline_id: str = MAIN_PIPELINE_ID) -> List[Dict[str, Any]]:
    """Return methods visible in preview but blocked from the main live executor."""

    profile = get_pipeline_profile(pipeline_id)
    executor_ids = set(profile.executor_method_ids if profile else ())
    main_method_ids = set(profile.method_ids if profile else ())
    blockers: List[Dict[str, Any]] = []

    for method in iter_processing_methods():
        if method.id in executor_ids:
            continue
        if method.id in main_method_ids and method.domain == "gain_fader":
            continue

        blocked_by_status = method.status in BLOCKED_METHOD_STATUSES
        blocked_by_domain = method.domain in PROPOSAL_ONLY_BLOCKED_DOMAINS
        if not blocked_by_status and not blocked_by_domain:
            continue

        reason = method.blocked_reason
        if not reason and blocked_by_status:
            reason = f"{method.status}_method_not_live_executor"
        if not reason:
            reason = f"{method.domain}_is_proposal_only_in_main_live_assist"

        blockers.append(
            {
                "id": f"{pipeline_id}:registry_blocker:{method.id}",
                "adapter_id": "registry_blockers",
                "pipeline_id": pipeline_id,
                "method_id": method.id,
                "method_label": method.label,
                "domain": method.domain,
                "status": method.status,
                "title": f"{method.label}: live apply unavailable",
                "reason": method.summary,
                "confidence": None,
                "risk": "blocked" if blocked_by_status else "medium",
                "queue_action": "not_queueable",
                "can_import_to_queue": False,
                "blockers": [reason],
                "evidence": {
                    "blocked_reason": reason,
                    "promotion_next": method.promotion_next,
                    "live_executor": False,
                    "requested_change_allowed": False,
                },
            }
        )

    return blockers
