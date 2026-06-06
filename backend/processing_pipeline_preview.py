"""Read-only preview for the selected AUTO-MIXER processing pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from agentic_method_foundation import (
        build_agentic_recommendations,
        build_inter_channel_context,
        feature_snapshot_from_channel_inventory,
    )
    from operator_analysis import build_operator_analysis_report
    from operator_product_state import build_channel_inventory
    from operator_recommendation_bridge import build_safe_gain_operator_proposals
    from processing_proposal_adapters import (
        MAIN_PIPELINE_ID,
        adapt_agentic_foundation_gain_fader_recommendations,
        adapt_operator_analysis_gain_fader_proposals,
        adapt_safe_gain_suggestions,
        adapt_source_identity_replay_calibration_proposals,
        build_proposal_only_method_contracts,
        build_registry_blockers,
    )
    from processing_registry import (
        LIVE_ALLOWED_WRITE_KINDS,
        build_pipeline_runtime_summary,
        build_processing_registry_snapshot,
        get_pipeline_profile,
        get_processing_method,
    )
    from source_identity_calibration_acceptance_ledger import (
        build_source_identity_calibration_acceptance_ledger,
    )
    from source_identity_calibration_persistence_plan import (
        build_source_identity_calibration_persistence_plan,
    )
    from source_identity_calibration_memory_draft import (
        build_source_identity_calibration_memory_draft,
    )
    from source_identity_calibration_memory_writer import (
        build_source_identity_calibration_memory_writer,
    )
    from source_identity_calibration_memory_index import (
        build_source_identity_calibration_memory_index,
    )
    from source_identity_calibration_memory_trust_review import (
        build_source_identity_calibration_memory_trust_review,
    )
    from source_identity_calibration_memory_trust_acceptance_ledger import (
        build_source_identity_calibration_memory_trust_acceptance_ledger,
    )
    from source_identity_calibration_memory_trust_promotion_plan import (
        build_source_identity_calibration_memory_trust_promotion_plan,
    )
    from source_identity_calibration_memory_trust_promotion_contract import (
        build_source_identity_calibration_memory_trust_promotion_contract,
    )
    from source_identity_calibration_memory_trust_promotion_acceptance_ledger import (
        build_source_identity_calibration_memory_trust_promotion_acceptance_ledger,
    )
    from source_identity_calibration_memory_trust_promotion_execution_plan import (
        build_source_identity_calibration_memory_trust_promotion_execution_plan,
    )
    from source_identity_replay_calibration import (
        SourceIdentityReplayEvent,
        analyze_source_identity_replay_calibration,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.agentic_method_foundation import (
        build_agentic_recommendations,
        build_inter_channel_context,
        feature_snapshot_from_channel_inventory,
    )
    from backend.operator_analysis import build_operator_analysis_report
    from backend.operator_product_state import build_channel_inventory
    from backend.operator_recommendation_bridge import build_safe_gain_operator_proposals
    from backend.processing_proposal_adapters import (
        MAIN_PIPELINE_ID,
        adapt_agentic_foundation_gain_fader_recommendations,
        adapt_operator_analysis_gain_fader_proposals,
        adapt_safe_gain_suggestions,
        adapt_source_identity_replay_calibration_proposals,
        build_proposal_only_method_contracts,
        build_registry_blockers,
    )
    from backend.processing_registry import (
        LIVE_ALLOWED_WRITE_KINDS,
        build_pipeline_runtime_summary,
        build_processing_registry_snapshot,
        get_pipeline_profile,
        get_processing_method,
    )
    from backend.source_identity_calibration_acceptance_ledger import (
        build_source_identity_calibration_acceptance_ledger,
    )
    from backend.source_identity_calibration_persistence_plan import (
        build_source_identity_calibration_persistence_plan,
    )
    from backend.source_identity_calibration_memory_draft import (
        build_source_identity_calibration_memory_draft,
    )
    from backend.source_identity_calibration_memory_writer import (
        build_source_identity_calibration_memory_writer,
    )
    from backend.source_identity_calibration_memory_index import (
        build_source_identity_calibration_memory_index,
    )
    from backend.source_identity_calibration_memory_trust_review import (
        build_source_identity_calibration_memory_trust_review,
    )
    from backend.source_identity_calibration_memory_trust_acceptance_ledger import (
        build_source_identity_calibration_memory_trust_acceptance_ledger,
    )
    from backend.source_identity_calibration_memory_trust_promotion_plan import (
        build_source_identity_calibration_memory_trust_promotion_plan,
    )
    from backend.source_identity_calibration_memory_trust_promotion_contract import (
        build_source_identity_calibration_memory_trust_promotion_contract,
    )
    from backend.source_identity_calibration_memory_trust_promotion_acceptance_ledger import (
        build_source_identity_calibration_memory_trust_promotion_acceptance_ledger,
    )
    from backend.source_identity_calibration_memory_trust_promotion_execution_plan import (
        build_source_identity_calibration_memory_trust_promotion_execution_plan,
    )
    from backend.source_identity_replay_calibration import (
        SourceIdentityReplayEvent,
        analyze_source_identity_replay_calibration,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_call(fn: Any, *args: Any, default: Any = None, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        return {"error": str(exc)} if default is None else default


def _safe_int(value: Any, default: int = 24) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _active_methods_for_main_pipeline() -> List[Dict[str, Any]]:
    profile = get_pipeline_profile(MAIN_PIPELINE_ID)
    if profile is None:
        return []
    proposal_ids = set(profile.proposal_method_ids)
    executor_ids = set(profile.executor_method_ids)
    active = []
    for method_id in profile.method_ids:
        method = get_processing_method(method_id)
        if method is None:
            continue
        role = "analysis"
        if method.id in executor_ids:
            role = "executor"
        elif method.id in proposal_ids:
            role = "proposal"
        active.append(
            {
                "method_id": method.id,
                "label": method.label,
                "domain": method.domain,
                "status": method.status,
                "role": role,
                "can_write_live": method.can_write_live,
                "live_write_kinds": list(method.live_write_kinds),
            }
        )
    return active


def _safe_gain_preview(server: Any) -> Dict[str, Any]:
    calibrator = getattr(server, "safe_gain_calibrator", None)
    if calibrator is None:
        return {
            "available": False,
            "reason": "safe_gain_calibrator_unavailable",
            "proposals": [],
        }

    try:
        suggestions = calibrator.get_suggestions()
    except Exception as exc:
        return {
            "available": False,
            "reason": "safe_gain_suggestions_unavailable",
            "error": str(exc),
            "proposals": [],
        }

    channel_mapping = getattr(calibrator, "channel_mapping", None)
    proposals = build_safe_gain_operator_proposals(
        suggestions or {},
        channel_mapping=channel_mapping,
        mixer_client=getattr(server, "mixer_client", None),
        source="safe_gain_preview",
    )
    return {
        "available": bool(suggestions),
        "reason": None if suggestions else "no_safe_gain_suggestions",
        "proposal_count": len(proposals),
        "proposals": proposals,
    }


def _agentic_foundation_preview(server: Any, *, total_channels: int) -> Dict[str, Any]:
    inventory = build_channel_inventory(server, total_channels=total_channels)
    audio_running = bool(inventory.get("audio_capture", {}).get("running"))
    if not audio_running:
        return {
            "available": False,
            "reason": "audio_capture_not_running",
            "recommendation_count": 0,
            "proposals": [],
        }

    snapshots = [
        feature_snapshot_from_channel_inventory(channel)
        for channel in inventory.get("channels", [])
        if isinstance(channel, dict)
    ]
    context = build_inter_channel_context(snapshots)
    recommendations = []
    for snapshot in snapshots:
        recommendations.extend(
            recommendation.to_dict()
            for recommendation in build_agentic_recommendations(snapshot, context=context)
        )

    proposals = adapt_agentic_foundation_gain_fader_recommendations(
        recommendations,
        pipeline_id=MAIN_PIPELINE_ID,
    )
    return {
        "available": bool(recommendations),
        "reason": None if recommendations else "no_agentic_foundation_recommendations",
        "recommendation_count": len(recommendations),
        "proposal_count": len(proposals),
        "proposals": proposals,
        "policy": {
            "preview_only": True,
            "queue_mutation": False,
            "live_write": False,
            "safe_action_types": ["gain_trim", "fader"],
            "direction": "down_only",
        },
    }


def _source_identity_replay_event_from_mapping(raw: Dict[str, Any]) -> Optional[SourceIdentityReplayEvent]:
    allowed = {
        "event_id",
        "source_id",
        "predicted_source_type",
        "predicted_confidence",
        "operator_decision",
        "accepted_source_type",
        "replay_matched",
        "downstream_family",
        "evidence_packet_id",
        "safety_regression",
        "manual_override",
        "session_id",
    }
    payload = {key: raw[key] for key in allowed if key in raw}
    required = {
        "event_id",
        "source_id",
        "predicted_source_type",
        "predicted_confidence",
        "operator_decision",
    }
    if not required.issubset(payload):
        return None
    try:
        return SourceIdentityReplayEvent(**payload)
    except TypeError:
        return None


def _source_identity_replay_events(server: Any) -> List[SourceIdentityReplayEvent]:
    raw_events = getattr(server, "source_identity_replay_events", None)
    if raw_events is None:
        config = getattr(server, "config", {})
        if isinstance(config, dict):
            raw_events = config.get("source_identity_replay_events")
    if not isinstance(raw_events, (list, tuple)):
        return []

    events: List[SourceIdentityReplayEvent] = []
    for raw in raw_events:
        if isinstance(raw, SourceIdentityReplayEvent):
            events.append(raw)
        elif isinstance(raw, dict):
            event = _source_identity_replay_event_from_mapping(raw)
            if event is not None:
                events.append(event)
    return events


def _source_identity_replay_freshness(server: Any) -> Dict[str, Any]:
    raw = getattr(server, "source_identity_replay_context_cache_freshness", None)
    if raw is None:
        config = getattr(server, "config", {})
        if isinstance(config, dict):
            raw = config.get("source_identity_replay_context_cache_freshness")
    return dict(raw) if isinstance(raw, dict) else {"status": "not_provided", "provided": False}


def _source_identity_replay_preview(server: Any) -> Dict[str, Any]:
    events = _source_identity_replay_events(server)
    freshness = _source_identity_replay_freshness(server)
    if not events:
        return {
            "available": False,
            "reason": "no_source_identity_replay_events",
            "event_count": 0,
            "proposal_count": 0,
            "proposals": [],
            "metadata": {
                "current_state_cache_freshness_status": str(
                    freshness.get("status") or "not_provided"
                ),
                "queue_mutation": False,
                "live_write": False,
            },
        }

    result = analyze_source_identity_replay_calibration(
        events,
        current_state_cache_freshness=freshness,
        operator_review_enabled=True,
    )
    raw_proposals = tuple(item.to_operator_proposal() for item in result.recommendations)
    proposals = adapt_source_identity_replay_calibration_proposals(
        raw_proposals,
        pipeline_id=MAIN_PIPELINE_ID,
    )
    return {
        "available": bool(proposals),
        "reason": None if proposals else result.status,
        "status": result.status,
        "event_count": len(events),
        "recommendation_count": len(result.recommendations),
        "proposal_count": len(proposals),
        "warnings": list(result.warnings),
        "metadata": {
            **dict(result.metadata),
            "queue_mutation": False,
            "live_write": False,
            "downstream_priority_mutation": False,
        },
        "proposals": proposals,
    }


def _source_identity_calibration_acceptance_records(server: Any) -> List[Dict[str, Any]]:
    raw_records = getattr(server, "source_identity_calibration_acceptance_records", None)
    if raw_records is None:
        config = getattr(server, "config", {})
        if isinstance(config, dict):
            raw_records = config.get("source_identity_calibration_acceptance_records")
    if not isinstance(raw_records, (list, tuple)):
        return []
    return [dict(record) for record in raw_records if isinstance(record, dict)]


def _source_identity_calibration_memory_text(server: Any) -> Optional[str]:
    raw_text = getattr(server, "source_identity_calibration_memory_text", None)
    if raw_text is None:
        config = getattr(server, "config", {})
        if isinstance(config, dict):
            raw_text = config.get("source_identity_calibration_memory_text")
    return raw_text if isinstance(raw_text, str) else None


def _source_identity_calibration_memory_root(server: Any) -> Optional[str]:
    raw_root = getattr(server, "source_identity_calibration_memory_root", None)
    if raw_root is None:
        config = getattr(server, "config", {})
        if isinstance(config, dict):
            raw_root = config.get("source_identity_calibration_memory_root")
    return str(raw_root) if raw_root else None


def _source_identity_memory_trust_acceptance_records(server: Any) -> List[Dict[str, Any]]:
    raw_records = getattr(server, "source_identity_memory_trust_acceptance_records", None)
    if raw_records is None:
        raw_records = getattr(
            server,
            "source_identity_calibration_memory_trust_acceptance_records",
            None,
        )
    if raw_records is None:
        config = getattr(server, "config", {})
        if isinstance(config, dict):
            raw_records = config.get("source_identity_memory_trust_acceptance_records")
            if raw_records is None:
                raw_records = config.get(
                    "source_identity_calibration_memory_trust_acceptance_records"
                )
    if not isinstance(raw_records, (list, tuple)):
        return []
    return [dict(record) for record in raw_records if isinstance(record, dict)]


def _source_identity_memory_trust_promotion_rollback_evidence(
    server: Any,
) -> Dict[str, Any]:
    raw_evidence = getattr(
        server,
        "source_identity_memory_trust_promotion_rollback_evidence",
        None,
    )
    if raw_evidence is None:
        raw_evidence = getattr(
            server,
            "source_identity_calibration_memory_trust_promotion_rollback_evidence",
            None,
        )
    if raw_evidence is None:
        config = getattr(server, "config", {})
        if isinstance(config, dict):
            raw_evidence = config.get(
                "source_identity_memory_trust_promotion_rollback_evidence"
            )
            if raw_evidence is None:
                raw_evidence = config.get(
                    "source_identity_calibration_memory_trust_promotion_rollback_evidence"
                )
    return dict(raw_evidence) if isinstance(raw_evidence, dict) else {}


def _source_identity_memory_trust_promotion_acceptance_records(
    server: Any,
) -> List[Dict[str, Any]]:
    raw_records = getattr(
        server,
        "source_identity_memory_trust_promotion_acceptance_records",
        None,
    )
    if raw_records is None:
        raw_records = getattr(
            server,
            "source_identity_calibration_memory_trust_promotion_acceptance_records",
            None,
        )
    if raw_records is None:
        config = getattr(server, "config", {})
        if isinstance(config, dict):
            raw_records = config.get(
                "source_identity_memory_trust_promotion_acceptance_records"
            )
            if raw_records is None:
                raw_records = config.get(
                    "source_identity_calibration_memory_trust_promotion_acceptance_records"
                )
    if not isinstance(raw_records, (list, tuple)):
        return []
    return [dict(record) for record in raw_records if isinstance(record, dict)]


def build_main_pipeline_preview(
    server: Any,
    *,
    total_channels: int = 24,
    include_methods: bool = False,
) -> Dict[str, Any]:
    """Build a read-only preview of the main live assist pipeline.

    The preview may produce proposal candidates, but it never imports them into
    the queue and never calls a live mixer write surface.
    """

    total_channels = max(1, min(_safe_int(total_channels, 24), 64))
    operator_mode = server.get_operator_mode_status()
    manual_write_supervision = (
        _safe_call(server.get_manual_write_supervision_status, default={})
        if hasattr(server, "get_manual_write_supervision_status")
        else {}
    )
    profile = get_pipeline_profile(MAIN_PIPELINE_ID)

    analysis_report = build_operator_analysis_report(
        server,
        total_channels=total_channels,
        create_proposals=False,
        include_proposal_previews=True,
    )
    operator_proposals = adapt_operator_analysis_gain_fader_proposals(
        analysis_report.get("proposals_preview", []),
        pipeline_id=MAIN_PIPELINE_ID,
    )

    safe_gain = _safe_gain_preview(server)
    safe_gain_proposals = adapt_safe_gain_suggestions(
        safe_gain.get("proposals", []),
        pipeline_id=MAIN_PIPELINE_ID,
    )

    agentic_foundation = _agentic_foundation_preview(server, total_channels=total_channels)
    agentic_foundation_proposals = agentic_foundation.get("proposals", [])

    source_identity_replay = _source_identity_replay_preview(server)
    source_identity_replay_proposals = source_identity_replay.get("proposals", [])
    source_identity_acceptance_ledger = build_source_identity_calibration_acceptance_ledger(
        calibration_candidates=source_identity_replay_proposals,
        acceptance_records=_source_identity_calibration_acceptance_records(server),
    )
    source_identity_acceptance_summary = source_identity_acceptance_ledger.get("summary", {})
    source_identity_persistence_plan = build_source_identity_calibration_persistence_plan(
        acceptance_ledger=source_identity_acceptance_ledger,
    )
    source_identity_persistence_summary = source_identity_persistence_plan.get("summary", {})
    source_identity_memory_draft = build_source_identity_calibration_memory_draft(
        persistence_plan=source_identity_persistence_plan,
    )
    source_identity_memory_draft_summary = source_identity_memory_draft.get("summary", {})
    source_identity_memory_writer = build_source_identity_calibration_memory_writer(
        memory_draft=source_identity_memory_draft,
        apply=False,
        operator_ack=False,
    )
    source_identity_memory_writer_summary = source_identity_memory_writer.get("summary", {})
    source_identity_memory_index = build_source_identity_calibration_memory_index(
        memory_text=_source_identity_calibration_memory_text(server),
        root=_source_identity_calibration_memory_root(server),
    )
    source_identity_memory_index_summary = source_identity_memory_index.get("summary", {})
    source_identity_memory_trust_review = (
        build_source_identity_calibration_memory_trust_review(
            memory_index=source_identity_memory_index,
            current_state_cache_freshness=_source_identity_replay_freshness(server),
        )
    )
    source_identity_memory_trust_review_summary = source_identity_memory_trust_review.get(
        "summary",
        {},
    )
    source_identity_memory_trust_acceptance_ledger = (
        build_source_identity_calibration_memory_trust_acceptance_ledger(
            memory_trust_review=source_identity_memory_trust_review,
            acceptance_records=_source_identity_memory_trust_acceptance_records(server),
        )
    )
    source_identity_memory_trust_acceptance_summary = (
        source_identity_memory_trust_acceptance_ledger.get("summary", {})
    )
    source_identity_memory_trust_promotion_plan = (
        build_source_identity_calibration_memory_trust_promotion_plan(
            acceptance_ledger=source_identity_memory_trust_acceptance_ledger,
        )
    )
    source_identity_memory_trust_promotion_summary = (
        source_identity_memory_trust_promotion_plan.get("summary", {})
    )
    source_identity_memory_trust_promotion_contract = (
        build_source_identity_calibration_memory_trust_promotion_contract(
            promotion_plan=source_identity_memory_trust_promotion_plan,
            current_state_cache_freshness=_source_identity_replay_freshness(server),
            rollback_evidence=(
                _source_identity_memory_trust_promotion_rollback_evidence(server)
            ),
        )
    )
    source_identity_memory_trust_promotion_contract_summary = (
        source_identity_memory_trust_promotion_contract.get("summary", {})
    )
    source_identity_memory_trust_promotion_acceptance_ledger = (
        build_source_identity_calibration_memory_trust_promotion_acceptance_ledger(
            promotion_contract=source_identity_memory_trust_promotion_contract,
            acceptance_records=(
                _source_identity_memory_trust_promotion_acceptance_records(server)
            ),
        )
    )
    source_identity_memory_trust_promotion_acceptance_summary = (
        source_identity_memory_trust_promotion_acceptance_ledger.get("summary", {})
    )
    source_identity_memory_trust_promotion_execution_plan = (
        build_source_identity_calibration_memory_trust_promotion_execution_plan(
            promotion_acceptance_ledger=(
                source_identity_memory_trust_promotion_acceptance_ledger
            ),
        )
    )
    source_identity_memory_trust_promotion_execution_summary = (
        source_identity_memory_trust_promotion_execution_plan.get("summary", {})
    )

    proposals = (
        operator_proposals
        + safe_gain_proposals
        + agentic_foundation_proposals
        + source_identity_replay_proposals
    )
    blockers = build_registry_blockers(pipeline_id=MAIN_PIPELINE_ID)
    proposal_only_contracts = build_proposal_only_method_contracts(pipeline_id=MAIN_PIPELINE_ID)
    registry_snapshot = build_processing_registry_snapshot(include_methods=include_methods)

    analysis_status = str(analysis_report.get("status") or "unknown")
    status = "ok"
    reason: Optional[str] = None
    if analysis_status == "no_audio":
        status = "no_audio"
        reason = "audio_capture_not_running"
    elif analysis_status == "clear" and not proposals:
        status = "clear"
        reason = "no_pipeline_preview_proposals"

    return {
        "type": "main_pipeline_preview",
        "generated_at": _utc_now(),
        "status": status,
        "reason": reason,
        "pipeline_id": MAIN_PIPELINE_ID,
        "pipeline_status": profile.status if profile else "missing",
        "operator_mode": operator_mode,
        "runtime": build_pipeline_runtime_summary(operator_mode, manual_write_supervision),
        "executor_policy": {
            "preview_only": True,
            "queue_mutation": False,
            "live_write": False,
            "live_apply_policy": "supervised_fader_gain_only",
            "apply_path": "apply_operator_proposal -> supervised_manual_gate",
            "executor_method_ids": list(profile.executor_method_ids if profile else ()),
            "allowed_requested_change_value_types": list(LIVE_ALLOWED_WRITE_KINDS),
            "requires_supervision": True,
        },
        "summary": {
            "total_channels": total_channels,
            "analysis_status": analysis_status,
            "analysis_observation_count": int(analysis_report.get("observation_count", 0) or 0),
            "proposal_count": len(proposals),
            "operator_analysis_proposal_count": len(operator_proposals),
            "safe_gain_proposal_count": len(safe_gain_proposals),
            "agentic_foundation_proposal_count": len(agentic_foundation_proposals),
            "agentic_foundation_recommendation_count": int(agentic_foundation.get("recommendation_count", 0) or 0),
            "source_identity_replay_event_count": int(source_identity_replay.get("event_count", 0) or 0),
            "source_identity_replay_calibration_proposal_count": len(source_identity_replay_proposals),
            "source_identity_calibration_acceptance_record_count": int(
                source_identity_acceptance_summary.get("acceptance_record_count", 0) or 0
            ),
            "source_identity_calibration_accepted_record_count": int(
                source_identity_acceptance_summary.get("accepted_record_count", 0) or 0
            ),
            "source_identity_calibration_invalid_record_count": int(
                source_identity_acceptance_summary.get("invalid_record_count", 0) or 0
            ),
            "source_identity_calibration_persistence_candidate_count": int(
                source_identity_persistence_summary.get("persistence_candidate_count", 0) or 0
            ),
            "source_identity_calibration_memory_draft_record_count": int(
                source_identity_memory_draft_summary.get("draft_record_count", 0) or 0
            ),
            "source_identity_calibration_memory_writer_write_result_count": int(
                source_identity_memory_writer_summary.get("write_result_count", 0) or 0
            ),
            "source_identity_calibration_memory_index_record_count": int(
                source_identity_memory_index_summary.get("memory_record_count", 0) or 0
            ),
            "source_identity_calibration_memory_index_invalid_record_count": int(
                source_identity_memory_index_summary.get("invalid_record_count", 0) or 0
            ),
            "source_identity_calibration_memory_trust_review_candidate_count": int(
                source_identity_memory_trust_review_summary.get(
                    "trust_review_candidate_count",
                    0,
                )
                or 0
            ),
            "source_identity_calibration_memory_trust_acceptance_record_count": int(
                source_identity_memory_trust_acceptance_summary.get(
                    "acceptance_record_count",
                    0,
                )
                or 0
            ),
            "source_identity_calibration_memory_trust_accepted_record_count": int(
                source_identity_memory_trust_acceptance_summary.get(
                    "accepted_record_count",
                    0,
                )
                or 0
            ),
            "source_identity_calibration_memory_trust_promotion_candidate_count": int(
                source_identity_memory_trust_promotion_summary.get(
                    "promotion_candidate_count",
                    0,
                )
                or 0
            ),
            "source_identity_calibration_memory_trust_promotion_contract_record_count": int(
                source_identity_memory_trust_promotion_contract_summary.get(
                    "contract_record_count",
                    0,
                )
                or 0
            ),
            "source_identity_calibration_memory_trust_promotion_acceptance_record_count": int(
                source_identity_memory_trust_promotion_acceptance_summary.get(
                    "acceptance_record_count",
                    0,
                )
                or 0
            ),
            "source_identity_calibration_memory_trust_promotion_accepted_record_count": int(
                source_identity_memory_trust_promotion_acceptance_summary.get(
                    "accepted_record_count",
                    0,
                )
                or 0
            ),
            "source_identity_calibration_memory_trust_promotion_execution_candidate_count": int(
                source_identity_memory_trust_promotion_execution_summary.get(
                    "execution_candidate_count",
                    0,
                )
                or 0
            ),
            "proposal_only_contract_count": len(proposal_only_contracts),
            "blocker_count": len(blockers),
            "queue_mutation": False,
            "live_write": False,
        },
        "active_methods": _active_methods_for_main_pipeline(),
        "adapters": [
            {
                "adapter_id": "operator_analysis_gain_fader",
                "method_id": "auto_soundcheck_engine",
                "status": "active",
                "output": "fader_gain_proposals_preview",
            },
            {
                "adapter_id": "safe_gain_suggestions",
                "method_id": "lufs_gain_staging",
                "status": "active" if safe_gain.get("available") else "standby",
                "output": "gain_proposals_preview",
                "reason": safe_gain.get("reason"),
            },
            {
                "adapter_id": "agentic_foundation_gain_fader",
                "method_id": "agentic_method_foundation",
                "status": "active" if agentic_foundation_proposals else "standby",
                "output": "bounded_gain_fader_down_preview",
                "reason": agentic_foundation.get("reason"),
            },
            {
                "adapter_id": "source_identity_replay_calibration",
                "method_id": "source_identity_replay_calibration",
                "status": "active" if source_identity_replay_proposals else "standby",
                "output": "source_identity_downstream_trust_review_preview",
                "reason": source_identity_replay.get("reason"),
            },
            {
                "adapter_id": "source_identity_calibration_acceptance_ledger",
                "method_id": "source_identity_calibration_acceptance_ledger",
                "status": source_identity_acceptance_ledger.get("status"),
                "output": "source_identity_operator_acceptance_ledger",
                "reason": (
                    ", ".join(source_identity_acceptance_summary.get("blocked_by", []))
                    if source_identity_acceptance_summary.get("blocked_by")
                    else None
                ),
            },
            {
                "adapter_id": "source_identity_calibration_persistence_plan",
                "method_id": "source_identity_calibration_persistence_plan",
                "status": source_identity_persistence_plan.get("status"),
                "output": "source_identity_manual_persistence_plan",
                "reason": (
                    ", ".join(source_identity_persistence_summary.get("blocked_by", []))
                    if source_identity_persistence_summary.get("blocked_by")
                    else None
                ),
            },
            {
                "adapter_id": "source_identity_calibration_memory_draft",
                "method_id": "source_identity_calibration_memory_draft",
                "status": source_identity_memory_draft.get("status"),
                "output": "source_identity_append_only_memory_draft",
                "reason": (
                    ", ".join(source_identity_memory_draft_summary.get("blocked_by", []))
                    if source_identity_memory_draft_summary.get("blocked_by")
                    else None
                ),
            },
            {
                "adapter_id": "source_identity_calibration_memory_writer",
                "method_id": "source_identity_calibration_memory_writer",
                "status": source_identity_memory_writer.get("status"),
                "output": "source_identity_append_only_memory_writer_dry_run",
                "reason": (
                    ", ".join(source_identity_memory_writer_summary.get("blocked_by", []))
                    if source_identity_memory_writer_summary.get("blocked_by")
                    else None
                ),
            },
            {
                "adapter_id": "source_identity_calibration_memory_index",
                "method_id": "source_identity_calibration_memory_index",
                "status": source_identity_memory_index.get("status"),
                "output": "source_identity_append_only_memory_index_review",
                "reason": (
                    ", ".join(source_identity_memory_index_summary.get("blocked_by", []))
                    if source_identity_memory_index_summary.get("blocked_by")
                    else None
                ),
            },
            {
                "adapter_id": "source_identity_calibration_memory_trust_review",
                "method_id": "source_identity_calibration_memory_trust_review",
                "status": source_identity_memory_trust_review.get("status"),
                "output": "source_identity_memory_trust_operator_review",
                "reason": (
                    ", ".join(
                        source_identity_memory_trust_review_summary.get("blocked_by", [])
                    )
                    if source_identity_memory_trust_review_summary.get("blocked_by")
                    else None
                ),
            },
            {
                "adapter_id": "source_identity_calibration_memory_trust_acceptance_ledger",
                "method_id": "source_identity_calibration_memory_trust_acceptance_ledger",
                "status": source_identity_memory_trust_acceptance_ledger.get("status"),
                "output": "source_identity_memory_trust_acceptance_ledger",
                "reason": (
                    ", ".join(
                        source_identity_memory_trust_acceptance_summary.get(
                            "blocked_by",
                            [],
                        )
                    )
                    if source_identity_memory_trust_acceptance_summary.get("blocked_by")
                    else None
                ),
            },
            {
                "adapter_id": "source_identity_calibration_memory_trust_promotion_plan",
                "method_id": "source_identity_calibration_memory_trust_promotion_plan",
                "status": source_identity_memory_trust_promotion_plan.get("status"),
                "output": "source_identity_memory_trust_manual_promotion_plan",
                "reason": (
                    ", ".join(
                        source_identity_memory_trust_promotion_summary.get(
                            "blocked_by",
                            [],
                        )
                    )
                    if source_identity_memory_trust_promotion_summary.get("blocked_by")
                    else None
                ),
            },
            {
                "adapter_id": (
                    "source_identity_calibration_memory_trust_promotion_contract"
                ),
                "method_id": (
                    "source_identity_calibration_memory_trust_promotion_contract"
                ),
                "status": source_identity_memory_trust_promotion_contract.get("status"),
                "output": "source_identity_memory_trust_promotion_contract_review",
                "reason": (
                    ", ".join(
                        source_identity_memory_trust_promotion_contract_summary.get(
                            "blocked_by",
                            [],
                        )
                    )
                    if source_identity_memory_trust_promotion_contract_summary.get(
                        "blocked_by"
                    )
                    else None
                ),
            },
            {
                "adapter_id": (
                    "source_identity_calibration_memory_trust_promotion_acceptance_ledger"
                ),
                "method_id": (
                    "source_identity_calibration_memory_trust_promotion_acceptance_ledger"
                ),
                "status": (
                    source_identity_memory_trust_promotion_acceptance_ledger.get(
                        "status"
                    )
                ),
                "output": "source_identity_memory_trust_promotion_acceptance_ledger",
                "reason": (
                    ", ".join(
                        source_identity_memory_trust_promotion_acceptance_summary.get(
                            "blocked_by",
                            [],
                        )
                    )
                    if source_identity_memory_trust_promotion_acceptance_summary.get(
                        "blocked_by"
                    )
                    else None
                ),
            },
            {
                "adapter_id": (
                    "source_identity_calibration_memory_trust_promotion_execution_plan"
                ),
                "method_id": (
                    "source_identity_calibration_memory_trust_promotion_execution_plan"
                ),
                "status": (
                    source_identity_memory_trust_promotion_execution_plan.get("status")
                ),
                "output": "source_identity_memory_trust_promotion_execution_plan",
                "reason": (
                    ", ".join(
                        source_identity_memory_trust_promotion_execution_summary.get(
                            "blocked_by",
                            [],
                        )
                    )
                    if source_identity_memory_trust_promotion_execution_summary.get(
                        "blocked_by"
                    )
                    else None
                ),
            },
            {
                "adapter_id": "registry_blockers",
                "method_id": "processing_registry",
                "status": "active",
                "output": "blocked_method_visibility",
            },
            {
                "adapter_id": "proposal_only_contracts",
                "method_id": "processing_registry",
                "status": "active",
                "output": "non_fader_gain_contract_visibility",
            },
        ],
        "proposals": proposals,
        "proposal_only_contracts": proposal_only_contracts,
        "blockers": blockers,
        "analysis": {
            "status": analysis_report.get("status"),
            "reason": analysis_report.get("reason"),
            "audio_running": analysis_report.get("audio_running"),
            "observations": analysis_report.get("observations", []),
            "create_proposals": False,
        },
        "safe_gain": {
            key: value
            for key, value in safe_gain.items()
            if key != "proposals"
        },
        "agentic_foundation": {
            key: value
            for key, value in agentic_foundation.items()
            if key != "proposals"
        },
        "source_identity_replay_calibration": {
            key: value
            for key, value in source_identity_replay.items()
            if key != "proposals"
        },
        "source_identity_calibration_acceptance_ledger": source_identity_acceptance_ledger,
        "source_identity_calibration_persistence_plan": source_identity_persistence_plan,
        "source_identity_calibration_memory_draft": source_identity_memory_draft,
        "source_identity_calibration_memory_writer": source_identity_memory_writer,
        "source_identity_calibration_memory_index": source_identity_memory_index,
        "source_identity_calibration_memory_trust_review": (
            source_identity_memory_trust_review
        ),
        "source_identity_calibration_memory_trust_acceptance_ledger": (
            source_identity_memory_trust_acceptance_ledger
        ),
        "source_identity_calibration_memory_trust_promotion_plan": (
            source_identity_memory_trust_promotion_plan
        ),
        "source_identity_calibration_memory_trust_promotion_contract": (
            source_identity_memory_trust_promotion_contract
        ),
        "source_identity_calibration_memory_trust_promotion_acceptance_ledger": (
            source_identity_memory_trust_promotion_acceptance_ledger
        ),
        "source_identity_calibration_memory_trust_promotion_execution_plan": (
            source_identity_memory_trust_promotion_execution_plan
        ),
        "registry": {
            "summary": registry_snapshot.get("summary", {}),
            "pipeline_profiles": registry_snapshot.get("pipeline_profiles", []),
            "methods": registry_snapshot.get("methods", []),
        },
    }
