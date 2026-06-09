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
    from operator_proposal_apply_sequence_policy_memory_index import (
        build_operator_proposal_apply_sequence_policy_memory_index,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_review import (
        build_operator_proposal_apply_sequence_policy_memory_trust_review,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger import (
        build_operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_plan import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_plan,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_contract import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_contract,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_executor import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer,
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
    from source_identity_calibration_memory_trust_promotion_executor_dry_run import (
        build_source_identity_calibration_memory_trust_promotion_executor_dry_run,
    )
    from source_identity_calibration_memory_trust_promotion_executor import (
        build_source_identity_calibration_memory_trust_promotion_executor,
    )
    from source_identity_calibration_memory_trust_promotion_apply_audit import (
        build_source_identity_calibration_memory_trust_promotion_apply_audit,
    )
    from source_identity_calibration_memory_trust_promotion_rollback_verification import (
        build_source_identity_calibration_memory_trust_promotion_rollback_verification,
    )
    from source_identity_calibration_memory_trust_promotion_priority_gate import (
        build_source_identity_calibration_memory_trust_promotion_priority_gate,
    )
    from source_identity_proposal_priority_policy import (
        build_source_identity_proposal_priority_policy,
    )
    from source_identity_proposal_ordering_preview import (
        build_source_identity_proposal_ordering_preview,
    )
    from source_identity_proposal_queue_ordering_plan import (
        build_source_identity_proposal_queue_ordering_plan,
    )
    from source_identity_proposal_queue_importer import (
        build_source_identity_proposal_queue_importer,
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
    from backend.operator_proposal_apply_sequence_policy_memory_index import (
        build_operator_proposal_apply_sequence_policy_memory_index,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_review import (
        build_operator_proposal_apply_sequence_policy_memory_trust_review,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger import (
        build_operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_plan import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_plan,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_contract import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_contract,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_executor import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer,
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
    from backend.source_identity_calibration_memory_trust_promotion_executor_dry_run import (
        build_source_identity_calibration_memory_trust_promotion_executor_dry_run,
    )
    from backend.source_identity_calibration_memory_trust_promotion_executor import (
        build_source_identity_calibration_memory_trust_promotion_executor,
    )
    from backend.source_identity_calibration_memory_trust_promotion_apply_audit import (
        build_source_identity_calibration_memory_trust_promotion_apply_audit,
    )
    from backend.source_identity_calibration_memory_trust_promotion_rollback_verification import (
        build_source_identity_calibration_memory_trust_promotion_rollback_verification,
    )
    from backend.source_identity_calibration_memory_trust_promotion_priority_gate import (
        build_source_identity_calibration_memory_trust_promotion_priority_gate,
    )
    from backend.source_identity_proposal_priority_policy import (
        build_source_identity_proposal_priority_policy,
    )
    from backend.source_identity_proposal_ordering_preview import (
        build_source_identity_proposal_ordering_preview,
    )
    from backend.source_identity_proposal_queue_ordering_plan import (
        build_source_identity_proposal_queue_ordering_plan,
    )
    from backend.source_identity_proposal_queue_importer import (
        build_source_identity_proposal_queue_importer,
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


def _operator_proposal_apply_sequence_policy_memory_text(server: Any) -> Optional[str]:
    raw_text = getattr(server, "operator_proposal_apply_sequence_policy_memory_text", None)
    if raw_text is None:
        config = getattr(server, "config", {})
        if isinstance(config, dict):
            raw_text = config.get("operator_proposal_apply_sequence_policy_memory_text")
    return raw_text if isinstance(raw_text, str) else None


def _operator_proposal_apply_sequence_policy_memory_root(server: Any) -> Optional[str]:
    raw_root = getattr(server, "operator_proposal_apply_sequence_policy_memory_root", None)
    if raw_root is None:
        config = getattr(server, "config", {})
        if isinstance(config, dict):
            raw_root = config.get("operator_proposal_apply_sequence_policy_memory_root")
    return str(raw_root) if raw_root else None


def _operator_proposal_apply_sequence_policy_review_freshness(
    server: Any,
) -> Dict[str, Any]:
    raw = getattr(server, "operator_proposal_apply_sequence_policy_review_freshness", None)
    if raw is None:
        config = getattr(server, "config", {})
        if isinstance(config, dict):
            raw = config.get("operator_proposal_apply_sequence_policy_review_freshness")
    return dict(raw) if isinstance(raw, dict) else {"status": "not_provided", "provided": False}


def _operator_proposal_apply_sequence_policy_memory_trust_acceptance_records(
    server: Any,
) -> List[Dict[str, Any]]:
    raw_records = getattr(
        server,
        "operator_proposal_apply_sequence_policy_memory_trust_acceptance_records",
        None,
    )
    if raw_records is None:
        raw_records = getattr(server, "sequence_policy_memory_trust_acceptance_records", None)
    if raw_records is None:
        config = getattr(server, "config", {})
        if isinstance(config, dict):
            raw_records = config.get(
                "operator_proposal_apply_sequence_policy_memory_trust_acceptance_records"
            )
            if raw_records is None:
                raw_records = config.get("sequence_policy_memory_trust_acceptance_records")
    if not isinstance(raw_records, (list, tuple)):
        return []
    return [dict(record) for record in raw_records if isinstance(record, dict)]


def _operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_evidence(
    server: Any,
) -> Dict[str, Any]:
    raw_evidence = getattr(
        server,
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_evidence",
        None,
    )
    if raw_evidence is None:
        raw_evidence = getattr(
            server,
            "sequence_policy_memory_trust_promotion_rollback_evidence",
            None,
        )
    if raw_evidence is None:
        config = getattr(server, "config", {})
        if isinstance(config, dict):
            raw_evidence = config.get(
                "operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_evidence"
            )
            if raw_evidence is None:
                raw_evidence = config.get(
                    "sequence_policy_memory_trust_promotion_rollback_evidence"
                )
    return dict(raw_evidence) if isinstance(raw_evidence, dict) else {}


def _operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_records(
    server: Any,
) -> List[Dict[str, Any]]:
    raw_records = getattr(
        server,
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_records",
        None,
    )
    if raw_records is None:
        raw_records = getattr(
            server,
            "sequence_policy_memory_trust_promotion_acceptance_records",
            None,
        )
    if raw_records is None:
        config = getattr(server, "config", {})
        if isinstance(config, dict):
            raw_records = config.get(
                "operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_records"
            )
            if raw_records is None:
                raw_records = config.get(
                    "sequence_policy_memory_trust_promotion_acceptance_records"
                )
    if not isinstance(raw_records, (list, tuple)):
        return []
    return [dict(record) for record in raw_records if isinstance(record, dict)]


def _operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification_records(
    server: Any,
) -> List[Dict[str, Any]]:
    raw_records = getattr(
        server,
        (
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_"
            "rollback_verification_records"
        ),
        None,
    )
    if raw_records is None:
        raw_records = getattr(
            server,
            "sequence_policy_memory_trust_promotion_rollback_verification_records",
            None,
        )
    if raw_records is None:
        config = getattr(server, "config", {})
        if isinstance(config, dict):
            raw_records = config.get(
                "operator_proposal_apply_sequence_policy_memory_trust_promotion_"
                "rollback_verification_records"
            )
            if raw_records is None:
                raw_records = config.get(
                    "sequence_policy_memory_trust_promotion_rollback_verification_records"
                )
    if not isinstance(raw_records, (list, tuple)):
        return []
    return [dict(record) for record in raw_records if isinstance(record, dict)]


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


def _source_identity_memory_trust_promotion_rollback_verification_records(
    server: Any,
) -> List[Dict[str, Any]]:
    raw_records = getattr(
        server,
        "source_identity_memory_trust_promotion_rollback_verification_records",
        None,
    )
    if raw_records is None:
        raw_records = getattr(
            server,
            (
                "source_identity_calibration_memory_trust_promotion_"
                "rollback_verification_records"
            ),
            None,
        )
    if raw_records is None:
        config = getattr(server, "config", {})
        if isinstance(config, dict):
            raw_records = config.get(
                "source_identity_memory_trust_promotion_rollback_verification_records"
            )
            if raw_records is None:
                raw_records = config.get(
                    "source_identity_calibration_memory_trust_promotion_"
                    "rollback_verification_records"
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
    sequence_policy_memory_index = (
        build_operator_proposal_apply_sequence_policy_memory_index(
            memory_text=_operator_proposal_apply_sequence_policy_memory_text(server),
            root=_operator_proposal_apply_sequence_policy_memory_root(server),
        )
    )
    sequence_policy_memory_index_summary = sequence_policy_memory_index.get("summary", {})
    sequence_policy_memory_trust_review = (
        build_operator_proposal_apply_sequence_policy_memory_trust_review(
            memory_index=sequence_policy_memory_index,
            current_sequence_review_freshness=(
                _operator_proposal_apply_sequence_policy_review_freshness(server)
            ),
        )
    )
    sequence_policy_memory_trust_review_summary = (
        sequence_policy_memory_trust_review.get("summary", {})
    )
    sequence_policy_memory_trust_acceptance_ledger = (
        build_operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger(
            memory_trust_review=sequence_policy_memory_trust_review,
            acceptance_records=(
                _operator_proposal_apply_sequence_policy_memory_trust_acceptance_records(
                    server
                )
            ),
        )
    )
    sequence_policy_memory_trust_acceptance_summary = (
        sequence_policy_memory_trust_acceptance_ledger.get("summary", {})
    )
    sequence_policy_memory_trust_promotion_plan = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_plan(
            acceptance_ledger=sequence_policy_memory_trust_acceptance_ledger,
        )
    )
    sequence_policy_memory_trust_promotion_summary = (
        sequence_policy_memory_trust_promotion_plan.get("summary", {})
    )
    sequence_policy_memory_trust_promotion_contract = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_contract(
            promotion_plan=sequence_policy_memory_trust_promotion_plan,
            current_sequence_policy_context=(
                _operator_proposal_apply_sequence_policy_review_freshness(server)
            ),
            rollback_evidence=(
                _operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_evidence(
                    server
                )
            ),
        )
    )
    sequence_policy_memory_trust_promotion_contract_summary = (
        sequence_policy_memory_trust_promotion_contract.get("summary", {})
    )
    sequence_policy_memory_trust_promotion_acceptance_ledger = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger(
            promotion_contract=sequence_policy_memory_trust_promotion_contract,
            acceptance_records=(
                _operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_records(
                    server
                )
            ),
        )
    )
    sequence_policy_memory_trust_promotion_acceptance_summary = (
        sequence_policy_memory_trust_promotion_acceptance_ledger.get("summary", {})
    )
    sequence_policy_memory_trust_promotion_execution_plan = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan(
            promotion_acceptance_ledger=(
                sequence_policy_memory_trust_promotion_acceptance_ledger
            ),
        )
    )
    sequence_policy_memory_trust_promotion_execution_summary = (
        sequence_policy_memory_trust_promotion_execution_plan.get("summary", {})
    )
    sequence_policy_memory_trust_promotion_executor_dry_run = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run(
            promotion_execution_plan=sequence_policy_memory_trust_promotion_execution_plan,
            apply=False,
            operator_ack=False,
        )
    )
    sequence_policy_memory_trust_promotion_executor_dry_run_summary = (
        sequence_policy_memory_trust_promotion_executor_dry_run.get("summary", {})
    )
    sequence_policy_memory_trust_promotion_executor = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor(
            executor_dry_run=sequence_policy_memory_trust_promotion_executor_dry_run,
            apply=False,
            operator_ack=False,
        )
    )
    sequence_policy_memory_trust_promotion_executor_summary = (
        sequence_policy_memory_trust_promotion_executor.get("summary", {})
    )
    sequence_policy_memory_trust_promotion_apply_audit = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit(
            promotion_executor=sequence_policy_memory_trust_promotion_executor,
        )
    )
    sequence_policy_memory_trust_promotion_apply_audit_summary = (
        sequence_policy_memory_trust_promotion_apply_audit.get("summary", {})
    )
    sequence_policy_memory_trust_promotion_rollback_verification = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification(
            apply_audit=sequence_policy_memory_trust_promotion_apply_audit,
            verification_records=(
                _operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification_records(
                    server
                )
            ),
        )
    )
    sequence_policy_memory_trust_promotion_rollback_verification_summary = (
        sequence_policy_memory_trust_promotion_rollback_verification.get("summary", {})
    )
    sequence_policy_memory_trust_promotion_sequence_policy_gate = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate(
            rollback_verification=(
                sequence_policy_memory_trust_promotion_rollback_verification
            ),
        )
    )
    sequence_policy_memory_trust_promotion_sequence_policy_gate_summary = (
        sequence_policy_memory_trust_promotion_sequence_policy_gate.get("summary", {})
    )
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
    source_identity_memory_trust_promotion_executor_dry_run = (
        build_source_identity_calibration_memory_trust_promotion_executor_dry_run(
            promotion_execution_plan=source_identity_memory_trust_promotion_execution_plan,
            apply=False,
            operator_ack=False,
        )
    )
    source_identity_memory_trust_promotion_executor_dry_run_summary = (
        source_identity_memory_trust_promotion_executor_dry_run.get("summary", {})
    )
    source_identity_memory_trust_promotion_executor = (
        build_source_identity_calibration_memory_trust_promotion_executor(
            executor_dry_run=source_identity_memory_trust_promotion_executor_dry_run,
            apply=False,
            operator_ack=False,
        )
    )
    source_identity_memory_trust_promotion_executor_summary = (
        source_identity_memory_trust_promotion_executor.get("summary", {})
    )
    source_identity_memory_trust_promotion_apply_audit = (
        build_source_identity_calibration_memory_trust_promotion_apply_audit(
            promotion_executor=source_identity_memory_trust_promotion_executor,
        )
    )
    source_identity_memory_trust_promotion_apply_audit_summary = (
        source_identity_memory_trust_promotion_apply_audit.get("summary", {})
    )
    source_identity_memory_trust_promotion_rollback_verification = (
        build_source_identity_calibration_memory_trust_promotion_rollback_verification(
            apply_audit=source_identity_memory_trust_promotion_apply_audit,
            verification_records=(
                _source_identity_memory_trust_promotion_rollback_verification_records(
                    server
                )
            ),
        )
    )
    source_identity_memory_trust_promotion_rollback_verification_summary = (
        source_identity_memory_trust_promotion_rollback_verification.get("summary", {})
    )
    source_identity_memory_trust_promotion_priority_gate = (
        build_source_identity_calibration_memory_trust_promotion_priority_gate(
            rollback_verification=(
                source_identity_memory_trust_promotion_rollback_verification
            ),
        )
    )
    source_identity_memory_trust_promotion_priority_gate_summary = (
        source_identity_memory_trust_promotion_priority_gate.get("summary", {})
    )

    proposals = (
        operator_proposals
        + safe_gain_proposals
        + agentic_foundation_proposals
        + source_identity_replay_proposals
    )
    sequence_policy_memory_trust_promotion_policy_consumer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer(
            proposals=proposals,
            sequence_policy_gate=(
                sequence_policy_memory_trust_promotion_sequence_policy_gate
            ),
        )
    )
    sequence_policy_memory_trust_promotion_policy_consumer_summary = (
        sequence_policy_memory_trust_promotion_policy_consumer.get("summary", {})
    )
    sequence_policy_memory_trust_promotion_ordering_preview = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview(
            proposals=proposals,
            policy_consumer=(
                sequence_policy_memory_trust_promotion_policy_consumer
            ),
        )
    )
    sequence_policy_memory_trust_promotion_ordering_preview_summary = (
        sequence_policy_memory_trust_promotion_ordering_preview.get("summary", {})
    )
    sequence_policy_memory_trust_promotion_queue_ordering_plan = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan(
            ordering_preview=sequence_policy_memory_trust_promotion_ordering_preview,
        )
    )
    sequence_policy_memory_trust_promotion_queue_ordering_plan_summary = (
        sequence_policy_memory_trust_promotion_queue_ordering_plan.get("summary", {})
    )
    sequence_policy_memory_trust_promotion_queue_importer = (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer(
            queue_ordering_plan=sequence_policy_memory_trust_promotion_queue_ordering_plan,
            proposal_payloads=proposals,
            operator_mode=operator_mode,
            apply=False,
            operator_ack=False,
        )
    )
    sequence_policy_memory_trust_promotion_queue_importer_summary = (
        sequence_policy_memory_trust_promotion_queue_importer.get("summary", {})
    )
    source_identity_proposal_priority_policy = (
        build_source_identity_proposal_priority_policy(
            proposals=proposals,
            priority_gate=source_identity_memory_trust_promotion_priority_gate,
        )
    )
    source_identity_proposal_priority_policy_summary = (
        source_identity_proposal_priority_policy.get("summary", {})
    )
    source_identity_proposal_ordering_preview = (
        build_source_identity_proposal_ordering_preview(
            proposals=proposals,
            priority_policy=source_identity_proposal_priority_policy,
        )
    )
    source_identity_proposal_ordering_preview_summary = (
        source_identity_proposal_ordering_preview.get("summary", {})
    )
    source_identity_proposal_queue_ordering_plan = (
        build_source_identity_proposal_queue_ordering_plan(
            ordering_preview=source_identity_proposal_ordering_preview,
        )
    )
    source_identity_proposal_queue_ordering_plan_summary = (
        source_identity_proposal_queue_ordering_plan.get("summary", {})
    )
    source_identity_proposal_queue_importer = (
        build_source_identity_proposal_queue_importer(
            queue_ordering_plan=source_identity_proposal_queue_ordering_plan,
            proposal_payloads=proposals,
            operator_mode=operator_mode,
            apply=False,
            operator_ack=False,
        )
    )
    source_identity_proposal_queue_importer_summary = (
        source_identity_proposal_queue_importer.get("summary", {})
    )
    blockers = build_registry_blockers(pipeline_id=MAIN_PIPELINE_ID)
    proposal_only_contracts = build_proposal_only_method_contracts(pipeline_id=MAIN_PIPELINE_ID)
    registry_snapshot = build_processing_registry_snapshot(include_methods=include_methods)
    project_goal_capability_matrix = registry_snapshot.get(
        "project_goal_capability_matrix",
        {},
    )
    project_goal_capability_matrix_summary = (
        project_goal_capability_matrix.get("summary", {})
    )
    project_goal_capability_progress_snapshot = registry_snapshot.get(
        "project_goal_capability_progress_snapshot",
        {},
    )
    project_goal_capability_progress_snapshot_summary = (
        project_goal_capability_progress_snapshot.get("summary", {})
    )
    project_goal_capability_objective_alignment_snapshot = registry_snapshot.get(
        "project_goal_capability_objective_alignment_snapshot",
        {},
    )
    project_goal_capability_objective_alignment_summary = (
        project_goal_capability_objective_alignment_snapshot.get("summary", {})
    )
    autonomy_promotion_readiness = registry_snapshot.get(
        "autonomy_promotion_readiness",
        {},
    )
    autonomy_promotion_readiness_summary = autonomy_promotion_readiness.get(
        "summary",
        {},
    )
    autonomy_mode_transition_gate = registry_snapshot.get(
        "autonomy_mode_transition_gate",
        {},
    )
    autonomy_mode_transition_gate_summary = (
        autonomy_mode_transition_gate.get("summary", {})
    )
    autonomy_promotion_summary = autonomy_promotion_readiness.get("summary", {})
    autonomy_promotion_safety = autonomy_promotion_readiness.get("safety", {})
    autonomy_mode_transition_safety = autonomy_mode_transition_gate.get("safety", {})

    analysis_status = str(analysis_report.get("status") or "unknown")
    status = "ok"
    reason: Optional[str] = None
    if analysis_status == "no_audio":
        status = "no_audio"
        reason = "audio_capture_not_running"
    elif analysis_status == "clear" and not proposals:
        status = "clear"
        reason = "no_pipeline_preview_proposals"

    runtime = build_pipeline_runtime_summary(operator_mode, manual_write_supervision)
    return {
        "type": "main_pipeline_preview",
        "generated_at": _utc_now(),
        "status": status,
        "reason": reason,
        "pipeline_id": MAIN_PIPELINE_ID,
        "pipeline_status": profile.status if profile else "missing",
        "operator_mode": operator_mode,
        "runtime": runtime,
        "executor_policy": {
            "preview_only": True,
            "queue_mutation": False,
            "live_write": False,
            "live_apply_policy": "supervised_fader_gain_only",
            "apply_path": "apply_operator_proposal -> supervised_manual_gate",
            "executor_method_ids": list(profile.executor_method_ids if profile else ()),
            "allowed_requested_change_value_types": list(
                runtime.get("allowed_live_write_kinds", LIVE_ALLOWED_WRITE_KINDS)
            ),
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
            "project_goal_capability_matrix_row_count": int(
                project_goal_capability_matrix_summary.get("row_count", 0) or 0
            ),
            "project_goal_capability_progress_row_count": int(
                project_goal_capability_progress_snapshot_summary.get("row_count", 0)
                or 0
            ),
            "project_goal_capability_objective_required_row_count": int(
                project_goal_capability_objective_alignment_summary.get(
                    "required_row_count", 0
                )
                or 0
            ),
            "project_goal_capability_objective_present_row_count": int(
                project_goal_capability_objective_alignment_summary.get(
                    "present_row_count", 0
                )
                or 0
            ),
            "project_goal_capability_objective_missing_row_count": int(
                project_goal_capability_objective_alignment_summary.get(
                    "missing_row_count", 0
                )
                or 0
            ),
            "project_goal_capability_objective_readiness_score": int(
                project_goal_capability_objective_alignment_summary.get(
                    "autonomy_readiness_score",
                    0,
                )
                or 0
            ),
            "project_goal_capability_objective_readiness_ready_row_count": int(
                project_goal_capability_objective_alignment_summary.get(
                    "autonomy_readiness_ready_row_count",
                    0,
                )
                or 0
            ),
            "project_goal_capability_objective_readiness_gap_row_count": int(
                project_goal_capability_objective_alignment_summary.get(
                    "autonomy_readiness_gap_row_count",
                    0,
                )
                or 0
            ),
            "project_goal_capability_objective_readiness_goal_area_priorities": list(
                project_goal_capability_objective_alignment_summary.get(
                    "autonomy_readiness_goal_area_priorities",
                    [],
                )
                or []
            ),
            "project_goal_capability_objective_readiness_by_goal_area": dict(
                project_goal_capability_objective_alignment_summary.get(
                    "autonomy_readiness_by_goal_area",
                    {},
                )
                or {}
            ),
            "project_goal_capability_objective_readiness_hotspots": list(
                project_goal_capability_progress_snapshot_summary.get(
                    "project_goal_objective_autonomy_readiness_hotspots",
                    [],
                )
                or []
            ),
            "project_goal_capability_objective_blocked_live_row_count": int(
                project_goal_capability_objective_alignment_summary.get(
                    "blocked_live_row_count",
                    0,
                )
                or 0
            ),
            "project_goal_capability_objective_currently_live_writable_row_count": int(
                project_goal_capability_objective_alignment_summary.get(
                    "currently_live_writable_row_count",
                    0,
                )
                or 0
            ),
            "project_goal_capability_objective_full_autonomous_status": (
                project_goal_capability_objective_alignment_summary.get(
                    "full_autonomous_status"
                )
            ),
            "autonomy_promotion_target_count": int(
                autonomy_promotion_readiness_summary.get("target_count", 0) or 0
            ),
            "autonomy_mode_transition_gate_mode_count": int(
                autonomy_mode_transition_gate_summary.get("mode_count", 0) or 0
            ),
            "autonomy_mode_transition_gate_blocked_mode_count": int(
                len(autonomy_mode_transition_gate_summary.get("blocked_modes", ()) or [])
            ),
            "autonomy_promotion_full_autonomous_ready": bool(
                autonomy_promotion_safety.get("full_autonomous_ready", False)
            ),
            "autonomy_mode_transition_full_autonomous_ready": bool(
                autonomy_mode_transition_safety.get("full_autonomous_ready", False)
            ),
            "autonomy_promotion_supervised_live_allowed_target_count": int(
                len(autonomy_promotion_summary.get("supervised_live_allowed_target_ids", ()))
            ),
            "autonomy_promotion_blocked_target_count": int(
                len(autonomy_promotion_summary.get("blocked_target_ids", ()))
            ),
            "autonomy_mode_transition_gate_blocked_modes": list(
                autonomy_mode_transition_gate_summary.get("blocked_modes", ())
            ),
            "autonomy_mode_transition_gate_live_apply_allowed_modes": list(
                autonomy_mode_transition_gate_summary.get("live_apply_allowed_modes", ())
            ),
            "autonomy_mode_transition_gate_supervised_current_live_write_kinds": list(
                autonomy_mode_transition_gate_summary.get(
                    "supervised_current_live_write_kinds",
                    [],
                )
            ),
            "autonomy_mode_transition_gate_supervised_current_live_write_surfaces": list(
                autonomy_mode_transition_gate_summary.get(
                    "supervised_current_live_write_surfaces",
                    (),
                )
            ),
            "source_identity_calibration_memory_index_record_count": int(
                source_identity_memory_index_summary.get("memory_record_count", 0) or 0
            ),
            "source_identity_calibration_memory_index_invalid_record_count": int(
                source_identity_memory_index_summary.get("invalid_record_count", 0) or 0
            ),
            "operator_proposal_apply_sequence_policy_memory_index_record_count": int(
                sequence_policy_memory_index_summary.get("memory_record_count", 0) or 0
            ),
            "operator_proposal_apply_sequence_policy_memory_index_invalid_record_count": int(
                sequence_policy_memory_index_summary.get("invalid_record_count", 0) or 0
            ),
            "operator_proposal_apply_sequence_policy_memory_trust_review_candidate_count": int(
                sequence_policy_memory_trust_review_summary.get(
                    "trust_review_candidate_count",
                    0,
                )
                or 0
            ),
            "operator_proposal_apply_sequence_policy_memory_trust_acceptance_record_count": int(
                sequence_policy_memory_trust_acceptance_summary.get(
                    "acceptance_record_count",
                    0,
                )
                or 0
            ),
            "operator_proposal_apply_sequence_policy_memory_trust_accepted_record_count": int(
                sequence_policy_memory_trust_acceptance_summary.get(
                    "accepted_record_count",
                    0,
                )
                or 0
            ),
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_candidate_count": int(
                sequence_policy_memory_trust_promotion_summary.get(
                    "promotion_candidate_count",
                    0,
                )
                or 0
            ),
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_contract_record_count": int(
                sequence_policy_memory_trust_promotion_contract_summary.get(
                    "contract_record_count",
                    0,
                )
                or 0
            ),
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_record_count": int(
                sequence_policy_memory_trust_promotion_acceptance_summary.get(
                    "acceptance_record_count",
                    0,
                )
                or 0
            ),
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_accepted_record_count": int(
                sequence_policy_memory_trust_promotion_acceptance_summary.get(
                    "accepted_record_count",
                    0,
                )
                or 0
            ),
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_candidate_count": int(
                sequence_policy_memory_trust_promotion_execution_summary.get(
                    "execution_candidate_count",
                    0,
                )
                or 0
            ),
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run_record_count": int(
                sequence_policy_memory_trust_promotion_executor_dry_run_summary.get(
                    "dry_run_record_count",
                    0,
                )
                or 0
            ),
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_apply_result_count": int(
                sequence_policy_memory_trust_promotion_executor_summary.get(
                    "apply_result_count",
                    0,
                )
                or 0
            ),
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit_record_count": int(
                sequence_policy_memory_trust_promotion_apply_audit_summary.get(
                    "apply_audit_record_count",
                    0,
                )
                or 0
            ),
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verified_record_count": int(
                sequence_policy_memory_trust_promotion_rollback_verification_summary.get(
                    "rollback_verified_record_count",
                    0,
                )
                or 0
            ),
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate_record_count": int(
                sequence_policy_memory_trust_promotion_sequence_policy_gate_summary.get(
                    "sequence_policy_gate_record_count",
                    0,
                )
                or 0
            ),
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer_record_count": int(
                sequence_policy_memory_trust_promotion_policy_consumer_summary.get(
                    "consumer_record_count",
                    0,
                )
                or 0
            ),
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer_matched_proposal_count": len(
                sequence_policy_memory_trust_promotion_policy_consumer_summary.get(
                    "policy_consumer_proposal_ids",
                    [],
                )
                or []
            ),
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview_count": int(
                sequence_policy_memory_trust_promotion_ordering_preview_summary.get(
                    "ordered_preview_count",
                    0,
                )
                or 0
            ),
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview_ranked_proposal_count": int(
                sequence_policy_memory_trust_promotion_ordering_preview_summary.get(
                    "policy_ranked_proposal_count",
                    0,
                )
                or 0
            ),
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan_record_count": int(
                sequence_policy_memory_trust_promotion_queue_ordering_plan_summary.get(
                    "queue_ordering_plan_record_count",
                    0,
                )
                or 0
            ),
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan_ranked_record_count": int(
                sequence_policy_memory_trust_promotion_queue_ordering_plan_summary.get(
                    "ranked_queue_candidate_count",
                    0,
                )
                or 0
            ),
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer_spec_count": int(
                sequence_policy_memory_trust_promotion_queue_importer_summary.get(
                    "queue_import_spec_count",
                    0,
                )
                or 0
            ),
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer_result_count": int(
                sequence_policy_memory_trust_promotion_queue_importer_summary.get(
                    "queue_import_result_count",
                    0,
                )
                or 0
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
            "source_identity_calibration_memory_trust_promotion_executor_dry_run_record_count": int(
                source_identity_memory_trust_promotion_executor_dry_run_summary.get(
                    "dry_run_record_count",
                    0,
                )
                or 0
            ),
            "source_identity_calibration_memory_trust_promotion_executor_apply_result_count": int(
                source_identity_memory_trust_promotion_executor_summary.get(
                    "apply_result_count",
                    0,
                )
                or 0
            ),
            "source_identity_calibration_memory_trust_promotion_apply_audit_record_count": int(
                source_identity_memory_trust_promotion_apply_audit_summary.get(
                    "apply_audit_record_count",
                    0,
                )
                or 0
            ),
            "source_identity_calibration_memory_trust_promotion_rollback_verified_record_count": int(
                source_identity_memory_trust_promotion_rollback_verification_summary.get(
                    "rollback_verified_record_count",
                    0,
                )
                or 0
            ),
            "source_identity_calibration_memory_trust_promotion_priority_gate_record_count": int(
                source_identity_memory_trust_promotion_priority_gate_summary.get(
                    "priority_gate_record_count",
                    0,
                )
                or 0
            ),
            "source_identity_calibration_memory_trust_promotion_priority_eligible_record_count": int(
                source_identity_memory_trust_promotion_priority_gate_summary.get(
                    "eligible_record_count",
                    0,
                )
                or 0
            ),
            "source_identity_proposal_priority_policy_record_count": int(
                source_identity_proposal_priority_policy_summary.get(
                    "policy_record_count",
                    0,
                )
                or 0
            ),
            "source_identity_proposal_priority_policy_ranked_proposal_count": len(
                source_identity_proposal_priority_policy_summary.get(
                    "ranked_proposal_ids",
                    [],
                )
                or []
            ),
            "source_identity_proposal_ordering_preview_count": int(
                source_identity_proposal_ordering_preview_summary.get(
                    "ordered_preview_count",
                    0,
                )
                or 0
            ),
            "source_identity_proposal_ordering_preview_ranked_proposal_count": int(
                source_identity_proposal_ordering_preview_summary.get(
                    "policy_ranked_proposal_count",
                    0,
                )
                or 0
            ),
            "source_identity_proposal_queue_ordering_plan_record_count": int(
                source_identity_proposal_queue_ordering_plan_summary.get(
                    "queue_ordering_plan_record_count",
                    0,
                )
                or 0
            ),
            "source_identity_proposal_queue_ordering_plan_ranked_record_count": int(
                source_identity_proposal_queue_ordering_plan_summary.get(
                    "ranked_queue_candidate_count",
                    0,
                )
                or 0
            ),
            "source_identity_proposal_queue_importer_spec_count": int(
                source_identity_proposal_queue_importer_summary.get(
                    "queue_import_spec_count",
                    0,
                )
                or 0
            ),
            "source_identity_proposal_queue_importer_result_count": int(
                source_identity_proposal_queue_importer_summary.get(
                    "queue_import_result_count",
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
                "adapter_id": "operator_proposal_apply_sequence_policy_memory_index",
                "method_id": "operator_proposal_apply_sequence_policy_memory_index",
                "status": sequence_policy_memory_index.get("status"),
                "output": "operator_proposal_apply_sequence_append_only_memory_index_review",
                "reason": (
                    ", ".join(sequence_policy_memory_index_summary.get("blocked_by", []))
                    if sequence_policy_memory_index_summary.get("blocked_by")
                    else None
                ),
            },
            {
                "adapter_id": "operator_proposal_apply_sequence_policy_memory_trust_review",
                "method_id": "operator_proposal_apply_sequence_policy_memory_trust_review",
                "status": sequence_policy_memory_trust_review.get("status"),
                "output": "operator_proposal_apply_sequence_memory_trust_operator_review",
                "reason": (
                    ", ".join(
                        sequence_policy_memory_trust_review_summary.get("blocked_by", [])
                    )
                    if sequence_policy_memory_trust_review_summary.get("blocked_by")
                    else None
                ),
            },
            {
                "adapter_id": (
                    "operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger"
                ),
                "method_id": (
                    "operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger"
                ),
                "status": sequence_policy_memory_trust_acceptance_ledger.get("status"),
                "output": "operator_proposal_apply_sequence_memory_trust_acceptance_ledger",
                "reason": (
                    ", ".join(
                        sequence_policy_memory_trust_acceptance_summary.get(
                            "blocked_by",
                            [],
                        )
                    )
                    if sequence_policy_memory_trust_acceptance_summary.get("blocked_by")
                    else None
                ),
            },
            {
                "adapter_id": (
                    "operator_proposal_apply_sequence_policy_memory_trust_promotion_plan"
                ),
                "method_id": (
                    "operator_proposal_apply_sequence_policy_memory_trust_promotion_plan"
                ),
                "status": sequence_policy_memory_trust_promotion_plan.get("status"),
                "output": "operator_proposal_apply_sequence_memory_trust_promotion_plan",
                "reason": (
                    ", ".join(
                        sequence_policy_memory_trust_promotion_summary.get(
                            "blocked_by",
                            [],
                        )
                    )
                    if sequence_policy_memory_trust_promotion_summary.get("blocked_by")
                    else None
                ),
            },
            {
                "adapter_id": (
                    "operator_proposal_apply_sequence_policy_memory_trust_promotion_contract"
                ),
                "method_id": (
                    "operator_proposal_apply_sequence_policy_memory_trust_promotion_contract"
                ),
                "status": sequence_policy_memory_trust_promotion_contract.get("status"),
                "output": "operator_proposal_apply_sequence_memory_trust_promotion_contract_review",
                "reason": (
                    ", ".join(
                        sequence_policy_memory_trust_promotion_contract_summary.get(
                            "blocked_by",
                            [],
                        )
                    )
                    if sequence_policy_memory_trust_promotion_contract_summary.get(
                        "blocked_by"
                    )
                    else None
                ),
            },
            {
                "adapter_id": (
                    "operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger"
                ),
                "method_id": (
                    "operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger"
                ),
                "status": sequence_policy_memory_trust_promotion_acceptance_ledger.get(
                    "status"
                ),
                "output": (
                    "operator_proposal_apply_sequence_memory_trust_promotion_acceptance_ledger"
                ),
                "reason": (
                    ", ".join(
                        sequence_policy_memory_trust_promotion_acceptance_summary.get(
                            "blocked_by",
                            [],
                        )
                    )
                    if sequence_policy_memory_trust_promotion_acceptance_summary.get(
                        "blocked_by"
                    )
                    else None
                ),
            },
            {
                "adapter_id": (
                    "operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan"
                ),
                "method_id": (
                    "operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan"
                ),
                "status": sequence_policy_memory_trust_promotion_execution_plan.get(
                    "status"
                ),
                "output": (
                    "operator_proposal_apply_sequence_memory_trust_promotion_execution_plan"
                ),
                "reason": (
                    ", ".join(
                        sequence_policy_memory_trust_promotion_execution_summary.get(
                            "blocked_by",
                            [],
                        )
                    )
                    if sequence_policy_memory_trust_promotion_execution_summary.get(
                        "blocked_by"
                    )
                    else None
                ),
            },
            {
                "adapter_id": (
                    "operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run"
                ),
                "method_id": (
                    "operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run"
                ),
                "status": (
                    sequence_policy_memory_trust_promotion_executor_dry_run.get(
                        "status"
                    )
                ),
                "output": (
                    "operator_proposal_apply_sequence_memory_trust_promotion_executor_dry_run"
                ),
                "reason": (
                    ", ".join(
                        sequence_policy_memory_trust_promotion_executor_dry_run_summary.get(
                            "blocked_by",
                            [],
                        )
                    )
                    if sequence_policy_memory_trust_promotion_executor_dry_run_summary.get(
                        "blocked_by"
                    )
                    else None
                ),
            },
            {
                "adapter_id": (
                    "operator_proposal_apply_sequence_policy_memory_trust_promotion_executor"
                ),
                "method_id": (
                    "operator_proposal_apply_sequence_policy_memory_trust_promotion_executor"
                ),
                "status": sequence_policy_memory_trust_promotion_executor.get(
                    "status"
                ),
                "output": (
                    "operator_proposal_apply_sequence_memory_trust_promotion_executor"
                ),
                "reason": (
                    ", ".join(
                        sequence_policy_memory_trust_promotion_executor_summary.get(
                            "blocked_by",
                            [],
                        )
                    )
                    if sequence_policy_memory_trust_promotion_executor_summary.get(
                        "blocked_by"
                    )
                    else None
                ),
            },
            {
                "adapter_id": (
                    "operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit"
                ),
                "method_id": (
                    "operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit"
                ),
                "status": sequence_policy_memory_trust_promotion_apply_audit.get(
                    "status"
                ),
                "output": (
                    "operator_proposal_apply_sequence_memory_trust_promotion_apply_audit"
                ),
                "reason": (
                    ", ".join(
                        sequence_policy_memory_trust_promotion_apply_audit_summary.get(
                            "blocked_by",
                            [],
                        )
                    )
                    if sequence_policy_memory_trust_promotion_apply_audit_summary.get(
                        "blocked_by"
                    )
                    else None
                ),
            },
            {
                "adapter_id": (
                    "operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification"
                ),
                "method_id": (
                    "operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification"
                ),
                "status": (
                    sequence_policy_memory_trust_promotion_rollback_verification.get(
                        "status"
                    )
                ),
                "output": (
                    "operator_proposal_apply_sequence_memory_trust_promotion_rollback_verification"
                ),
                "reason": (
                    ", ".join(
                        sequence_policy_memory_trust_promotion_rollback_verification_summary.get(
                            "blocked_by",
                            [],
                        )
                    )
                    if sequence_policy_memory_trust_promotion_rollback_verification_summary.get(
                        "blocked_by"
                    )
                    else None
                ),
            },
            {
                "adapter_id": (
                    "operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate"
                ),
                "method_id": (
                    "operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate"
                ),
                "status": (
                    sequence_policy_memory_trust_promotion_sequence_policy_gate.get(
                        "status"
                    )
                ),
                "output": (
                    "operator_proposal_apply_sequence_memory_trust_promotion_sequence_policy_gate"
                ),
                "reason": (
                    ", ".join(
                        sequence_policy_memory_trust_promotion_sequence_policy_gate_summary.get(
                            "blocked_by",
                            [],
                        )
                    )
                    if sequence_policy_memory_trust_promotion_sequence_policy_gate_summary.get(
                        "blocked_by"
                    )
                    else None
                ),
            },
            {
                "adapter_id": (
                    "operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer"
                ),
                "method_id": (
                    "operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer"
                ),
                "status": (
                    sequence_policy_memory_trust_promotion_policy_consumer.get(
                        "status"
                    )
                ),
                "output": (
                    "operator_proposal_apply_sequence_memory_trust_promotion_policy_consumer"
                ),
                "reason": (
                    ", ".join(
                        sequence_policy_memory_trust_promotion_policy_consumer_summary.get(
                            "blocked_by",
                            [],
                        )
                    )
                    if sequence_policy_memory_trust_promotion_policy_consumer_summary.get(
                        "blocked_by"
                    )
                    else None
                ),
            },
            {
                "adapter_id": (
                    "operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview"
                ),
                "method_id": (
                    "operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview"
                ),
                "status": (
                    sequence_policy_memory_trust_promotion_ordering_preview.get(
                        "status"
                    )
                ),
                "output": (
                    "operator_proposal_apply_sequence_memory_trust_promotion_ordering_preview"
                ),
                "reason": (
                    ", ".join(
                        sequence_policy_memory_trust_promotion_ordering_preview_summary.get(
                            "blocked_by",
                            [],
                        )
                    )
                    if sequence_policy_memory_trust_promotion_ordering_preview_summary.get(
                        "blocked_by"
                    )
                    else None
                ),
            },
            {
                "adapter_id": (
                    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan"
                ),
                "method_id": (
                    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan"
                ),
                "status": (
                    sequence_policy_memory_trust_promotion_queue_ordering_plan.get(
                        "status"
                    )
                ),
                "output": (
                    "operator_proposal_apply_sequence_memory_trust_promotion_queue_ordering_plan"
                ),
                "reason": (
                    ", ".join(
                        sequence_policy_memory_trust_promotion_queue_ordering_plan_summary.get(
                            "blocked_by",
                            [],
                        )
                    )
                    if sequence_policy_memory_trust_promotion_queue_ordering_plan_summary.get(
                        "blocked_by"
                    )
                    else None
                ),
            },
            {
                "adapter_id": (
                    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer"
                ),
                "method_id": (
                    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer"
                ),
                "status": (
                    sequence_policy_memory_trust_promotion_queue_importer.get(
                        "status"
                    )
                ),
                "output": (
                    "operator_proposal_apply_sequence_memory_trust_promotion_queue_importer_dry_run"
                ),
                "reason": (
                    ", ".join(
                        sequence_policy_memory_trust_promotion_queue_importer_summary.get(
                            "blocked_by",
                            [],
                        )
                    )
                    if sequence_policy_memory_trust_promotion_queue_importer_summary.get(
                        "blocked_by"
                    )
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
                "adapter_id": (
                    "source_identity_calibration_memory_trust_promotion_executor_dry_run"
                ),
                "method_id": (
                    "source_identity_calibration_memory_trust_promotion_executor_dry_run"
                ),
                "status": (
                    source_identity_memory_trust_promotion_executor_dry_run.get(
                        "status"
                    )
                ),
                "output": "source_identity_memory_trust_promotion_executor_dry_run",
                "reason": (
                    ", ".join(
                        source_identity_memory_trust_promotion_executor_dry_run_summary.get(
                            "blocked_by",
                            [],
                        )
                    )
                    if source_identity_memory_trust_promotion_executor_dry_run_summary.get(
                        "blocked_by"
                    )
                    else None
                ),
            },
            {
                "adapter_id": (
                    "source_identity_calibration_memory_trust_promotion_executor"
                ),
                "method_id": (
                    "source_identity_calibration_memory_trust_promotion_executor"
                ),
                "status": source_identity_memory_trust_promotion_executor.get("status"),
                "output": "source_identity_memory_trust_promotion_executor",
                "reason": (
                    ", ".join(
                        source_identity_memory_trust_promotion_executor_summary.get(
                            "blocked_by",
                            [],
                        )
                    )
                    if source_identity_memory_trust_promotion_executor_summary.get(
                        "blocked_by"
                    )
                    else None
                ),
            },
            {
                "adapter_id": (
                    "source_identity_calibration_memory_trust_promotion_apply_audit"
                ),
                "method_id": (
                    "source_identity_calibration_memory_trust_promotion_apply_audit"
                ),
                "status": source_identity_memory_trust_promotion_apply_audit.get(
                    "status"
                ),
                "output": "source_identity_memory_trust_promotion_apply_audit",
                "reason": (
                    ", ".join(
                        source_identity_memory_trust_promotion_apply_audit_summary.get(
                            "blocked_by",
                            [],
                        )
                    )
                    if source_identity_memory_trust_promotion_apply_audit_summary.get(
                        "blocked_by"
                    )
                    else None
                ),
            },
            {
                "adapter_id": (
                    "source_identity_calibration_memory_trust_promotion_rollback_verification"
                ),
                "method_id": (
                    "source_identity_calibration_memory_trust_promotion_rollback_verification"
                ),
                "status": (
                    source_identity_memory_trust_promotion_rollback_verification.get(
                        "status"
                    )
                ),
                "output": (
                    "source_identity_memory_trust_promotion_rollback_verification"
                ),
                "reason": (
                    ", ".join(
                        source_identity_memory_trust_promotion_rollback_verification_summary.get(
                            "blocked_by",
                            [],
                        )
                    )
                    if source_identity_memory_trust_promotion_rollback_verification_summary.get(
                        "blocked_by"
                    )
                    else None
                ),
            },
            {
                "adapter_id": (
                    "source_identity_calibration_memory_trust_promotion_priority_gate"
                ),
                "method_id": (
                    "source_identity_calibration_memory_trust_promotion_priority_gate"
                ),
                "status": (
                    source_identity_memory_trust_promotion_priority_gate.get("status")
                ),
                "output": "source_identity_memory_trust_promotion_priority_gate",
                "reason": (
                    ", ".join(
                        source_identity_memory_trust_promotion_priority_gate_summary.get(
                            "blocked_by",
                            [],
                        )
                    )
                    if source_identity_memory_trust_promotion_priority_gate_summary.get(
                        "blocked_by"
                    )
                    else None
                ),
            },
            {
                "adapter_id": "source_identity_proposal_priority_policy",
                "method_id": "source_identity_proposal_priority_policy",
                "status": source_identity_proposal_priority_policy.get("status"),
                "output": "source_identity_proposal_priority_policy_projection",
                "reason": (
                    ", ".join(
                        source_identity_proposal_priority_policy_summary.get(
                            "blocked_by",
                            [],
                        )
                    )
                    if source_identity_proposal_priority_policy_summary.get("blocked_by")
                    else None
                ),
            },
            {
                "adapter_id": "source_identity_proposal_ordering_preview",
                "method_id": "source_identity_proposal_ordering_preview",
                "status": source_identity_proposal_ordering_preview.get("status"),
                "output": "source_identity_proposal_ordering_preview",
                "reason": (
                    ", ".join(
                        source_identity_proposal_ordering_preview_summary.get(
                            "blocked_by",
                            [],
                        )
                    )
                    if source_identity_proposal_ordering_preview_summary.get("blocked_by")
                    else None
                ),
            },
            {
                "adapter_id": "source_identity_proposal_queue_ordering_plan",
                "method_id": "source_identity_proposal_queue_ordering_plan",
                "status": source_identity_proposal_queue_ordering_plan.get("status"),
                "output": "source_identity_proposal_queue_ordering_plan",
                "reason": (
                    ", ".join(
                        source_identity_proposal_queue_ordering_plan_summary.get(
                            "blocked_by",
                            [],
                        )
                    )
                    if source_identity_proposal_queue_ordering_plan_summary.get(
                        "blocked_by"
                    )
                    else None
                ),
            },
            {
                "adapter_id": "source_identity_proposal_queue_importer",
                "method_id": "source_identity_proposal_queue_importer",
                "status": source_identity_proposal_queue_importer.get("status"),
                "output": "source_identity_proposal_queue_importer_dry_run",
                "reason": (
                    ", ".join(
                        source_identity_proposal_queue_importer_summary.get(
                            "blocked_by",
                            [],
                        )
                    )
                    if source_identity_proposal_queue_importer_summary.get("blocked_by")
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
        "operator_proposal_apply_sequence_policy_memory_index": (
            sequence_policy_memory_index
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_review": (
            sequence_policy_memory_trust_review
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger": (
            sequence_policy_memory_trust_acceptance_ledger
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_plan": (
            sequence_policy_memory_trust_promotion_plan
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_contract": (
            sequence_policy_memory_trust_promotion_contract
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger": (
            sequence_policy_memory_trust_promotion_acceptance_ledger
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan": (
            sequence_policy_memory_trust_promotion_execution_plan
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run": (
            sequence_policy_memory_trust_promotion_executor_dry_run
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_executor": (
            sequence_policy_memory_trust_promotion_executor
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit": (
            sequence_policy_memory_trust_promotion_apply_audit
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification": (
            sequence_policy_memory_trust_promotion_rollback_verification
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate": (
            sequence_policy_memory_trust_promotion_sequence_policy_gate
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer": (
            sequence_policy_memory_trust_promotion_policy_consumer
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview": (
            sequence_policy_memory_trust_promotion_ordering_preview
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan": (
            sequence_policy_memory_trust_promotion_queue_ordering_plan
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer": (
            sequence_policy_memory_trust_promotion_queue_importer
        ),
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
        "source_identity_calibration_memory_trust_promotion_executor_dry_run": (
            source_identity_memory_trust_promotion_executor_dry_run
        ),
        "source_identity_calibration_memory_trust_promotion_executor": (
            source_identity_memory_trust_promotion_executor
        ),
        "source_identity_calibration_memory_trust_promotion_apply_audit": (
            source_identity_memory_trust_promotion_apply_audit
        ),
        "source_identity_calibration_memory_trust_promotion_rollback_verification": (
            source_identity_memory_trust_promotion_rollback_verification
        ),
        "source_identity_calibration_memory_trust_promotion_priority_gate": (
            source_identity_memory_trust_promotion_priority_gate
        ),
        "source_identity_proposal_priority_policy": (
            source_identity_proposal_priority_policy
        ),
        "source_identity_proposal_ordering_preview": (
            source_identity_proposal_ordering_preview
        ),
        "source_identity_proposal_queue_ordering_plan": (
            source_identity_proposal_queue_ordering_plan
        ),
        "source_identity_proposal_queue_importer": (
            source_identity_proposal_queue_importer
        ),
        "registry": {
            "summary": registry_snapshot.get("summary", {}),
            "pipeline_profiles": registry_snapshot.get("pipeline_profiles", []),
            "methods": registry_snapshot.get("methods", []),
            "project_goal_capability_matrix": registry_snapshot.get(
                "project_goal_capability_matrix",
                {},
            ),
            "project_goal_capability_progress_snapshot": registry_snapshot.get(
                "project_goal_capability_progress_snapshot",
                {},
            ),
            "project_goal_capability_objective_alignment_snapshot": (
                registry_snapshot.get(
                    "project_goal_capability_objective_alignment_snapshot",
                    {},
                )
            ),
            "autonomy_promotion_readiness": registry_snapshot.get(
                "autonomy_promotion_readiness",
                {},
            ),
            "autonomy_mode_transition_gate": registry_snapshot.get(
                "autonomy_mode_transition_gate",
                {},
            ),
        },
    }
