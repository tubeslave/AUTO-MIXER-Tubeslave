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
PROPOSAL_ONLY_BLOCKED_DOMAINS = {
    "eq",
    "dynamics",
    "gate",
    "pan",
    "phase",
    "fx",
    "master_bus",
    "bus_group",
    "mute",
    "routing",
    "scene_preset",
    "mix_variant",
    "adaptation",
    "source_identity",
    "arrangement",
    "decision_evidence",
    "memory_learning",
}
BLOCKED_METHOD_STATUSES = {"quarantined", "placeholder", "deprecated"}
ALLOWED_REQUESTED_CHANGE_TYPES = set(LIVE_ALLOWED_WRITE_KINDS)
PROPOSAL_ONLY_CONTRACTS = {
    "auto_eq": {
        "value_type": "eq",
        "target_kind": "channel_eq_band",
        "old_value_source": "cached_or_physical_eq_band_read_required_before_promotion",
        "safety_evidence": ["frequency_match", "cut_or_bounded_boost", "feedback_notch_preserved", "true_peak_headroom"],
    },
    "corrective_eq_proposal": {
        "value_type": "eq",
        "target_kind": "channel_eq_band",
        "old_value_source": "cached_or_physical_eq_band_read_required_before_promotion",
        "safety_evidence": [
            "band_identity",
            "frequency_match",
            "protected_band_policy",
            "filter_type_policy",
            "feedback_notch_preserved",
            "manual_band_preserved",
            "bounded_gain_delta",
            "true_peak_headroom",
            "current_eq_context_cache_freshness",
            "current_eq_context_cache_no_live_authority",
            "current_phase_audio_context_cache_freshness",
            "current_phase_audio_context_cache_no_live_authority",
            "source_identity_downstream_trust",
            "old_value_source",
            "rollback_eq_band_readback",
        ],
    },
    "cross_adaptive_eq": {
        "value_type": "eq",
        "target_kind": "masking_eq_band",
        "old_value_source": "cached_or_physical_eq_band_read_required_before_promotion",
        "safety_evidence": [
            "masking_pair",
            "linked_cut_boost_pair",
            "frequency_match",
            "bounded_gain_delta",
            "current_masking_context_cache_freshness",
            "source_identity_downstream_trust",
            "sum_bus_check",
            "rollback_eq_band_readback",
        ],
    },
    "frequency_separation_proposal": {
        "value_type": "frequency_split",
        "target_kind": "source_stem_frequency_lane",
        "old_value_source": "cached_or_physical_eq_band_and_stem_spectrum_read_required_before_promotion",
        "safety_evidence": [
            "current_frequency_context_cache_freshness",
            "source_stem_contribution",
            "lane_conflict_pair",
            "bounded_subtractive_delta",
            "sum_bus_check",
            "source_identity_downstream_trust",
            "rollback_eq_band_readback",
        ],
    },
    "source_identity_confidence_proposal": {
        "value_type": "source_identity",
        "target_kind": "source_identity_review",
        "old_value_source": "classifier_output_and_operator_accepted_label_required_before_promotion",
        "safety_evidence": [
            "classifier_source",
            "match_type",
            "source_type_confidence",
            "alternative_margin",
            "current_source_identity_context_cache_freshness",
            "operator_acceptance",
            "downstream_rejection_reasons",
            "replay_outcome",
        ],
    },
    "source_identity_replay_calibration": {
        "value_type": "source_identity_calibration",
        "target_kind": "source_identity_replay_trust_state",
        "old_value_source": "accepted_rejected_source_identity_replay_outcomes_required_before_promotion",
        "safety_evidence": [
            "accepted_rejected_outcomes",
            "acceptance_rate",
            "replay_match_rate",
            "average_predicted_confidence",
            "current_source_identity_replay_context_cache_freshness",
            "safety_regression_count",
            "operator_acceptance",
            "downstream_trust_state",
        ],
    },
    "auto_compressor": {
        "value_type": "compressor",
        "target_kind": "channel_dynamics",
        "old_value_source": "cached_or_physical_dynamics_read_required_before_promotion",
        "safety_evidence": ["gain_reduction_estimate", "makeup_headroom", "true_peak_headroom", "bypass_state"],
    },
    "compressor_dynamics_proposal": {
        "value_type": "compressor",
        "target_kind": "channel_dynamics",
        "old_value_source": "cached_or_physical_dynamics_read_required_before_promotion",
        "safety_evidence": [
            "crest_factor",
            "loudness_range",
            "gain_reduction_estimate",
            "bounded_makeup_headroom",
            "true_peak_headroom",
            "feedback_risk",
            "current_dynamics_cache_freshness",
            "current_dynamics_cache_no_live_authority",
            "source_identity_downstream_trust",
            "old_value_source",
            "rollback_dynamics_readback",
            "bypass_state",
        ],
    },
    "deesser_vocal_sibilance": {
        "value_type": "deesser",
        "target_kind": "channel_deesser_or_dynamic_eq",
        "old_value_source": "cached_or_physical_dynamics_or_dynamic_eq_read_required_before_promotion",
        "safety_evidence": [
            "vocal_source_role",
            "sibilance_ratio",
            "sibilance_persistence",
            "current_dynamics_or_dynamic_eq_cache_freshness",
            "source_identity_downstream_trust",
            "true_peak_headroom",
        ],
    },
    "auto_gate_caig": {
        "value_type": "gate",
        "target_kind": "channel_gate",
        "old_value_source": "cached_or_physical_gate_read_required_before_promotion",
        "safety_evidence": [
            "noise_floor",
            "hysteresis",
            "source_role",
            "bleed_risk",
            "current_gate_bleed_context_cache_freshness",
            "current_gate_bleed_context_cache_no_live_authority",
            "source_identity_downstream_trust",
        ],
    },
    "context_aware_gate_agent": {
        "value_type": "gate",
        "target_kind": "channel_gate_or_expander",
        "old_value_source": "cached_or_physical_gate_read_required_before_promotion",
        "safety_evidence": [
            "bleed_telemetry",
            "false_open_risk",
            "false_close_risk",
            "chatter_risk",
            "tail_risk",
            "chatter_guard",
            "current_gate_bleed_context_cache_freshness",
            "current_gate_bleed_context_cache_no_live_authority",
            "source_identity_downstream_trust",
            "hysteresis",
            "old_value_source",
            "rollback_gate_readback",
        ],
    },
    "auto_panner": {
        "value_type": "pan",
        "target_kind": "channel_pan",
        "old_value_source": "cached_or_physical_pan_read_required_before_promotion",
        "safety_evidence": ["stereo_pair", "mono_compatibility", "operator_context"],
    },
    "pan_spatial_proposal": {
        "value_type": "pan",
        "target_kind": "channel_pan",
        "old_value_source": "cached_or_physical_pan_read_required_before_promotion",
        "safety_evidence": [
            "source_role",
            "stereo_pair",
            "stage_position",
            "mono_compatibility",
            "phase_correlation",
            "mono_phase_safety",
            "current_pan",
            "current_pan_cache_freshness",
            "current_pan_cache_no_live_authority",
            "mono_phase_check_no_live_authority",
            "source_identity_downstream_trust",
            "old_value_source",
            "rollback_pan_readback",
        ],
    },
    "phase_alignment": {
        "value_type": "phase_delay",
        "target_kind": "channel_delay_or_polarity",
        "old_value_source": "cached_or_physical_delay_polarity_read_required_before_promotion",
        "safety_evidence": ["correlation_delta", "gcc_phat_confidence", "polarity_readback", "delay_readback"],
    },
    "phase_aware_eq_guard": {
        "value_type": "phase_eq_guard",
        "target_kind": "group_phase_before_eq_policy",
        "old_value_source": "fresh_multichannel_phase_audio_context_required_before_promotion",
        "safety_evidence": [
            "current_phase_audio_context_cache_freshness",
            "group_phase_measurements",
            "gcc_phat_confidence",
            "low_frequency_sum_loss",
            "bleed_risk",
            "eq_hold_policy",
        ],
    },
    "auto_fx": {
        "value_type": "fx_send",
        "target_kind": "send_or_fx_parameter",
        "old_value_source": "cached_or_physical_fx_read_required_before_promotion",
        "safety_evidence": ["send_level_limit", "fx_return_headroom", "dry_wet_bounds", "feedback_risk"],
    },
    "fx_send_proposal": {
        "value_type": "fx_send",
        "target_kind": "send_or_fx_parameter",
        "old_value_source": "cached_or_physical_fx_read_required_before_promotion",
        "safety_evidence": [
            "source_role",
            "send_level_limit",
            "fx_return_headroom",
            "feedback_risk",
            "intelligibility_risk",
            "dry_wet_bounds",
            "current_fx_send_return_cache_freshness",
            "current_fx_parameter_schema_hints",
            "source_identity_downstream_trust",
        ],
    },
    "master_bus_optimization_proposal": {
        "value_type": "master_bus",
        "target_kind": "main_bus_processing",
        "old_value_source": "cached_or_physical_main_bus_processing_read_required_before_promotion",
        "safety_evidence": [
            "integrated_lufs",
            "true_peak_headroom",
            "crest_factor",
            "loudness_range",
            "spectral_tilt",
            "reference_profile",
            "limiter_or_dynamics_proof",
            "current_main_bus_processing_cache_freshness",
        ],
    },
    "bus_group_balance_proposal": {
        "value_type": "bus_group_balance",
        "target_kind": "bus_group_dca_matrix_fader",
        "old_value_source": "cached_or_physical_bus_group_dca_matrix_fader_read_required_before_promotion",
        "safety_evidence": [
            "ayaic_target_plane",
            "integrated_lufs",
            "summed_child_lufs",
            "assignment_readback",
            "current_bus_fader",
            "current_bus_group_dca_matrix_cache_freshness",
            "true_peak_headroom",
            "main_bus_headroom",
            "feedback_risk",
            "source_identity_downstream_trust",
        ],
    },
    "mute_management_proposal": {
        "value_type": "mute_state",
        "target_kind": "channel_mute_state",
        "old_value_source": "cached_or_physical_channel_mute_read_required_before_promotion",
        "safety_evidence": [
            "source_role",
            "current_mute_state",
            "current_mute_cache_freshness",
            "auto_muted_by_system",
            "signal_activity",
            "idle_duration",
            "operator_intent",
            "dropout_risk",
            "feedback_risk",
            "source_identity_downstream_trust",
            "rollback_value",
        ],
    },
    "routing_management_proposal": {
        "value_type": "routing",
        "target_kind": "route_diff",
        "old_value_source": "cached_or_physical_route_read_required_before_promotion",
        "safety_evidence": [
            "route_diff",
            "current_route_readback",
            "current_route_cache_freshness",
            "proposed_source_readback",
            "rollback_route",
            "operator_intent",
            "active_performance_guard",
            "destructive_scope_check",
            "current_route_activity",
            "source_identity_downstream_trust",
        ],
    },
    "scene_preset_management_proposal": {
        "value_type": "scene_preset",
        "target_kind": "scoped_scene_or_user_preset",
        "old_value_source": "cached_or_physical_full_state_read_required_before_promotion",
        "safety_evidence": [
            "current_console_state_readback",
            "current_scene_preset_cache_freshness",
            "snapshots_scenes_snapshot_cache_freshness",
            "user_presets_snapshot_cache_freshness",
            "target_preview",
            "scoped_diff",
            "read_only_scoped_diff_evidence",
            "rollback_snapshot",
            "read_only_rollback_evidence",
            "operator_intent",
            "active_performance_guard",
            "destructive_scope_check",
            "active_signal_guard",
        ],
    },
    "mix_variant_profile_proposal": {
        "value_type": "mix_variant_profile",
        "target_kind": "genre_task_mix_profile",
        "old_value_source": "operator_selected_or_current_mix_profile_read_required_before_promotion",
        "safety_evidence": [
            "genre_confidence",
            "task_confidence",
            "arrangement_context",
            "current_mix_variant_context_cache_freshness",
            "reference_metrics",
            "ayaic_level_planes",
            "source_identity_downstream_trust",
            "true_peak_headroom",
            "feedback_risk",
            "no_named_artist_style_clone",
            "followup_proposal_methods",
        ],
    },
    "performance_adaptation_proposal": {
        "value_type": "performance_adaptation",
        "target_kind": "channel_or_bus_level_trend",
        "old_value_source": "cached_or_physical_level_readback_required_before_promotion",
        "safety_evidence": [
            "baseline_level",
            "current_level_readback",
            "current_performance_context_cache_freshness",
            "level_drift",
            "trend_persistence",
            "cooldown",
            "true_peak_headroom",
            "feedback_risk",
            "manual_override_guard",
            "rollback_state",
            "latency_budget",
            "source_identity_downstream_trust",
        ],
    },
    "arrangement_role_proposal": {
        "value_type": "arrangement_role",
        "target_kind": "source_arrangement_role",
        "old_value_source": "current_or_operator_accepted_role_required_before_promotion",
        "safety_evidence": [
            "source_type_confidence",
            "role_confidence",
            "arrangement_context",
            "current_arrangement_role_context_cache_freshness",
            "inter_channel_context",
            "reference_role",
            "accepted_history",
            "role_persistence",
            "role_ambiguity",
            "masking_impact",
            "source_identity_downstream_trust",
        ],
    },
    "decision_evidence_packet_proposal": {
        "value_type": "decision_evidence_packet",
        "target_kind": "proposal_or_shadow_decision_evidence",
        "old_value_source": "current_analysis_trace_and_operator_review_required_before_promotion",
        "safety_evidence": [
            "audio_analysis",
            "psychoacoustic_model",
            "statistical_model",
            "project_history",
            "reference_comparison",
            "previous_decision_outcomes",
            "inter_channel_context",
            "arrangement_context",
            "genre_context",
            "current_decision_evidence_context_cache_freshness",
            "source_identity_downstream_trust",
            "readback_state",
            "rollback_plan",
            "true_peak_headroom",
            "feedback_risk",
        ],
    },
    "long_term_memory_learning_proposal": {
        "value_type": "long_term_memory_learning",
        "target_kind": "decision_memory_learning_record",
        "old_value_source": "verified_decision_trace_or_offline_replay_required_before_promotion",
        "safety_evidence": [
            "verified_decisions",
            "offline_replay_runs",
            "accepted_rejected_outcomes",
            "decision_evidence_packets",
            "telemetry_sessions",
            "engineer_actions",
            "current_memory_learning_context_cache_freshness",
            "source_identity_downstream_trust",
            "reference_comparisons",
            "genre_context",
            "arrangement_context",
            "replay_correlation",
            "outcome_consistency",
            "safety_regressions",
            "manual_overrides",
        ],
    },
    "routing_snapshot_reset_writes": {
        "value_type": "routing",
        "target_kind": "route_or_snapshot_state",
        "old_value_source": "full_route_snapshot_required_before_promotion",
        "safety_evidence": ["route_diff", "destructive_state_check", "rollback_snapshot", "operator_confirmation"],
    },
    "fx_direct_apply": {
        "value_type": "fx_parameter",
        "target_kind": "fx_rack_or_insert",
        "old_value_source": "cached_or_physical_fx_read_required_before_promotion",
        "safety_evidence": ["insert_bypass_state", "parameter_bounds", "return_headroom", "rollback_value"],
    },
}
AGENTIC_FOUNDATION_SAFE_ACTION_TYPES = {
    "gain_trim": "gain",
    "fader": "fader",
}
AGENTIC_FOUNDATION_MAX_DOWN_STEP_DB = {
    "gain": 0.5,
    "fader": 1.0,
}


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


def adapt_agentic_foundation_gain_fader_recommendations(
    recommendations: Iterable[Dict[str, Any]],
    *,
    pipeline_id: str = MAIN_PIPELINE_ID,
) -> List[Dict[str, Any]]:
    """Adapt bounded foundation gain/fader-down recommendations for preview.

    The foundation layer is dry-run by design. This adapter only exposes
    recommendations that are already safe in direction and size: known current
    value, known proposed value, negative delta, and fader/gain value type.
    Queue import still stays disabled by the Product Layer preview contract.
    """

    outputs: List[Dict[str, Any]] = []
    for recommendation in recommendations:
        if not isinstance(recommendation, dict):
            continue
        action_type = str(recommendation.get("action_type") or "").strip()
        value_type = AGENTIC_FOUNDATION_SAFE_ACTION_TYPES.get(action_type)
        if not value_type:
            continue
        channel = _coerce_int(recommendation.get("channel_id") or recommendation.get("channel"))
        current_value = _coerce_float(recommendation.get("current_value"))
        proposed_value = _coerce_float(recommendation.get("suggested_value"))
        if channel is None or current_value is None or proposed_value is None:
            continue
        delta_db = round(proposed_value - current_value, 1)
        max_step = AGENTIC_FOUNDATION_MAX_DOWN_STEP_DB[value_type]
        if delta_db >= 0.0 or abs(delta_db) > max_step:
            continue

        title = (
            f"Agentic foundation: channel {channel} {value_type} "
            f"{current_value:.1f} -> {proposed_value:.1f} dB"
        )
        proposal = {
            "id": recommendation.get("idempotency_key") or f"agentic_foundation:{value_type}:{channel}",
            "title": title,
            "target": channel,
            "channel": channel,
            "kind": value_type,
            "severity": "medium",
            "confidence": recommendation.get("confidence"),
            "source": "agentic_method_foundation",
            "reason": recommendation.get("reason") or "Foundation produced a bounded gain/fader-down recommendation.",
            "requested_change": {
                "value_type": value_type,
                "channel": channel,
                "value": round(proposed_value, 1),
                "current_value": round(current_value, 1),
                "old_value": round(current_value, 1),
                "proposed_value": round(proposed_value, 1),
                "rollback_value": round(current_value, 1),
                "delta_db": delta_db,
                "readback_required": True,
                "safety_evidence": [
                    {"label": "Source", "value": "agentic_foundation_v1"},
                    {"label": "Direction", "value": f"{value_type}_down_only"},
                    {"label": "Max step dB", "value": max_step},
                    {"label": "Dry run", "value": True},
                ],
            },
            "raw": {
                "rule": "agentic_foundation_bounded_downward_preview",
                "action_type": action_type,
                "dry_run": recommendation.get("dry_run", True),
                "safety_status": recommendation.get("safety_status"),
                "risk_score": recommendation.get("risk_score"),
                "evidence": recommendation.get("evidence", {}),
                "idempotency_key": recommendation.get("idempotency_key"),
                "live_write": False,
                "queue_mutation": False,
            },
        }
        output = _proposal_adapter_output(
            proposal,
            adapter_id="agentic_foundation_gain_fader",
            method_id="agentic_method_foundation",
            pipeline_id=pipeline_id,
        )
        output["can_import_to_queue"] = False
        output["queue_action"] = "preview_only_not_queued"
        output.setdefault("blockers", [])
        output["blockers"] = sorted(set([*output["blockers"], "foundation_preview_requires_explicit_promotion"]))
        outputs.append(output)
    return outputs


def adapt_source_identity_confidence_proposals(
    proposals: Iterable[Dict[str, Any]],
    *,
    pipeline_id: str = MAIN_PIPELINE_ID,
) -> List[Dict[str, Any]]:
    """Adapt source-identity review proposals for preview-only display."""

    outputs: List[Dict[str, Any]] = []
    for proposal in proposals:
        if not isinstance(proposal, dict):
            continue
        evidence = proposal.get("evidence") if isinstance(proposal.get("evidence"), dict) else {}
        proposed_action = (
            proposal.get("proposed_action")
            if isinstance(proposal.get("proposed_action"), dict)
            else {}
        )
        blockers = {
            "unsupported_value_type:source_identity",
            "not_in_supervised_fader_gain_executor",
            "source_identity_requires_operator_acceptance",
        }
        raw_blockers = proposal.get("blockers")
        if isinstance(raw_blockers, list):
            blockers.update(str(item) for item in raw_blockers)
        review_reasons = proposed_action.get("review_reasons")
        if isinstance(review_reasons, list):
            blockers.update(str(item) for item in review_reasons)
        downstream_reasons = proposed_action.get("downstream_rejection_reasons")
        if isinstance(downstream_reasons, list):
            blockers.update(str(item) for item in downstream_reasons)

        outputs.append(
            {
                "id": f"{pipeline_id}:source_identity_confidence:{proposal.get('id')}",
                "adapter_id": "source_identity_confidence",
                "pipeline_id": pipeline_id,
                **_method_contract("source_identity_confidence_proposal"),
                "title": str(proposal.get("title") or "Source identity review"),
                "reason": str(
                    proposal.get("reason")
                    or "Classifier confidence requires operator review before downstream processing."
                ),
                "confidence": proposal.get("confidence"),
                "risk": _risk_from_severity(proposal.get("severity")),
                "target": proposal.get("target"),
                "queue_action": "preview_only_not_queued",
                "can_import_to_queue": False,
                "can_apply_live": False,
                "blockers": sorted(blockers),
                "proposal_priority": proposal.get("proposal_priority"),
                "proposed_action": proposed_action,
                "evidence": {
                    **evidence,
                    "live_executor": False,
                    "requested_change_allowed": False,
                    "queue_mutation": False,
                },
                "source_proposal_id": proposal.get("id"),
            }
        )
    return outputs


def adapt_source_identity_replay_calibration_proposals(
    proposals: Iterable[Dict[str, Any]],
    *,
    pipeline_id: str = MAIN_PIPELINE_ID,
) -> List[Dict[str, Any]]:
    """Adapt replay-calibration proposals for preview-only operator review."""

    outputs: List[Dict[str, Any]] = []
    for proposal in proposals:
        if not isinstance(proposal, dict):
            continue
        evidence = proposal.get("evidence") if isinstance(proposal.get("evidence"), dict) else {}
        proposed_action = (
            proposal.get("proposed_action")
            if isinstance(proposal.get("proposed_action"), dict)
            else {}
        )
        blockers = {
            "unsupported_value_type:source_identity_calibration",
            "not_in_supervised_fader_gain_executor",
            "calibration_requires_operator_review",
            "source_identity_replay_calibration_requires_operator_acceptance",
        }
        raw_blockers = proposal.get("blockers")
        if isinstance(raw_blockers, list):
            blockers.update(str(item) for item in raw_blockers)

        outputs.append(
            {
                "id": f"{pipeline_id}:source_identity_replay_calibration:{proposal.get('id')}",
                "adapter_id": "source_identity_replay_calibration",
                "pipeline_id": pipeline_id,
                **_method_contract("source_identity_replay_calibration"),
                "title": str(proposal.get("title") or "Source identity replay calibration"),
                "reason": str(
                    proposal.get("reason")
                    or "Replay-calibrated source identity trust requires operator review before downstream priority changes."
                ),
                "confidence": proposal.get("confidence"),
                "risk": _risk_from_severity(proposal.get("severity")),
                "target": proposal.get("target"),
                "queue_action": "preview_only_not_queued",
                "can_import_to_queue": False,
                "can_apply_live": False,
                "blockers": sorted(blockers),
                "proposed_action": proposed_action,
                "evidence": {
                    **evidence,
                    "live_executor": False,
                    "requested_change_allowed": False,
                    "queue_mutation": False,
                    "model_mutation": False,
                    "memory_mutation": False,
                    "downstream_priority_mutation": False,
                },
                "source_proposal_id": proposal.get("id"),
            }
        )
    return outputs


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


def build_proposal_only_method_contracts(*, pipeline_id: str = MAIN_PIPELINE_ID) -> List[Dict[str, Any]]:
    """Expose future correction domains as display-safe proposal contracts.

    These contracts intentionally stay separate from queueable fader/gain
    proposals. They describe the minimum evidence needed before a domain can be
    promoted, but they do not create live apply requests.
    """

    contracts: List[Dict[str, Any]] = []
    for method_id, contract in PROPOSAL_ONLY_CONTRACTS.items():
        method = get_processing_method(method_id)
        if method is None:
            continue

        value_type = str(contract["value_type"])
        promotion_next = method.promotion_next or "proposal_adapter_and_hardware_proof_required"
        blockers = [
            f"unsupported_value_type:{value_type}",
            "not_in_supervised_fader_gain_executor",
            "hardware_readback_and_rollback_not_proven",
        ]
        if method.status in BLOCKED_METHOD_STATUSES:
            blockers.append(method.blocked_reason or f"{method.status}_method_not_live_executor")

        contracts.append(
            {
                "id": f"{pipeline_id}:proposal_only_contract:{method.id}",
                "adapter_id": "proposal_only_contracts",
                "pipeline_id": pipeline_id,
                "method_id": method.id,
                "method_label": method.label,
                "domain": method.domain,
                "status": method.status,
                "title": f"{method.label}: proposal contract only",
                "reason": method.summary,
                "risk": "blocked" if method.status in BLOCKED_METHOD_STATUSES else "medium",
                "target": contract["target_kind"],
                "queue_action": "not_queueable_until_promoted",
                "can_import_to_queue": False,
                "can_apply_live": False,
                "blockers": blockers,
                "proposal_contract": {
                    "value_type": value_type,
                    "target_kind": contract["target_kind"],
                    "apply_state": "proposal_only",
                    "required_fields": [
                        "target",
                        "old_value",
                        "proposed_value",
                        "safety_evidence",
                        "required_approval",
                        "readback_required",
                        "rollback_value",
                    ],
                    "old_value_source": contract["old_value_source"],
                    "safety_evidence_required": list(contract["safety_evidence"]),
                    "required_approval": "operator_supervised",
                    "readback_required": True,
                    "rollback_required": True,
                    "live_apply_allowed": False,
                    "allowed_live_write_kinds": list(LIVE_ALLOWED_WRITE_KINDS),
                    "promotion_next": promotion_next,
                },
                "evidence": {
                    "existing_method": True,
                    "can_analyze": bool(method.can_analyze),
                    "can_propose": bool(method.can_propose),
                    "live_executor": False,
                    "live_apply_policy": "supervised_fader_gain_only",
                    "promotion_next": promotion_next,
                },
            }
        )

    return contracts
