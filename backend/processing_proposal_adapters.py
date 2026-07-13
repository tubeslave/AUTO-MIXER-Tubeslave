"""Proposal-only adapters for processing methods.

Adapters in this module translate existing analysis artifacts into the Product
Layer preview contract. They must remain write-free: no mixer clients, no OSC
clients, and no live executor imports.
"""

from __future__ import annotations

import math

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
FUTURE_LIVE_REQUIRED_PROOFS = [
    "real_wing_readonly_connection_artifact",
    "physical_console_readback_for_old_and_proposed_values",
    "operator_review_acceptance_record",
    "rollback_readback_proof",
    "emergency_stop_proof",
    "connection_loss_timeout_fail_closed_proof",
    "safety_kernel_trace_for_target_action",
]
PROPOSAL_ONLY_CONTRACTS = {
    "gain_v3": {
        "value_type": "gain_trim",
        "target_kind": "channel_input_trim_or_preamp_gain",
        "old_value_source": "cached_or_physical_input_trim_read_required_before_promotion",
        "safety_evidence": [
            "class_peak_target_with_show_margin",
            "true_peak_allowance",
            "quantize_toward_headroom",
            "pad_rule_precheck",
            "noise_floor_snr_verification",
            "gain_lock_dependency_graph",
            "trim_compensation_law",
            "no_agc_on_input_gain",
            "trim_down_only_boost_operator_opt_in",
            "shared_preamps_hard_lock",
            "old_value_source",
            "rollback_gain",
        ],
    },
    "lufs_gain_staging": {
        "value_type": "gain_trim",
        "target_kind": "channel_input_trim_or_preamp_gain",
        "old_value_source": "cached_or_physical_input_trim_read_required_before_promotion",
        "safety_evidence": [
            "input_level_evidence",
            "overload_or_weak_signal_problem",
            "headroom_margin",
            "true_peak_headroom",
            "bleed_risk",
            "bounded_gain_delta",
            "gain_down_or_operator_review_gain_up",
            "rollback_input_trim_readback",
        ],
    },
    "live_shared_mix": {
        "value_type": "fader_balance",
        "target_kind": "shared_mix_channel_or_stem_balance",
        "old_value_source": "cached_or_physical_channel_fader_and_stem_sum_read_required_before_promotion",
        "safety_evidence": [
            "master_spectrum_reference",
            "source_or_stem_contributor",
            "ayaic_level_plane",
            "bounded_fader_delta",
            "true_peak_headroom",
            "feedback_risk",
            "routing_fix_disabled",
            "rollback_fader_readback",
        ],
    },
    "balance_v3": {
        "value_type": "fader_balance",
        "target_kind": "channel_fader",
        "old_value_source": "cached_or_physical_fader_read_required_before_promotion",
        "safety_evidence": [
            "lead_vocal_reference_target",
            "ayaic_level_plane",
            "fader_ceiling",
            "downward_normalization",
            "bleed_gated_measurement",
            "leakage_matrix_nnls",
            "musical_dynamics_freeze",
            "preamp_frozen_live",
            "old_value_source",
            "rollback_fader_readback",
        ],
    },
    "auto_fader_v1": {
        "value_type": "fader_balance",
        "target_kind": "channel_fader_balance",
        "old_value_source": "cached_or_physical_channel_fader_read_required_before_promotion",
        "safety_evidence": [
            "current_fader_readback",
            "ayaic_level_plane",
            "bounded_fader_delta",
            "true_peak_headroom",
            "feedback_risk",
            "fader_ceiling",
            "rollback_fader_readback",
        ],
    },
    "auto_soundcheck_engine": {
        "value_type": "gain_fader_preview",
        "target_kind": "bounded_soundcheck_gain_or_fader_preview",
        "old_value_source": "cached_or_physical_gain_fader_read_required_before_promotion",
        "safety_evidence": [
            "requested_change",
            "bounded_gain_or_fader_delta",
            "operator_review_policy",
            "live_write",
            "queue_mutation",
        ],
    },
    "auto_eq": {
        "value_type": "eq",
        "target_kind": "channel_eq_band",
        "old_value_source": "cached_or_physical_eq_band_read_required_before_promotion",
        "safety_evidence": [
            "frequency_match",
            "cut_or_bounded_boost",
            "feedback_notch_preserved",
            "manual_band_preserved",
            "true_peak_headroom",
            "eq_band_readback",
            "current_eq_context_cache_freshness",
            "current_eq_context_cache_no_live_authority",
            "rollback_eq_band_readback",
        ],
    },
    "eq_v3": {
        "value_type": "eq",
        "target_kind": "channel_eq_band",
        "old_value_source": "cached_or_physical_eq_band_read_required_before_promotion",
        "safety_evidence": [
            "resonance_harmonic_filter",
            "frequency_match",
            "cut_or_bounded_boost",
            "erb_masking_sum_of_others",
            "vocal_priority_protected",
            "feedback_notch_preserved",
            "manual_band_preserved",
            "bounded_gain_delta",
            "true_peak_headroom",
            "predictive_bs1770_ab_gate",
            "old_value_source",
            "rollback_eq_band_readback",
        ],
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
            "current_masking_context_cache_no_live_authority",
            "current_arrangement_context_cache_freshness",
            "current_arrangement_context_cache_no_live_authority",
            "arrangement_priority_context",
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
            "current_frequency_context_cache_no_live_authority",
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
            "current_source_identity_context_cache_no_live_authority",
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
            "current_source_identity_replay_context_cache_no_live_authority",
            "safety_regression_count",
            "operator_acceptance",
            "downstream_trust_state",
        ],
    },
    "auto_compressor": {
        "value_type": "compressor",
        "target_kind": "channel_dynamics",
        "old_value_source": "cached_or_physical_dynamics_read_required_before_promotion",
        "safety_evidence": [
            "gain_reduction_estimate",
            "target_gain_reduction_db",
            "makeup_headroom",
            "requested_makeup_gain_db",
            "true_peak_headroom",
            "current_dynamics_cache_freshness",
            "current_dynamics_cache_no_live_authority",
            "old_value_source",
            "rollback_dynamics_readback",
            "bypass_state",
        ],
    },
    "comp_v3": {
        "value_type": "compressor",
        "target_kind": "channel_dynamics",
        "old_value_source": "cached_or_physical_dynamics_read_required_before_promotion",
        "safety_evidence": [
            "compress_only_with_reason",
            "crest_factor_class",
            "loud_section_gr_target",
            "giannoulis_simulation_verify",
            "pumping_metric_guard",
            "bleed_lift_guard",
            "lf_release_guard",
            "bounded_makeup_headroom",
            "true_peak_headroom",
            "never_raise_ratio_live",
            "old_value_source",
            "rollback_dynamics_readback",
        ],
    },
    "auto_compressor_cf": {
        "value_type": "compressor",
        "target_kind": "channel_dynamics",
        "old_value_source": "cached_or_physical_dynamics_read_required_before_promotion",
        "safety_evidence": [
            "crest_factor_class",
            "lufs_lra_context",
            "gain_reduction_estimate",
            "bounded_makeup_headroom",
            "true_peak_headroom",
            "bypass_state",
        ],
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
            "arrangement_role",
            "sibilance_ratio",
            "sibilance_persistence",
            "current_arrangement_context_cache_freshness",
            "current_arrangement_context_cache_no_live_authority",
            "current_dynamics_or_dynamic_eq_cache_freshness",
            "current_dynamics_or_dynamic_eq_cache_no_live_authority",
            "source_identity_downstream_trust",
            "true_peak_headroom",
            "bounded_attenuation_only",
            "max_gain_reduction_db",
            "speech_intelligibility",
            "quiet_words_tail_attack_preservation",
            "dynamic_eq_or_deesser_readback",
            "old_value_source",
            "rollback_dynamics_or_dynamic_eq_readback",
            "bypass_state",
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
            "chatter_guard",
            "quiet_words_tail_attack_preservation",
            "current_gate_bleed_context_cache_freshness",
            "current_gate_bleed_context_cache_no_live_authority",
            "source_identity_downstream_trust",
            "old_value_source",
            "rollback_gate_readback",
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
            "quiet_words_tail_attack_preservation",
            "chatter_guard",
            "current_gate_bleed_context_cache_freshness",
            "current_gate_bleed_context_cache_no_live_authority",
            "source_identity_downstream_trust",
            "hysteresis",
            "old_value_source",
            "rollback_gate_readback",
        ],
    },
    "pan_v3": {
        "value_type": "pan",
        "target_kind": "channel_pan",
        "old_value_source": "cached_or_physical_pan_read_required_before_promotion",
        "safety_evidence": [
            "mono_pa_relevance_gate",
            "critical_roles_center_anchor",
            "lf_dominant_center_anchor",
            "venue_class_width_budget",
            "stereo_pair_width_only",
            "lr_symmetry_verification",
            "mono_fold_verification",
            "live_pans_frozen",
            "rate_limited_optin_lanes",
            "old_value_source",
            "rollback_pan_readback",
        ],
    },
    "auto_panner": {
        "value_type": "pan",
        "target_kind": "channel_pan",
        "old_value_source": "cached_or_physical_pan_read_required_before_promotion",
        "safety_evidence": [
            "stereo_pair",
            "mono_compatibility",
            "mono_phase_safety",
            "bounded_pan_delta",
            "current_pan_cache_freshness",
            "current_pan_cache_no_live_authority",
            "mono_phase_check_no_live_authority",
            "source_identity_downstream_trust",
            "rollback_pan_readback",
            "operator_context",
        ],
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
    "phase_v3": {
        "value_type": "phase_delay",
        "target_kind": "channel_polarity_or_input_delay",
        "old_value_source": "cached_or_physical_delay_polarity_read_required_before_promotion",
        "safety_evidence": [
            "alignment_cluster_membership",
            "stereo_pair_interior_never_aligned",
            "vocal_zero_added_delay_iem",
            "isolated_hit_gating",
            "mad_0_5ms_apply_rule",
            "iqr_sanity",
            "cap_and_console_step_quantization",
            "sum_energy_verification_rollback",
            "silence_window_apply_only",
            "old_value_source",
            "rollback_delay_polarity_readback",
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
            "current_phase_audio_context_cache_no_live_authority",
            "group_phase_measurements",
            "gcc_phat_confidence",
            "low_frequency_sum_loss",
            "bleed_risk",
            "eq_hold_policy",
        ],
    },
    "fx_v3": {
        "value_type": "fx_send",
        "target_kind": "fx_slot_or_send_level",
        "old_value_source": "cached_or_physical_fx_read_required_before_promotion",
        "safety_evidence": [
            "venue_rt_budget_clamp",
            "console_meter_send_calibration",
            "send_rate_limit_and_snapshot_cap",
            "speech_duck_confidence_gate",
            "sends_muted_tails_ring_out",
            "delay_runaway_feedback_cap",
            "tempo_resync_pause_only",
            "mdl_silence_only_until_glitch_test",
            "creative_throws_forbidden",
            "old_value_source",
            "rollback_fx_readback",
        ],
    },
    "auto_fx": {
        "value_type": "fx_send",
        "target_kind": "send_or_fx_parameter",
        "old_value_source": "cached_or_physical_fx_read_required_before_promotion",
        "safety_evidence": [
            "send_level_limit",
            "fx_return_headroom",
            "return_level_safety",
            "dry_wet_bounds",
            "feedback_risk",
            "monitor_safety",
            "intelligibility_risk",
            "current_fx_send_return_cache_freshness",
            "current_fx_send_return_cache_no_live_authority",
            "current_fx_parameter_schema_hints",
            "fx_parameter_schema_no_live_authority",
            "source_identity_downstream_trust",
            "old_value_source",
            "rollback_fx_readback",
        ],
    },
    "fx_send_proposal": {
        "value_type": "fx_send",
        "target_kind": "send_or_fx_parameter",
        "old_value_source": "cached_or_physical_fx_read_required_before_promotion",
        "safety_evidence": [
            "source_role",
            "send_level_limit",
            "fx_return_headroom",
            "return_level_safety",
            "feedback_risk",
            "intelligibility_risk",
            "monitor_safety",
            "dry_wet_bounds",
            "current_fx_send_return_cache_freshness",
            "current_fx_send_return_cache_no_live_authority",
            "current_fx_parameter_schema_hints",
            "fx_parameter_schema_no_live_authority",
            "source_identity_downstream_trust",
            "old_value_source",
            "rollback_fx_readback",
        ],
    },
    "pa_v3": {
        "value_type": "master_bus",
        "target_kind": "system_output_processing",
        "old_value_source": "cached_or_physical_output_processing_read_required_before_promotion",
        "safety_evidence": [
            "coherence_gate_setup_0_7_live_0_6",
            "spatial_consistency_mask",
            "prefer_cut_filter_budget",
            "sub_top_alignment_setup_only",
            "no_live_delay_polarity_crossover_changes",
            "system_snapshot_gates_channel_pipelines",
            "outputs_excluded_from_song_scenes",
            "spl_jurisdiction_supervision",
            "hf_lane_opt_in_rate_limited",
            "old_value_source",
            "rollback_output_readback",
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
            "pa_feed_readback",
            "pa_feed_safety",
            "pa_feed_cache_freshness",
            "pa_feed_cache_no_live_authority",
            "reference_feed_exclusion",
            "current_main_bus_processing_cache_freshness",
            "current_main_bus_processing_cache_no_live_authority",
            "rollback_master_processing_readback",
            "rollback_pa_feed_matrix_readback",
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
            "bounded_bus_group_delta",
            "assignment_readback",
            "current_bus_fader",
            "current_bus_group_dca_matrix_cache_freshness",
            "current_bus_group_dca_matrix_cache_no_live_authority",
            "true_peak_headroom",
            "main_bus_headroom",
            "feedback_risk",
            "source_identity_downstream_trust",
            "rollback_bus_group_dca_matrix_readback",
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
            "current_mute_cache_no_live_authority",
            "auto_muted_by_system",
            "signal_activity",
            "idle_duration",
            "operator_intent",
            "dropout_risk",
            "feedback_risk",
            "source_identity_downstream_trust",
            "rollback_value",
            "rollback_mute_readback",
            "manual_mute_authority_preserved",
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
            "current_route_cache_no_live_authority",
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
            "current_scene_preset_cache_no_live_authority",
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
            "current_mix_variant_context_cache_no_live_authority",
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
            "current_performance_context_cache_no_live_authority",
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
            "current_arrangement_role_context_cache_no_live_authority",
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
            "current_decision_evidence_context_cache_no_live_authority",
            "source_identity_downstream_trust",
            "readback_state",
            "rollback_plan",
            "true_peak_headroom",
            "feedback_risk",
        ],
    },
    "smart_dsp_strategy_engine": {
        "value_type": "smart_dsp_strategy",
        "target_kind": "offline_soundcheck_live_strategy_selection",
        "old_value_source": "offline_strategy_evidence_required_before_live_promotion",
        "safety_evidence": [
            "source_identity",
            "current_smart_dsp_context_cache_no_live_authority",
            "current_state_readback_available",
            "rollback_plan_available",
            "detected_problems",
            "strategy_search",
            "selected_strategy",
            "alternatives",
            "true_peak_headroom",
            "feedback_risk",
            "phase_mono_safety",
            "operator_explanation",
            "residual_risk",
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
            "current_memory_learning_context_cache_no_live_authority",
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
    "agentic_method_foundation": {
        "value_type": "gain_fader_preview",
        "target_kind": "bounded_gain_or_fader_down_preview",
        "old_value_source": "cached_or_physical_gain_fader_read_required_before_promotion",
        "safety_evidence": [
            "feature_snapshot",
            "inter_channel_context",
            "bounded_downward_delta",
            "dry_run",
            "true_peak_headroom",
            "feedback_risk",
            "rollback_gain_fader_readback",
        ],
    },
    "routing_snapshot_reset_writes": {
        "value_type": "routing",
        "target_kind": "route_or_snapshot_state",
        "old_value_source": "full_route_snapshot_required_before_promotion",
        "safety_evidence": [
            "route_diff",
            "current_route_readback",
            "current_route_cache_freshness",
            "current_route_cache_no_live_authority",
            "proposed_source_readback",
            "active_performance_guard",
            "destructive_scope_check",
            "destructive_state_check",
            "rollback_route",
            "rollback_snapshot",
            "operator_confirmation",
        ],
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
REQUESTED_CHANGE_OPERATOR_CONTRACTS = {
    "gain": {
        "value_type": "gain_trim",
        "target_kind": "channel_input_trim_or_preamp_gain",
        "old_value_source": "cached_or_physical_input_trim_read_required_before_promotion",
        "safety_evidence": [
            "input_level_evidence",
            "overload_or_weak_signal_problem",
            "headroom_margin",
            "true_peak_headroom",
            "bleed_risk",
            "bounded_gain_delta",
            "gain_down_or_operator_review_gain_up",
            "rollback_input_trim_readback",
        ],
    },
    "fader": {
        "value_type": "fader_balance",
        "target_kind": "channel_fader_balance",
        "old_value_source": "cached_or_physical_channel_fader_read_required_before_promotion",
        "safety_evidence": [
            "current_fader_readback",
            "ayaic_level_plane",
            "bounded_fader_delta",
            "true_peak_headroom",
            "feedback_risk",
            "fader_ceiling",
            "rollback_fader_readback",
        ],
    },
}

BASE_PROPOSAL_REQUIRED_FIELDS = [
    "target",
    "old_value",
    "proposed_value",
    "safety_evidence",
    "required_approval",
    "readback_required",
    "rollback_value",
]

OPERATOR_CORRECTION_REQUIRED_FIELDS = [
    "method_id",
    "source",
    "target",
    "value_type",
    "target_kind",
    "old_value_source",
    "objective_family",
    "bounded_delta_unit",
    "bounded_delta_policy",
    "source_evidence",
    "source_evidence_details",
    "measured_or_audible_problem",
    "proposed_correction",
    "operator_evidence_packet",
    "confidence",
    "confidence_review",
    "confidence_review_details",
    "operator_proposal_review_gate",
    "bounded_delta",
    "bounded_delta_details",
    "safety_blockers",
    "effective_safety_blockers",
    "safety_blocker_details",
    "why_safe_details",
    "why_blocked_details",
    "headroom_feedback_true_peak_checks",
    "headroom_feedback_true_peak_check_details",
    "operator_safety_review_matrix",
    "operator_readable_explanation",
    "operator_explanation_packet",
    "operator_next_action_details",
    "operator_review_summary",
    "operator_review_checklist",
    "operator_review_checklist_details",
    "operator_review_decision_options",
    "operator_review_decision_option_details",
    "operator_review_record_template",
    "operator_review_record_field_details",
    "rollback_expectation",
    "rollback_readback_contract",
    "rollback_readback_details",
    "future_live_promotion_requirements",
    "future_live_requirement_details",
    "live_authority_boundary",
    "future_wing_readback_contract",
    "denial_reason",
    "operator_next_step",
]

OPERATOR_REVIEW_RECORD_FIELD_PROFILES = {
    "reviewer": ("operator_identity", "operator_identity"),
    "reviewed_at_utc": ("operator_review_timestamp", "utc_timestamp"),
    "decision": ("operator_review_decision_options", "enum"),
    "rationale": ("operator_rationale", "text"),
    "evidence_packet_reference": ("operator_evidence_packet", "reference"),
    "source_material_reference": ("source_evidence", "reference"),
    "source_evidence_acknowledgement": ("source_evidence", "acknowledgement"),
    "test_evidence_reference": ("test_evidence", "reference"),
    "test_evidence_acknowledgement": ("test_evidence", "acknowledgement"),
    "verification_report_reference": (
        "operator_review_verification_report",
        "reference",
    ),
    "verification_report_acknowledgement": (
        "operator_review_verification_report",
        "acknowledgement",
    ),
    "measured_problem_acknowledgement": (
        "measured_or_audible_problem",
        "acknowledgement",
    ),
    "proposed_correction_acknowledgement": (
        "proposed_correction",
        "acknowledgement",
    ),
    "bounded_delta_acknowledgement": ("bounded_delta", "acknowledgement"),
    "confidence_acknowledgement": ("confidence", "acknowledgement"),
    "operator_explanation_acknowledgement": (
        "operator_explanation_packet",
        "acknowledgement",
    ),
    "future_wing_readback_acknowledgement": (
        "future_wing_readback_contract",
        "acknowledgement",
    ),
    "headroom_feedback_true_peak_acknowledgement": (
        "headroom_feedback_true_peak_checks",
        "acknowledgement",
    ),
    "rollback_readback_acknowledgement": (
        "rollback_readback_contract",
        "acknowledgement",
    ),
    "safety_blockers_acknowledgement": ("safety_blockers", "acknowledgement"),
    "required_confirmations_acknowledgement": (
        "what_operator_must_confirm",
        "acknowledgement",
    ),
    "offline_acceptance_boundary_acknowledgement": (
        "live_authority_boundary",
        "acknowledgement",
    ),
    "operator_notes": ("operator_notes", "text"),
}

OPERATOR_REVIEW_CHECKLIST_DETAIL_PROFILES = {
    "verify_source_evidence_matches_current_offline_material": (
        "source_evidence",
        "verify_source_evidence",
        "source_evidence_acknowledgement",
    ),
    "confirm_measured_or_audible_problem": (
        "measured_or_audible_problem",
        "confirm_problem_statement",
        "measured_problem_acknowledgement",
    ),
    "inspect_proposed_correction_and_bounded_delta": (
        "proposed_correction",
        "inspect_proposed_correction",
        "proposed_correction_acknowledgement,bounded_delta_acknowledgement",
    ),
    "verify_required_safety_checks": (
        "operator_safety_review_matrix",
        "verify_required_safety_checks",
        "headroom_feedback_true_peak_acknowledgement,safety_blockers_acknowledgement",
    ),
    "confirm_rollback_expectation": (
        "rollback_readback_contract",
        "confirm_rollback_expectation",
        "rollback_readback_acknowledgement",
    ),
    "do_not_queue_or_apply_live": (
        "live_authority_boundary",
        "hold_live_and_queue_authority",
        "offline_acceptance_boundary_acknowledgement",
    ),
    "capture_real_wing_readback_and_rollback_proof_before_live_promotion": (
        "future_wing_readback_contract",
        "capture_future_wing_readback",
        "future_wing_readback_acknowledgement",
    ),
    "collect_missing_required_evidence_before_operator_review": (
        "evidence_gate",
        "collect_missing_required_evidence",
        "required_confirmations_acknowledgement",
    ),
    "operator_review_required_before_future_live_promotion": (
        "operator_proposal_review_gate",
        "operator_review_before_future_live",
        "required_confirmations_acknowledgement",
    ),
}

REQUIRED_CONFIRMATION_DETAIL_PROFILES = {
    "source_evidence_matches_current_offline_material": (
        "source_evidence",
        "confirm_source_evidence_matches_material",
        "source_evidence_acknowledgement",
    ),
    "measured_or_audible_problem_is_stated": (
        "measured_or_audible_problem",
        "confirm_problem_statement",
        "measured_problem_acknowledgement",
    ),
    "proposed_correction_matches_the_problem": (
        "proposed_correction",
        "confirm_proposed_correction_matches_problem",
        "proposed_correction_acknowledgement",
    ),
    "confidence_and_confidence_blockers_are_reviewed": (
        "confidence",
        "review_confidence_and_blockers",
        "confidence_acknowledgement",
    ),
    "bounded_delta_is_safe_for_the_target": (
        "bounded_delta",
        "review_bounded_delta",
        "bounded_delta_acknowledgement",
    ),
    "safety_blockers_and_safety_checks_are_reviewed": (
        "safety_blockers,headroom_feedback_true_peak_checks",
        "review_safety_blockers_and_checks",
        "safety_blockers_acknowledgement,headroom_feedback_true_peak_acknowledgement",
    ),
    "rollback_expectation_is_clear": (
        "rollback_readback_contract",
        "confirm_rollback_expectation",
        "rollback_readback_acknowledgement",
    ),
    "future_wing_readback_targets_are_known_before_live_promotion": (
        "future_wing_readback_contract",
        "confirm_future_wing_readback_targets",
        "future_wing_readback_acknowledgement",
    ),
    "offline_method_tests_pass_before_acceptance": (
        "test_evidence",
        "verify_offline_method_tests",
        "test_evidence_acknowledgement",
    ),
    "live_apply_and_queue_import_remain_blocked": (
        "live_authority_boundary",
        "confirm_offline_acceptance_boundary",
        "offline_acceptance_boundary_acknowledgement",
    ),
}

OPERATOR_REVIEW_DECISION_OPTION_DETAIL_PROFILES = {
    "request_more_evidence": (
        "keeps_proposal_blocked_until_required_evidence_complete",
        "request_more_evidence",
        "record_operator_decision:request_more_evidence",
        True,
    ),
    "accept_as_offline_proposal": (
        "records_operator_review_for_offline_plan_only",
        "accept_offline_proposal_only",
        "record_operator_decision:accept_as_offline_proposal",
        False,
    ),
    "reject_proposal": (
        "keeps_or_discards_proposal_without_console_mutation",
        "reject_offline_proposal",
        "record_operator_decision:reject_proposal",
        True,
    ),
    "defer_until_real_wing_proof": (
        "requires_future_wing_readback_rollback_and_operator_review",
        "defer_until_real_wing_proof",
        "record_operator_decision:defer_until_real_wing_proof",
        True,
    ),
}

CORRECTION_METHOD_REQUIRED_CONFIRMATIONS = (
    "source_evidence_matches_current_offline_material",
    "measured_or_audible_problem_is_stated",
    "proposed_correction_matches_the_problem",
    "confidence_and_confidence_blockers_are_reviewed",
    "bounded_delta_is_safe_for_the_target",
    "safety_blockers_and_safety_checks_are_reviewed",
    "rollback_expectation_is_clear",
    "future_wing_readback_targets_are_known_before_live_promotion",
    "offline_method_tests_pass_before_acceptance",
    "live_apply_and_queue_import_remain_blocked",
)

CORRECTION_METHOD_GOAL_PRIORITY_GROUPS = [
    {
        "priority": 1,
        "label": "Gain / Trim and Fader / Balance",
        "objective_families": [
            "gain_trim",
            "gain_trim_and_fader_balance",
            "fader_balance",
        ],
    },
    {
        "priority": 2,
        "label": "EQ and Compressor",
        "objective_families": ["eq", "compressor_dynamics"],
    },
    {
        "priority": 3,
        "label": "Gate / Expander / De-esser",
        "objective_families": ["gate_expander", "gate_expander_deesser"],
    },
    {
        "priority": 4,
        "label": "FX / Sends",
        "objective_families": ["fx_sends"],
    },
    {
        "priority": 5,
        "label": "Pan / Stereo",
        "objective_families": ["pan_stereo"],
    },
    {
        "priority": 6,
        "label": "Routing / Groups / Scenes",
        "objective_families": ["routing_groups_scenes"],
    },
    {
        "priority": 7,
        "label": "Master / Output",
        "objective_families": ["master_output"],
    },
]

CORRECTION_METHOD_GOAL_REQUIRED_OBJECTIVE_FAMILIES = [
    family
    for group in CORRECTION_METHOD_GOAL_PRIORITY_GROUPS
    for family in group["objective_families"]
]

CORRECTION_METHOD_GOAL_REQUIRED_OPERATOR_SURFACES = [
    {
        "surface": "source_evidence",
        "flag": "source_evidence_required",
    },
    {
        "surface": "source_evidence_details",
        "flag": "source_evidence_details_required",
    },
    {
        "surface": "measured_or_audible_problem",
        "flag": "measured_or_audible_problem_required",
    },
    {
        "surface": "proposed_correction",
        "flag": "proposed_correction_required",
    },
    {
        "surface": "confidence",
        "flag": "confidence_required",
    },
    {
        "surface": "confidence_review",
        "flag": "confidence_review_required",
    },
    {
        "surface": "bounded_delta",
        "flag": "bounded_delta_required",
    },
    {
        "surface": "bounded_delta_details",
        "flag": "bounded_delta_details_required",
    },
    {
        "surface": "safety_blockers",
        "flag": "safety_blockers_required",
    },
    {
        "surface": "safety_blocker_details",
        "flag": "safety_blocker_details_required",
    },
    {
        "surface": "headroom_feedback_true_peak_checks",
        "flag": "headroom_feedback_true_peak_checks_required",
    },
    {
        "surface": "headroom_feedback_true_peak_check_details",
        "flag": "headroom_feedback_true_peak_check_details_required",
    },
    {
        "surface": "operator_safety_review_matrix",
        "flag": "operator_safety_review_matrix_required",
    },
    {
        "surface": "operator_readable_explanation",
        "flag": "operator_readable_explanation_required",
    },
    {
        "surface": "operator_explanation_packet",
        "flag": "operator_explanation_packet_required",
    },
    {
        "surface": "operator_evidence_packet",
        "flag": "operator_evidence_packet_required",
    },
    {
        "surface": "rollback_expectation",
        "flag": "rollback_expectation_required",
    },
    {
        "surface": "rollback_readback_contract",
        "flag": "rollback_readback_contract_required",
    },
    {
        "surface": "rollback_readback_details",
        "flag": "rollback_readback_details_required",
    },
    {
        "surface": "future_wing_readback_contract",
        "flag": "future_wing_readback_contract_required",
    },
    {
        "surface": "future_live_promotion_requirements",
        "flag": "future_live_promotion_requirements_required",
    },
    {
        "surface": "live_authority_boundary",
        "flag": "live_authority_boundary_required",
    },
]

DEFAULT_OPERATOR_CORRECTION_CONTRACT = {
    "objective_family": "proposal_only_method",
    "bounded_delta_unit": "method_specific",
    "bounded_delta_policy": "proposal_must_state_method_specific_bounds_before_operator_review",
    "applicable_safety_checks": ["safety_blockers", "rollback", "future_wing_readback"],
    "future_wing_readback_targets": ["target_parameter", "old_value", "proposed_value", "rollback_value"],
    "rollback_expectation": "restore_previous_readback_value_or_hold_for_operator_manual_rollback",
    "blocked_live_actions": ["live_apply"],
}

OPERATOR_CORRECTION_CONTRACTS_BY_VALUE_TYPE = {
    "gain_trim": {
        "objective_family": "gain_trim",
        "bounded_delta_unit": "db",
        "bounded_delta_policy": "gain_down_is_bounded; gain_up_requires_operator_review_and_explicit_future_proof",
        "applicable_safety_checks": ["headroom", "feedback", "true_peak", "bleed", "source_identity"],
        "future_wing_readback_targets": ["channel_input_trim", "preamp_gain", "channel_gain", "rollback_gain"],
        "rollback_expectation": "restore_previous_input_trim_or_a_safer_lower_value_after_physical_readback",
        "blocked_live_actions": ["preamp_gain_write", "input_trim_write", "gain_up_without_operator_opt_in"],
    },
    "fader_balance": {
        "objective_family": "fader_balance",
        "bounded_delta_unit": "db",
        "bounded_delta_policy": "small_fader_delta_with_fader_ceiling_and_headroom_limits",
        "applicable_safety_checks": ["headroom", "feedback", "true_peak", "fader_ceiling", "ayaic_level_plane"],
        "future_wing_readback_targets": ["channel_fader", "group_fader", "dca_fader", "rollback_fader"],
        "rollback_expectation": "restore_previous_fader_value_after_physical_readback",
        "blocked_live_actions": ["unsupervised_fader_write", "fader_above_zero_dbfs"],
    },
    "gain_fader_preview": {
        "objective_family": "gain_trim_and_fader_balance",
        "bounded_delta_unit": "db",
        "bounded_delta_policy": "preview_only_downward_gain_or_fader_delta",
        "applicable_safety_checks": ["headroom", "feedback", "true_peak", "dry_run"],
        "future_wing_readback_targets": ["channel_gain", "channel_fader", "rollback_gain_or_fader"],
        "rollback_expectation": "restore_previous_gain_or_fader_value_after_supervised_readback",
        "blocked_live_actions": ["live_preview_apply", "upward_delta_without_operator_proof"],
    },
    "source_identity": {
        "objective_family": "source_identity",
        "bounded_delta_unit": "source_label",
        "bounded_delta_policy": "operator_reviewed_source_identity_label_only_with_classifier_confidence_and_replay_calibration_before_downstream_trust",
        "applicable_safety_checks": ["classifier_confidence", "operator_acceptance", "context_freshness", "downstream_rejection", "replay_outcome"],
        "future_wing_readback_targets": ["operator_accepted_source_label", "classifier_evidence", "replay_outcome", "downstream_trust_state"],
        "rollback_expectation": "discard_unaccepted_source_identity_label_or_restore_previous_operator_accepted_label",
        "blocked_live_actions": ["source_identity_memory_write", "downstream_source_specific_processing_apply", "live_console_write_from_identity"],
    },
    "bus_group_balance": {
        "objective_family": "fader_balance",
        "bounded_delta_unit": "db",
        "bounded_delta_policy": "small_bus_group_dca_matrix_delta_with_ayaic_and_sum_bus_limits",
        "applicable_safety_checks": ["headroom", "feedback", "true_peak", "main_bus_headroom", "ayaic_level_plane"],
        "future_wing_readback_targets": ["bus_fader", "group_fader", "dca_fader", "matrix_fader", "assignment_map"],
        "rollback_expectation": "restore_previous_bus_group_dca_matrix_fader_after_physical_readback",
        "blocked_live_actions": ["bus_group_dca_matrix_write"],
    },
    "eq": {
        "objective_family": "eq",
        "bounded_delta_unit": "db_hz_q",
        "bounded_delta_policy": "bounded_cut_or_small_boost_with_protected_band_policy",
        "applicable_safety_checks": ["headroom", "feedback", "true_peak", "manual_band", "phase"],
        "future_wing_readback_targets": ["eq_on", "band_id", "band_frequency", "band_gain", "band_q", "filter_type"],
        "rollback_expectation": "restore_previous_eq_band_or_leave_feedback_manual_band_unchanged",
        "blocked_live_actions": ["eq_band_write", "eq_on_write"],
    },
    "frequency_split": {
        "objective_family": "eq",
        "bounded_delta_unit": "db_hz_lane",
        "bounded_delta_policy": "subtractive_or_mirror_eq_only_until_sum_bus_readback_exists",
        "applicable_safety_checks": ["headroom", "feedback", "true_peak", "sum_bus", "source_identity"],
        "future_wing_readback_targets": ["eq_band", "stem_spectrum", "sum_bus"],
        "rollback_expectation": "restore_previous_eq_band_and_stem_lane_plan_after_readback",
        "blocked_live_actions": ["frequency_split_live_write", "linked_eq_write"],
    },
    "compressor": {
        "objective_family": "compressor_dynamics",
        "bounded_delta_unit": "threshold_ratio_attack_release_makeup_db",
        "bounded_delta_policy": "bounded_threshold_ratio_timing_and_makeup_gain_with_gr_estimate",
        "applicable_safety_checks": ["headroom", "feedback", "true_peak", "gain_reduction", "makeup_gain"],
        "future_wing_readback_targets": ["dyn_on", "threshold", "ratio", "attack", "release", "makeup_gain"],
        "rollback_expectation": "restore_previous_dynamics_values_or_bypass_state_after_physical_readback",
        "blocked_live_actions": ["compressor_write", "makeup_gain_write"],
    },
    "gate": {
        "objective_family": "gate_expander",
        "bounded_delta_unit": "threshold_range_hysteresis_timing_db_ms",
        "bounded_delta_policy": "chatter_safe_expander_or_gate_delta_only",
        "applicable_safety_checks": ["source_identity", "chatter_guard", "hysteresis", "tails_and_attack"],
        "future_wing_readback_targets": ["gate_on", "threshold", "range", "attack", "hold", "release", "hysteresis"],
        "rollback_expectation": "restore_previous_gate_or_expander_values_or_bypass_after_readback",
        "blocked_live_actions": ["gate_write", "expander_write"],
    },
    "deesser": {
        "objective_family": "gate_expander_deesser",
        "bounded_delta_unit": "frequency_threshold_range_db",
        "bounded_delta_policy": "bounded_sibilance_attenuation_only",
        "applicable_safety_checks": ["source_identity", "arrangement_role", "true_peak", "speech_intelligibility", "tails_and_attack"],
        "future_wing_readback_targets": ["deesser_band", "dynamic_eq_band", "threshold", "range"],
        "rollback_expectation": "restore_previous_dynamics_or_dynamic_eq_values_after_readback",
        "blocked_live_actions": ["deesser_write", "dynamic_eq_write"],
    },
    "fx_send": {
        "objective_family": "fx_sends",
        "bounded_delta_unit": "send_return_db_and_fx_params",
        "bounded_delta_policy": "bounded_send_with_return_level_monitor_and_feedback_limits",
        "applicable_safety_checks": ["headroom", "feedback", "return_level", "monitor_safety", "intelligibility"],
        "future_wing_readback_targets": ["send_level", "fx_return_level", "fx_slot_parameters", "monitor_send"],
        "rollback_expectation": "restore_previous_send_and_return_values_after_physical_readback",
        "blocked_live_actions": ["fx_send_write", "fx_parameter_write", "monitor_send_write"],
    },
    "fx_parameter": {
        "objective_family": "fx_sends",
        "bounded_delta_unit": "fx_insert_or_rack_parameter",
        "bounded_delta_policy": "read_only_fx_parameter_contract_requires_insert_bypass_return_headroom_and_rollback_readback",
        "applicable_safety_checks": ["headroom", "feedback", "return_level", "insert_bypass_state", "parameter_bounds"],
        "future_wing_readback_targets": ["fx_slot_parameter", "insert_bypass", "fx_return_level", "rollback_fx_parameter"],
        "rollback_expectation": "restore_previous_fx_parameter_or_insert_bypass_state_after_physical_readback",
        "blocked_live_actions": ["fx_parameter_write", "insert_bypass_write", "fx_rack_write"],
    },
    "pan": {
        "objective_family": "pan_stereo",
        "bounded_delta_unit": "pan_percent",
        "bounded_delta_policy": "bounded_pan_delta_with_mono_phase_check_for_non_center_moves",
        "applicable_safety_checks": ["mono_compatibility", "phase", "source_identity"],
        "future_wing_readback_targets": ["channel_pan", "stereo_pair_link", "mono_phase_check"],
        "rollback_expectation": "restore_previous_pan_value_after_physical_readback",
        "blocked_live_actions": ["pan_write", "stereo_width_write"],
    },
    "phase_delay": {
        "objective_family": "phase_alignment",
        "bounded_delta_unit": "delay_ms_or_polarity",
        "bounded_delta_policy": "proposal_only_delay_or_polarity_change_requires_correlation_readback_and_manual_confirmation",
        "applicable_safety_checks": ["phase_correlation", "polarity_readback", "delay_readback", "sum_bus"],
        "future_wing_readback_targets": ["channel_delay", "polarity_invert", "correlation_meter", "rollback_delay_polarity"],
        "rollback_expectation": "restore_previous_delay_and_polarity_after_physical_readback",
        "blocked_live_actions": ["delay_write", "polarity_write"],
    },
    "phase_eq_guard": {
        "objective_family": "phase_alignment",
        "bounded_delta_unit": "phase_guard_policy",
        "bounded_delta_policy": "read_only_phase_guard_can_hold_eq_proposals_until_multichannel_phase_context_is_fresh",
        "applicable_safety_checks": ["phase_correlation", "sum_bus", "bleed", "eq_hold_policy"],
        "future_wing_readback_targets": ["phase_measurements", "eq_proposal_hold_state", "sum_bus", "rollback_eq_hold"],
        "rollback_expectation": "release_or_restore_held_eq_proposals_after_fresh_phase_context_review",
        "blocked_live_actions": ["eq_write_from_stale_phase_context", "delay_write", "polarity_write"],
    },
    "routing": {
        "objective_family": "routing_groups_scenes",
        "bounded_delta_unit": "route_diff",
        "bounded_delta_policy": "read_only_map_and_non_queueable_route_diff_only",
        "applicable_safety_checks": ["read_only_map", "destructive_scope", "active_performance_guard"],
        "future_wing_readback_targets": ["routing_map", "bus_assignment", "matrix_send", "rollback_route"],
        "rollback_expectation": "operator-reviewed route snapshot required before any future manual rollback",
        "blocked_live_actions": ["routing_write", "bus_assignment_write", "matrix_write"],
    },
    "scene_preset": {
        "objective_family": "routing_groups_scenes",
        "bounded_delta_unit": "scoped_scene_diff",
        "bounded_delta_policy": "read_only_inventory_and_scoped_diff_only",
        "applicable_safety_checks": ["read_only_inventory", "destructive_scope", "active_signal_guard"],
        "future_wing_readback_targets": ["scene_inventory", "preset_inventory", "full_console_state", "rollback_snapshot"],
        "rollback_expectation": "operator-reviewed full-state snapshot required before any future scene or preset action",
        "blocked_live_actions": ["scene_load", "scene_save", "preset_load", "preset_save"],
    },
    "mute_state": {
        "objective_family": "routing_groups_scenes",
        "bounded_delta_unit": "mute_state_diff",
        "bounded_delta_policy": "read_only_mute_state_proposal_requires_operator_confirmation_and_readback",
        "applicable_safety_checks": ["source_identity", "active_signal_guard", "feedback", "destructive_scope"],
        "future_wing_readback_targets": ["channel_mute", "mute_group", "rollback_mute_state"],
        "rollback_expectation": "restore_previous_mute_state_after_physical_readback_or_operator_manual_action",
        "blocked_live_actions": ["mute_write", "unmute_write", "mute_group_write"],
    },
    "master_bus": {
        "objective_family": "master_output",
        "bounded_delta_unit": "db_and_processor_params",
        "bounded_delta_policy": "offline_static_trim_eq_limiter_or_dynamics_recommendation_only",
        "applicable_safety_checks": ["headroom", "true_peak", "pa_feed_safety", "reference_feed_exclusion", "limiter_or_dynamics_proof"],
        "future_wing_readback_targets": ["main_bus_processing", "limiter", "matrix_or_pa_feed", "reference_feed"],
        "rollback_expectation": "restore_previous_master_processing_or_hold_for_operator_manual_rollback",
        "blocked_live_actions": ["master_bus_write", "limiter_write", "matrix_pa_feed_write"],
    },
    "mix_variant_profile": {
        "objective_family": "mix_variant_profile",
        "bounded_delta_unit": "genre_task_profile",
        "bounded_delta_policy": "operator_selected_mix_variant_profile_only_with_level_plane_reference_and_no_named_artist_style_clone",
        "applicable_safety_checks": ["headroom", "feedback", "true_peak", "arrangement_context", "source_identity", "style_clone"],
        "future_wing_readback_targets": ["current_mix_profile", "operator_selected_profile", "level_plane_targets", "rollback_profile"],
        "rollback_expectation": "restore_previous_operator_selected_mix_profile_or_hold_for_manual_review",
        "blocked_live_actions": ["mix_profile_apply", "scene_load", "snapshot_recall", "unsupervised_style_clone"],
    },
    "arrangement_role": {
        "objective_family": "arrangement_role",
        "bounded_delta_unit": "role_label",
        "bounded_delta_policy": "operator_reviewed_arrangement_role_label_only_with_context_persistence_and_source_identity_trust",
        "applicable_safety_checks": ["source_identity", "arrangement_context", "inter_channel_context", "role_persistence", "masking"],
        "future_wing_readback_targets": ["operator_accepted_source_role", "current_arrangement_role", "decision_evidence_packet"],
        "rollback_expectation": "restore_previous_operator_accepted_arrangement_role_or_hold_for_manual_review",
        "blocked_live_actions": ["role_memory_mutation", "downstream_priority_apply", "unsupervised_arrangement_role_write"],
    },
    "decision_evidence_packet": {
        "objective_family": "decision_evidence_packet",
        "bounded_delta_unit": "evidence_packet",
        "bounded_delta_policy": "operator_reviewed_evidence_packet_only_with_readback_rollback_headroom_and_feedback_context",
        "applicable_safety_checks": ["readback", "rollback", "headroom", "feedback", "source_identity", "context_freshness"],
        "future_wing_readback_targets": ["decision_trace", "current_state_readback", "rollback_plan", "operator_review_record"],
        "rollback_expectation": "restore_or_discard_unaccepted_decision_packet_without_mutating_mixer_or_memory",
        "blocked_live_actions": ["decision_packet_apply", "queue_import", "live_console_write_from_packet"],
    },
    "long_term_memory_learning": {
        "objective_family": "long_term_memory_learning",
        "bounded_delta_unit": "memory_learning_record",
        "bounded_delta_policy": "operator_accepted_offline_learning_record_only_with_replay_correlation_and_no_safety_regressions",
        "applicable_safety_checks": ["offline_replay", "operator_outcomes", "safety_regressions", "manual_overrides", "source_identity", "context_freshness"],
        "future_wing_readback_targets": ["accepted_learning_record", "offline_replay_summary", "operator_acceptance_record", "memory_rollback_plan"],
        "rollback_expectation": "discard_unaccepted_learning_record_or_restore_previous_memory_snapshot_after_operator_review",
        "blocked_live_actions": ["memory_write", "model_update", "live_console_write_from_learning"],
    },
    "source_identity_calibration": {
        "objective_family": "source_identity_calibration",
        "bounded_delta_unit": "trust_state",
        "bounded_delta_policy": "operator_reviewed_replay_calibration_only_with_accepted_rejected_outcomes_and_no_live_authority_context",
        "applicable_safety_checks": ["operator_outcomes", "replay_match", "safety_regressions", "operator_acceptance", "context_freshness"],
        "future_wing_readback_targets": ["accepted_source_identity_label", "replay_outcome_summary", "operator_acceptance_record", "downstream_trust_state"],
        "rollback_expectation": "discard_unaccepted_trust_state_or_restore_previous_operator_accepted_source_identity_calibration",
        "blocked_live_actions": ["source_identity_memory_write", "downstream_trust_apply", "live_console_write_from_identity"],
    },
    "performance_adaptation": {
        "objective_family": "performance_adaptation",
        "bounded_delta_unit": "db",
        "bounded_delta_policy": "persistent_level_drift_proposal_only_with_headroom_feedback_cooldown_and_operator_review",
        "applicable_safety_checks": ["headroom", "feedback", "true_peak", "cooldown", "manual_override", "source_identity"],
        "future_wing_readback_targets": ["channel_level", "channel_fader", "bus_fader", "rollback_level_or_fader"],
        "rollback_expectation": "restore_previous_level_or_fader_state_after_physical_readback",
        "blocked_live_actions": ["performance_adaptation_live_write", "trend_based_fader_write", "unsupervised_level_adaptation"],
    },
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


def _unique_fields(*field_groups: Iterable[str]) -> List[str]:
    fields: List[str] = []
    seen = set()
    for group in field_groups:
        for field in group:
            if field in seen:
                continue
            seen.add(field)
            fields.append(field)
    return fields


def _operator_correction_contract(
    value_type: str,
    target_kind: str,
    old_value_source: str,
    required_evidence_fields: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    profile = {
        **DEFAULT_OPERATOR_CORRECTION_CONTRACT,
        **OPERATOR_CORRECTION_CONTRACTS_BY_VALUE_TYPE.get(
            value_type, {}
        ),
    }
    required_evidence = list(required_evidence_fields or [])
    return {
        "method_id_required": True,
        "source_required": True,
        "target_required": True,
        "value_type_required": True,
        "target_kind_required": True,
        "old_value_source_required": True,
        "objective_family_required": True,
        "value_type": value_type,
        "target_kind": target_kind,
        "old_value_source": old_value_source,
        "objective_family": profile["objective_family"],
        "bounded_delta_unit_required": True,
        "bounded_delta_policy_required": True,
        "source_evidence_required": True,
        "source_evidence_details_required": True,
        "measured_or_audible_problem_required": True,
        "proposed_correction_required": True,
        "confidence_required": True,
        "confidence_review_required": True,
        "confidence_review_details_required": True,
        "operator_proposal_review_gate_required": True,
        "bounded_delta_required": True,
        "bounded_delta_details_required": True,
        "required_evidence_fields": required_evidence,
        "bounded_delta_unit": profile["bounded_delta_unit"],
        "bounded_delta_policy": profile["bounded_delta_policy"],
        "safety_blockers_required": True,
        "safety_blocker_details_required": True,
        "why_safe_details_required": True,
        "why_blocked_details_required": True,
        "applicable_safety_checks": list(profile["applicable_safety_checks"]),
        "headroom_feedback_true_peak_checks_required": True,
        "headroom_feedback_true_peak_check_details_required": True,
        "operator_safety_review_matrix_required": True,
        "operator_readable_explanation_required": True,
        "operator_explanation_packet_required": True,
        "operator_next_action_details_required": True,
        "operator_evidence_packet_required": True,
        "operator_review_summary_required": True,
        "operator_review_checklist_required": True,
        "operator_review_checklist_details_required": True,
        "operator_review_decision_options_required": True,
        "operator_review_decision_option_details_required": True,
        "operator_review_record_template_required": True,
        "operator_review_record_field_details_required": True,
        "future_live_promotion_requirements_required": True,
        "future_live_requirement_details_required": True,
        "live_authority_boundary_required": True,
        "denial_reason_required": True,
        "operator_next_step_required": True,
        "rollback_readback_contract_required": True,
        "rollback_readback_details_required": True,
        "rollback_expectation": profile["rollback_expectation"],
        "rollback_required": True,
        "future_wing_readback_contract": {
            "required_before_live_promotion": True,
            "old_value_source": old_value_source,
            "target_kind": target_kind,
            "readback_targets": list(profile["future_wing_readback_targets"]),
            "blocked_live_actions": list(profile["blocked_live_actions"]),
            "required_evidence_fields": required_evidence,
            "read_only_or_dry_run_only": True,
            "live_apply_allowed": False,
            "queue_import_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "readiness_closure_allowed": False,
            "can_authorize_live": False,
            "can_close_readiness": False,
            "mutates_mixer": False,
            "operator_review_required": True,
            "operator_approval_required": True,
            "hardware_proof_required": True,
            "blocks_live_promotion": True,
            "blocks_queue_import": True,
            "future_wing_readback_required": True,
            "denial_reason": "real_wing_hardware_readback_rollback_and_operator_review_required",
            "operator_next_step": "capture_real_wing_readback_and_rollback_proof_before_live_promotion",
            "future_live_promotion_requires": list(FUTURE_LIVE_REQUIRED_PROOFS),
            "why_blocked": [
                "live_apply_denied_until_real_wing_proof",
                "queue_import_denied_until_operator_review",
                "future_wing_readback_required",
            ],
        },
        "live_authority_boundary": {
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "readiness_closure_allowed": False,
            "wing_proof_required": True,
            "hardware_proof_required": True,
            "operator_review_required": True,
            "allowed_current_live_write_kinds": list(LIVE_ALLOWED_WRITE_KINDS),
            "read_only_or_dry_run_only": True,
            "live_apply_allowed": False,
            "queue_import_allowed": False,
            "can_authorize_live": False,
            "can_close_readiness": False,
            "mutates_mixer": False,
            "blocks_live_promotion": True,
            "blocks_queue_import": True,
            "future_wing_readback_required": True,
            "blocked_live_actions": list(profile["blocked_live_actions"]),
            "denial_reason": "real_wing_hardware_readback_rollback_and_operator_review_required",
        },
        "future_live_promotion_requirements": {
            "required_before_live_promotion": True,
            "read_only_or_dry_run_only": True,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "readiness_closure_allowed": False,
            "live_apply_allowed": False,
            "queue_import_allowed": False,
            "can_authorize_live": False,
            "can_close_readiness": False,
            "mutates_mixer": False,
            "operator_approval_required": True,
            "hardware_proof_required": True,
            "required_proofs": list(FUTURE_LIVE_REQUIRED_PROOFS),
            "blocked_until": "real_wing_hardware_readback_rollback_and_operator_review",
        },
        "operator_evidence_packet": {
            "packet_type": "offline_operator_correction_review",
            "read_only_or_dry_run_only": True,
            "live_apply_allowed": False,
            "queue_import_allowed": False,
            "source_evidence_required": True,
            "source_evidence_details_required": True,
            "problem_statement_required": True,
            "proposed_correction_required": True,
            "confidence_required": True,
            "confidence_review_required": True,
            "confidence_review_details_required": True,
            "operator_proposal_review_gate_required": True,
            "bounded_delta_required": True,
            "bounded_delta_details_required": True,
            "safety_blockers_required": True,
            "safety_blocker_details_required": True,
            "why_safe_details_required": True,
            "why_blocked_details_required": True,
            "safety_checks_required": True,
            "headroom_feedback_true_peak_checks_required": True,
            "headroom_feedback_true_peak_check_details_required": True,
            "operator_safety_review_matrix_required": True,
            "operator_readable_explanation_required": True,
            "operator_explanation_packet_required": True,
            "operator_next_action_details_required": True,
            "rollback_expectation_required": True,
            "rollback_readback_contract_required": True,
            "rollback_readback_details_required": True,
            "future_live_promotion_requirements_required": True,
            "future_live_requirement_details_required": True,
            "future_wing_readback_contract_required": True,
            "live_authority_boundary_required": True,
            "operator_review_required": True,
            "operator_review_record_template_required": True,
            "operator_review_record_field_details_required": True,
            "operator_review_checklist_details_required": True,
            "operator_review_decision_option_details_required": True,
            "test_evidence_required": True,
        },
        "operator_review_record_template": {
            "record_type": "offline_operator_correction_review_record",
            "read_only_or_dry_run_only": True,
            "live_apply_allowed": False,
            "queue_import_allowed": False,
            "can_authorize_live": False,
            "can_close_readiness": False,
            "operator_review_required": True,
            "allowed_decisions": [
                "request_more_evidence",
                "accept_as_offline_proposal",
                "reject_proposal",
                "defer_until_real_wing_proof",
            ],
            "required_record_fields": [
                "reviewer",
                "reviewed_at_utc",
                "decision",
                "rationale",
                "evidence_packet_reference",
                "source_material_reference",
                "source_evidence_acknowledgement",
                "test_evidence_reference",
                "test_evidence_acknowledgement",
                "verification_report_reference",
                "verification_report_acknowledgement",
                "measured_problem_acknowledgement",
                "proposed_correction_acknowledgement",
                "bounded_delta_acknowledgement",
                "confidence_acknowledgement",
                "operator_explanation_acknowledgement",
                "future_wing_readback_acknowledgement",
                "headroom_feedback_true_peak_acknowledgement",
                "rollback_readback_acknowledgement",
                "safety_blockers_acknowledgement",
                "required_confirmations_acknowledgement",
                "offline_acceptance_boundary_acknowledgement",
                "operator_notes",
            ],
            "future_live_promotion_requires_real_wing_proof": True,
        },
        "operator_review_checklist": [
            "verify_source_evidence_matches_current_offline_material",
            "confirm_measured_or_audible_problem",
            "inspect_proposed_correction_and_bounded_delta",
            "verify_required_safety_checks",
            "confirm_rollback_expectation",
            "do_not_queue_or_apply_live",
            "capture_real_wing_readback_and_rollback_proof_before_live_promotion",
        ],
        "operator_review_decision_options": [
            {
                "decision": "request_more_evidence",
                "effect": "keeps_proposal_blocked_until_required_evidence_complete",
                "live_apply_allowed": False,
                "queue_import_allowed": False,
            },
            {
                "decision": "accept_as_offline_proposal",
                "effect": "records_operator_review_for_offline_plan_only",
                "live_apply_allowed": False,
                "queue_import_allowed": False,
            },
            {
                "decision": "reject_proposal",
                "effect": "keeps_or_discards_proposal_without_console_mutation",
                "live_apply_allowed": False,
                "queue_import_allowed": False,
            },
            {
                "decision": "defer_until_real_wing_proof",
                "effect": "requires_future_wing_readback_rollback_and_operator_review",
                "live_apply_allowed": False,
                "queue_import_allowed": False,
            },
        ],
        "read_only_or_dry_run_only": True,
        "live_apply_allowed": False,
        "queue_import_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "readiness_closure_allowed": False,
        "can_authorize_live": False,
        "can_close_readiness": False,
        "mutates_mixer": False,
        "operator_review_required": True,
        "blocks_live_promotion": True,
        "blocks_queue_import": True,
        "future_wing_readback_required": True,
    }


def _operator_contract_identity_profile(
    *,
    proposal: Dict[str, Any],
    method_id: str,
    target: Any,
    operator_contract: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "method_id": method_id,
        "source": proposal.get("source"),
        "target": target if target is not None else proposal.get("target"),
        "value_type": operator_contract["value_type"],
        "target_kind": operator_contract["target_kind"],
        "old_value_source": operator_contract["old_value_source"],
        "objective_family": operator_contract["objective_family"],
        "bounded_delta_unit": operator_contract["bounded_delta_unit"],
        "bounded_delta_policy": operator_contract["bounded_delta_policy"],
    }


def _operator_correction_context_profile(
    identity_profile: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "correction_method_id": identity_profile["method_id"],
        "correction_source": identity_profile["source"],
        "correction_target": identity_profile["target"],
        "correction_value_type": identity_profile["value_type"],
        "correction_target_kind": identity_profile["target_kind"],
        "correction_old_value_source": identity_profile["old_value_source"],
        "correction_objective_family": identity_profile["objective_family"],
        "correction_bounded_delta_unit": identity_profile["bounded_delta_unit"],
        "correction_bounded_delta_policy": identity_profile[
            "bounded_delta_policy"
        ],
    }


def _operator_contract_inputs(
    *,
    method_id: str,
    requested_change_value_type: Optional[str],
    value_type: Optional[str] = None,
    target_kind: Optional[str] = None,
    old_value_source: Optional[str] = None,
) -> Dict[str, str]:
    method_contract = PROPOSAL_ONLY_CONTRACTS.get(method_id, {})
    requested_contract = REQUESTED_CHANGE_OPERATOR_CONTRACTS.get(
        str(requested_change_value_type or "").strip().lower(),
        {},
    )
    resolved_value_type = (
        value_type
        or method_contract.get("value_type")
        or requested_contract.get("value_type")
        or str(requested_change_value_type or "proposal_only_method")
    )
    return {
        "value_type": resolved_value_type,
        "target_kind": (
            target_kind
            or method_contract.get("target_kind")
            or requested_contract.get("target_kind")
            or "proposal_target"
        ),
        "old_value_source": (
            old_value_source
            or method_contract.get("old_value_source")
            or requested_contract.get("old_value_source")
            or "operator_or_physical_read_required_before_promotion"
        ),
    }


def _contract_specific_problem_summary(
    *,
    proposal: Dict[str, Any],
    method_id: str,
    operator_contract: Dict[str, Any],
) -> str:
    proposal_detail = ""
    value = proposal.get("reason")
    if isinstance(value, str) and value.strip():
        proposal_detail = value.strip()
        if proposal_detail[-1] not in ".!?":
            proposal_detail = f"{proposal_detail}."
    proposal_detail_sentence = (
        f" Proposal detail: {proposal_detail}" if proposal_detail else ""
    )
    return (
        f"{method_id} produced proposal-only evidence for "
        f"{operator_contract['objective_family']} review.{proposal_detail_sentence} "
        "The measured or audible problem must be reviewed as part of the "
        "offline correction packet; a concrete "
        "correction must still be reviewed by the operator before any future "
        "live promotion."
    )


def _contract_specific_proposed_correction(
    *,
    method_id: str,
    operator_contract: Dict[str, Any],
    evidence_completeness: Dict[str, Any],
    evidence_gate: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "status": "proposal_contract_evidence_review_only",
        "method_id": method_id,
        "objective_family": operator_contract["objective_family"],
        "target_kind": operator_contract["future_wing_readback_contract"][
            "target_kind"
        ],
        "bounded_delta_unit": operator_contract["bounded_delta_unit"],
        "bounded_delta_policy": operator_contract["bounded_delta_policy"],
        "required_evidence_fields": list(
            operator_contract["required_evidence_fields"]
        ),
        "provided_evidence_fields": list(evidence_completeness.get("provided") or []),
        "missing_required_evidence": list(
            evidence_gate.get("missing_required_evidence") or []
        ),
        "operator_action": (
            "review_complete_evidence_and_supply_concrete_proposed_action"
            if evidence_completeness.get("complete")
            else evidence_gate["operator_action"]
        ),
        "concrete_proposed_action_required": True,
        "read_only_or_dry_run_only": True,
        "live_apply_allowed": False,
        "queue_import_allowed": False,
        "can_apply_live": False,
        "can_import_to_queue": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "readiness_closure_allowed": False,
        "can_authorize_live": False,
        "can_close_readiness": False,
        "mutates_mixer": False,
        "operator_review_required": True,
        "blocks_live_promotion": True,
        "blocks_queue_import": True,
        "future_wing_readback_required": True,
        "denial_reason": operator_contract["future_live_promotion_requirements"][
            "blocked_until"
        ],
        "operator_next_step": "review_required_safety_checks_and_collect_missing_evidence",
        "future_live_promotion_requires": list(
            operator_contract["future_live_promotion_requirements"]["required_proofs"]
        ),
        "why_blocked": [
            "live_apply_denied_until_real_wing_proof",
            "queue_import_denied_until_operator_review",
            "future_wing_readback_required",
            "operator_safety_review_required",
        ],
    }


def _proposed_correction_review_payload(
    *,
    proposed_correction: Dict[str, Any],
    source: Any,
    target: Any,
    operator_contract: Dict[str, Any],
    evidence_completeness: Dict[str, Any],
    evidence_gate: Dict[str, Any],
    bounded_delta: Dict[str, Any],
    future_live_promotion_requirements: Dict[str, Any],
    rollback_readback_contract: Dict[str, Any],
) -> Dict[str, Any]:
    required_evidence_status = _required_evidence_status(
        evidence_completeness,
        evidence_gate,
    )
    proposed_status = proposed_correction.get("status")
    if isinstance(proposed_status, str) and proposed_status.startswith("blocked"):
        review_only_status = proposed_status
    else:
        review_only_status = "proposal_contract_evidence_review_only"
    identity_contract = (
        operator_contract.get("future_wing_readback_contract")
        if isinstance(operator_contract.get("future_wing_readback_contract"), dict)
        else {}
    )
    requested_method_id = proposed_correction.get("method_id")
    requested_source = proposed_correction.get("source")
    requested_target = proposed_correction.get("target")
    requested_value_type = proposed_correction.get("value_type")
    requested_target_kind = proposed_correction.get("target_kind")
    requested_old_value_source = proposed_correction.get("old_value_source")
    requested_readback_required = proposed_correction.get("readback_required")
    requested_queue_action = proposed_correction.get("queue_action")
    requested_apply_now = proposed_correction.get("apply_now")
    requested_dispatch_supervised_apply = proposed_correction.get(
        "dispatch_supervised_apply"
    )
    contract_method_id = identity_contract.get("method_id") or requested_method_id
    contract_source = identity_contract.get("source")
    if contract_source is None:
        contract_source = source
    contract_target = identity_contract.get("target")
    if contract_target is None:
        contract_target = target
    return {
        **proposed_correction,
        "status": review_only_status,
        "method_id": contract_method_id,
        "source": contract_source,
        "target": contract_target,
        "value_type": operator_contract["value_type"],
        "target_kind": operator_contract["target_kind"],
        "old_value_source": operator_contract["old_value_source"],
        "requested_method_id": requested_method_id,
        "requested_source": requested_source,
        "requested_target": requested_target,
        "requested_value_type": requested_value_type,
        "requested_target_kind": requested_target_kind,
        "requested_old_value_source": requested_old_value_source,
        "requested_readback_required": requested_readback_required,
        "requested_queue_action": requested_queue_action,
        "requested_apply_now": requested_apply_now,
        "requested_dispatch_supervised_apply": requested_dispatch_supervised_apply,
        "contract_method_id": contract_method_id,
        "contract_source": contract_source,
        "contract_target": contract_target,
        "contract_value_type": operator_contract["value_type"],
        "contract_target_kind": operator_contract["target_kind"],
        "contract_old_value_source": operator_contract["old_value_source"],
        "proposal_review_status": "proposed_correction_requires_operator_review",
        "bounded_delta": bounded_delta,
        "bounded_delta_policy": operator_contract["bounded_delta_policy"],
        "bounded_delta_unit": operator_contract["bounded_delta_unit"],
        "rollback_expectation": operator_contract["rollback_expectation"],
        "rollback_readback_contract": rollback_readback_contract,
        "required_evidence_fields": list(operator_contract["required_evidence_fields"]),
        "provided_evidence_fields": list(evidence_completeness.get("provided") or []),
        "missing_required_evidence": list(
            evidence_gate.get("missing_required_evidence") or []
        ),
        "required_evidence_status": required_evidence_status,
        "required_evidence_actions": required_evidence_status[
            "required_evidence_actions"
        ],
        "evidence_gate_status": evidence_gate["status"],
        "operator_action": evidence_gate["operator_action"],
        "readback_required": True,
        "readback_required_before_live_promotion": True,
        "future_wing_readback_contract": operator_contract[
            "future_wing_readback_contract"
        ],
        "future_live_promotion_requires": list(
            future_live_promotion_requirements.get("required_proofs") or []
        ),
        "read_only_or_dry_run_only": True,
        "live_apply_allowed": False,
        "queue_import_allowed": False,
        "queue_action": "not_queueable_until_promoted",
        "apply_now": False,
        "dispatch_supervised_apply": False,
        "can_apply_live": False,
        "can_import_to_queue": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "readiness_closure_allowed": False,
        "can_authorize_live": False,
        "can_close_readiness": False,
        "mutates_mixer": False,
        "operator_review_required": True,
        "blocks_live_promotion": True,
        "blocks_queue_import": True,
        "future_wing_readback_required": True,
        "denial_reason": future_live_promotion_requirements["blocked_until"],
        "operator_next_step": "review_proposed_correction_bounded_delta_and_required_evidence",
        "why_blocked": _unique_fields(
            [
                "proposed_correction_requires_operator_review",
                "live_apply_denied_until_real_wing_proof",
                "queue_import_denied_until_operator_review",
                "future_wing_readback_required",
            ],
            evidence_gate.get("blockers") or [],
        ),
    }


def _contract_specific_operator_explanation(
    *,
    proposal: Dict[str, Any],
    method_id: str,
    operator_contract: Dict[str, Any],
    evidence_gate: Dict[str, Any],
) -> str:
    proposal_detail = ""
    for field in ("reason", "title"):
        value = proposal.get(field)
        if isinstance(value, str) and value.strip():
            proposal_detail = value.strip()
            break
    if proposal_detail and proposal_detail[-1] not in ".!?":
        proposal_detail = f"{proposal_detail}."
    proposal_detail_sentence = (
        f" Proposal detail: {proposal_detail}" if proposal_detail else ""
    )
    target_kind = operator_contract["future_wing_readback_contract"]["target_kind"]
    missing_required_evidence = list(
        evidence_gate.get("missing_required_evidence") or []
    )
    evidence_sentence = (
        (
            "Required evidence is still missing for "
            f"{target_kind}: {', '.join(missing_required_evidence)}; "
        )
        if missing_required_evidence
        else f"Required evidence is complete for {target_kind}; "
    )
    return (
        f"{method_id} is an offline/dry-run proposal-only review for "
        f"{operator_contract['objective_family']}.{proposal_detail_sentence} "
        f"{evidence_sentence}{evidence_gate['operator_action']}. "
        "Live apply and queue import remain "
        "blocked until real WING readback, rollback proof, and operator approval."
    )


def _change_old_value(change: Dict[str, Any]) -> Any:
    return change.get("old_value") if "old_value" in change else change.get("current_value")


def _change_proposed_value(change: Dict[str, Any]) -> Any:
    return change.get("proposed_value") if "proposed_value" in change else change.get("value")


def _change_delta_db(change: Dict[str, Any]) -> Optional[float]:
    delta = _coerce_float(change.get("delta_db"))
    if delta is not None:
        return round(delta, 1)
    old_value = _coerce_float(_change_old_value(change))
    proposed_value = _coerce_float(_change_proposed_value(change))
    if old_value is None or proposed_value is None:
        return None
    return round(proposed_value - old_value, 1)


def _target_from_change(change: Dict[str, Any]) -> Any:
    return change.get("target") if "target" in change else change.get("channel")


def _bounded_delta_denial_metadata(
    operator_contract: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "read_only_or_dry_run_only": True,
        "live_apply_allowed": False,
        "queue_import_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "readiness_closure_allowed": False,
        "can_authorize_live": False,
        "can_close_readiness": False,
        "mutates_mixer": False,
        "operator_review_required": True,
        "blocks_live_promotion": True,
        "blocks_queue_import": True,
        "future_wing_readback_required": True,
        "denial_reason": operator_contract["future_live_promotion_requirements"][
            "blocked_until"
        ],
        "operator_next_step": "review_bounded_delta_and_required_evidence",
        "future_live_promotion_requires": list(
            operator_contract["future_live_promotion_requirements"]["required_proofs"]
        ),
        "why_blocked": [
            "live_apply_denied_until_real_wing_proof",
            "queue_import_denied_until_operator_review",
            "future_wing_readback_required",
        ],
    }


def _bounded_delta_contract_target(
    change: Dict[str, Any],
    operator_contract: Dict[str, Any],
) -> Any:
    identity_contract = operator_contract.get("future_wing_readback_contract")
    if (
        isinstance(identity_contract, dict)
        and identity_contract.get("target") is not None
    ):
        return identity_contract["target"]
    if change.get("channel") is not None:
        return change["channel"]
    if change.get("channel_id") is not None:
        return change["channel_id"]
    return _target_from_change(change)


def _bounded_delta_entry(
    change: Dict[str, Any],
    operator_contract: Dict[str, Any],
) -> Dict[str, Any]:
    requested_target = _target_from_change(change)
    contract_target = _bounded_delta_contract_target(change, operator_contract)
    entry = {
        "value_type": operator_contract["value_type"],
        "target": contract_target,
        "requested_value_type": change.get("value_type"),
        "requested_target": requested_target,
        "requested_readback_required": change.get("readback_required"),
        "contract_value_type": operator_contract["value_type"],
        "contract_target": contract_target,
        "old_value": _change_old_value(change),
        "proposed_value": _change_proposed_value(change),
        "delta_db": _change_delta_db(change),
        "unit": operator_contract["bounded_delta_unit"],
        "policy": operator_contract["bounded_delta_policy"],
        "readback_required": True,
    }
    entry.update(_bounded_delta_denial_metadata(operator_contract))
    if isinstance(change.get("safety_evidence"), list):
        entry["safety_evidence"] = change["safety_evidence"]
    return entry


def _bounded_delta_from_action(
    proposed_action: Dict[str, Any],
    operator_contract: Dict[str, Any],
) -> Dict[str, Any]:
    nested_params = (
        proposed_action.get("params")
        if isinstance(proposed_action.get("params"), dict)
        else {}
    )
    delta = None
    for key in (
        "suggested_delta_db",
        "delta_db",
        "change_db",
        "gain_db",
        "delta_send_db",
        "delta_pan_percent",
        "suggested_trim_db",
        "broad_eq_tilt_db",
        "range_db",
        "max_gain_reduction_db",
        "makeup_gain_db",
        "suggested_eq_delta_db",
    ):
        delta = _coerce_float(proposed_action.get(key))
        if delta is None:
            delta = _coerce_float(nested_params.get(key))
        if delta is not None:
            break
    parameter_keys = (
        "filter_type",
        "frequency_hz",
        "gain_db",
        "q",
        "q_factor",
        "threshold_db",
        "ratio",
        "attack_ms",
        "hold_ms",
        "release_ms",
        "knee_db",
        "range_db",
        "hysteresis_db",
        "makeup_gain_db",
        "max_gain_reduction_db",
        "target_gain_reduction_db",
        "current_send_db",
        "target_send_db",
        "delta_send_db",
        "delta_db",
        "send_level_db",
        "fx_type",
        "bus_name",
        "return_level_db",
        "hpf_hz",
        "current_hpf_hz",
        "lpf_hz",
        "predelay_ms",
        "decay_s",
        "delay_ms",
        "measured_delay_ms",
        "phase_invert",
        "correlation_delta",
        "gcc_phat_confidence",
        "ducking_db",
        "current_pan_percent",
        "target_pan_percent",
        "delta_pan_percent",
        "pan_percent",
        "target_plane_lufs",
        "boost_allowed",
        "route_id",
        "route_kind",
        "current_source",
        "proposed_source",
        "artifact_id",
        "artifact_name",
        "artifact_type",
        "target_scope",
        "active_scene_id",
        "active_scene_name",
        "target_integrated_lufs",
        "max_true_peak_dbfs",
        "suggested_trim_db",
        "limiter_ceiling_dbfs",
        "broad_eq_tilt_db",
        "low_shelf_db",
        "high_shelf_db",
        "bus_compression_ratio",
        "bus_compression_threshold_db",
        "target_source_id",
        "protected_source_id",
        "stem_id",
        "band",
        "center_hz",
        "suggested_eq_delta_db",
        "channel",
        "channel_id",
        "current_muted",
        "proposed_muted",
        "mute_reason",
    )
    parameter_deltas = {
        key: proposed_action[key]
        for key in parameter_keys
        if key in proposed_action
    }
    parameter_deltas.update({
        key: nested_params[key]
        for key in parameter_keys
        if key in nested_params
    })
    return {
        "action_type": proposed_action.get("action_type"),
        "target": (
            proposed_action.get("bus_id")
            or proposed_action.get("channel")
            or proposed_action.get("channel_id")
            or proposed_action.get("target_source_id")
            or proposed_action.get("target")
        ),
        "old_value": (
            proposed_action.get("current_fader_db")
            if "current_fader_db" in proposed_action
            else proposed_action.get(
                "current_value",
                proposed_action.get("current_muted", proposed_action.get("old_value")),
            )
        ),
        "proposed_value": (
            proposed_action.get("target_fader_db")
            if "target_fader_db" in proposed_action
            else proposed_action.get(
                "proposed_value",
                proposed_action.get("proposed_muted"),
            )
        ),
        "delta_db": round(delta, 1) if delta is not None else None,
        "parameter_deltas": parameter_deltas,
        "unit": operator_contract["bounded_delta_unit"],
        "policy": operator_contract["bounded_delta_policy"],
        "readback_required": True,
        **_bounded_delta_denial_metadata(operator_contract),
    }


def _bounded_delta_payload(
    *,
    requested_change: Optional[Dict[str, Any]],
    requested_changes: Optional[List[Dict[str, Any]]],
    proposed_action: Optional[Dict[str, Any]],
    operator_contract: Dict[str, Any],
) -> Dict[str, Any]:
    if requested_change:
        return _bounded_delta_entry(requested_change, operator_contract)
    if requested_changes:
        return {
            "changes": [
                _bounded_delta_entry(change, operator_contract)
                for change in requested_changes
            ],
            "unit": operator_contract["bounded_delta_unit"],
            "policy": operator_contract["bounded_delta_policy"],
            "readback_required": True,
            **_bounded_delta_denial_metadata(operator_contract),
        }
    if proposed_action:
        return _bounded_delta_from_action(proposed_action, operator_contract)
    return {
        "status": "missing_requested_change_or_proposed_action",
        "unit": operator_contract["bounded_delta_unit"],
        "policy": operator_contract["bounded_delta_policy"],
        "readback_required": True,
        **_bounded_delta_denial_metadata(operator_contract),
    }


def _bounded_delta_details(
    *,
    bounded_delta: Dict[str, Any],
    operator_contract: Dict[str, Any],
    live_authority_boundary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    changes = bounded_delta.get("changes")
    delta_rows = (
        list(changes)
        if isinstance(changes, list) and changes
        else [bounded_delta]
    )
    denial_reason = (
        live_authority_boundary.get("denial_reason")
        or bounded_delta.get("denial_reason")
        or "operator_review_and_future_wing_readback_required"
    )
    details: List[Dict[str, Any]] = []

    for index, delta in enumerate(delta_rows):
        if not isinstance(delta, dict):
            delta = {"raw_delta": delta}
        parameter_deltas = (
            dict(delta.get("parameter_deltas") or {})
            if isinstance(delta.get("parameter_deltas"), dict)
            else {}
        )
        has_concrete_delta = any(
            delta.get(key) is not None
            for key in ("delta_db", "old_value", "proposed_value")
        ) or bool(parameter_deltas)
        status = (
            str(delta.get("status"))
            if delta.get("status")
            else (
                "bounded_delta_review_required"
                if has_concrete_delta
                else "concrete_delta_required_before_operator_review"
            )
        )
        details.append(
            {
                "detail": "bounded_delta_change",
                "source": "bounded_delta",
                "sources": ["bounded_delta"],
                "change_index": index,
                "value_type": delta.get("value_type"),
                "delta_action_type": delta.get("action_type"),
                "target": delta.get("target"),
                "requested_value_type": delta.get("requested_value_type"),
                "requested_target": delta.get("requested_target"),
                "requested_readback_required": delta.get(
                    "requested_readback_required"
                ),
                "contract_value_type": delta.get("contract_value_type")
                or operator_contract["value_type"],
                "contract_target": delta.get("contract_target")
                if delta.get("contract_target") is not None
                else delta.get("target"),
                "old_value": delta.get("old_value"),
                "proposed_value": delta.get("proposed_value"),
                "delta_db": delta.get("delta_db"),
                "parameter_deltas": parameter_deltas,
                "unit": delta.get("unit") or bounded_delta.get("unit"),
                "delta_unit": delta.get("unit") or bounded_delta.get("unit"),
                "policy": delta.get("policy") or bounded_delta.get("policy"),
                "bounded_delta_policy": (
                    delta.get("policy")
                    or bounded_delta.get("policy")
                    or operator_contract["bounded_delta_policy"]
                ),
                "target_kind": operator_contract["future_wing_readback_contract"][
                    "target_kind"
                ],
                "old_value_source": operator_contract[
                    "future_wing_readback_contract"
                ]["old_value_source"],
                "readback_required": True,
                "readback_required_before_live_promotion": True,
                "has_concrete_delta": has_concrete_delta,
                "status": status,
                "action_type": "review_bounded_delta",
                "operator_action": "review_bounded_delta_and_required_evidence",
                "safety_evidence": list(delta.get("safety_evidence") or []),
                "required_before_operator_acceptance": True,
                "required_before_live_promotion": True,
                "required_before_queue_import": True,
                "bounded_delta_required": True,
                "read_only_or_dry_run_only": True,
                "live_apply_allowed": False,
                "queue_import_allowed": False,
                "live_authority_granted": False,
                "full_autonomous_ready": False,
                "readiness_closure_allowed": False,
                "can_authorize_live": False,
                "can_close_readiness": False,
                "can_import_to_queue": False,
                "mutates_mixer": False,
                "operator_review_required": True,
                "future_wing_readback_required": True,
                "blocks_live_promotion": True,
                "blocks_queue_import": True,
                "denial_reason": denial_reason,
            }
        )
    return details


def _safety_evidence_items(
    requested_change: Optional[Dict[str, Any]],
    requested_changes: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    changes = []
    if requested_change:
        changes.append(requested_change)
    if requested_changes:
        changes.extend(requested_changes)

    evidence_items: List[Dict[str, Any]] = []
    for change in changes:
        safety_evidence = change.get("safety_evidence")
        if isinstance(safety_evidence, list):
            evidence_items.extend(
                item for item in safety_evidence if isinstance(item, dict)
            )
    return evidence_items


def _normalize_evidence_field_name(value: Any) -> str:
    normalized = []
    previous_separator = False
    for char in str(value or "").strip().lower():
        if char.isalnum():
            normalized.append(char)
            previous_separator = False
        elif normalized and not previous_separator:
            normalized.append("_")
            previous_separator = True
    return "".join(normalized).strip("_")


def _add_evidence_field_names(names: List[str], seen: set, value: Any) -> None:
    normalized = _normalize_evidence_field_name(value)
    if not normalized or normalized in seen:
        return
    seen.add(normalized)
    names.append(normalized)


def _collect_provided_evidence_fields(
    *,
    proposal: Dict[str, Any],
    requested_change: Optional[Dict[str, Any]],
    requested_changes: Optional[List[Dict[str, Any]]],
    proposed_action: Optional[Dict[str, Any]],
) -> List[str]:
    names: List[str] = []
    seen = set()

    def add_mapping_keys(mapping: Any) -> None:
        if not isinstance(mapping, dict):
            return
        for key, value in mapping.items():
            _add_evidence_field_names(names, seen, key)
            if isinstance(value, dict):
                for nested_key in value:
                    _add_evidence_field_names(names, seen, nested_key)

    add_mapping_keys(proposal.get("raw"))
    add_mapping_keys(proposal.get("evidence"))
    add_mapping_keys(proposed_action)
    if isinstance(proposed_action, dict):
        add_mapping_keys(proposed_action.get("params"))

    changes = []
    if requested_change:
        changes.append(requested_change)
    if requested_changes:
        changes.extend(requested_changes)
    for change in changes:
        add_mapping_keys(change)
        for item in change.get("safety_evidence") or []:
            if not isinstance(item, dict):
                continue
            add_mapping_keys(item)
            _add_evidence_field_names(names, seen, item.get("label"))

    return names


def _evidence_field_satisfied(required: str, provided_fields: Iterable[str]) -> bool:
    normalized_required = _normalize_evidence_field_name(required)
    for field in provided_fields:
        if (
            field == normalized_required
            or field.startswith(f"{normalized_required}_")
            or normalized_required.startswith(f"{field}_")
        ):
            return True
    return False


def _evidence_completeness(
    *,
    required_evidence_fields: Iterable[str],
    proposal: Dict[str, Any],
    requested_change: Optional[Dict[str, Any]],
    requested_changes: Optional[List[Dict[str, Any]]],
    proposed_action: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    required = list(required_evidence_fields)
    provided = _collect_provided_evidence_fields(
        proposal=proposal,
        requested_change=requested_change,
        requested_changes=requested_changes,
        proposed_action=proposed_action,
    )
    missing = [
        field
        for field in required
        if not _evidence_field_satisfied(field, provided)
    ]
    if not required:
        status = "not_declared"
    elif not missing:
        status = "complete"
    elif len(missing) == len(required):
        status = "missing"
    else:
        status = "partial"
    return {
        "status": status,
        "required": required,
        "provided": provided,
        "missing": missing,
        "complete": status == "complete",
    }


def _required_evidence_status(
    evidence_completeness: Dict[str, Any],
    evidence_gate: Dict[str, Any],
) -> Dict[str, Any]:
    required = list(evidence_completeness.get("required") or [])
    provided = list(evidence_completeness.get("provided") or [])
    missing = list(evidence_gate.get("missing_required_evidence") or [])
    missing_set = set(missing)
    correction_context_profile = dict(
        evidence_completeness.get("correction_context_profile") or {}
    )
    rows = []
    for field in required:
        matched_provided = [
            provided_field
            for provided_field in provided
            if _evidence_field_satisfied(field, [provided_field])
        ]
        is_missing = field in missing_set
        rows.append(
            {
                **correction_context_profile,
                "field": field,
                "status": (
                    "missing_required_evidence"
                    if is_missing
                    else "provided_operator_review_required"
                ),
                "provided": not is_missing,
                "missing": is_missing,
                "matched_provided_fields": matched_provided,
                "operator_action": (
                    "collect_missing_required_evidence_before_operator_review"
                    if is_missing
                    else "operator_review_required_before_future_live_promotion"
                ),
                "read_only_or_dry_run_only": True,
                "live_apply_allowed": False,
                "queue_import_allowed": False,
                "live_authority_granted": False,
                "full_autonomous_ready": False,
                "readiness_closure_allowed": False,
                "can_authorize_live": False,
                "can_close_readiness": False,
                "mutates_mixer": False,
                "operator_review_required": True,
                "future_wing_readback_required": True,
                "blocks_live_promotion": True,
                "blocks_queue_import": True,
                "denial_reason": "operator_review_and_future_wing_readback_required",
            }
        )
    action_rows = []
    for row in rows:
        field = row["field"]
        missing_field = bool(row["missing"])
        action = (
            f"collect_missing_required_evidence:{field}"
            if missing_field
            else f"review_required_evidence:{field}"
        )
        action_rows.append(
            {
                **correction_context_profile,
                "field": field,
                "action": action,
                "action_type": (
                    "collect_missing_required_evidence"
                    if missing_field
                    else "review_provided_required_evidence"
                ),
                "status": row["status"],
                "provided": row["provided"],
                "missing": row["missing"],
                "matched_provided_fields": list(row["matched_provided_fields"]),
                "required_before_operator_acceptance": missing_field,
                "required_before_live_promotion": True,
                "operator_action": row["operator_action"],
                "read_only_or_dry_run_only": True,
                "live_apply_allowed": False,
                "queue_import_allowed": False,
                "live_authority_granted": False,
                "full_autonomous_ready": False,
                "readiness_closure_allowed": False,
                "can_authorize_live": False,
                "can_close_readiness": False,
                "mutates_mixer": False,
                "operator_review_required": True,
                "future_wing_readback_required": True,
                "blocks_live_promotion": True,
                "blocks_queue_import": True,
                "denial_reason": "operator_review_and_future_wing_readback_required",
            }
        )

    return {
        **correction_context_profile,
        "status": evidence_completeness.get("status"),
        "complete": bool(evidence_completeness.get("complete")),
        "required_fields": required,
        "provided_fields": provided,
        "missing_fields": missing,
        "required_count": len(required),
        "provided_count": len(provided),
        "missing_count": len(missing),
        "rows": rows,
        "required_evidence_actions": action_rows,
        "operator_next_actions": [row["action"] for row in action_rows],
        "operator_action": evidence_gate["operator_action"],
        "read_only_or_dry_run_only": True,
        "live_apply_allowed": False,
        "queue_import_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "readiness_closure_allowed": False,
        "can_authorize_live": False,
        "can_close_readiness": False,
        "mutates_mixer": False,
        "operator_review_required": True,
        "future_wing_readback_required": True,
        "blocks_live_promotion": True,
        "blocks_queue_import": True,
        "denial_reason": "operator_review_and_future_wing_readback_required",
    }


def _evidence_gate(evidence_completeness: Dict[str, Any]) -> Dict[str, Any]:
    missing = list(evidence_completeness.get("missing") or [])
    if missing:
        blockers = [
            "required_evidence_incomplete",
            *[f"required_evidence_missing:{field}" for field in missing],
        ]
        status = "blocked_until_required_evidence_complete"
        operator_action = "collect_missing_required_evidence_before_operator_review"
    else:
        blockers = []
        status = "required_evidence_complete_operator_review_still_required"
        operator_action = "operator_review_required_before_future_live_promotion"

    return {
        "status": status,
        "blocks_live_promotion": True,
        "blocks_queue_import": True,
        "operator_action": operator_action,
        "blockers": blockers,
        "missing_required_evidence": missing,
        "readback_required_before_live_promotion": True,
        "read_only_or_dry_run_only": True,
        "live_apply_allowed": False,
        "queue_import_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "readiness_closure_allowed": False,
        "can_authorize_live": False,
        "can_close_readiness": False,
        "mutates_mixer": False,
        "operator_review_required": True,
        "future_wing_readback_required": True,
        "denial_reason": "operator_review_and_future_wing_readback_required",
    }


def _live_authority_boundary(
    *,
    operator_contract: Dict[str, Any],
    evidence_gate: Dict[str, Any],
) -> Dict[str, Any]:
    contract_boundary = operator_contract["live_authority_boundary"]
    return {
        **contract_boundary,
        "readback_required_before_live_promotion": True,
        "evidence_gate_status": evidence_gate["status"],
        "read_only_or_dry_run_only": True,
        "can_authorize_live": False,
        "can_close_readiness": False,
        "mutates_mixer": False,
        "blocks_live_promotion": True,
        "blocks_queue_import": True,
        "future_wing_readback_required": True,
        "blockers": _unique_fields(
            [
                "live_authority_denied",
                "real_wing_proof_required",
                "operator_review_required",
            ],
            evidence_gate.get("blockers") or [],
        ),
    }


def _operator_review_checklist(
    *,
    operator_contract: Dict[str, Any],
    evidence_gate: Dict[str, Any],
    operator_proposal_review_gate: Optional[Dict[str, Any]] = None,
) -> List[str]:
    checklist = list(operator_contract["operator_review_checklist"])
    missing = list(evidence_gate.get("missing_required_evidence") or [])
    if missing:
        checklist.extend(
            [
                "collect_missing_required_evidence_before_operator_review",
                *[f"missing_required_evidence:{field}" for field in missing],
            ]
        )
    else:
        checklist.append("operator_review_required_before_future_live_promotion")
    if operator_proposal_review_gate:
        gate_action = operator_proposal_review_gate.get("operator_action")
        if gate_action and gate_action not in checklist:
            checklist.append(str(gate_action))
        for blocker in operator_proposal_review_gate.get("confidence_blockers") or []:
            checklist.append(f"confidence_blocker:{blocker}")
    return _unique_fields(checklist)


def _operator_review_decision_options(
    *,
    operator_contract: Dict[str, Any],
    operator_proposal_review_gate: Dict[str, Any],
) -> List[Dict[str, Any]]:
    recommended = operator_proposal_review_gate["recommended_decision"]
    options: List[Dict[str, Any]] = []
    for option in operator_contract["operator_review_decision_options"]:
        decision = option["decision"]
        options.append(
            {
                **option,
                "recommended": decision == recommended,
                "requires_real_wing_proof_before_live": True,
                "read_only_or_dry_run_only": True,
                "live_apply_allowed": False,
                "queue_import_allowed": False,
                "live_authority_granted": False,
                "full_autonomous_ready": False,
                "readiness_closure_allowed": False,
                "can_authorize_live": False,
                "can_close_readiness": False,
                "mutates_mixer": False,
                "operator_review_required": True,
                "future_wing_readback_required": True,
                "blocks_live_promotion": True,
                "blocks_queue_import": True,
                "denial_reason": (
                    "operator_review_and_future_wing_readback_required"
                ),
            }
        )
    return options


def _operator_review_decision_option_details(
    *,
    operator_review_decision_options: Iterable[Dict[str, Any]],
    operator_proposal_review_gate: Dict[str, Any],
    evidence_gate: Dict[str, Any],
    live_authority_boundary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    details: List[Dict[str, Any]] = []
    recommended_decision = operator_proposal_review_gate.get("recommended_decision")
    for option in operator_review_decision_options:
        decision = str(option.get("decision") or "")
        recommended = bool(option.get("recommended")) or decision == recommended_decision
        if decision == "request_more_evidence":
            action_type = "request_more_evidence"
            status = "recommended_until_required_evidence_complete" if recommended else "available"
            operator_action = "record_operator_decision:request_more_evidence"
            blocks_offline_acceptance = True
        elif decision == "accept_as_offline_proposal":
            action_type = "accept_offline_proposal_only"
            status = "recommended_offline_acceptance" if recommended else "available_after_operator_review"
            operator_action = "record_operator_decision:accept_as_offline_proposal"
            blocks_offline_acceptance = False
        elif decision == "reject_proposal":
            action_type = "reject_offline_proposal"
            status = "available_rejection"
            operator_action = "record_operator_decision:reject_proposal"
            blocks_offline_acceptance = True
        elif decision == "defer_until_real_wing_proof":
            action_type = "defer_until_real_wing_proof"
            status = "available_future_wing_deferral"
            operator_action = "record_operator_decision:defer_until_real_wing_proof"
            blocks_offline_acceptance = True
        else:
            action_type = "operator_review_decision"
            status = "available_operator_decision"
            operator_action = f"record_operator_decision:{decision}"
            blocks_offline_acceptance = True

        details.append(
            {
                "decision": decision,
                "option": decision,
                "effect": option.get("effect"),
                "recommended": recommended,
                "recommended_decision": recommended_decision,
                "action_type": action_type,
                "status": status,
                "operator_action": operator_action,
                "evidence_gate_status": evidence_gate.get("status"),
                "proposal_review_gate_status": operator_proposal_review_gate.get(
                    "status"
                ),
                "offline_acceptance_allowed": bool(
                    operator_proposal_review_gate.get("offline_acceptance_allowed")
                )
                and decision == "accept_as_offline_proposal",
                "blocks_offline_acceptance": blocks_offline_acceptance,
                "records_operator_review_only": True,
                "decision_records_live_authority": False,
                "decision_records_queue_import": False,
                "requires_real_wing_proof_before_live": True,
                "required_before_live_promotion": True,
                "required_before_queue_import": True,
                "read_only_or_dry_run_only": True,
                "live_apply_allowed": False,
                "queue_import_allowed": False,
                "live_authority_granted": False,
                "full_autonomous_ready": False,
                "readiness_closure_allowed": False,
                "can_authorize_live": False,
                "can_close_readiness": False,
                "can_import_to_queue": False,
                "mutates_mixer": False,
                "operator_review_required": True,
                "future_wing_readback_required": True,
                "blocks_live_promotion": True,
                "blocks_queue_import": True,
                "denial_reason": (
                    live_authority_boundary.get("denial_reason")
                    or "operator_review_and_future_wing_readback_required"
                ),
            }
        )
    return details


def _operator_review_record_template(
    *,
    proposal: Dict[str, Any],
    method_id: str,
    target: Any,
    operator_contract: Dict[str, Any],
    evidence_gate: Dict[str, Any],
    operator_proposal_review_gate: Dict[str, Any],
    operator_review_decision_options: List[Dict[str, Any]],
) -> Dict[str, Any]:
    recommended_decision = next(
        (
            option["decision"]
            for option in operator_review_decision_options
            if option.get("recommended")
        ),
        "request_more_evidence",
    )
    return {
        **operator_contract["operator_review_record_template"],
        "method_id": method_id,
        "objective_family": operator_contract["objective_family"],
        "source": proposal.get("source"),
        "target": target if target is not None else proposal.get("target"),
        "evidence_gate_status": evidence_gate["status"],
        "operator_proposal_review_gate_status": operator_proposal_review_gate[
            "status"
        ],
        "offline_acceptance_allowed": operator_proposal_review_gate[
            "offline_acceptance_allowed"
        ],
        "confidence_review_status": operator_proposal_review_gate[
            "confidence_review_status"
        ],
        "missing_required_evidence": list(
            evidence_gate.get("missing_required_evidence") or []
        ),
        "operator_action": operator_proposal_review_gate["operator_action"],
        "recommended_decision": recommended_decision,
        "decision_options": [
            option["decision"] for option in operator_review_decision_options
        ],
        "decision_records_live_authority": False,
        "decision_records_queue_import": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "readiness_closure_allowed": False,
        "mutates_mixer": False,
        "future_wing_readback_required": True,
        "blocks_live_promotion": True,
        "blocks_queue_import": True,
        "denial_reason": operator_proposal_review_gate["denial_reason"],
    }


def _operator_review_record_field_details(
    *,
    operator_review_record_template: Dict[str, Any],
    operator_proposal_review_gate: Dict[str, Any],
    evidence_gate: Dict[str, Any],
    live_authority_boundary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    details: List[Dict[str, Any]] = []
    required_fields = list(
        operator_review_record_template.get("required_record_fields") or []
    )
    allowed_decisions = list(
        operator_review_record_template.get("allowed_decisions") or []
    )
    recommended_decision = operator_review_record_template.get(
        "recommended_decision"
    )
    denial_reason = (
        live_authority_boundary.get("denial_reason")
        or operator_review_record_template.get("denial_reason")
        or "operator_review_and_future_wing_readback_required"
    )

    for field in required_fields:
        source, field_type = OPERATOR_REVIEW_RECORD_FIELD_PROFILES.get(
            str(field),
            ("operator_review_record_template", "text"),
        )
        is_decision_field = field == "decision"
        details.append(
            {
                "field": field,
                "record_field": field,
                "field_type": field_type,
                "record_field_type": field_type,
                "source": source,
                "sources": [source, "operator_review_record_template"],
                "status": (
                    "required_for_recommended_operator_decision_record"
                    if is_decision_field
                    else "required_before_operator_decision_record"
                ),
                "operator_action": f"fill_operator_review_record_field:{field}",
                "allowed_decisions": allowed_decisions if is_decision_field else [],
                "recommended_decision": (
                    recommended_decision if is_decision_field else None
                ),
                "evidence_gate_status": evidence_gate.get("status"),
                "proposal_review_gate_status": operator_proposal_review_gate.get(
                    "status"
                ),
                "offline_acceptance_allowed": bool(
                    operator_proposal_review_gate.get("offline_acceptance_allowed")
                ),
                "missing_required_evidence": list(
                    evidence_gate.get("missing_required_evidence") or []
                ),
                "field_required": True,
                "required_before_operator_acceptance": True,
                "required_before_live_promotion": True,
                "required_before_queue_import": True,
                "records_operator_review_only": True,
                "decision_records_live_authority": False,
                "decision_records_queue_import": False,
                "requires_real_wing_proof_before_live": True,
                "read_only_or_dry_run_only": True,
                "live_apply_allowed": False,
                "queue_import_allowed": False,
                "live_authority_granted": False,
                "full_autonomous_ready": False,
                "readiness_closure_allowed": False,
                "can_authorize_live": False,
                "can_close_readiness": False,
                "can_import_to_queue": False,
                "mutates_mixer": False,
                "operator_review_required": True,
                "future_wing_readback_required": True,
                "blocks_live_promotion": True,
                "blocks_queue_import": True,
                "denial_reason": denial_reason,
            }
        )
    return details


def _confidence_review(proposal: Dict[str, Any]) -> Dict[str, Any]:
    raw_confidence = proposal.get("confidence")
    confidence_value = _coerce_float(raw_confidence)

    if confidence_value is None:
        status = "missing_confidence"
        operator_action = "provide_or_recompute_confidence_before_operator_review"
        blockers = ["confidence_missing"]
    elif confidence_value < 0.0 or confidence_value > 1.0:
        status = "invalid_confidence_range"
        operator_action = "normalize_confidence_to_zero_one_before_operator_review"
        blockers = ["confidence_out_of_range"]
    elif confidence_value < 0.5:
        status = "low_confidence_operator_review_required"
        operator_action = "operator_review_required_before_accepting_low_confidence_proposal"
        blockers = ["low_confidence"]
    else:
        status = "confidence_available_operator_review_required"
        operator_action = "operator_review_required_before_future_live_promotion"
        blockers = []

    return {
        "raw_confidence": raw_confidence,
        "confidence": confidence_value,
        "expected_range": [0.0, 1.0],
        "low_confidence_threshold": 0.5,
        "status": status,
        "operator_action": operator_action,
        "blockers": blockers,
        "read_only_or_dry_run_only": True,
        "live_apply_allowed": False,
        "queue_import_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "readiness_closure_allowed": False,
        "can_authorize_live": False,
        "can_close_readiness": False,
        "mutates_mixer": False,
        "blocks_live_promotion": True,
        "blocks_queue_import": True,
        "operator_review_required": True,
        "future_wing_readback_required": True,
    }


def _confidence_review_details(
    *,
    confidence_review: Dict[str, Any],
    operator_proposal_review_gate: Dict[str, Any],
    live_authority_boundary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    confidence_value = confidence_review.get("confidence")
    expected_range = list(confidence_review.get("expected_range") or [0.0, 1.0])
    low_threshold = confidence_review.get("low_confidence_threshold")
    blockers = list(confidence_review.get("blockers") or [])
    confidence_available = confidence_value is not None
    confidence_in_expected_range = bool(
        confidence_available
        and expected_range[0] <= confidence_value <= expected_range[1]
    )
    low_confidence = bool(
        confidence_in_expected_range
        and low_threshold is not None
        and confidence_value < low_threshold
    )
    denial_reason = (
        live_authority_boundary.get("denial_reason")
        or "operator_review_and_future_wing_readback_required"
    )

    def base_row(
        *,
        detail: str,
        status: str,
        operator_action: str,
        blocker: Optional[str] = None,
        action_type: str = "review_confidence",
    ) -> Dict[str, Any]:
        return {
            "detail": detail,
            "source": "confidence_review",
            "sources": ["confidence_review"],
            "field": "confidence",
            "fields": ["confidence"],
            "raw_confidence": confidence_review.get("raw_confidence"),
            "confidence": confidence_value,
            "expected_range": expected_range,
            "low_confidence_threshold": low_threshold,
            "confidence_available": confidence_available,
            "confidence_in_expected_range": confidence_in_expected_range,
            "low_confidence": low_confidence,
            "status": status,
            "blocker": blocker,
            "blockers": [blocker] if blocker else list(blockers),
            "action_type": action_type,
            "operator_action": operator_action,
            "proposal_review_gate_status": operator_proposal_review_gate.get(
                "status"
            ),
            "offline_acceptance_allowed": bool(
                operator_proposal_review_gate.get("offline_acceptance_allowed")
            ),
            "recommended_decision": operator_proposal_review_gate.get(
                "recommended_decision"
            ),
            "required_before_operator_acceptance": True,
            "required_before_live_promotion": True,
            "required_before_queue_import": True,
            "read_only_or_dry_run_only": True,
            "live_apply_allowed": False,
            "queue_import_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "readiness_closure_allowed": False,
            "can_authorize_live": False,
            "can_close_readiness": False,
            "can_import_to_queue": False,
            "mutates_mixer": False,
            "operator_review_required": True,
            "future_wing_readback_required": True,
            "blocks_live_promotion": True,
            "blocks_queue_import": True,
            "denial_reason": denial_reason,
        }

    details = [
        base_row(
            detail="confidence_value_review",
            status=str(confidence_review.get("status") or "confidence_review"),
            operator_action=str(
                confidence_review.get("operator_action")
                or "operator_review_required_before_future_live_promotion"
            ),
        )
    ]
    for blocker in blockers:
        details.append(
            base_row(
                detail="confidence_blocker",
                status=str(confidence_review.get("status") or "blocked"),
                operator_action=f"resolve_confidence_blocker:{blocker}",
                blocker=str(blocker),
                action_type="resolve_confidence_blocker",
            )
        )
    return details


def _operator_proposal_review_gate(
    *,
    evidence_gate: Dict[str, Any],
    confidence_review: Dict[str, Any],
    safety_blockers: Iterable[str],
) -> Dict[str, Any]:
    evidence_blockers = list(evidence_gate.get("blockers") or [])
    confidence_blockers = list(confidence_review.get("blockers") or [])
    safety_blocker_list = list(safety_blockers or [])
    confidence_status = confidence_review.get("status")

    if evidence_gate.get("missing_required_evidence"):
        status = "blocked_until_required_evidence_complete"
        operator_action = evidence_gate["operator_action"]
        recommended_decision = "request_more_evidence"
    elif confidence_status == "missing_confidence":
        status = "blocked_until_confidence_available"
        operator_action = confidence_review["operator_action"]
        recommended_decision = "request_more_evidence"
    elif confidence_status == "invalid_confidence_range":
        status = "blocked_until_confidence_normalized"
        operator_action = confidence_review["operator_action"]
        recommended_decision = "request_more_evidence"
    elif confidence_status == "low_confidence_operator_review_required":
        status = "low_confidence_requires_operator_recheck"
        operator_action = confidence_review["operator_action"]
        recommended_decision = "request_more_evidence"
    else:
        status = "offline_proposal_review_ready"
        operator_action = evidence_gate["operator_action"]
        recommended_decision = "accept_as_offline_proposal"

    offline_acceptance_allowed = status == "offline_proposal_review_ready"
    return {
        "status": status,
        "recommended_decision": recommended_decision,
        "operator_action": operator_action,
        "offline_acceptance_allowed": offline_acceptance_allowed,
        "operator_review_required": True,
        "requires_operator_decision_record": True,
        "evidence_gate_status": evidence_gate["status"],
        "confidence_review_status": confidence_status,
        "missing_required_evidence": list(
            evidence_gate.get("missing_required_evidence") or []
        ),
        "confidence_blockers": confidence_blockers,
        "blockers": _unique_fields(evidence_blockers, confidence_blockers),
        "safety_blockers": safety_blocker_list,
        "live_authority_blockers": _unique_fields(
            [
                "live_authority_denied",
                "real_wing_proof_required",
            ],
            safety_blocker_list,
        ),
        "read_only_or_dry_run_only": True,
        "live_apply_allowed": False,
        "queue_import_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "readiness_closure_allowed": False,
        "blocks_live_promotion": True,
        "blocks_queue_import": True,
        "can_authorize_live": False,
        "can_close_readiness": False,
        "can_import_to_queue": False,
        "mutates_mixer": False,
        "future_wing_readback_required": True,
        "denial_reason": "operator_review_and_future_wing_readback_required",
    }


def _effective_operator_safety_blockers(
    *,
    safety_blockers: Iterable[str],
    evidence_gate: Dict[str, Any],
    confidence_review: Dict[str, Any],
    operator_proposal_review_gate: Dict[str, Any],
    live_authority_boundary: Dict[str, Any],
) -> List[str]:
    return _unique_fields(
        safety_blockers,
        evidence_gate.get("blockers") or [],
        confidence_review.get("blockers") or [],
        operator_proposal_review_gate.get("blockers") or [],
        operator_proposal_review_gate.get("live_authority_blockers") or [],
        live_authority_boundary.get("blockers") or [],
    )


def _safety_blocker_action(
    blocker: str,
    *,
    source: str,
    evidence_gate: Dict[str, Any],
    confidence_review: Dict[str, Any],
    operator_proposal_review_gate: Dict[str, Any],
) -> str:
    if source == "input_safety_blocker":
        return f"review_safety_blocker:{blocker}"
    if source == "evidence_gate":
        prefix = "required_evidence_missing:"
        if blocker.startswith(prefix):
            return f"collect_missing_required_evidence:{blocker[len(prefix):]}"
        return str(
            evidence_gate.get("operator_action")
            or "collect_missing_required_evidence_before_operator_review"
        )
    if source == "confidence_review":
        return f"resolve_confidence_blocker:{blocker}"
    if source == "proposal_review_gate":
        return str(
            operator_proposal_review_gate.get("operator_action")
            or confidence_review.get("operator_action")
            or "operator_review_required_before_future_live_promotion"
        )
    if source == "proposal_review_live_authority_boundary":
        return "do_not_queue_or_apply_live"
    if source == "live_authority_boundary":
        return "capture_real_wing_readback_and_rollback_proof_before_live_promotion"
    return "operator_review_required_before_future_live_promotion"


def _safety_blocker_details(
    *,
    safety_blockers: Iterable[str],
    effective_safety_blockers: Iterable[str],
    evidence_gate: Dict[str, Any],
    confidence_review: Dict[str, Any],
    operator_proposal_review_gate: Dict[str, Any],
    live_authority_boundary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    evidence_gate = evidence_gate if isinstance(evidence_gate, dict) else {}
    confidence_review = (
        confidence_review if isinstance(confidence_review, dict) else {}
    )
    operator_proposal_review_gate = (
        operator_proposal_review_gate
        if isinstance(operator_proposal_review_gate, dict)
        else {}
    )
    live_authority_boundary = (
        live_authority_boundary if isinstance(live_authority_boundary, dict) else {}
    )

    rows: Dict[str, Dict[str, Any]] = {}

    def add_sources(
        blockers: Iterable[str],
        *,
        source: str,
        status: Optional[str],
    ) -> None:
        for raw_blocker in blockers or []:
            blocker = str(raw_blocker)
            if not blocker:
                continue
            row = rows.setdefault(
                blocker,
                {
                    "blocker": blocker,
                    "sources": [],
                    "source_statuses": {},
                    "operator_actions": [],
                },
            )
            row["sources"] = _unique_fields(row["sources"], [source])
            if status:
                row["source_statuses"][source] = status
            action = _safety_blocker_action(
                blocker,
                source=source,
                evidence_gate=evidence_gate,
                confidence_review=confidence_review,
                operator_proposal_review_gate=operator_proposal_review_gate,
            )
            row["operator_actions"] = _unique_fields(
                row["operator_actions"],
                [action],
            )

    add_sources(
        safety_blockers,
        source="input_safety_blocker",
        status="operator_review_required",
    )
    add_sources(
        evidence_gate.get("blockers") or [],
        source="evidence_gate",
        status=evidence_gate.get("status"),
    )
    add_sources(
        confidence_review.get("blockers") or [],
        source="confidence_review",
        status=confidence_review.get("status"),
    )
    add_sources(
        operator_proposal_review_gate.get("blockers") or [],
        source="proposal_review_gate",
        status=operator_proposal_review_gate.get("status"),
    )
    add_sources(
        operator_proposal_review_gate.get("live_authority_blockers") or [],
        source="proposal_review_live_authority_boundary",
        status=operator_proposal_review_gate.get("status"),
    )
    add_sources(
        live_authority_boundary.get("blockers") or [],
        source="live_authority_boundary",
        status=live_authority_boundary.get("evidence_gate_status"),
    )

    details: List[Dict[str, Any]] = []
    for blocker in effective_safety_blockers or []:
        blocker = str(blocker)
        row = rows.setdefault(
            blocker,
            {
                "blocker": blocker,
                "sources": ["effective_safety_blockers"],
                "source_statuses": {},
                "operator_actions": [
                    "operator_review_required_before_future_live_promotion"
                ],
            },
        )
        operator_actions = _unique_fields(row.get("operator_actions") or [])
        details.append(
            {
                "blocker": blocker,
                "sources": list(row.get("sources") or []),
                "source_statuses": dict(row.get("source_statuses") or {}),
                "operator_actions": operator_actions,
                "operator_action": operator_actions[0],
                "read_only_or_dry_run_only": True,
                "live_apply_allowed": False,
                "queue_import_allowed": False,
                "live_authority_granted": False,
                "full_autonomous_ready": False,
                "readiness_closure_allowed": False,
                "can_authorize_live": False,
                "can_close_readiness": False,
                "can_import_to_queue": False,
                "mutates_mixer": False,
                "operator_review_required": True,
                "future_wing_readback_required": True,
                "blocks_live_promotion": True,
                "blocks_queue_import": True,
                "denial_reason": (
                    "operator_review_and_future_wing_readback_required"
                ),
            }
        )
    return details


def _authority_denied_detail(reason: str, source: str) -> Dict[str, Any]:
    return {
        "reason": reason,
        "source": source,
        "read_only_or_dry_run_only": True,
        "live_apply_allowed": False,
        "queue_import_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "readiness_closure_allowed": False,
        "can_authorize_live": False,
        "can_close_readiness": False,
        "can_import_to_queue": False,
        "mutates_mixer": False,
        "operator_review_required": True,
        "future_wing_readback_required": True,
        "blocks_live_promotion": True,
        "blocks_queue_import": True,
        "denial_reason": "operator_review_and_future_wing_readback_required",
    }


def _operator_why_safe_details(
    *,
    operator_contract: Dict[str, Any],
    bounded_delta: Dict[str, Any],
    rollback_readback_contract: Dict[str, Any],
    live_authority_boundary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return [
        {
            **_authority_denied_detail(
                "read_only_or_dry_run_only",
                "operator_correction_contract",
            ),
            "explanation": (
                "This payload is an offline/dry-run proposal and does not "
                "mutate mixer state."
            ),
            "verified_by": [
                "read_only_or_dry_run_only:true",
                "mutates_mixer:false",
            ],
            "operator_action": "review_operator_evidence_packet",
        },
        {
            **_authority_denied_detail(
                "live_apply_denied",
                "live_authority_boundary",
            ),
            "explanation": (
                "Current live apply is denied until real WING proof, operator "
                "approval, readback and rollback exist."
            ),
            "verified_by": [
                "live_apply_allowed:false",
                "live_authority_granted:false",
            ],
            "blocked_live_actions": list(
                live_authority_boundary.get("blocked_live_actions") or []
            ),
            "operator_action": "do_not_apply_live",
        },
        {
            **_authority_denied_detail(
                "queue_import_denied",
                "operator_proposal_queue_boundary",
            ),
            "explanation": (
                "The proposal is not importable into a live queue before "
                "operator review and future WING readback proof."
            ),
            "verified_by": [
                "queue_import_allowed:false",
                "can_import_to_queue:false",
            ],
            "operator_action": "do_not_import_to_queue",
        },
        {
            **_authority_denied_detail(
                f"bounded_delta_policy:{operator_contract['bounded_delta_policy']}",
                "bounded_delta_contract",
            ),
            "explanation": (
                "Any proposed move remains bounded and must be reviewed before "
                "future promotion."
            ),
            "bounded_delta_policy": operator_contract["bounded_delta_policy"],
            "bounded_delta": dict(bounded_delta),
            "verified_by": [
                "bounded_delta_required:true",
                "operator_review_required:true",
            ],
            "operator_action": "review_bounded_delta_and_required_evidence",
        },
        {
            **_authority_denied_detail(
                f"rollback_expectation:{operator_contract['rollback_expectation']}",
                "rollback_readback_contract",
            ),
            "explanation": (
                "Future promotion still requires rollback/readback proof; the "
                "offline proposal cannot stand in for that proof."
            ),
            "rollback_expectation": operator_contract["rollback_expectation"],
            "rollback_readback_contract": rollback_readback_contract,
            "verified_by": [
                "rollback_required_before_promotion:true",
                "future_wing_readback_required:true",
            ],
            "operator_action": "confirm_rollback_expectation",
        },
    ]


def _why_blocked_action(
    reason: str,
    *,
    evidence_gate: Dict[str, Any],
    confidence_review: Dict[str, Any],
    operator_proposal_review_gate: Dict[str, Any],
) -> str:
    if reason.startswith("required_evidence_missing:"):
        field = reason[len("required_evidence_missing:") :]
        return f"collect_missing_required_evidence:{field}"
    if reason in {
        "required_evidence_incomplete",
        "missing_required_evidence",
    }:
        return str(
            evidence_gate.get("operator_action")
            or "collect_missing_required_evidence_before_operator_review"
        )
    if reason in confidence_review.get("blockers", []):
        return f"resolve_confidence_blocker:{reason}"
    if reason.startswith("live_apply_denied"):
        return "capture_real_wing_readback_and_rollback_proof_before_live_promotion"
    if reason.startswith("queue_import_denied"):
        return "record_operator_decision_before_promotion"
    if reason in operator_proposal_review_gate.get("live_authority_blockers", []):
        return "do_not_queue_or_apply_live"
    return "operator_review_required_before_future_live_promotion"


def _operator_why_blocked_details(
    *,
    why_blocked: Iterable[str],
    safety_blocker_details: List[Dict[str, Any]],
    evidence_gate: Dict[str, Any],
    confidence_review: Dict[str, Any],
    operator_proposal_review_gate: Dict[str, Any],
    live_authority_boundary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    safety_detail_by_blocker = {
        str(row.get("blocker")): row
        for row in safety_blocker_details
        if isinstance(row, dict) and row.get("blocker")
    }
    details: List[Dict[str, Any]] = []

    for reason in _unique_fields(why_blocked or []):
        source = "operator_review_gate"
        sources = [source]
        source_statuses: Dict[str, Any] = {}
        operator_actions: List[str] = []
        explanation = "Operator review is required before future promotion."

        safety_detail = safety_detail_by_blocker.get(str(reason))
        if safety_detail:
            sources = list(safety_detail.get("sources") or [])
            source = sources[0] if sources else "safety_blocker_details"
            source_statuses = dict(safety_detail.get("source_statuses") or {})
            operator_actions = list(safety_detail.get("operator_actions") or [])
            explanation = "This blocker is part of the effective safety gate."
        elif reason in evidence_gate.get("blockers", []):
            source = "evidence_gate"
            sources = [source]
            source_statuses[source] = evidence_gate.get("status")
            explanation = "Required evidence is incomplete for operator review."
        elif reason in confidence_review.get("blockers", []):
            source = "confidence_review"
            sources = [source]
            source_statuses[source] = confidence_review.get("status")
            explanation = "Confidence must be reviewed before acceptance."
        elif reason in live_authority_boundary.get("blockers", []):
            source = "live_authority_boundary"
            sources = [source]
            source_statuses[source] = live_authority_boundary.get(
                "evidence_gate_status"
            )
            explanation = "Live authority remains denied before real WING proof."
        elif reason.startswith("live_apply_denied"):
            source = "live_authority_boundary"
            sources = [source]
            source_statuses[source] = live_authority_boundary.get(
                "evidence_gate_status"
            )
            explanation = "Live apply is denied until future WING proof exists."
        elif reason.startswith("queue_import_denied"):
            source = "operator_proposal_review_gate"
            sources = [source]
            source_statuses[source] = operator_proposal_review_gate.get("status")
            explanation = "Queue import is denied until operator review is recorded."

        action = _why_blocked_action(
            str(reason),
            evidence_gate=evidence_gate,
            confidence_review=confidence_review,
            operator_proposal_review_gate=operator_proposal_review_gate,
        )
        operator_actions = _unique_fields(operator_actions, [action])

        details.append(
            {
                "reason": str(reason),
                "source": source,
                "sources": sources,
                "source_statuses": source_statuses,
                "explanation": explanation,
                "operator_actions": operator_actions,
                "operator_action": operator_actions[0],
                "read_only_or_dry_run_only": True,
                "live_apply_allowed": False,
                "queue_import_allowed": False,
                "live_authority_granted": False,
                "full_autonomous_ready": False,
                "readiness_closure_allowed": False,
                "can_authorize_live": False,
                "can_close_readiness": False,
                "can_import_to_queue": False,
                "mutates_mixer": False,
                "operator_review_required": True,
                "future_wing_readback_required": True,
                "blocks_live_promotion": True,
                "blocks_queue_import": True,
                "denial_reason": (
                    "operator_review_and_future_wing_readback_required"
                ),
            }
        )

    return details


def _summarize_proposed_correction(proposed_correction: Dict[str, Any]) -> str:
    if not isinstance(proposed_correction, dict):
        return "no_applyable_change_provided"
    if proposed_correction.get("status"):
        return str(proposed_correction["status"])
    if proposed_correction.get("action_type"):
        return str(proposed_correction["action_type"])
    if "changes" in proposed_correction:
        changes = proposed_correction.get("changes") or []
        return f"multiple_proposed_changes:{len(changes)}"

    value_type = proposed_correction.get("value_type") or "proposal"
    target = _target_from_change(proposed_correction)
    old_value = _change_old_value(proposed_correction)
    proposed_value = _change_proposed_value(proposed_correction)
    if target is not None and (old_value is not None or proposed_value is not None):
        return f"{value_type} proposal for {target}: {old_value} -> {proposed_value}"
    return str(value_type)


def _operator_review_summary(
    *,
    source_evidence_summary: Dict[str, Any],
    problem_summary: str,
    proposed_correction: Dict[str, Any],
    safety_blockers: Iterable[str],
    effective_safety_blockers: Iterable[str],
    safety_blocker_details: List[Dict[str, Any]],
    why_safe_details: List[Dict[str, Any]],
    why_blocked_details: List[Dict[str, Any]],
    evidence_gate: Dict[str, Any],
    confidence_review: Dict[str, Any],
    confidence_review_details: List[Dict[str, Any]],
    bounded_delta_details: List[Dict[str, Any]],
    source_evidence_details: List[Dict[str, Any]],
    headroom_feedback_true_peak_check_details: List[Dict[str, Any]],
    operator_proposal_review_gate: Dict[str, Any],
    operator_contract: Dict[str, Any],
    rollback_readback_contract: Dict[str, Any],
    rollback_readback_details: List[Dict[str, Any]],
    future_live_requirement_details: List[Dict[str, Any]],
    operator_next_action_details: List[Dict[str, Any]],
    operator_review_checklist_details: List[Dict[str, Any]],
    operator_review_decision_option_details: List[Dict[str, Any]],
    operator_review_record_field_details: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "method_id": source_evidence_summary["method_id"],
        "source": source_evidence_summary["source"],
        "target": source_evidence_summary["target"],
        "value_type": operator_contract["value_type"],
        "target_kind": source_evidence_summary["target_kind"],
        "old_value_source": source_evidence_summary["old_value_source"],
        "objective_family": source_evidence_summary["objective_family"],
        "bounded_delta_unit": operator_contract["bounded_delta_unit"],
        "bounded_delta_policy": operator_contract["bounded_delta_policy"],
        "problem": problem_summary,
        "proposed_correction": _summarize_proposed_correction(proposed_correction),
        "why_safe": [
            "read_only_or_dry_run_only",
            "live_apply_denied",
            "queue_import_denied",
            f"bounded_delta_policy:{operator_contract['bounded_delta_policy']}",
            f"rollback_expectation:{operator_contract['rollback_expectation']}",
        ],
        "why_blocked": _unique_fields(
            [
                "live_apply_denied_until_future_wing_readback",
                "queue_import_denied_until_operator_review",
            ],
            effective_safety_blockers,
            evidence_gate.get("blockers") or [],
            confidence_review.get("blockers") or [],
        ),
        "effective_safety_blockers": list(effective_safety_blockers or []),
        "safety_blocker_details": [dict(row) for row in safety_blocker_details],
        "why_safe_details": [dict(row) for row in why_safe_details],
        "why_blocked_details": [dict(row) for row in why_blocked_details],
        "proposal_review_gate_status": operator_proposal_review_gate["status"],
        "offline_acceptance_allowed": operator_proposal_review_gate[
            "offline_acceptance_allowed"
        ],
        "recommended_decision": operator_proposal_review_gate[
            "recommended_decision"
        ],
        "confidence_review_status": confidence_review["status"],
        "confidence_blockers": list(confidence_review.get("blockers") or []),
        "confidence_review_details": [
            dict(row) for row in confidence_review_details
        ],
        "source_evidence_details": [dict(row) for row in source_evidence_details],
        "bounded_delta_details": [dict(row) for row in bounded_delta_details],
        "headroom_feedback_true_peak_check_details": [
            dict(row) for row in headroom_feedback_true_peak_check_details
        ],
        "next_required_evidence": list(
            evidence_gate.get("missing_required_evidence") or []
        ),
        "future_live_requirement_details": [
            dict(row) for row in future_live_requirement_details
        ],
        "operator_next_action_details": [
            dict(row) for row in operator_next_action_details
        ],
        "operator_review_checklist_details": [
            dict(row) for row in operator_review_checklist_details
        ],
        "operator_review_decision_option_details": [
            dict(row) for row in operator_review_decision_option_details
        ],
        "operator_review_record_field_details": [
            dict(row) for row in operator_review_record_field_details
        ],
        "rollback_expectation": operator_contract["rollback_expectation"],
        "rollback_readback_contract": rollback_readback_contract,
        "rollback_readback_details": [
            dict(row) for row in rollback_readback_details
        ],
        "read_only_or_dry_run_only": True,
        "live_apply_allowed": False,
        "queue_import_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "readiness_closure_allowed": False,
        "can_authorize_live": False,
        "can_close_readiness": False,
        "mutates_mixer": False,
        "operator_review_required": True,
        "blocks_live_promotion": True,
        "blocks_queue_import": True,
        "future_wing_readback_required": True,
        "future_live_requirements": {
            "readback_required_before_live_promotion": True,
            "read_only_or_dry_run_only": True,
            "live_apply_allowed": False,
            "queue_import_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "readiness_closure_allowed": False,
            "can_authorize_live": False,
            "can_close_readiness": False,
            "mutates_mixer": False,
            "operator_review_required": True,
            "blocks_live_promotion": True,
            "blocks_queue_import": True,
            "future_wing_readback_required": True,
            "wing_proof_required": True,
            "readback_targets": list(
                operator_contract["future_wing_readback_contract"]["readback_targets"]
            ),
            "blocked_live_actions": list(
                operator_contract["future_wing_readback_contract"]["blocked_live_actions"]
            ),
            "required_evidence_status": rollback_readback_contract[
                "required_evidence_status"
            ],
            "rollback_readback_details": [
                dict(row) for row in rollback_readback_details
            ],
            "requirement_details": [
                dict(row) for row in future_live_requirement_details
            ],
        },
        "operator_action": operator_proposal_review_gate["operator_action"],
    }


def _measured_or_audible_problem_payload(
    *,
    problem_summary: str,
    proposal: Dict[str, Any],
    method_id: str,
    target: Any,
    operator_contract: Dict[str, Any],
    evidence_completeness: Dict[str, Any],
    evidence_gate: Dict[str, Any],
) -> Dict[str, Any]:
    required_evidence_status = _required_evidence_status(
        evidence_completeness,
        evidence_gate,
    )
    return {
        "summary": problem_summary,
        "severity": proposal.get("severity"),
        "source": proposal.get("source"),
        "method_id": method_id,
        "target": target if target is not None else proposal.get("target"),
        "value_type": operator_contract["value_type"],
        "objective_family": operator_contract["objective_family"],
        "target_kind": operator_contract["future_wing_readback_contract"][
            "target_kind"
        ],
        "old_value_source": operator_contract["old_value_source"],
        "bounded_delta_unit": operator_contract["bounded_delta_unit"],
        "bounded_delta_policy": operator_contract["bounded_delta_policy"],
        "evidence_completeness_status": evidence_completeness["status"],
        "evidence_complete": bool(evidence_completeness["complete"]),
        "required_evidence_fields": list(operator_contract["required_evidence_fields"]),
        "provided_evidence_fields": list(evidence_completeness.get("provided") or []),
        "missing_required_evidence": list(
            evidence_gate.get("missing_required_evidence") or []
        ),
        "required_evidence_status": required_evidence_status,
        "required_evidence_actions": required_evidence_status[
            "required_evidence_actions"
        ],
        "evidence_gate_status": evidence_gate["status"],
        "operator_action": evidence_gate["operator_action"],
        "problem_statement_requires_operator_review": True,
        "problem_statement_can_authorize_live": False,
        "read_only_or_dry_run_only": True,
        "live_apply_allowed": False,
        "queue_import_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "readiness_closure_allowed": False,
        "can_authorize_live": False,
        "can_close_readiness": False,
        "mutates_mixer": False,
        "operator_review_required": True,
        "future_wing_readback_required": True,
        "blocks_live_promotion": True,
        "blocks_queue_import": True,
        "denial_reason": operator_contract["future_live_promotion_requirements"][
            "blocked_until"
        ],
        "why_blocked": _unique_fields(
            [
                "problem_statement_is_offline_evidence_only",
                "live_apply_denied_until_real_wing_proof",
                "queue_import_denied_until_operator_review",
                "future_wing_readback_required",
            ],
            evidence_gate.get("blockers") or [],
        ),
    }


def _source_evidence_summary(
    *,
    proposal: Dict[str, Any],
    method_id: str,
    target: Any,
    operator_contract: Dict[str, Any],
    evidence_completeness: Dict[str, Any],
    evidence_gate: Dict[str, Any],
) -> Dict[str, Any]:
    raw = proposal.get("raw") if isinstance(proposal.get("raw"), dict) else {}
    evidence = (
        proposal.get("evidence") if isinstance(proposal.get("evidence"), dict) else {}
    )
    raw_evidence_fields = sorted(raw.keys())
    structured_evidence_fields = sorted(evidence.keys())
    raw_input_authority_claims = _raw_input_authority_claims(evidence)
    required_evidence_status = _required_evidence_status(
        evidence_completeness,
        evidence_gate,
    )
    return {
        "source": proposal.get("source"),
        "method_id": method_id,
        "target": target if target is not None else proposal.get("target"),
        "value_type": operator_contract["value_type"],
        "objective_family": operator_contract["objective_family"],
        "target_kind": operator_contract["future_wing_readback_contract"][
            "target_kind"
        ],
        "old_value_source": operator_contract["future_wing_readback_contract"][
            "old_value_source"
        ],
        "required_evidence_fields": list(operator_contract["required_evidence_fields"]),
        "provided_evidence_fields": list(evidence_completeness.get("provided") or []),
        "missing_required_evidence": list(
            evidence_gate.get("missing_required_evidence") or []
        ),
        "evidence_completeness_status": evidence_completeness["status"],
        "evidence_complete": bool(evidence_completeness["complete"]),
        "evidence_gate_status": evidence_gate["status"],
        "required_evidence_status": required_evidence_status,
        "operator_action": evidence_gate["operator_action"],
        "raw_evidence_fields": raw_evidence_fields,
        "structured_evidence_fields": structured_evidence_fields,
        "raw_input_authority_claims": raw_input_authority_claims,
        "raw_input_evidence_boundary": _raw_input_evidence_boundary(
            surface="source_evidence",
            evidence_fields=_unique_fields(
                raw_evidence_fields,
                structured_evidence_fields,
            ),
        ),
        "read_only_or_dry_run_only": True,
        "live_apply_allowed": False,
        "queue_import_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "readiness_closure_allowed": False,
        "can_authorize_live": False,
        "can_close_readiness": False,
        "mutates_mixer": False,
        "operator_review_required": True,
        "blocks_live_promotion": True,
        "blocks_queue_import": True,
        "future_wing_readback_required": True,
        "denial_reason": operator_contract["future_live_promotion_requirements"][
            "blocked_until"
        ],
        "operator_next_step": evidence_gate["operator_action"],
        "future_live_promotion_requires": list(
            operator_contract["future_live_promotion_requirements"]["required_proofs"]
        ),
        "why_blocked": _unique_fields(
            [
                "live_apply_denied_until_real_wing_proof",
                "queue_import_denied_until_operator_review",
                "future_wing_readback_required",
            ],
            evidence_gate.get("blockers") or [],
        ),
    }


def _matching_source_values(
    field: str,
    mapping: Dict[str, Any],
    *,
    source: str,
) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for key, value in mapping.items():
        normalized_key = _normalize_evidence_field_name(key)
        if _evidence_field_satisfied(field, [normalized_key]):
            values[f"{source}:{key}"] = value
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                normalized_nested = _normalize_evidence_field_name(nested_key)
                if _evidence_field_satisfied(field, [normalized_nested]):
                    values[f"{source}:{key}.{nested_key}"] = nested_value
    return values


def _source_evidence_details(
    *,
    raw: Dict[str, Any],
    evidence: Dict[str, Any],
    operator_contract: Dict[str, Any],
    evidence_completeness: Dict[str, Any],
    evidence_gate: Dict[str, Any],
    live_authority_boundary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    required_status = _required_evidence_status(evidence_completeness, evidence_gate)
    status_by_field = {row["field"]: row for row in required_status["rows"]}
    provided = set(evidence_completeness.get("provided") or [])
    common = {
        "objective_family": operator_contract["objective_family"],
        "target_kind": operator_contract["future_wing_readback_contract"][
            "target_kind"
        ],
        "old_value_source": operator_contract["future_wing_readback_contract"][
            "old_value_source"
        ],
        "all_required_evidence_fields": list(
            operator_contract["required_evidence_fields"]
        ),
        "provided_evidence_fields": list(evidence_completeness.get("provided") or []),
        "missing_required_evidence": list(
            evidence_gate.get("missing_required_evidence") or []
        ),
        "evidence_gate_status": evidence_gate["status"],
        "required_before_operator_acceptance": True,
        "required_before_live_promotion": True,
        "required_before_queue_import": True,
        "read_only_or_dry_run_only": True,
        "live_apply_allowed": False,
        "queue_import_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "readiness_closure_allowed": False,
        "can_authorize_live": False,
        "can_close_readiness": False,
        "can_import_to_queue": False,
        "mutates_mixer": False,
        "operator_review_required": True,
        "future_wing_readback_required": True,
        "blocks_live_promotion": True,
        "blocks_queue_import": True,
        "denial_reason": (
            live_authority_boundary.get("denial_reason")
            or "operator_review_and_future_wing_readback_required"
        ),
    }
    details: List[Dict[str, Any]] = []
    for field in operator_contract["required_evidence_fields"]:
        status_row = status_by_field[field]
        raw_values = _matching_source_values(field, raw, source="raw")
        structured_values = _matching_source_values(
            field, evidence, source="structured_evidence"
        )
        source_locations: List[str] = []
        if raw_values:
            source_locations.append("raw")
        if structured_values:
            source_locations.append("structured_evidence")
        if (
            status_row["matched_provided_fields"]
            and not raw_values
            and not structured_values
        ):
            source_locations.append("derived_request_or_action")
        if status_row["missing"]:
            source_locations.append("missing")

        action_type = (
            "collect_source_evidence"
            if status_row["missing"]
            else "review_source_evidence"
        )
        action = f"{action_type}:{field}"
        details.append(
            {
                **common,
                "field": field,
                "required_evidence_field": field,
                "status": status_row["status"],
                "provided": status_row["provided"],
                "missing": status_row["missing"],
                "matched_provided_fields": list(
                    status_row["matched_provided_fields"]
                ),
                "source_locations": source_locations,
                "raw_evidence_values": raw_values,
                "structured_evidence_values": structured_values,
                "source_value_available": bool(raw_values or structured_values),
                "operator_action": action,
                "action": action,
                "action_type": action_type,
                "matched_provided_fields_in_completeness": [
                    field_name
                    for field_name in status_row["matched_provided_fields"]
                    if field_name in provided
                ],
            }
        )
    return details


def _raw_input_evidence_boundary(
    *,
    surface: str,
    evidence_fields: Iterable[str],
) -> Dict[str, Any]:
    return {
        "surface": surface,
        "raw_input_evidence_only": True,
        "raw_input_cannot_grant_authority": True,
        "evidence_fields": sorted(str(field) for field in evidence_fields),
        "read_only_or_dry_run_only": True,
        "live_apply_allowed": False,
        "queue_import_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "readiness_closure_allowed": False,
        "can_authorize_live": False,
        "can_close_readiness": False,
        "mutates_mixer": False,
        "operator_review_required": True,
        "future_wing_readback_required": True,
        "blocks_live_promotion": True,
        "blocks_queue_import": True,
        "denial_reason": "raw_input_evidence_requires_operator_review_and_future_wing_readback",
    }


_OPERATOR_VISIBLE_EVIDENCE_FALSE_FIELDS = {
    "apply_now",
    "can_apply_live",
    "can_authorize_live",
    "can_close_readiness",
    "can_import_to_queue",
    "dispatch_supervised_apply",
    "full_autonomous_ready",
    "live_apply_allowed",
    "live_authority_granted",
    "live_write",
    "mutates_mixer",
    "queue_import_allowed",
    "queue_mutation",
    "readiness_closure_allowed",
}

_OPERATOR_VISIBLE_EVIDENCE_TRUE_FIELDS = {
    "blocks_live_promotion",
    "blocks_queue_import",
    "future_wing_readback_required",
    "operator_review_required",
    "read_only_or_dry_run_only",
    "readback_required",
}

_OPERATOR_VISIBLE_EVIDENCE_QUEUE_FIELDS = {"queue_action"}


def _operator_visible_authority_claim_value(field: str, value: Any) -> Any:
    normalized = str(field)
    if normalized in _OPERATOR_VISIBLE_EVIDENCE_FALSE_FIELDS:
        return False
    if normalized in _OPERATOR_VISIBLE_EVIDENCE_TRUE_FIELDS:
        return True
    if normalized in _OPERATOR_VISIBLE_EVIDENCE_QUEUE_FIELDS:
        return "not_queueable_until_promoted"
    return value


def _operator_visible_source_evidence(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _operator_visible_source_evidence(
                _operator_visible_authority_claim_value(key, nested)
            )
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [_operator_visible_source_evidence(item) for item in value]
    return value


def _raw_input_authority_claims(value: Any, path: str = "") -> List[Dict[str, Any]]:
    claims: List[Dict[str, Any]] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            visible_value = _operator_visible_authority_claim_value(str(key), nested)
            if visible_value != nested:
                claims.append(
                    {
                        "path": child_path,
                        "field": str(key),
                        "requested_value": nested,
                        "operator_visible_value": visible_value,
                        "status": "raw_input_claim_denied",
                    }
                )
            claims.extend(_raw_input_authority_claims(nested, child_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            claims.extend(_raw_input_authority_claims(item, f"{path}[{index}]"))
    return claims


def _operator_next_action_details(
    *,
    operator_next_actions: Iterable[str],
    required_evidence_actions: Iterable[Dict[str, Any]],
    safety_blocker_details: Iterable[Dict[str, Any]],
    why_blocked_details: Iterable[Dict[str, Any]],
    future_live_requirement_details: Iterable[Dict[str, Any]],
    evidence_gate: Dict[str, Any],
    confidence_review: Dict[str, Any],
    operator_proposal_review_gate: Dict[str, Any],
    rollback_readback_contract: Dict[str, Any],
    live_authority_boundary: Dict[str, Any],
    operator_review_record_field_details: Optional[Iterable[Dict[str, Any]]] = None,
    source_evidence_details: Optional[Iterable[Dict[str, Any]]] = None,
    confidence_review_details: Optional[Iterable[Dict[str, Any]]] = None,
    bounded_delta_details: Optional[Iterable[Dict[str, Any]]] = None,
    rollback_readback_details: Optional[Iterable[Dict[str, Any]]] = None,
    headroom_feedback_true_peak_check_details: Optional[
        Iterable[Dict[str, Any]]
    ] = None,
) -> List[Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}

    def register(
        action: Any,
        *,
        source: str,
        action_type: str,
        status: Optional[str] = None,
        field: Optional[str] = None,
        blocker: Optional[str] = None,
        reason: Optional[str] = None,
        requirement: Optional[str] = None,
        requirement_type: Optional[str] = None,
        proof: Optional[str] = None,
        readback_target: Optional[str] = None,
        required_before_operator_acceptance: bool = True,
    ) -> None:
        normalized = str(action or "").strip()
        if not normalized:
            return
        row = indexed.setdefault(
            normalized,
            {
                "action": normalized,
                "operator_action": normalized,
                "source": source,
                "sources": [],
                "action_type": action_type,
                "status": status or "operator_review_required",
                "fields": [],
                "blockers": [],
                "reasons": [],
                "requirements": [],
                "requirement_types": [],
                "proofs": [],
                "readback_targets": [],
                "required_before_operator_acceptance": (
                    required_before_operator_acceptance
                ),
                "required_before_live_promotion": True,
                "required_before_queue_import": True,
                "read_only_or_dry_run_only": True,
                "live_apply_allowed": False,
                "queue_import_allowed": False,
                "live_authority_granted": False,
                "full_autonomous_ready": False,
                "readiness_closure_allowed": False,
                "can_authorize_live": False,
                "can_close_readiness": False,
                "can_import_to_queue": False,
                "mutates_mixer": False,
                "operator_review_required": True,
                "future_wing_readback_required": True,
                "blocks_live_promotion": True,
                "blocks_queue_import": True,
                "denial_reason": (
                    live_authority_boundary.get("denial_reason")
                    or "operator_review_and_future_wing_readback_required"
                ),
            },
        )
        row["sources"] = _unique_fields(row["sources"], [source])
        row["required_before_operator_acceptance"] = (
            row["required_before_operator_acceptance"]
            or required_before_operator_acceptance
        )
        if status and row["status"] == "operator_review_required":
            row["status"] = status
        for key, value in (
            ("fields", field),
            ("blockers", blocker),
            ("reasons", reason),
            ("requirements", requirement),
            ("requirement_types", requirement_type),
            ("proofs", proof),
            ("readback_targets", readback_target),
        ):
            if value is not None:
                row[key] = _unique_fields(row[key], [str(value)])

    register(
        operator_proposal_review_gate.get("operator_action"),
        source="operator_proposal_review_gate",
        action_type="proposal_review_gate_action",
        status=operator_proposal_review_gate.get("status"),
    )
    register(
        "review_operator_evidence_packet",
        source="operator_evidence_packet",
        action_type="review_operator_evidence_packet",
    )
    register(
        "record_operator_decision_before_promotion",
        source="operator_review_record_template",
        action_type="record_operator_review_decision",
    )
    register(
        "do_not_queue_or_apply_live",
        source="live_authority_boundary",
        action_type="hold_live_and_queue_authority",
        status=live_authority_boundary.get("evidence_gate_status"),
    )
    for row in source_evidence_details or []:
        register(
            row.get("operator_action"),
            source="source_evidence_details",
            action_type=str(row.get("action_type") or "review_source_evidence"),
            status=row.get("status"),
            field=row.get("field"),
            required_before_operator_acceptance=bool(
                row.get("required_before_operator_acceptance", True)
            ),
        )
    register(
        "confirm_rollback_expectation",
        source="rollback_readback_contract",
        action_type="confirm_rollback_expectation",
        status=rollback_readback_contract.get("required_evidence_status", {}).get(
            "status"
        ),
    )
    for blocker in confidence_review.get("blockers") or []:
        register(
            f"resolve_confidence_blocker:{blocker}",
            source="confidence_review",
            action_type="resolve_confidence_blocker",
            status=confidence_review.get("status"),
            blocker=blocker,
        )
    for row in confidence_review_details or []:
        register(
            row.get("operator_action"),
            source="confidence_review_details",
            action_type=str(row.get("action_type") or "review_confidence"),
            status=row.get("status"),
            field=row.get("field"),
            blocker=row.get("blocker"),
            required_before_operator_acceptance=bool(
                row.get("required_before_operator_acceptance", True)
            ),
        )
    for row in bounded_delta_details or []:
        register(
            row.get("operator_action"),
            source="bounded_delta_details",
            action_type=str(row.get("action_type") or "review_bounded_delta"),
            status=row.get("status"),
            required_before_operator_acceptance=bool(
                row.get("required_before_operator_acceptance", True)
            ),
        )
    for row in rollback_readback_details or []:
        register(
            row.get("operator_action"),
            source="rollback_readback_details",
            action_type=str(row.get("action_type") or "review_rollback_readback"),
            status=row.get("status"),
            requirement=row.get("detail"),
            requirement_type=row.get("detail_type"),
            readback_target=row.get("readback_target"),
            required_before_operator_acceptance=bool(
                row.get("required_before_operator_acceptance", True)
            ),
        )
    for row in headroom_feedback_true_peak_check_details or []:
        register(
            row.get("operator_action"),
            source="headroom_feedback_true_peak_check_details",
            action_type=str(row.get("action_type") or "review_safety_check"),
            status=row.get("status"),
            requirement=row.get("check"),
            requirement_type=row.get("evidence_mapping_status"),
            required_before_operator_acceptance=bool(
                row.get("required_before_operator_acceptance", True)
            ),
        )
    for field in evidence_gate.get("missing_required_evidence") or []:
        register(
            f"collect_missing_required_evidence:{field}",
            source="evidence_gate",
            action_type="collect_missing_required_evidence",
            status=evidence_gate.get("status"),
            field=field,
            blocker=f"required_evidence_missing:{field}",
        )
    for row in required_evidence_actions:
        register(
            row.get("action"),
            source="required_evidence_status",
            action_type=str(row.get("action_type") or "required_evidence_action"),
            status=row.get("status"),
            field=row.get("field"),
            required_before_operator_acceptance=bool(
                row.get("required_before_operator_acceptance")
            ),
        )
    for row in safety_blocker_details:
        blocker = row.get("blocker")
        for action in row.get("operator_actions") or []:
            register(
                action,
                source="safety_blocker_details",
                action_type="review_safety_blocker",
                status=row.get("status"),
                blocker=blocker,
            )
    for row in why_blocked_details:
        reason = row.get("reason")
        for action in row.get("operator_actions") or []:
            register(
                action,
                source="why_blocked_details",
                action_type="resolve_or_acknowledge_blocker",
                status=row.get("status"),
                reason=reason,
            )
    for row in future_live_requirement_details:
        register(
            row.get("operator_action"),
            source="future_live_requirement_details",
            action_type=str(row.get("requirement_type") or "future_live_requirement"),
            status=row.get("status"),
            requirement=row.get("requirement"),
            requirement_type=row.get("requirement_type"),
            proof=row.get("proof"),
            readback_target=row.get("readback_target"),
        )
    for row in operator_review_record_field_details or []:
        register(
            row.get("operator_action"),
            source="operator_review_record_field_details",
            action_type="fill_operator_review_record_field",
            status=row.get("status"),
            field=row.get("field"),
            required_before_operator_acceptance=bool(
                row.get("required_before_operator_acceptance", True)
            ),
        )

    details: List[Dict[str, Any]] = []
    for action in _unique_fields(operator_next_actions):
        if action not in indexed:
            action_type = str(action).split(":", 1)[0] or "operator_next_action"
            register(
                action,
                source="operator_next_actions",
                action_type=action_type,
                status="operator_review_required",
            )
        details.append(dict(indexed[action]))
    return details


def _refresh_operator_next_action_details(correction: Dict[str, Any]) -> None:
    explanation_packet = correction.get("operator_explanation_packet")
    if not isinstance(explanation_packet, dict):
        return
    details = _operator_next_action_details(
        operator_next_actions=explanation_packet.get("operator_next_actions") or [],
        required_evidence_actions=explanation_packet.get(
            "required_evidence_actions"
        )
        or [],
        safety_blocker_details=correction.get("safety_blocker_details") or [],
        why_blocked_details=correction.get("why_blocked_details") or [],
        future_live_requirement_details=correction.get(
            "future_live_requirement_details"
        )
        or [],
        operator_review_record_field_details=correction.get(
            "operator_review_record_field_details"
        )
        or [],
        confidence_review_details=correction.get("confidence_review_details")
        or [],
        bounded_delta_details=correction.get("bounded_delta_details") or [],
        rollback_readback_details=correction.get("rollback_readback_details") or [],
        headroom_feedback_true_peak_check_details=correction.get(
            "headroom_feedback_true_peak_check_details"
        )
        or [],
        evidence_gate=correction.get("evidence_gate") or {},
        confidence_review=correction.get("confidence_review") or {},
        operator_proposal_review_gate=correction.get(
            "operator_proposal_review_gate"
        )
        or {},
        rollback_readback_contract=correction.get("rollback_readback_contract")
        or {},
        live_authority_boundary=correction.get("live_authority_boundary") or {},
        source_evidence_details=correction.get("source_evidence_details") or [],
    )
    correction_context_profile = _operator_correction_context_profile(correction)
    details = [
        {
            **row,
            **correction_context_profile,
        }
        for row in details
    ]
    correction["operator_next_action_details"] = details
    explanation_packet["operator_next_action_details"] = details

    evidence_packet = correction.get("operator_evidence_packet")
    if isinstance(evidence_packet, dict):
        evidence_packet["operator_next_action_details"] = details
        evidence_packet["operator_explanation_packet"] = explanation_packet

    review_summary = correction.get("operator_review_summary")
    if isinstance(review_summary, dict):
        review_summary["operator_next_action_details"] = details


def _operator_review_checklist_details(
    *,
    operator_review_checklist: Iterable[str],
    required_evidence_status: Dict[str, Any],
    operator_next_action_details: Iterable[Dict[str, Any]],
    evidence_gate: Dict[str, Any],
    confidence_review: Dict[str, Any],
    operator_proposal_review_gate: Dict[str, Any],
    rollback_readback_contract: Dict[str, Any],
    live_authority_boundary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    next_actions_by_action = {
        row["action"]: row
        for row in operator_next_action_details
        if isinstance(row, dict) and row.get("action")
    }
    required_actions_by_field = {
        row["field"]: row
        for row in required_evidence_status.get("required_evidence_actions") or []
        if isinstance(row, dict) and row.get("field")
    }

    def base_row(
        item: str,
        *,
        source: str,
        action_type: str,
        operator_action: str,
        status: Optional[str] = None,
        field: Optional[str] = None,
        blocker: Optional[str] = None,
        required_before_operator_acceptance: bool = True,
    ) -> Dict[str, Any]:
        related_next_actions = []
        if operator_action in next_actions_by_action:
            related_next_actions.append(operator_action)
        return {
            "item": item,
            "checklist_item": item,
            "source": source,
            "sources": [source],
            "action_type": action_type,
            "status": status or "operator_review_required",
            "field": field,
            "fields": [field] if field else [],
            "blocker": blocker,
            "blockers": [blocker] if blocker else [],
            "operator_action": operator_action,
            "related_next_actions": related_next_actions,
            "required_before_operator_acceptance": required_before_operator_acceptance,
            "required_before_live_promotion": True,
            "required_before_queue_import": True,
            "read_only_or_dry_run_only": True,
            "live_apply_allowed": False,
            "queue_import_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "readiness_closure_allowed": False,
            "can_authorize_live": False,
            "can_close_readiness": False,
            "can_import_to_queue": False,
            "mutates_mixer": False,
            "operator_review_required": True,
            "future_wing_readback_required": True,
            "blocks_live_promotion": True,
            "blocks_queue_import": True,
            "denial_reason": (
                live_authority_boundary.get("denial_reason")
                or "operator_review_and_future_wing_readback_required"
            ),
        }

    details: List[Dict[str, Any]] = []
    for item in _unique_fields(operator_review_checklist):
        source = "operator_review_checklist"
        action_type = "operator_review_check"
        operator_action = "review_operator_evidence_packet"
        status = "operator_review_required"
        field = None
        blocker = None
        required_before_operator_acceptance = True

        if item == "verify_source_evidence_matches_current_offline_material":
            source = "source_evidence"
            action_type = "verify_source_evidence"
        elif item == "confirm_measured_or_audible_problem":
            source = "measured_or_audible_problem"
            action_type = "confirm_problem_statement"
        elif item == "inspect_proposed_correction_and_bounded_delta":
            source = "proposed_correction"
            action_type = "inspect_proposed_correction"
            operator_action = "review_bounded_delta_and_required_evidence"
        elif item == "verify_required_safety_checks":
            source = "operator_safety_review_matrix"
            action_type = "verify_required_safety_checks"
            operator_action = evidence_gate.get("operator_action") or operator_action
            status = evidence_gate.get("status") or status
        elif item == "confirm_rollback_expectation":
            source = "rollback_readback_contract"
            action_type = "confirm_rollback_expectation"
            operator_action = "confirm_rollback_expectation"
            status = rollback_readback_contract.get("required_evidence_status", {}).get(
                "status"
            ) or status
        elif item == "do_not_queue_or_apply_live":
            source = "live_authority_boundary"
            action_type = "hold_live_and_queue_authority"
            operator_action = "do_not_queue_or_apply_live"
            status = live_authority_boundary.get("evidence_gate_status") or status
        elif item == "capture_real_wing_readback_and_rollback_proof_before_live_promotion":
            source = "future_wing_readback_contract"
            action_type = "capture_future_wing_readback"
            operator_action = item
        elif item == "collect_missing_required_evidence_before_operator_review":
            source = "evidence_gate"
            action_type = "collect_missing_required_evidence"
            operator_action = evidence_gate.get("operator_action") or item
            status = evidence_gate.get("status") or status
        elif item == "operator_review_required_before_future_live_promotion":
            source = "operator_proposal_review_gate"
            action_type = "operator_review_before_future_live"
            operator_action = operator_proposal_review_gate.get(
                "operator_action"
            ) or item
            status = operator_proposal_review_gate.get("status") or status
        elif item.startswith("missing_required_evidence:"):
            field = item.split(":", 1)[1]
            action = required_actions_by_field.get(field, {})
            source = "required_evidence_status"
            action_type = str(
                action.get("action_type") or "collect_missing_required_evidence"
            )
            operator_action = str(
                action.get("action") or f"collect_missing_required_evidence:{field}"
            )
            status = str(action.get("status") or evidence_gate.get("status") or status)
            blocker = f"required_evidence_missing:{field}"
            required_before_operator_acceptance = bool(
                action.get("required_before_operator_acceptance", True)
            )
        elif item.startswith("confidence_blocker:"):
            blocker = item.split(":", 1)[1]
            source = "confidence_review"
            action_type = "resolve_confidence_blocker"
            operator_action = f"resolve_confidence_blocker:{blocker}"
            status = confidence_review.get("status") or status
        elif item == operator_proposal_review_gate.get("operator_action"):
            source = "operator_proposal_review_gate"
            action_type = "proposal_review_gate_action"
            operator_action = item
            status = operator_proposal_review_gate.get("status") or status

        details.append(
            base_row(
                item,
                source=source,
                action_type=action_type,
                operator_action=operator_action,
                status=status,
                field=field,
                blocker=blocker,
                required_before_operator_acceptance=required_before_operator_acceptance,
            )
        )
    return details


def _operator_explanation_packet(
    *,
    problem_summary: str,
    proposed_correction: Dict[str, Any],
    source_evidence_summary: Dict[str, Any],
    source_evidence_details: List[Dict[str, Any]],
    bounded_delta: Dict[str, Any],
    bounded_delta_details: List[Dict[str, Any]],
    confidence_review: Dict[str, Any],
    confidence_review_details: List[Dict[str, Any]],
    operator_proposal_review_gate: Dict[str, Any],
    evidence_gate: Dict[str, Any],
    safety_checks: Dict[str, Any],
    headroom_feedback_true_peak_check_details: List[Dict[str, Any]],
    operator_safety_review_matrix: Dict[str, Any],
    safety_blockers: Iterable[str],
    effective_safety_blockers: Iterable[str],
    safety_blocker_details: List[Dict[str, Any]],
    why_safe_details: List[Dict[str, Any]],
    why_blocked_details: List[Dict[str, Any]],
    operator_review_checklist: List[str],
    operator_review_decision_options: List[Dict[str, Any]],
    operator_review_decision_option_details: List[Dict[str, Any]],
    operator_review_record_template: Dict[str, Any],
    operator_review_record_field_details: List[Dict[str, Any]],
    operator_contract: Dict[str, Any],
    future_live_promotion_requirements: Dict[str, Any],
    future_live_requirement_details: List[Dict[str, Any]],
    rollback_readback_contract: Dict[str, Any],
    rollback_readback_details: List[Dict[str, Any]],
    live_authority_boundary: Dict[str, Any],
) -> Dict[str, Any]:
    required_evidence_status = source_evidence_summary["required_evidence_status"]
    required_evidence_actions = list(
        required_evidence_status.get("required_evidence_actions") or []
    )
    operator_next_actions = [
        operator_proposal_review_gate["operator_action"],
        "review_operator_evidence_packet",
        "record_operator_decision_before_promotion",
        "do_not_queue_or_apply_live",
    ]
    for blocker in confidence_review.get("blockers") or []:
        operator_next_actions.append(f"resolve_confidence_blocker:{blocker}")
    for detail in source_evidence_details:
        operator_next_actions.append(detail["operator_action"])
    for detail in confidence_review_details:
        operator_next_actions.append(detail["operator_action"])
    for detail in bounded_delta_details:
        operator_next_actions.append(detail["operator_action"])
    for detail in rollback_readback_details:
        operator_next_actions.append(detail["operator_action"])
    for detail in headroom_feedback_true_peak_check_details:
        operator_next_actions.append(detail["operator_action"])
    for field in evidence_gate.get("missing_required_evidence") or []:
        operator_next_actions.append(f"collect_missing_required_evidence:{field}")
    for action in required_evidence_actions:
        operator_next_actions.append(action["action"])
    for blocker in safety_blockers or []:
        operator_next_actions.append(f"review_safety_blocker:{blocker}")
    for detail in safety_blocker_details:
        for action in detail.get("operator_actions") or []:
            operator_next_actions.append(action)
    operator_next_actions.append("confirm_rollback_expectation")
    for target in operator_contract["future_wing_readback_contract"][
        "readback_targets"
    ]:
        operator_next_actions.append(f"verify_future_wing_readback_target:{target}")
    for proof in future_live_promotion_requirements.get("required_proofs") or []:
        operator_next_actions.append(f"collect_future_live_required_proof:{proof}")
    for detail in future_live_requirement_details:
        operator_next_actions.append(detail["operator_action"])
    for detail in operator_review_record_field_details:
        operator_next_actions.append(detail["operator_action"])
    operator_next_actions = _unique_fields(operator_next_actions)
    operator_next_action_details = _operator_next_action_details(
        operator_next_actions=operator_next_actions,
        required_evidence_actions=required_evidence_actions,
        safety_blocker_details=safety_blocker_details,
        why_blocked_details=why_blocked_details,
        future_live_requirement_details=future_live_requirement_details,
        evidence_gate=evidence_gate,
        confidence_review=confidence_review,
        operator_proposal_review_gate=operator_proposal_review_gate,
        rollback_readback_contract=rollback_readback_contract,
        live_authority_boundary=live_authority_boundary,
        operator_review_record_field_details=operator_review_record_field_details,
        source_evidence_details=source_evidence_details,
        confidence_review_details=confidence_review_details,
        bounded_delta_details=bounded_delta_details,
        rollback_readback_details=rollback_readback_details,
        headroom_feedback_true_peak_check_details=(
            headroom_feedback_true_peak_check_details
        ),
    )
    operator_review_checklist_details = _operator_review_checklist_details(
        operator_review_checklist=operator_review_checklist,
        required_evidence_status=required_evidence_status,
        operator_next_action_details=operator_next_action_details,
        evidence_gate=evidence_gate,
        confidence_review=confidence_review,
        operator_proposal_review_gate=operator_proposal_review_gate,
        rollback_readback_contract=rollback_readback_contract,
        live_authority_boundary=live_authority_boundary,
    )

    return {
        "packet_type": "operator_readable_correction_explanation",
        "audience": "sound_engineer",
        "mode": "offline_dry_run_proposal_only",
        "method_id": source_evidence_summary["method_id"],
        "source": source_evidence_summary["source"],
        "target": source_evidence_summary["target"],
        "value_type": operator_contract["value_type"],
        "target_kind": source_evidence_summary["target_kind"],
        "old_value_source": source_evidence_summary["old_value_source"],
        "objective_family": source_evidence_summary["objective_family"],
        "bounded_delta_unit": operator_contract["bounded_delta_unit"],
        "bounded_delta_policy": operator_contract["bounded_delta_policy"],
        "problem": problem_summary,
        "proposed_correction": _summarize_proposed_correction(proposed_correction),
        "source_evidence_summary": source_evidence_summary,
        "source_evidence_details": [dict(row) for row in source_evidence_details],
        "bounded_delta": bounded_delta,
        "bounded_delta_details": [dict(row) for row in bounded_delta_details],
        "evidence_status": evidence_gate["status"],
        "required_evidence_status": source_evidence_summary[
            "required_evidence_status"
        ],
        "required_evidence_actions": required_evidence_actions,
        "missing_required_evidence": list(
            evidence_gate.get("missing_required_evidence") or []
        ),
        "confidence_status": confidence_review["status"],
        "confidence_blockers": list(confidence_review.get("blockers") or []),
        "confidence_review_details": [
            dict(row) for row in confidence_review_details
        ],
        "confidence_review_summary": {
            "raw_confidence": confidence_review.get("raw_confidence"),
            "confidence": confidence_review.get("confidence"),
            "status": confidence_review["status"],
            "expected_range": list(confidence_review["expected_range"]),
            "low_confidence_threshold": confidence_review[
                "low_confidence_threshold"
            ],
            "blockers": list(confidence_review.get("blockers") or []),
            "operator_action": confidence_review["operator_action"],
            "operator_review_required": confidence_review[
                "operator_review_required"
            ],
            "blocks_live_promotion": confidence_review["blocks_live_promotion"],
            "blocks_queue_import": confidence_review["blocks_queue_import"],
            "read_only_or_dry_run_only": True,
            "live_apply_allowed": False,
            "queue_import_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "readiness_closure_allowed": False,
            "can_authorize_live": False,
            "can_close_readiness": False,
            "mutates_mixer": False,
            "future_wing_readback_required": True,
            "detail_rows": [dict(row) for row in confidence_review_details],
        },
        "proposal_review_gate_status": operator_proposal_review_gate["status"],
        "offline_acceptance_allowed": operator_proposal_review_gate[
            "offline_acceptance_allowed"
        ],
        "recommended_decision": operator_proposal_review_gate[
            "recommended_decision"
        ],
        "why_safe": [
            "read_only_or_dry_run_only",
            "live_apply_denied",
            "queue_import_denied",
            f"bounded_delta_policy:{operator_contract['bounded_delta_policy']}",
            f"rollback_expectation:{operator_contract['rollback_expectation']}",
        ],
        "why_safe_details": [dict(row) for row in why_safe_details],
        "why_blocked": _unique_fields(
            [
                "live_apply_denied_until_real_wing_proof",
                "queue_import_denied_until_operator_review",
            ],
            evidence_gate.get("blockers") or [],
            confidence_review.get("blockers") or [],
            operator_proposal_review_gate.get("live_authority_blockers") or [],
            safety_blockers,
        ),
        "why_blocked_details": [dict(row) for row in why_blocked_details],
        "safety_checks": list(safety_checks.get("required_checks") or []),
        "headroom_feedback_true_peak_check_details": [
            dict(row) for row in headroom_feedback_true_peak_check_details
        ],
        "safety_review_status": operator_safety_review_matrix["evidence_gate_status"],
        "safety_review_operator_action": operator_safety_review_matrix[
            "operator_action"
        ],
        "safety_review_summary": dict(operator_safety_review_matrix["summary"]),
        "safety_review_checks": [
            dict(row) for row in operator_safety_review_matrix["checks"]
        ],
        "missing_safety_evidence": list(
            operator_safety_review_matrix.get("missing_required_evidence") or []
        ),
        "safety_blockers": list(safety_blockers or []),
        "effective_safety_blockers": list(effective_safety_blockers or []),
        "safety_blocker_details": [dict(row) for row in safety_blocker_details],
        "rollback_expectation": operator_contract["rollback_expectation"],
        "rollback_readback_contract": rollback_readback_contract,
        "rollback_readback_details": [
            dict(row) for row in rollback_readback_details
        ],
        "operator_review_checklist": list(operator_review_checklist),
        "operator_review_checklist_details": operator_review_checklist_details,
        "operator_next_actions": operator_next_actions,
        "operator_next_action_details": operator_next_action_details,
        "future_live_requirement_details": [
            dict(row) for row in future_live_requirement_details
        ],
        "future_live_requirements": {
            "required_before_live_promotion": True,
            "required_proofs": list(
                future_live_promotion_requirements.get("required_proofs") or []
            ),
            "readback_targets": list(
                future_live_promotion_requirements.get("readback_targets") or []
            ),
            "blocked_live_actions": list(
                future_live_promotion_requirements.get("blocked_live_actions") or []
            ),
            "required_evidence_status": future_live_promotion_requirements[
                "required_evidence_status"
            ],
            "read_only_or_dry_run_only": True,
            "live_apply_allowed": False,
            "queue_import_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "readiness_closure_allowed": False,
            "can_authorize_live": False,
            "can_close_readiness": False,
            "mutates_mixer": False,
            "operator_review_required": True,
            "future_wing_readback_required": True,
            "blocks_live_promotion": True,
            "blocks_queue_import": True,
            "denial_reason": live_authority_boundary["denial_reason"],
            "wing_proof_required": True,
            "live_authority_blockers": list(live_authority_boundary["blockers"]),
            "requirement_details": [
                dict(row) for row in future_live_requirement_details
            ],
        },
        "live_authority_boundary_summary": {
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "readiness_closure_allowed": False,
            "wing_proof_required": True,
            "hardware_proof_required": live_authority_boundary[
                "hardware_proof_required"
            ],
            "operator_review_required": live_authority_boundary[
                "operator_review_required"
            ],
            "readback_required_before_live_promotion": live_authority_boundary[
                "readback_required_before_live_promotion"
            ],
            "allowed_current_live_write_kinds": list(
                live_authority_boundary["allowed_current_live_write_kinds"]
            ),
            "blocked_live_actions": list(
                live_authority_boundary["blocked_live_actions"]
            ),
            "blockers": list(live_authority_boundary["blockers"]),
            "denial_reason": live_authority_boundary["denial_reason"],
            "evidence_gate_status": live_authority_boundary["evidence_gate_status"],
            "live_apply_allowed": False,
            "queue_import_allowed": False,
            "read_only_or_dry_run_only": True,
            "mutates_mixer": False,
            "future_wing_readback_required": True,
            "blocks_live_promotion": True,
            "blocks_queue_import": True,
            "can_authorize_live": False,
            "can_close_readiness": False,
        },
        "future_wing_readback_summary": {
            "required_before_live_promotion": True,
            "target_kind": operator_contract["future_wing_readback_contract"][
                "target_kind"
            ],
            "old_value_source": operator_contract["future_wing_readback_contract"][
                "old_value_source"
            ],
            "readback_targets": list(
                operator_contract["future_wing_readback_contract"][
                    "readback_targets"
                ]
            ),
            "required_evidence_fields": list(
                operator_contract["future_wing_readback_contract"][
                    "required_evidence_fields"
                ]
            ),
            "missing_required_evidence": list(
                evidence_gate.get("missing_required_evidence") or []
            ),
            "required_evidence_status": future_live_promotion_requirements[
                "required_evidence_status"
            ],
            "blocked_live_actions": list(
                operator_contract["future_wing_readback_contract"][
                    "blocked_live_actions"
                ]
            ),
            "evidence_gate_status": evidence_gate["status"],
            "operator_action": evidence_gate["operator_action"],
            "requirement_details": [
                dict(row)
                for row in future_live_requirement_details
                if row["requirement_type"] == "future_wing_readback_target"
            ],
            "rollback_readback_details": [
                dict(row) for row in rollback_readback_details
            ],
            "read_only_or_dry_run_only": True,
            "live_apply_allowed": False,
            "queue_import_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "readiness_closure_allowed": False,
            "can_authorize_live": False,
            "can_close_readiness": False,
            "mutates_mixer": False,
            "operator_review_required": True,
            "future_wing_readback_required": True,
            "blocks_live_promotion": True,
            "blocks_queue_import": True,
            "denial_reason": live_authority_boundary["denial_reason"],
        },
        "future_live_proof_contract_summary": {
            "required_before_live_promotion": True,
            "bounded_delta_policy": operator_contract["bounded_delta_policy"],
            "bounded_delta": dict(bounded_delta),
            "rollback_expectation": operator_contract["rollback_expectation"],
            "rollback_readback_contract": rollback_readback_contract,
            "rollback_readback_details": [
                dict(row) for row in rollback_readback_details
            ],
            "target_kind": operator_contract["future_wing_readback_contract"][
                "target_kind"
            ],
            "old_value_source": operator_contract["future_wing_readback_contract"][
                "old_value_source"
            ],
            "readback_targets": list(
                operator_contract["future_wing_readback_contract"][
                    "readback_targets"
                ]
            ),
            "required_proofs": list(
                future_live_promotion_requirements.get("required_proofs") or []
            ),
            "required_evidence_fields": list(
                operator_contract["future_wing_readback_contract"][
                    "required_evidence_fields"
                ]
            ),
            "missing_required_evidence": list(
                evidence_gate.get("missing_required_evidence") or []
            ),
            "required_evidence_status": future_live_promotion_requirements[
                "required_evidence_status"
            ],
            "safety_blockers": list(safety_blockers or []),
            "effective_safety_blockers": list(effective_safety_blockers or []),
            "safety_blocker_details": [
                dict(row) for row in safety_blocker_details
            ],
            "blocked_live_actions": list(
                operator_contract["future_wing_readback_contract"][
                    "blocked_live_actions"
                ]
            ),
            "live_authority_blockers": list(live_authority_boundary["blockers"]),
            "evidence_gate_status": evidence_gate["status"],
            "confidence_review_status": confidence_review["status"],
            "operator_review_gate_status": operator_proposal_review_gate["status"],
            "operator_action": operator_proposal_review_gate["operator_action"],
            "requirement_details": [
                dict(row) for row in future_live_requirement_details
            ],
            "read_only_or_dry_run_only": True,
            "live_apply_allowed": False,
            "queue_import_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "readiness_closure_allowed": False,
            "can_authorize_live": False,
            "can_close_readiness": False,
            "mutates_mixer": False,
            "operator_review_required": True,
            "future_wing_readback_required": True,
            "blocks_live_promotion": True,
            "blocks_queue_import": True,
            "denial_reason": live_authority_boundary["denial_reason"],
        },
        "operator_review_decision_options": [
            dict(option) for option in operator_review_decision_options
        ],
        "operator_review_decision_option_details": [
            dict(row) for row in operator_review_decision_option_details
        ],
        "operator_review_record_field_details": [
            dict(row) for row in operator_review_record_field_details
        ],
        "operator_review_record_summary": {
            "record_type": operator_review_record_template["record_type"],
            "required_record_fields": list(
                operator_review_record_template["required_record_fields"]
            ),
            "allowed_decisions": list(
                operator_review_record_template["allowed_decisions"]
            ),
            "recommended_decision": operator_review_record_template[
                "recommended_decision"
            ],
            "operator_action": operator_review_record_template["operator_action"],
            "evidence_gate_status": operator_review_record_template[
                "evidence_gate_status"
            ],
            "missing_required_evidence": list(
                operator_review_record_template.get("missing_required_evidence") or []
            ),
            "offline_acceptance_allowed": operator_review_record_template[
                "offline_acceptance_allowed"
            ],
            "operator_review_required": operator_review_record_template[
                "operator_review_required"
            ],
            "decision_records_live_authority": operator_review_record_template[
                "decision_records_live_authority"
            ],
            "decision_records_queue_import": operator_review_record_template[
                "decision_records_queue_import"
            ],
            "read_only_or_dry_run_only": True,
            "live_apply_allowed": False,
            "queue_import_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "readiness_closure_allowed": False,
            "can_authorize_live": False,
            "can_close_readiness": False,
            "mutates_mixer": False,
            "future_wing_readback_required": True,
            "blocks_live_promotion": True,
            "blocks_queue_import": True,
            "denial_reason": operator_review_record_template["denial_reason"],
            "field_details": [
                dict(row) for row in operator_review_record_field_details
            ],
        },
        "read_only_or_dry_run_only": True,
        "live_apply_allowed": False,
        "queue_import_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "readiness_closure_allowed": False,
        "can_authorize_live": False,
        "can_close_readiness": False,
        "mutates_mixer": False,
        "operator_review_required": True,
        "future_wing_readback_required": True,
        "blocks_live_promotion": True,
        "blocks_queue_import": True,
        "denial_reason": live_authority_boundary["denial_reason"],
    }


_EVIDENCE_TOKEN_STOPWORDS = {
    "and",
    "or",
    "the",
    "with",
    "current",
    "required",
    "before",
    "promotion",
    "cache",
    "no",
    "live",
    "authority",
}


def _evidence_tokens(value: Any) -> set:
    normalized = "".join(
        char.lower() if char.isalnum() else "_"
        for char in str(value or "")
    )
    return {
        token
        for token in normalized.split("_")
        if token and token not in _EVIDENCE_TOKEN_STOPWORDS
    }


def _safety_check_matches_evidence(check: str, evidence_field: str) -> bool:
    check_tokens = _evidence_tokens(check)
    field_tokens = _evidence_tokens(evidence_field)
    if not check_tokens or not field_tokens:
        return False
    return bool(check_tokens & field_tokens)


def _operator_safety_review_matrix(
    *,
    operator_contract: Dict[str, Any],
    evidence_completeness: Dict[str, Any],
    evidence_gate: Dict[str, Any],
) -> Dict[str, Any]:
    required_evidence = list(operator_contract["required_evidence_fields"])
    provided = set(evidence_completeness.get("provided") or [])
    missing = set(evidence_gate.get("missing_required_evidence") or [])
    rows: List[Dict[str, Any]] = []

    for check in operator_contract["applicable_safety_checks"]:
        matching_required = [
            field
            for field in required_evidence
            if _safety_check_matches_evidence(str(check), field)
        ]
        provided_matching = [
            field for field in matching_required if field in provided
        ]
        missing_matching = [
            field for field in matching_required if field in missing
        ]
        if missing_matching:
            status = "missing_required_evidence"
        elif matching_required:
            status = "evidence_available_operator_review_required"
        else:
            status = "operator_review_required_no_direct_evidence_mapping"
        mapping_review_required = not bool(matching_required)

        rows.append(
            {
                "check": check,
                "status": status,
                "evidence_mapping_status": "direct"
                if matching_required
                else "no_direct_required_evidence_field_match",
                "required_evidence_fields": matching_required,
                "provided_evidence_fields": provided_matching,
                "missing_required_evidence": missing_matching,
                "available_contract_evidence_fields": sorted(provided),
                "all_contract_evidence_fields": required_evidence,
                "mapping_review_required": mapping_review_required,
                "review_note": (
                    "Operator must review the full evidence packet because this "
                    "safety check has no direct required-evidence field mapping."
                    if mapping_review_required
                    else "Direct required evidence is available for operator review."
                ),
                "operator_action": evidence_gate["operator_action"],
                "read_only_or_dry_run_only": True,
                "live_apply_allowed": False,
                "queue_import_allowed": False,
                "live_authority_granted": False,
                "full_autonomous_ready": False,
                "readiness_closure_allowed": False,
                "can_authorize_live": False,
                "can_close_readiness": False,
                "mutates_mixer": False,
                "operator_review_required": True,
                "future_wing_readback_required": True,
                "denial_reason": (
                    "operator_review_and_future_wing_readback_required"
                ),
                "blocks_live_promotion": True,
                "blocks_queue_import": True,
            }
        )

    return {
        "read_only_or_dry_run_only": True,
        "live_apply_allowed": False,
        "queue_import_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "readiness_closure_allowed": False,
        "can_authorize_live": False,
        "can_close_readiness": False,
        "mutates_mixer": False,
        "operator_review_required": True,
        "future_wing_readback_required": True,
        "blocks_live_promotion": True,
        "blocks_queue_import": True,
        "denial_reason": "operator_review_and_future_wing_readback_required",
        "evidence_gate_status": evidence_gate["status"],
        "operator_action": evidence_gate["operator_action"],
        "applicable_safety_checks": list(operator_contract["applicable_safety_checks"]),
        "required_evidence_fields": required_evidence,
        "provided_evidence_fields": list(evidence_completeness.get("provided") or []),
        "missing_required_evidence": list(
            evidence_gate.get("missing_required_evidence") or []
        ),
        "checks": rows,
        "summary": {
            "check_count": len(rows),
            "checks_with_missing_evidence": sum(
                1
                for row in rows
                if row["status"] == "missing_required_evidence"
            ),
            "checks_with_direct_evidence": sum(
                1
                for row in rows
                if row["required_evidence_fields"]
            ),
            "checks_requiring_mapping_review": sum(
                1
                for row in rows
                if row["mapping_review_required"]
            ),
        },
    }


def _future_live_promotion_requirements(
    *,
    operator_contract: Dict[str, Any],
    evidence_completeness: Dict[str, Any],
    evidence_gate: Dict[str, Any],
    live_authority_boundary: Dict[str, Any],
) -> Dict[str, Any]:
    future_readback = operator_contract["future_wing_readback_contract"]
    return {
        **operator_contract["future_live_promotion_requirements"],
        "method_id": future_readback["method_id"],
        "source": future_readback["source"],
        "target": future_readback["target"],
        "value_type": future_readback["value_type"],
        "target_kind": future_readback["target_kind"],
        "old_value_source": future_readback["old_value_source"],
        "objective_family": future_readback["objective_family"],
        "bounded_delta_unit": future_readback["bounded_delta_unit"],
        "bounded_delta_policy": future_readback["bounded_delta_policy"],
        "readback_targets": list(future_readback["readback_targets"]),
        "blocked_live_actions": list(future_readback["blocked_live_actions"]),
        "required_evidence_fields": list(future_readback["required_evidence_fields"]),
        "missing_required_evidence": list(
            evidence_gate.get("missing_required_evidence") or []
        ),
        "required_evidence_status": _required_evidence_status(
            evidence_completeness,
            evidence_gate,
        ),
        "evidence_gate_status": evidence_gate["status"],
        "operator_action": evidence_gate["operator_action"],
        "allowed_current_live_write_kinds": list(
            live_authority_boundary["allowed_current_live_write_kinds"]
        ),
        "live_authority_blockers": list(live_authority_boundary["blockers"]),
        "live_apply_allowed": False,
        "queue_import_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "readiness_closure_allowed": False,
        "can_authorize_live": False,
        "can_close_readiness": False,
        "mutates_mixer": False,
        "operator_review_required": True,
        "blocks_live_promotion": True,
        "blocks_queue_import": True,
        "future_wing_readback_required": True,
    }


def _future_live_requirement_details(
    *,
    operator_contract: Dict[str, Any],
    future_live_promotion_requirements: Dict[str, Any],
    live_authority_boundary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    future_readback = operator_contract["future_wing_readback_contract"]
    common = {
        "method_id": future_readback["method_id"],
        "source": future_readback["source"],
        "target": future_readback["target"],
        "value_type": future_readback["value_type"],
        "target_kind": future_readback["target_kind"],
        "old_value_source": future_readback["old_value_source"],
        "objective_family": future_readback["objective_family"],
        "bounded_delta_unit": future_readback["bounded_delta_unit"],
        "bounded_delta_policy": future_readback["bounded_delta_policy"],
        "blocked_live_actions": list(future_readback["blocked_live_actions"]),
        "live_authority_blockers": list(live_authority_boundary["blockers"]),
        "required_before_live_promotion": True,
        "required_before_queue_import": True,
        "read_only_or_dry_run_only": True,
        "live_apply_allowed": False,
        "queue_import_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "readiness_closure_allowed": False,
        "can_authorize_live": False,
        "can_close_readiness": False,
        "mutates_mixer": False,
        "operator_review_required": True,
        "operator_approval_required": True,
        "hardware_proof_required": True,
        "future_wing_readback_required": True,
        "blocks_live_promotion": True,
        "blocks_queue_import": True,
        "denial_reason": live_authority_boundary["denial_reason"],
    }
    details: List[Dict[str, Any]] = []

    for proof in future_live_promotion_requirements.get("required_proofs") or []:
        action = f"collect_future_live_required_proof:{proof}"
        details.append(
            {
                **common,
                "requirement": proof,
                "proof": proof,
                "readback_target": None,
                "requirement_type": "future_live_required_proof",
                "status": "pending_real_wing_proof",
                "operator_action": action,
                "action": action,
            }
        )

    for target in future_readback.get("readback_targets") or []:
        action = f"verify_future_wing_readback_target:{target}"
        details.append(
            {
                **common,
                "requirement": target,
                "proof": None,
                "readback_target": target,
                "requirement_type": "future_wing_readback_target",
                "status": "pending_future_wing_readback",
                "operator_action": action,
                "action": action,
            }
        )

    return details


def _rollback_readback_contract(
    *,
    operator_contract: Dict[str, Any],
    future_live_promotion_requirements: Dict[str, Any],
    evidence_completeness: Dict[str, Any],
    evidence_gate: Dict[str, Any],
) -> Dict[str, Any]:
    future_readback = operator_contract["future_wing_readback_contract"]
    return {
        "description": operator_contract["rollback_expectation"],
        "rollback_expectation": operator_contract["rollback_expectation"],
        "target_kind": future_readback["target_kind"],
        "old_value_source": future_readback["old_value_source"],
        "readback_targets": list(future_readback["readback_targets"]),
        "blocked_live_actions": list(future_readback["blocked_live_actions"]),
        "required_evidence_fields": list(future_readback["required_evidence_fields"]),
        "missing_required_evidence": list(
            evidence_gate.get("missing_required_evidence") or []
        ),
        "required_evidence_status": _required_evidence_status(
            evidence_completeness,
            evidence_gate,
        ),
        "required_proofs": list(
            future_live_promotion_requirements.get("required_proofs") or []
        ),
        "rollback_required_before_promotion": True,
        "readback_required_before_live_promotion": True,
        "operator_review_required": True,
        "operator_approval_required": True,
        "hardware_proof_required": True,
        "read_only_or_dry_run_only": True,
        "live_apply_allowed": False,
        "queue_import_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "readiness_closure_allowed": False,
        "can_authorize_live": False,
        "can_close_readiness": False,
        "mutates_mixer": False,
        "blocks_live_promotion": True,
        "blocks_queue_import": True,
        "future_wing_readback_required": True,
        "denial_reason": future_live_promotion_requirements["blocked_until"],
        "operator_next_step": (
            "capture_real_wing_readback_and_rollback_proof_before_live_promotion"
        ),
        "why_blocked": _unique_fields(
            [
                "live_apply_denied_until_real_wing_proof",
                "queue_import_denied_until_operator_review",
                "rollback_readback_proof_required",
                "future_wing_readback_required",
            ],
            evidence_gate.get("blockers") or [],
        ),
    }


def _rollback_readback_details(
    *,
    rollback_readback_contract: Dict[str, Any],
    future_live_promotion_requirements: Dict[str, Any],
    live_authority_boundary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    required_evidence_status = rollback_readback_contract.get(
        "required_evidence_status", {}
    )
    common = {
        "rollback_expectation": rollback_readback_contract["rollback_expectation"],
        "method_id": rollback_readback_contract["method_id"],
        "source": rollback_readback_contract["source"],
        "target": rollback_readback_contract["target"],
        "value_type": rollback_readback_contract["value_type"],
        "target_kind": rollback_readback_contract["target_kind"],
        "old_value_source": rollback_readback_contract["old_value_source"],
        "objective_family": rollback_readback_contract["objective_family"],
        "bounded_delta_unit": rollback_readback_contract["bounded_delta_unit"],
        "bounded_delta_policy": rollback_readback_contract["bounded_delta_policy"],
        "readback_targets": list(rollback_readback_contract["readback_targets"]),
        "blocked_live_actions": list(rollback_readback_contract["blocked_live_actions"]),
        "required_evidence_fields": list(
            rollback_readback_contract["required_evidence_fields"]
        ),
        "missing_required_evidence": list(
            rollback_readback_contract.get("missing_required_evidence") or []
        ),
        "required_evidence_status": required_evidence_status,
        "required_proofs": list(rollback_readback_contract.get("required_proofs") or []),
        "future_live_promotion_requires": list(
            future_live_promotion_requirements.get("required_proofs") or []
        ),
        "evidence_gate_status": future_live_promotion_requirements.get(
            "evidence_gate_status"
        )
        or required_evidence_status.get("status")
        or "operator_review_required",
        "required_before_operator_acceptance": True,
        "required_before_live_promotion": True,
        "required_before_queue_import": True,
        "rollback_required": True,
        "rollback_required_before_promotion": True,
        "readback_required": True,
        "readback_required_before_live_promotion": True,
        "operator_review_required": True,
        "operator_approval_required": True,
        "hardware_proof_required": True,
        "read_only_or_dry_run_only": True,
        "live_apply_allowed": False,
        "queue_import_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "readiness_closure_allowed": False,
        "can_authorize_live": False,
        "can_close_readiness": False,
        "can_import_to_queue": False,
        "mutates_mixer": False,
        "future_wing_readback_required": True,
        "blocks_live_promotion": True,
        "blocks_queue_import": True,
        "denial_reason": (
            live_authority_boundary.get("denial_reason")
            or rollback_readback_contract.get("denial_reason")
        ),
        "why_blocked": _unique_fields(
            rollback_readback_contract.get("why_blocked") or [],
            live_authority_boundary.get("blockers") or [],
        ),
    }
    details = [
        {
            **common,
            "detail": "rollback_expectation",
            "detail_type": "rollback_expectation",
            "description": rollback_readback_contract["description"],
            "status": required_evidence_status.get("status")
            or "operator_review_required",
            "readback_target": None,
            "operator_action": "confirm_rollback_expectation",
            "action": "confirm_rollback_expectation",
            "action_type": "confirm_rollback_expectation",
        }
    ]
    for target in rollback_readback_contract.get("readback_targets") or []:
        action = f"verify_rollback_readback_target:{target}"
        details.append(
            {
                **common,
                "detail": "rollback_readback_target",
                "detail_type": "rollback_readback_target",
                "description": (
                    "Future live promotion must read back this target before "
                    "any rollback-capable apply path can be authorized."
                ),
                "status": "pending_future_wing_readback",
                "readback_target": target,
                "operator_action": action,
                "action": action,
                "action_type": "verify_rollback_readback_target",
            }
        )
    return details


def _operator_evidence_packet(
    *,
    proposal: Dict[str, Any],
    method_id: str,
    target: Any,
    problem_summary: str,
    proposed_correction: Dict[str, Any],
    bounded_delta: Dict[str, Any],
    bounded_delta_details: List[Dict[str, Any]],
    safety_blockers: List[str],
    effective_safety_blockers: List[str],
    safety_blocker_details: List[Dict[str, Any]],
    why_safe_details: List[Dict[str, Any]],
    why_blocked_details: List[Dict[str, Any]],
    confidence_review: Dict[str, Any],
    confidence_review_details: List[Dict[str, Any]],
    source_evidence_details: List[Dict[str, Any]],
    operator_proposal_review_gate: Dict[str, Any],
    safety_checks: Dict[str, Any],
    headroom_feedback_true_peak_check_details: List[Dict[str, Any]],
    operator_safety_review_matrix: Dict[str, Any],
    future_live_promotion_requirements: Dict[str, Any],
    future_live_requirement_details: List[Dict[str, Any]],
    operator_next_action_details: List[Dict[str, Any]],
    operator_review_checklist_details: List[Dict[str, Any]],
    operator_review_decision_option_details: List[Dict[str, Any]],
    operator_review_record_field_details: List[Dict[str, Any]],
    rollback_readback_contract: Dict[str, Any],
    rollback_readback_details: List[Dict[str, Any]],
    operator_review_record_template: Dict[str, Any],
    operator_readable_explanation: str,
    operator_explanation_packet: Dict[str, Any],
    operator_contract: Dict[str, Any],
    evidence_completeness: Dict[str, Any],
    evidence_gate: Dict[str, Any],
    live_authority_boundary: Dict[str, Any],
) -> Dict[str, Any]:
    required_evidence_status = _required_evidence_status(
        evidence_completeness,
        evidence_gate,
    )
    return {
        **operator_contract["operator_evidence_packet"],
        "method_id": method_id,
        "objective_family": operator_contract["objective_family"],
        "source": proposal.get("source"),
        "target": target if target is not None else proposal.get("target"),
        "value_type": operator_contract["value_type"],
        "target_kind": operator_contract["target_kind"],
        "old_value_source": operator_contract["old_value_source"],
        "problem_statement": problem_summary,
        "proposed_correction_summary": _summarize_proposed_correction(
            proposed_correction
        ),
        "confidence": proposal.get("confidence"),
        "confidence_review": confidence_review,
        "confidence_review_details": [
            dict(row) for row in confidence_review_details
        ],
        "source_evidence_details": [dict(row) for row in source_evidence_details],
        "operator_proposal_review_gate": operator_proposal_review_gate,
        "bounded_delta": bounded_delta,
        "bounded_delta_details": [dict(row) for row in bounded_delta_details],
        "bounded_delta_unit": operator_contract["bounded_delta_unit"],
        "bounded_delta_policy": operator_contract["bounded_delta_policy"],
        "safety_blockers": safety_blockers,
        "effective_safety_blockers": effective_safety_blockers,
        "safety_blocker_details": [dict(row) for row in safety_blocker_details],
        "why_safe_details": [dict(row) for row in why_safe_details],
        "why_blocked_details": [dict(row) for row in why_blocked_details],
        "headroom_feedback_true_peak_checks": safety_checks,
        "headroom_feedback_true_peak_check_details": [
            dict(row) for row in headroom_feedback_true_peak_check_details
        ],
        "operator_safety_review_matrix": operator_safety_review_matrix,
        "future_live_promotion_requirements": future_live_promotion_requirements,
        "future_live_requirement_details": [
            dict(row) for row in future_live_requirement_details
        ],
        "operator_next_action_details": [
            dict(row) for row in operator_next_action_details
        ],
        "operator_review_checklist_details": [
            dict(row) for row in operator_review_checklist_details
        ],
        "operator_review_decision_option_details": [
            dict(row) for row in operator_review_decision_option_details
        ],
        "operator_review_record_field_details": [
            dict(row) for row in operator_review_record_field_details
        ],
        "operator_review_record_template": operator_review_record_template,
        "operator_readable_explanation": operator_readable_explanation,
        "operator_explanation_packet": operator_explanation_packet,
        "rollback_expectation": operator_contract["rollback_expectation"],
        "rollback_readback_contract": rollback_readback_contract,
        "rollback_readback_details": [
            dict(row) for row in rollback_readback_details
        ],
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "readiness_closure_allowed": False,
        "can_authorize_live": False,
        "can_close_readiness": False,
        "mutates_mixer": False,
        "future_wing_readback_required": True,
        "blocks_live_promotion": True,
        "blocks_queue_import": True,
        "evidence_completeness": evidence_completeness,
        "evidence_gate_status": evidence_gate["status"],
        "required_evidence_status": required_evidence_status,
        "required_evidence_actions": required_evidence_status[
            "required_evidence_actions"
        ],
        "missing_required_evidence": list(
            evidence_gate.get("missing_required_evidence") or []
        ),
        "required_evidence_fields": list(operator_contract["required_evidence_fields"]),
        "provided_evidence_fields": list(evidence_completeness.get("provided") or []),
        "operator_action": evidence_gate["operator_action"],
        "blocked_live_actions": list(
            operator_contract["future_wing_readback_contract"]["blocked_live_actions"]
        ),
        "live_authority_blockers": list(live_authority_boundary["blockers"]),
        "live_authority_boundary": live_authority_boundary,
        "future_wing_readback_contract": operator_contract[
            "future_wing_readback_contract"
        ],
    }


def _headroom_feedback_true_peak_checks(
    *,
    proposal: Dict[str, Any],
    requested_change: Optional[Dict[str, Any]],
    requested_changes: Optional[List[Dict[str, Any]]],
    proposed_action: Optional[Dict[str, Any]],
    operator_contract: Dict[str, Any],
) -> Dict[str, Any]:
    raw = proposal.get("raw") if isinstance(proposal.get("raw"), dict) else {}
    checks = {
        "required_checks": list(operator_contract["applicable_safety_checks"]),
        "required_evidence_fields": list(operator_contract["required_evidence_fields"]),
        "readback_required_before_live_promotion": True,
        "read_only_or_dry_run_only": True,
        "live_write": False,
        "queue_mutation": False,
        "live_apply_allowed": False,
        "queue_import_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "readiness_closure_allowed": False,
        "can_authorize_live": False,
        "can_close_readiness": False,
        "mutates_mixer": False,
        "operator_review_required": True,
        "blocks_live_promotion": True,
        "blocks_queue_import": True,
        "future_wing_readback_required": True,
        "denial_reason": operator_contract["future_live_promotion_requirements"][
            "blocked_until"
        ],
        "operator_next_step": "review_required_safety_checks_and_collect_missing_evidence",
        "future_live_promotion_requires": list(
            operator_contract["future_live_promotion_requirements"]["required_proofs"]
        ),
        "why_blocked": [
            "live_apply_denied_until_real_wing_proof",
            "queue_import_denied_until_operator_review",
            "future_wing_readback_required",
            "operator_safety_review_required",
        ],
    }
    evidence_items = _safety_evidence_items(requested_change, requested_changes)
    if evidence_items:
        checks["provided_evidence"] = evidence_items
    if proposed_action:
        checks["proposed_action_checks"] = {
            key: proposed_action[key]
            for key in (
                "true_peak_headroom_db",
                "main_bus_headroom_db",
                "feedback_risk",
                "target_plane_lufs",
            )
            if key in proposed_action
        }
    evidence = proposal.get("evidence") if isinstance(proposal.get("evidence"), dict) else {}
    evidence_check_keys = _unique_fields(
        operator_contract["required_evidence_fields"],
        (
            "true_peak_headroom_db",
            "true_peak_headroom",
            "feedback_risk",
            "feedback_safety",
            "input_level_evidence",
            "overload_or_weak_signal_problem",
            "headroom_margin",
            "bleed_risk",
            "gain_down_or_operator_review_gain_up",
            "rollback_input_trim_readback",
            "current_fader_readback",
            "ayaic_level_plane",
            "ayaic_target_plane",
            "master_spectrum_reference",
            "source_or_stem_contributor",
            "bounded_fader_delta",
            "fader_ceiling",
            "routing_fix_disabled",
            "rollback_fader_readback",
            "integrated_lufs",
            "summed_child_lufs",
            "assignment_readback",
            "current_bus_fader",
            "current_bus_group_dca_matrix_cache_freshness",
            "bounded_bus_group_delta",
            "rollback_bus_group_dca_matrix_readback",
            "phase_risk",
            "phase_safety",
            "masking_pair",
            "linked_cut_boost_pair",
            "frequency_match",
            "cut_or_bounded_boost",
            "feedback_notch_preserved",
            "manual_band_preserved",
            "bounded_gain_delta",
            "current_eq_context_cache_freshness",
            "current_masking_context_cache_freshness",
            "current_masking_context_cache_no_live_authority",
            "current_arrangement_context_cache_freshness",
            "makeup_gain_headroom_reserve_db",
            "requested_makeup_gain_db",
            "target_gain_reduction_db",
            "gain_reduction_estimate",
            "current_state_cache_freshness_status",
            "current_dynamics_cache_freshness",
            "current_eq_context_cache_no_live_authority",
            "current_frequency_context_cache_freshness",
            "current_frequency_context_cache_no_live_authority",
            "source_stem_contribution",
            "lane_conflict_pair",
            "bounded_subtractive_delta",
            "sum_bus_check",
            "eq_band_readback",
            "rollback_eq_band_readback",
            "current_phase_audio_context_cache_no_live_authority",
            "current_dynamics_cache_no_live_authority",
            "current_gate_bleed_context_cache_freshness",
            "current_gate_bleed_context_cache_no_live_authority",
            "current_dynamics_or_dynamic_eq_cache_freshness",
            "current_dynamics_or_dynamic_eq_cache_no_live_authority",
            "arrangement_context_cache_freshness_status",
            "current_arrangement_context_cache_no_live_authority",
            "noise_floor",
            "bleed_telemetry",
            "chatter_risk",
            "chatter_guard",
            "hysteresis",
            "false_open_risk",
            "false_close_risk",
            "tail_risk",
            "quiet_words_tail_attack_preservation",
            "vocal_source_role",
            "arrangement_role",
            "sibilance_ratio",
            "sibilance_persistence",
            "max_gain_reduction_db",
            "makeup_headroom",
            "bypass_state",
            "old_value_source",
            "rollback_dynamics_readback",
            "rollback_gate_readback",
            "dynamic_eq_or_deesser_readback",
            "rollback_dynamics_or_dynamic_eq_readback",
            "bounded_attenuation_only",
            "speech_intelligibility",
            "send_level_limit",
            "fx_return_headroom",
            "return_level_safety",
            "dry_wet_bounds",
            "return_headroom_db",
            "return_level_db",
            "feedback_risk",
            "intelligibility_risk",
            "monitor_safety_risk",
            "monitor_safety",
            "current_fx_send_return_cache_freshness",
            "current_fx_send_return_cache_no_live_authority",
            "current_fx_parameter_schema_hints",
            "fx_parameter_schema_no_live_authority",
            "current_fx_parameter_schema_no_live_authority",
            "current_fx_inserts_snapshot_no_live_authority",
            "source_identity_downstream_trust",
            "rollback_fx_readback",
            "stereo_pair",
            "stage_position",
            "mono_compatibility",
            "bounded_pan_delta",
            "current_pan",
            "current_pan_cache_freshness",
            "mono_compatibility_risk",
            "phase_correlation",
            "mono_sum_delta_db",
            "mono_phase_safety",
            "current_pan_cache_no_live_authority",
            "mono_phase_check_no_live_authority",
            "operator_context",
            "rollback_pan_readback",
            "route_readback_available",
            "rollback_route_available",
            "route_diff",
            "current_route_readback",
            "current_route_cache_freshness",
            "current_state_cache_freshness",
            "proposed_source_readback",
            "rollback_route",
            "destructive_scope_score",
            "active_performance",
            "active_performance_guard",
            "destructive_scope_check",
            "current_route_activity",
            "current_route_cache_no_live_authority",
            "current_route_map_cache_no_live_authority",
            "current_state_readback_available",
            "current_console_state_readback",
            "current_scene_preset_cache_freshness",
            "snapshots_scenes_snapshot_cache_freshness",
            "user_presets_snapshot_cache_freshness",
            "target_preview_available",
            "target_preview",
            "scoped_diff_available",
            "scoped_diff",
            "read_only_scoped_diff_evidence",
            "rollback_snapshot_available",
            "rollback_snapshot",
            "read_only_rollback_evidence",
            "active_signal_guard",
            "current_scene_preset_cache_no_live_authority",
            "current_scene_inventory_cache_no_live_authority",
            "current_user_presets_inventory_cache_no_live_authority",
            "scoped_diff_no_live_authority",
            "rollback_snapshot_no_live_authority",
            "limiter_or_dynamics_proof",
            "pa_feed_readback",
            "pa_feed_safety",
            "pa_feed_safe_for_master_change",
            "pa_feed_cache_freshness",
            "pa_feed_cache_no_live_authority",
            "reference_feed_exclusion",
            "is_reference_feed",
            "current_main_bus_processing_cache_freshness",
            "current_main_bus_processing_cache_no_live_authority",
            "rollback_master_processing_readback",
            "rollback_pa_feed_matrix_readback",
            "source_role",
            "current_mute_state",
            "current_mute_cache_freshness",
            "current_mute_cache_no_live_authority",
            "auto_muted_by_system",
            "signal_activity",
            "idle_duration",
            "operator_intent",
            "dropout_risk",
            "rollback_value",
            "rollback_mute_readback",
            "manual_mute_authority_preserved",
        ),
    )
    evidence_checks = {}
    for key in evidence_check_keys:
        if key in evidence:
            evidence_checks[key] = evidence[key]
        elif key in raw:
            evidence_checks[key] = raw[key]
    if evidence_checks:
        checks["evidence_checks"] = evidence_checks
        checks["evidence_checks_boundary"] = _raw_input_evidence_boundary(
            surface="headroom_feedback_true_peak_checks.evidence_checks",
            evidence_fields=evidence_checks.keys(),
        )
    for key in ("cache_freshness", "safety_status", "apply_status", "risk_score"):
        if key in raw:
            checks[key] = raw[key]
    return checks


def _headroom_feedback_true_peak_check_details(
    *,
    method_id: str,
    safety_checks: Dict[str, Any],
    operator_contract: Dict[str, Any],
    evidence_completeness: Dict[str, Any],
    evidence_gate: Dict[str, Any],
    live_authority_boundary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    required_evidence = list(safety_checks.get("required_evidence_fields") or [])
    provided = set(evidence_completeness.get("provided") or [])
    missing = set(evidence_gate.get("missing_required_evidence") or [])
    evidence_checks = (
        safety_checks.get("evidence_checks")
        if isinstance(safety_checks.get("evidence_checks"), dict)
        else {}
    )
    common = {
        "method_id": method_id,
        "objective_family": operator_contract["objective_family"],
        "target_kind": operator_contract["future_wing_readback_contract"][
            "target_kind"
        ],
        "old_value_source": operator_contract["future_wing_readback_contract"][
            "old_value_source"
        ],
        "bounded_delta_policy": operator_contract["bounded_delta_policy"],
        "safety_check_source": (
            "operator_correction_contract.applicable_safety_checks"
        ),
        "all_contract_evidence_fields": required_evidence,
        "available_contract_evidence_fields": sorted(provided),
        "missing_contract_evidence_fields": list(
            evidence_gate.get("missing_required_evidence") or []
        ),
        "evidence_gate_status": evidence_gate["status"],
        "required_before_operator_acceptance": True,
        "required_before_live_promotion": True,
        "required_before_queue_import": True,
        "read_only_or_dry_run_only": True,
        "live_apply_allowed": False,
        "queue_import_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "readiness_closure_allowed": False,
        "can_authorize_live": False,
        "can_close_readiness": False,
        "can_import_to_queue": False,
        "mutates_mixer": False,
        "operator_review_required": True,
        "future_wing_readback_required": True,
        "blocks_live_promotion": True,
        "blocks_queue_import": True,
        "denial_reason": (
            live_authority_boundary.get("denial_reason")
            or safety_checks.get("denial_reason")
        ),
        "why_blocked": _unique_fields(
            safety_checks.get("why_blocked") or [],
            evidence_gate.get("blockers") or [],
            ["operator_safety_review_required"],
        ),
    }
    details: List[Dict[str, Any]] = []
    for check in safety_checks.get("required_checks") or []:
        check_name = str(check)
        matching_required = [
            field
            for field in required_evidence
            if _safety_check_matches_evidence(check_name, field)
        ]
        provided_matching = [
            field for field in matching_required if field in provided
        ]
        missing_matching = [
            field for field in matching_required if field in missing
        ]
        evidence_values = {
            field: evidence_checks[field]
            for field in matching_required
            if field in evidence_checks
        }
        if missing_matching:
            status = "missing_required_evidence"
            action_type = "collect_safety_check_evidence"
            operator_action = f"collect_safety_check_evidence:{check_name}"
            mapping_status = "direct"
        elif matching_required:
            status = "evidence_available_operator_review_required"
            action_type = "review_safety_check_evidence"
            operator_action = f"review_safety_check_evidence:{check_name}"
            mapping_status = "direct"
        else:
            status = "operator_review_required_no_direct_evidence_mapping"
            action_type = "review_safety_check_mapping"
            operator_action = f"review_safety_check_mapping:{check_name}"
            mapping_status = "no_direct_required_evidence_field_match"

        details.append(
            {
                **common,
                "check": check_name,
                "safety_check": check_name,
                "method_safety_check": f"{method_id}:{check_name}",
                "status": status,
                "evidence_mapping_status": mapping_status,
                "required_evidence_fields": matching_required,
                "provided_evidence_fields": provided_matching,
                "missing_required_evidence": missing_matching,
                "evidence_checks": evidence_values,
                "operator_action": operator_action,
                "action": operator_action,
                "action_type": action_type,
            }
        )
    return details


def _apply_top_level_authority_denial(proposal: Dict[str, Any]) -> None:
    proposal["can_apply_live"] = False
    proposal["can_import_to_queue"] = False
    proposal["live_apply_allowed"] = False
    proposal["queue_import_allowed"] = False
    proposal["read_only_or_dry_run_only"] = True
    proposal["live_authority_granted"] = False
    proposal["readiness_closure_allowed"] = False
    proposal.setdefault("queue_action", "not_queueable_until_promoted")


def build_operator_correction_payload(
    *,
    proposal: Dict[str, Any],
    method_id: str,
    target: Any = None,
    requested_change: Optional[Dict[str, Any]] = None,
    requested_changes: Optional[List[Dict[str, Any]]] = None,
    proposed_action: Optional[Dict[str, Any]] = None,
    blockers: Optional[Iterable[str]] = None,
    value_type: Optional[str] = None,
    target_kind: Optional[str] = None,
    old_value_source: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the user-facing correction explanation without granting authority."""

    _apply_top_level_authority_denial(proposal)
    first_change = requested_change or (requested_changes or [None])[0]
    requested_value_type = (
        first_change.get("value_type") if isinstance(first_change, dict) else None
    )
    inputs = _operator_contract_inputs(
        method_id=method_id,
        requested_change_value_type=requested_value_type,
        value_type=value_type,
        target_kind=target_kind,
        old_value_source=old_value_source,
    )
    method_contract = PROPOSAL_ONLY_CONTRACTS.get(method_id, {})
    requested_contract = REQUESTED_CHANGE_OPERATOR_CONTRACTS.get(
        str(requested_value_type or "").strip().lower(),
        {},
    )
    required_evidence_fields = list(
        method_contract.get("safety_evidence")
        or requested_contract.get("safety_evidence")
        or []
    )
    operator_contract = _operator_correction_contract(
        inputs["value_type"],
        inputs["target_kind"],
        inputs["old_value_source"],
        required_evidence_fields,
    )
    identity_profile = _operator_contract_identity_profile(
        proposal=proposal,
        method_id=method_id,
        target=target,
        operator_contract=operator_contract,
    )
    correction_context_profile = _operator_correction_context_profile(
        identity_profile
    )
    operator_contract = dict(operator_contract)
    operator_contract["future_wing_readback_contract"] = {
        **operator_contract["future_wing_readback_contract"],
        **identity_profile,
        **correction_context_profile,
    }
    raw = proposal.get("raw") if isinstance(proposal.get("raw"), dict) else {}
    evidence = proposal.get("evidence") if isinstance(proposal.get("evidence"), dict) else {}
    operator_visible_evidence = _operator_visible_source_evidence(evidence)
    raw_input_authority_claims = _raw_input_authority_claims(evidence)
    evidence_completeness = _evidence_completeness(
        required_evidence_fields=required_evidence_fields,
        proposal=proposal,
        requested_change=requested_change,
        requested_changes=requested_changes,
        proposed_action=proposed_action,
    )
    evidence_completeness["correction_context_profile"] = (
        correction_context_profile
    )
    safety_checks = _headroom_feedback_true_peak_checks(
        proposal=proposal,
        requested_change=requested_change,
        requested_changes=requested_changes,
        proposed_action=proposed_action,
        operator_contract=operator_contract,
    )
    safety_checks["evidence_completeness"] = evidence_completeness
    evidence_gate = _evidence_gate(evidence_completeness)
    evidence_gate = {
        **evidence_gate,
        **correction_context_profile,
    }
    safety_checks["evidence_gate"] = evidence_gate
    safety_checks["required_evidence_status"] = _required_evidence_status(
        evidence_completeness,
        evidence_gate,
    )
    safety_checks["operator_next_step"] = evidence_gate["operator_action"]
    safety_checks["why_blocked"] = _unique_fields(
        safety_checks.get("why_blocked") or [],
        evidence_gate.get("blockers") or [],
    )
    safety_checks = {
        **safety_checks,
        **correction_context_profile,
    }
    confidence_review = _confidence_review(proposal)
    confidence_review = {
        **confidence_review,
        **correction_context_profile,
    }
    safety_blockers = list(blockers or [])
    operator_proposal_review_gate = _operator_proposal_review_gate(
        evidence_gate=evidence_gate,
        confidence_review=confidence_review,
        safety_blockers=safety_blockers,
    )
    operator_proposal_review_gate = {
        **operator_proposal_review_gate,
        **identity_profile,
        **correction_context_profile,
    }
    operator_safety_review_matrix = _operator_safety_review_matrix(
        operator_contract=operator_contract,
        evidence_completeness=evidence_completeness,
        evidence_gate=evidence_gate,
    )
    operator_safety_review_matrix = {
        **operator_safety_review_matrix,
        **identity_profile,
        **correction_context_profile,
        "summary": {
            **operator_safety_review_matrix["summary"],
            **correction_context_profile,
        },
        "checks": [
            {
                **row,
                **identity_profile,
                **correction_context_profile,
            }
            for row in operator_safety_review_matrix["checks"]
        ],
    }
    live_authority_boundary = _live_authority_boundary(
        operator_contract=operator_contract,
        evidence_gate=evidence_gate,
    )
    live_authority_boundary = {
        **live_authority_boundary,
        **identity_profile,
        **correction_context_profile,
    }
    headroom_feedback_true_peak_check_details = (
        _headroom_feedback_true_peak_check_details(
            method_id=method_id,
            safety_checks=safety_checks,
            operator_contract=operator_contract,
            evidence_completeness=evidence_completeness,
            evidence_gate=evidence_gate,
            live_authority_boundary=live_authority_boundary,
        )
    )
    headroom_feedback_true_peak_check_details = [
        {
            **row,
            **correction_context_profile,
        }
        for row in headroom_feedback_true_peak_check_details
    ]
    safety_checks["headroom_feedback_true_peak_check_details"] = [
        dict(row) for row in headroom_feedback_true_peak_check_details
    ]
    confidence_review_details = _confidence_review_details(
        confidence_review=confidence_review,
        operator_proposal_review_gate=operator_proposal_review_gate,
        live_authority_boundary=live_authority_boundary,
    )
    confidence_review_details = [
        {
            **row,
            **correction_context_profile,
        }
        for row in confidence_review_details
    ]
    effective_safety_blockers = _effective_operator_safety_blockers(
        safety_blockers=safety_blockers,
        evidence_gate=evidence_gate,
        confidence_review=confidence_review,
        operator_proposal_review_gate=operator_proposal_review_gate,
        live_authority_boundary=live_authority_boundary,
    )
    operator_proposal_review_gate[
        "effective_safety_blockers"
    ] = effective_safety_blockers
    safety_blocker_details = _safety_blocker_details(
        safety_blockers=safety_blockers,
        effective_safety_blockers=effective_safety_blockers,
        evidence_gate=evidence_gate,
        confidence_review=confidence_review,
        operator_proposal_review_gate=operator_proposal_review_gate,
        live_authority_boundary=live_authority_boundary,
    )
    safety_blocker_details = [
        {
            **row,
            **correction_context_profile,
        }
        for row in safety_blocker_details
    ]
    operator_proposal_review_gate[
        "safety_blocker_details"
    ] = safety_blocker_details
    future_live_promotion_requirements = _future_live_promotion_requirements(
        operator_contract=operator_contract,
        evidence_completeness=evidence_completeness,
        evidence_gate=evidence_gate,
        live_authority_boundary=live_authority_boundary,
    )
    future_live_promotion_requirements = {
        **future_live_promotion_requirements,
        **correction_context_profile,
    }
    future_live_requirement_details = _future_live_requirement_details(
        operator_contract=operator_contract,
        future_live_promotion_requirements=future_live_promotion_requirements,
        live_authority_boundary=live_authority_boundary,
    )
    future_live_requirement_details = [
        {
            **row,
            **correction_context_profile,
        }
        for row in future_live_requirement_details
    ]
    future_live_promotion_requirements["future_live_requirement_details"] = [
        dict(row) for row in future_live_requirement_details
    ]
    rollback_readback_contract = _rollback_readback_contract(
        operator_contract=operator_contract,
        future_live_promotion_requirements=future_live_promotion_requirements,
        evidence_completeness=evidence_completeness,
        evidence_gate=evidence_gate,
    )
    rollback_readback_contract = {
        **rollback_readback_contract,
        **identity_profile,
        **correction_context_profile,
    }
    rollback_readback_details = _rollback_readback_details(
        rollback_readback_contract=rollback_readback_contract,
        future_live_promotion_requirements=future_live_promotion_requirements,
        live_authority_boundary=live_authority_boundary,
    )
    rollback_readback_details = [
        {
            **row,
            **correction_context_profile,
        }
        for row in rollback_readback_details
    ]
    operator_review_checklist = _operator_review_checklist(
        operator_contract=operator_contract,
        evidence_gate=evidence_gate,
        operator_proposal_review_gate=operator_proposal_review_gate,
    )
    operator_review_decision_options = _operator_review_decision_options(
        operator_contract=operator_contract,
        operator_proposal_review_gate=operator_proposal_review_gate,
    )
    operator_review_decision_options = [
        {
            **option,
            **identity_profile,
            **correction_context_profile,
        }
        for option in operator_review_decision_options
    ]
    operator_review_decision_option_details = (
        _operator_review_decision_option_details(
            operator_review_decision_options=operator_review_decision_options,
            operator_proposal_review_gate=operator_proposal_review_gate,
            evidence_gate=evidence_gate,
            live_authority_boundary=live_authority_boundary,
        )
    )
    operator_review_decision_option_details = [
        {
            **row,
            **identity_profile,
            **correction_context_profile,
        }
        for row in operator_review_decision_option_details
    ]
    operator_review_record_template = _operator_review_record_template(
        proposal=proposal,
        method_id=method_id,
        target=target,
        operator_contract=operator_contract,
        evidence_gate=evidence_gate,
        operator_proposal_review_gate=operator_proposal_review_gate,
        operator_review_decision_options=operator_review_decision_options,
    )
    operator_review_record_template = {
        **operator_review_record_template,
        **identity_profile,
        **correction_context_profile,
    }
    operator_review_record_field_details = _operator_review_record_field_details(
        operator_review_record_template=operator_review_record_template,
        operator_proposal_review_gate=operator_proposal_review_gate,
        evidence_gate=evidence_gate,
        live_authority_boundary=live_authority_boundary,
    )
    operator_review_record_field_details = [
        {
            **row,
            **correction_context_profile,
        }
        for row in operator_review_record_field_details
    ]
    problem_summary = _contract_specific_problem_summary(
        proposal=proposal,
        method_id=method_id,
        operator_contract=operator_contract,
    )
    measured_or_audible_problem = _measured_or_audible_problem_payload(
        problem_summary=problem_summary,
        proposal=proposal,
        method_id=method_id,
        target=target,
        operator_contract=operator_contract,
        evidence_completeness=evidence_completeness,
        evidence_gate=evidence_gate,
    )
    measured_or_audible_problem = {
        **measured_or_audible_problem,
        **correction_context_profile,
    }
    base_proposed_correction = (
        requested_change
        or ({"changes": requested_changes} if requested_changes else None)
        or proposed_action
        or _contract_specific_proposed_correction(
            method_id=method_id,
            operator_contract=operator_contract,
            evidence_completeness=evidence_completeness,
            evidence_gate=evidence_gate,
        )
    )
    bounded_delta = _bounded_delta_payload(
        requested_change=requested_change,
        requested_changes=requested_changes,
        proposed_action=proposed_action,
        operator_contract=operator_contract,
    )
    bounded_delta = {
        **bounded_delta,
        **correction_context_profile,
    }
    bounded_delta_details = _bounded_delta_details(
        bounded_delta=bounded_delta,
        operator_contract=operator_contract,
        live_authority_boundary=live_authority_boundary,
    )
    bounded_delta_details = [
        {
            **row,
            **correction_context_profile,
        }
        for row in bounded_delta_details
    ]
    proposed_correction = _proposed_correction_review_payload(
        proposed_correction=base_proposed_correction,
        source=proposal.get("source"),
        target=target if target is not None else proposal.get("target"),
        operator_contract=operator_contract,
        evidence_completeness=evidence_completeness,
        evidence_gate=evidence_gate,
        bounded_delta=bounded_delta,
        future_live_promotion_requirements=future_live_promotion_requirements,
        rollback_readback_contract=rollback_readback_contract,
    )
    proposed_correction = {
        **proposed_correction,
        **correction_context_profile,
    }
    why_safe_details = _operator_why_safe_details(
        operator_contract=operator_contract,
        bounded_delta=bounded_delta,
        rollback_readback_contract=rollback_readback_contract,
        live_authority_boundary=live_authority_boundary,
    )
    why_safe_details = [
        {
            **row,
            **correction_context_profile,
        }
        for row in why_safe_details
    ]
    why_blocked_reasons = _unique_fields(
        [
            "live_apply_denied_until_real_wing_proof",
            "live_apply_denied_until_future_wing_readback",
            "queue_import_denied_until_operator_review",
        ],
        effective_safety_blockers,
        evidence_gate.get("blockers") or [],
        confidence_review.get("blockers") or [],
        operator_proposal_review_gate.get("live_authority_blockers") or [],
        safety_blockers,
    )
    why_blocked_details = _operator_why_blocked_details(
        why_blocked=why_blocked_reasons,
        safety_blocker_details=safety_blocker_details,
        evidence_gate=evidence_gate,
        confidence_review=confidence_review,
        operator_proposal_review_gate=operator_proposal_review_gate,
        live_authority_boundary=live_authority_boundary,
    )
    why_blocked_details = [
        {
            **row,
            **correction_context_profile,
        }
        for row in why_blocked_details
    ]
    operator_readable_explanation = _contract_specific_operator_explanation(
        proposal=proposal,
        method_id=method_id,
        operator_contract=operator_contract,
        evidence_gate=evidence_gate,
    )
    source_evidence_summary = _source_evidence_summary(
        proposal=proposal,
        method_id=method_id,
        target=target,
        operator_contract=operator_contract,
        evidence_completeness=evidence_completeness,
        evidence_gate=evidence_gate,
    )
    source_evidence_details = _source_evidence_details(
        raw=raw,
        evidence=operator_visible_evidence,
        operator_contract=operator_contract,
        evidence_completeness=evidence_completeness,
        evidence_gate=evidence_gate,
        live_authority_boundary=live_authority_boundary,
    )
    source_evidence_details = [
        {
            **row,
            **identity_profile,
            **correction_context_profile,
        }
        for row in source_evidence_details
    ]
    source_evidence_summary = {
        **source_evidence_summary,
        **correction_context_profile,
        "source_evidence_details": [
            dict(row) for row in source_evidence_details
        ],
    }
    operator_explanation_packet = _operator_explanation_packet(
        problem_summary=problem_summary,
        proposed_correction=proposed_correction,
        source_evidence_summary=source_evidence_summary,
        source_evidence_details=source_evidence_details,
        bounded_delta=bounded_delta,
        bounded_delta_details=bounded_delta_details,
        confidence_review=confidence_review,
        confidence_review_details=confidence_review_details,
        operator_proposal_review_gate=operator_proposal_review_gate,
        evidence_gate=evidence_gate,
        safety_checks=safety_checks,
        headroom_feedback_true_peak_check_details=(
            headroom_feedback_true_peak_check_details
        ),
        operator_safety_review_matrix=operator_safety_review_matrix,
        safety_blockers=safety_blockers,
        effective_safety_blockers=effective_safety_blockers,
        safety_blocker_details=safety_blocker_details,
        why_safe_details=why_safe_details,
        why_blocked_details=why_blocked_details,
        operator_review_checklist=operator_review_checklist,
        operator_review_decision_options=operator_review_decision_options,
        operator_review_decision_option_details=operator_review_decision_option_details,
        operator_review_record_template=operator_review_record_template,
        operator_review_record_field_details=operator_review_record_field_details,
        operator_contract=operator_contract,
        future_live_promotion_requirements=future_live_promotion_requirements,
        future_live_requirement_details=future_live_requirement_details,
        rollback_readback_contract=rollback_readback_contract,
        rollback_readback_details=rollback_readback_details,
        live_authority_boundary=live_authority_boundary,
    )
    operator_next_action_details = operator_explanation_packet[
        "operator_next_action_details"
    ]
    operator_next_action_details = [
        {
            **row,
            **correction_context_profile,
        }
        for row in operator_next_action_details
    ]
    operator_review_checklist_details = operator_explanation_packet[
        "operator_review_checklist_details"
    ]
    operator_review_checklist_details = [
        {
            **row,
            **correction_context_profile,
        }
        for row in operator_review_checklist_details
    ]
    operator_explanation_packet = {
        **operator_explanation_packet,
        **correction_context_profile,
        "source_evidence_summary": {
            **operator_explanation_packet["source_evidence_summary"],
            **correction_context_profile,
        },
        "confidence_review_summary": {
            **operator_explanation_packet["confidence_review_summary"],
            **correction_context_profile,
        },
        "safety_review_summary": {
            **operator_explanation_packet["safety_review_summary"],
            **correction_context_profile,
        },
        "future_live_requirements": {
            **operator_explanation_packet["future_live_requirements"],
            **correction_context_profile,
        },
        "live_authority_boundary_summary": {
            **operator_explanation_packet["live_authority_boundary_summary"],
            **correction_context_profile,
        },
        "future_wing_readback_summary": {
            **operator_explanation_packet["future_wing_readback_summary"],
            **correction_context_profile,
        },
        "future_live_proof_contract_summary": {
            **operator_explanation_packet["future_live_proof_contract_summary"],
            **correction_context_profile,
        },
        "operator_review_record_summary": {
            **operator_explanation_packet["operator_review_record_summary"],
            **correction_context_profile,
        },
        "operator_next_action_details": operator_next_action_details,
        "operator_review_checklist_details": operator_review_checklist_details,
    }
    operator_evidence_packet = _operator_evidence_packet(
        proposal=proposal,
        method_id=method_id,
        target=target,
        problem_summary=problem_summary,
        proposed_correction=proposed_correction,
        bounded_delta=bounded_delta,
        bounded_delta_details=bounded_delta_details,
        safety_blockers=safety_blockers,
        effective_safety_blockers=effective_safety_blockers,
        safety_blocker_details=safety_blocker_details,
        why_safe_details=why_safe_details,
        why_blocked_details=why_blocked_details,
        confidence_review=confidence_review,
        confidence_review_details=confidence_review_details,
        operator_proposal_review_gate=operator_proposal_review_gate,
        safety_checks=safety_checks,
        headroom_feedback_true_peak_check_details=(
            headroom_feedback_true_peak_check_details
        ),
        operator_safety_review_matrix=operator_safety_review_matrix,
        future_live_promotion_requirements=future_live_promotion_requirements,
        future_live_requirement_details=future_live_requirement_details,
        operator_next_action_details=operator_next_action_details,
        operator_review_checklist_details=operator_review_checklist_details,
        operator_review_decision_option_details=operator_review_decision_option_details,
        operator_review_record_field_details=operator_review_record_field_details,
        rollback_readback_contract=rollback_readback_contract,
        rollback_readback_details=rollback_readback_details,
        operator_review_record_template=operator_review_record_template,
        operator_readable_explanation=operator_readable_explanation,
        operator_explanation_packet=operator_explanation_packet,
        source_evidence_details=source_evidence_details,
        operator_contract=operator_contract,
        evidence_completeness=evidence_completeness,
        evidence_gate=evidence_gate,
        live_authority_boundary=live_authority_boundary,
    )
    operator_evidence_packet = {
        **operator_evidence_packet,
        **correction_context_profile,
    }
    operator_review_summary = _operator_review_summary(
        source_evidence_summary=source_evidence_summary,
        problem_summary=problem_summary,
        proposed_correction=proposed_correction,
        safety_blockers=safety_blockers,
        effective_safety_blockers=effective_safety_blockers,
        safety_blocker_details=safety_blocker_details,
        why_safe_details=why_safe_details,
        why_blocked_details=why_blocked_details,
        evidence_gate=evidence_gate,
        confidence_review=confidence_review,
        confidence_review_details=confidence_review_details,
        source_evidence_details=source_evidence_details,
        bounded_delta_details=bounded_delta_details,
        headroom_feedback_true_peak_check_details=(
            headroom_feedback_true_peak_check_details
        ),
        operator_proposal_review_gate=operator_proposal_review_gate,
        operator_contract=operator_contract,
        rollback_readback_contract=rollback_readback_contract,
        rollback_readback_details=rollback_readback_details,
        future_live_requirement_details=future_live_requirement_details,
        operator_next_action_details=operator_next_action_details,
        operator_review_checklist_details=operator_review_checklist_details,
        operator_review_decision_option_details=operator_review_decision_option_details,
        operator_review_record_field_details=operator_review_record_field_details,
    )
    operator_review_summary = {
        **operator_review_summary,
        **correction_context_profile,
        "future_live_requirements": {
            **operator_review_summary["future_live_requirements"],
            **correction_context_profile,
        },
    }
    correction = {
        "method_id": method_id,
        "source": proposal.get("source"),
        "target": target if target is not None else proposal.get("target"),
        "value_type": inputs["value_type"],
        "target_kind": inputs["target_kind"],
        "old_value_source": inputs["old_value_source"],
        "denial_reason": future_live_promotion_requirements["blocked_until"],
        "operator_next_step": evidence_gate["operator_action"],
        "source_evidence": {
            **correction_context_profile,
            "source": proposal.get("source"),
            "method_id": method_id,
            "target": target if target is not None else proposal.get("target"),
            "contract_profile": {
                "value_type": inputs["value_type"],
                "target_kind": inputs["target_kind"],
                "old_value_source": inputs["old_value_source"],
            },
            "required_evidence_fields": required_evidence_fields,
            "evidence_completeness": evidence_completeness,
            "evidence_gate": evidence_gate,
            "required_evidence_status": _required_evidence_status(
                evidence_completeness,
                evidence_gate,
            ),
            "raw": raw,
            "evidence": operator_visible_evidence,
            "raw_input_authority_claims": raw_input_authority_claims,
            "raw_input_evidence_boundary": _raw_input_evidence_boundary(
                surface="source_evidence",
                evidence_fields=_unique_fields(
                    raw.keys(),
                    evidence.keys(),
                ),
            ),
            "source_evidence_details": source_evidence_details,
            "read_only_or_dry_run_only": True,
            "live_apply_allowed": False,
            "queue_import_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "readiness_closure_allowed": False,
            "can_authorize_live": False,
            "can_close_readiness": False,
            "mutates_mixer": False,
            "operator_review_required": True,
            "blocks_live_promotion": True,
            "blocks_queue_import": True,
            "future_wing_readback_required": True,
            "denial_reason": future_live_promotion_requirements["blocked_until"],
            "operator_next_step": evidence_gate["operator_action"],
            "future_live_promotion_requires": list(
                future_live_promotion_requirements["required_proofs"]
            ),
            "why_blocked": _unique_fields(
                [
                    "live_apply_denied_until_real_wing_proof",
                    "queue_import_denied_until_operator_review",
                    "future_wing_readback_required",
                ],
                evidence_gate.get("blockers") or [],
            ),
        },
        "source_evidence_details": source_evidence_details,
        "measured_or_audible_problem": {
            **measured_or_audible_problem,
        },
        "proposed_correction": proposed_correction,
        "operator_evidence_packet": operator_evidence_packet,
        "operator_review_summary": operator_review_summary,
        "operator_review_checklist": operator_review_checklist,
        "operator_review_checklist_details": operator_review_checklist_details,
        "operator_review_decision_options": operator_review_decision_options,
        "operator_review_decision_option_details": operator_review_decision_option_details,
        "operator_review_record_template": operator_review_record_template,
        "operator_review_record_field_details": operator_review_record_field_details,
        "confidence": proposal.get("confidence"),
        "confidence_review": confidence_review,
        "confidence_review_details": confidence_review_details,
        "operator_proposal_review_gate": operator_proposal_review_gate,
        "bounded_delta": bounded_delta,
        "bounded_delta_details": bounded_delta_details,
        "safety_blockers": safety_blockers,
        "effective_safety_blockers": effective_safety_blockers,
        "safety_blocker_details": safety_blocker_details,
        "why_safe_details": why_safe_details,
        "why_blocked_details": why_blocked_details,
        "evidence_gate": evidence_gate,
        "live_authority_boundary": live_authority_boundary,
        "headroom_feedback_true_peak_checks": safety_checks,
        "headroom_feedback_true_peak_check_details": (
            headroom_feedback_true_peak_check_details
        ),
        "operator_safety_review_matrix": operator_safety_review_matrix,
        "operator_readable_explanation": operator_readable_explanation,
        "operator_explanation_packet": operator_explanation_packet,
        "operator_next_action_details": operator_next_action_details,
        "rollback_expectation": operator_contract["rollback_expectation"],
        "rollback_readback_contract": rollback_readback_contract,
        "rollback_readback_details": rollback_readback_details,
        "future_live_promotion_requirements": future_live_promotion_requirements,
        "future_live_requirement_details": future_live_requirement_details,
        "future_wing_readback_contract": operator_contract["future_wing_readback_contract"],
        "objective_family": operator_contract["objective_family"],
        "bounded_delta_unit": operator_contract["bounded_delta_unit"],
        "bounded_delta_policy": operator_contract["bounded_delta_policy"],
        "read_only_or_dry_run_only": True,
        "live_apply_allowed": False,
        "queue_import_allowed": False,
        "can_apply_live": False,
        "can_import_to_queue": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "readiness_closure_allowed": False,
        "can_authorize_live": False,
        "can_close_readiness": False,
        "mutates_mixer": False,
        "operator_review_required": True,
        "future_wing_readback_required": True,
        "blocks_live_promotion": True,
        "blocks_queue_import": True,
    }
    return correction


def _sync_operator_correction_safety_blockers(output: Dict[str, Any]) -> None:
    correction = output.get("operator_correction")
    if not isinstance(correction, dict):
        return

    blockers = list(output.get("blockers") or [])
    correction["safety_blockers"] = blockers
    effective_blockers = _unique_fields(
        correction.get("effective_safety_blockers") or [],
        blockers,
    )
    correction["effective_safety_blockers"] = effective_blockers
    evidence_gate = correction.get("evidence_gate")
    confidence_review = correction.get("confidence_review")
    live_authority_boundary = correction.get("live_authority_boundary")

    review_gate = correction.get("operator_proposal_review_gate")
    if isinstance(review_gate, dict):
        review_gate["safety_blockers"] = blockers
        review_gate["effective_safety_blockers"] = effective_blockers
        review_gate["live_authority_blockers"] = _unique_fields(
            review_gate.get("live_authority_blockers") or [],
            blockers,
        )
    safety_blocker_details = _safety_blocker_details(
        safety_blockers=blockers,
        effective_safety_blockers=effective_blockers,
        evidence_gate=evidence_gate if isinstance(evidence_gate, dict) else {},
        confidence_review=confidence_review
        if isinstance(confidence_review, dict)
        else {},
        operator_proposal_review_gate=review_gate
        if isinstance(review_gate, dict)
        else {},
        live_authority_boundary=live_authority_boundary
        if isinstance(live_authority_boundary, dict)
        else {},
    )
    correction_context_profile = _operator_correction_context_profile(correction)
    safety_blocker_details = [
        {
            **row,
            **correction_context_profile,
        }
        for row in safety_blocker_details
    ]
    correction["safety_blocker_details"] = safety_blocker_details
    if isinstance(review_gate, dict):
        review_gate["safety_blocker_details"] = safety_blocker_details

    explanation_packet = correction.get("operator_explanation_packet")
    if isinstance(explanation_packet, dict):
        explanation_packet["safety_blockers"] = blockers
        explanation_packet["effective_safety_blockers"] = effective_blockers
        explanation_packet["safety_blocker_details"] = safety_blocker_details
        explanation_packet["why_blocked"] = _unique_fields(
            explanation_packet.get("why_blocked") or [],
            effective_blockers,
        )
        explanation_packet["operator_next_actions"] = _unique_fields(
            explanation_packet.get("operator_next_actions") or [],
            [f"review_safety_blocker:{blocker}" for blocker in blockers],
        )
        future_requirements = explanation_packet.get("future_live_requirements")
        if isinstance(future_requirements, dict):
            future_requirements["live_authority_blockers"] = _unique_fields(
                future_requirements.get("live_authority_blockers") or [],
                blockers,
            )
        future_proof_summary = explanation_packet.get(
            "future_live_proof_contract_summary"
        )
        if isinstance(future_proof_summary, dict):
            future_proof_summary["safety_blockers"] = blockers
            future_proof_summary["effective_safety_blockers"] = effective_blockers
            future_proof_summary["safety_blocker_details"] = safety_blocker_details
            future_proof_summary["live_authority_blockers"] = _unique_fields(
                future_proof_summary.get("live_authority_blockers") or [],
                blockers,
            )

    evidence_packet = correction.get("operator_evidence_packet")
    if isinstance(evidence_packet, dict):
        evidence_packet["safety_blockers"] = blockers
        evidence_packet["effective_safety_blockers"] = effective_blockers
        evidence_packet["safety_blocker_details"] = safety_blocker_details
        if isinstance(review_gate, dict):
            evidence_packet["operator_proposal_review_gate"] = review_gate
        if isinstance(explanation_packet, dict):
            evidence_packet["operator_explanation_packet"] = explanation_packet

    review_summary = correction.get("operator_review_summary")
    if isinstance(review_summary, dict):
        review_summary["effective_safety_blockers"] = effective_blockers
        review_summary["safety_blocker_details"] = safety_blocker_details
        review_summary["why_blocked"] = _unique_fields(
            review_summary.get("why_blocked") or [],
            effective_blockers,
        )

    why_blocked_details = _operator_why_blocked_details(
        why_blocked=_unique_fields(
            correction.get("why_blocked") or [],
            explanation_packet.get("why_blocked") or []
            if isinstance(explanation_packet, dict)
            else [],
            review_summary.get("why_blocked") or []
            if isinstance(review_summary, dict)
            else [],
            effective_blockers,
        ),
        safety_blocker_details=safety_blocker_details,
        evidence_gate=evidence_gate if isinstance(evidence_gate, dict) else {},
        confidence_review=confidence_review
        if isinstance(confidence_review, dict)
        else {},
        operator_proposal_review_gate=review_gate
        if isinstance(review_gate, dict)
        else {},
        live_authority_boundary=live_authority_boundary
        if isinstance(live_authority_boundary, dict)
        else {},
    )
    why_blocked_details = [
        {
            **row,
            **correction_context_profile,
        }
        for row in why_blocked_details
    ]
    correction["why_blocked_details"] = why_blocked_details
    if isinstance(explanation_packet, dict):
        explanation_packet["why_blocked_details"] = why_blocked_details
        explanation_packet["operator_next_actions"] = _unique_fields(
            explanation_packet.get("operator_next_actions") or [],
            [
                action
                for detail in why_blocked_details
                for action in detail.get("operator_actions") or []
            ],
        )
    if isinstance(evidence_packet, dict):
        evidence_packet["why_blocked_details"] = why_blocked_details
        if isinstance(explanation_packet, dict):
            evidence_packet["operator_explanation_packet"] = explanation_packet
    if isinstance(review_summary, dict):
        review_summary["why_blocked_details"] = why_blocked_details

    _refresh_operator_next_action_details(correction)


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
    requested_readback_required = (
        requested_change.get("readback_required") if requested_change else None
    )
    return {
        "source": proposal.get("source"),
        "severity": proposal.get("severity"),
        "target": proposal.get("target"),
        "current_value": (requested_change or {}).get("current_value"),
        "proposed_value": (requested_change or {}).get("value"),
        "rollback_value": (requested_change or {}).get("rollback_value")
        or (requested_change or {}).get("current_value"),
        "requested_readback_required": requested_readback_required,
        "readback_required": True if requested_change else False,
        "raw": raw,
    }


def _gain_fader_preview_safety_summary(
    *,
    proposal: Dict[str, Any],
    method_id: str,
    requested_change: Optional[Dict[str, Any]],
    requested_changes: List[Dict[str, Any]],
    has_applyable_change: bool,
    blockers: List[str],
) -> Dict[str, Any]:
    change = requested_change or (requested_changes[0] if requested_changes else {})
    value_type = str(change.get("value_type") or proposal.get("kind") or "")
    requested_readback_required = change.get("readback_required")
    return {
        "summary_type": "gain_fader_preview_review",
        "status": "preview_only_not_queued",
        "method_id": method_id,
        "source": proposal.get("source"),
        "target": proposal.get("target") or proposal.get("channel") or proposal.get("channels"),
        "value_type": value_type,
        "current_value": change.get("current_value"),
        "proposed_value": change.get("value") or change.get("proposed_value"),
        "rollback_value": change.get("rollback_value") or change.get("current_value"),
        "delta_db": change.get("delta_db"),
        "requested_readback_required": requested_readback_required,
        "readback_required": True if has_applyable_change else False,
        "bounded_delta_required": True,
        "headroom_check": "required_before_apply",
        "feedback_safety": "required_before_apply",
        "true_peak_headroom": "required_before_apply",
        "ayaic_level_plane": "operator_evidence_required"
        if value_type in {"fader", "fader_balance"}
        else None,
        "rollback_required_before_promotion": True,
        "operator_review_required": True,
        "read_only_or_dry_run_only": True,
        "live_write": False,
        "queue_mutation": False,
        "live_apply_allowed": False,
        "queue_import_allowed": False,
        "can_import_to_queue": False,
        "can_apply_live": False,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "queue_action": "preview_only_not_queued",
        "mutates_mixer": False,
        "can_authorize_live": False,
        "blockers": list(blockers),
    }


def _safety_evidence_value(change: Dict[str, Any], *labels: str) -> Any:
    evidence_items = change.get("safety_evidence")
    if not isinstance(evidence_items, list):
        return None
    wanted = {_normalize_evidence_field_name(label) for label in labels}
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        label = _normalize_evidence_field_name(item.get("label"))
        if label in wanted:
            return item.get("value")
    return None


def _gain_fader_operator_review_policy(change: Dict[str, Any]) -> str:
    delta_db = _change_delta_db(change)
    if delta_db is not None and delta_db <= 0.0:
        return "gain_down_or_fader_down_review_required_before_apply"
    return "gain_up_or_fader_up_requires_operator_review_and_real_wing_proof"


def _gain_fader_preview_generic_contract_evidence(change: Dict[str, Any]) -> Dict[str, Any]:
    value_type = str(change.get("value_type") or "").strip().lower()
    old_value = _change_old_value(change)
    proposed_value = _change_proposed_value(change)
    delta_db = _change_delta_db(change)
    return {
        "requested_change": dict(change),
        "bounded_gain_or_fader_delta": {
            "value_type": value_type or None,
            "channel": change.get("channel"),
            "old_value": old_value,
            "proposed_value": proposed_value,
            "delta_db": delta_db,
            "bounded": True,
            "operator_review_required": True,
            "read_only_or_dry_run_only": True,
            "live_write": False,
        },
        "operator_review_policy": _gain_fader_operator_review_policy(change),
    }


def _gain_trim_preview_contract_evidence(
    *,
    proposal: Dict[str, Any],
    requested_change: Dict[str, Any],
) -> Dict[str, Any]:
    raw = proposal.get("raw") if isinstance(proposal.get("raw"), dict) else {}
    channel = requested_change.get("channel")
    delta_db = _change_delta_db(requested_change)
    current_value = _change_old_value(requested_change)
    proposed_value = _change_proposed_value(requested_change)
    rollback_value = requested_change.get("rollback_value", current_value)
    peak = (
        raw.get("peak_dbfs")
        if "peak_dbfs" in raw
        else raw.get("peak_db", _safety_evidence_value(requested_change, "Peak"))
    )
    peak_value = _coerce_float(peak)
    headroom_margin = raw.get("headroom_margin")
    if headroom_margin is None and peak_value is not None:
        headroom_margin = round(-1.0 - peak_value, 2)
    direction = (
        raw.get("gain_down_or_operator_review_gain_up")
        or ("gain_down" if (delta_db or 0.0) <= 0.0 else "gain_up_requires_operator_review")
    )
    problem = raw.get("overload_or_weak_signal_problem")
    if problem is None:
        problem = "overload" if (delta_db or 0.0) <= 0.0 else "weak_signal_requires_operator_review"
    true_peak_headroom = raw.get("true_peak_headroom")
    if true_peak_headroom is None:
        true_peak_headroom = {
            "status": "operator_review_required",
            "headroom_margin_db": headroom_margin,
            "required_before_live_promotion": True,
        }
    bleed_risk = raw.get("bleed_risk", "operator_review_required")
    return {
        "input_level_evidence": {
            "channel": channel,
            "peak_dbfs": peak_value,
            "source": proposal.get("source"),
            "read_only": True,
            "live_write": False,
        },
        "overload_or_weak_signal_problem": problem,
        "headroom_margin": headroom_margin,
        "true_peak_headroom": true_peak_headroom,
        "bleed_risk": bleed_risk,
        "bounded_gain_delta": {
            "channel": channel,
            "current_value_db": current_value,
            "proposed_value_db": proposed_value,
            "delta_db": delta_db,
            "direction": direction,
            "read_only_or_dry_run_only": True,
            "live_write": False,
        },
        "gain_down_or_operator_review_gain_up": direction,
        "rollback_input_trim_readback": {
            "channel": channel,
            "value_db": rollback_value,
            "source": "cached_or_operator_supplied_current_value",
            "required_before_live_promotion": True,
            "live_write": False,
        },
    }


def _fader_balance_preview_contract_evidence(
    *,
    proposal: Dict[str, Any],
    requested_change: Dict[str, Any],
) -> Dict[str, Any]:
    raw = proposal.get("raw") if isinstance(proposal.get("raw"), dict) else {}
    channel = requested_change.get("channel")
    delta_db = _change_delta_db(requested_change)
    current_value = _change_old_value(requested_change)
    proposed_value = _change_proposed_value(requested_change)
    rollback_value = requested_change.get("rollback_value", current_value)
    ayaic_value = (
        raw.get("ayaic_level_plane")
        if "ayaic_level_plane" in raw
        else _safety_evidence_value(requested_change, "Ayaic plane", "Ayaic level plane")
    )
    ayaic_level_plane = (
        ayaic_value
        if isinstance(ayaic_value, dict)
        else {
            "status": "available" if ayaic_value is not None else "operator_review_required",
            "target_plane_lufs": ayaic_value,
            "read_only_or_dry_run_only": True,
            "live_write": False,
        }
    )
    true_peak = (
        raw.get("true_peak_headroom")
        if "true_peak_headroom" in raw
        else _safety_evidence_value(requested_change, "True peak headroom")
    )
    if true_peak is None:
        true_peak = {
            "status": "operator_review_required",
            "required_before_live_promotion": True,
        }
    feedback_risk = raw.get("feedback_risk", "operator_review_required")
    fader_ceiling = raw.get("fader_ceiling", raw.get("fader_ceiling_db", 0.0))
    return {
        "current_fader_readback": {
            "channel": channel,
            "value_db": current_value,
            "source": "cached_or_operator_supplied_current_value",
            "physical_readback_required_before_live_promotion": True,
            "live_write": False,
        },
        "ayaic_level_plane": ayaic_level_plane,
        "bounded_fader_delta": {
            "channel": channel,
            "current_value_db": current_value,
            "proposed_value_db": proposed_value,
            "delta_db": delta_db,
            "fader_ceiling_db": fader_ceiling,
            "read_only_or_dry_run_only": True,
            "live_write": False,
        },
        "true_peak_headroom": true_peak,
        "feedback_risk": feedback_risk,
        "fader_ceiling": fader_ceiling,
        "rollback_fader_readback": {
            "channel": channel,
            "value_db": rollback_value,
            "source": "cached_or_operator_supplied_current_value",
            "required_before_live_promotion": True,
            "live_write": False,
        },
    }


def _gain_fader_preview_contract_evidence(
    *,
    proposal: Dict[str, Any],
    requested_change: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not requested_change:
        return {}
    value_type = str(requested_change.get("value_type") or proposal.get("kind") or "").strip().lower()
    generic_evidence = _gain_fader_preview_generic_contract_evidence(requested_change)
    if value_type == "gain":
        return {
            **_gain_trim_preview_contract_evidence(
                proposal=proposal,
                requested_change=requested_change,
            ),
            **generic_evidence,
        }
    if value_type == "fader":
        return {
            **_fader_balance_preview_contract_evidence(
                proposal=proposal,
                requested_change=requested_change,
            ),
            **generic_evidence,
        }
    return generic_evidence


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
    safety_summary = _gain_fader_preview_safety_summary(
        proposal=proposal,
        method_id=method_id,
        requested_change=requested_change,
        requested_changes=requested_changes,
        has_applyable_change=has_applyable_change,
        blockers=blockers,
    )
    evidence = {
        **_proposal_evidence(proposal, requested_change),
        **_gain_fader_preview_contract_evidence(
            proposal=proposal,
            requested_change=requested_change,
        ),
        "gain_fader_preview_safety_summary": safety_summary,
        "live_write": False,
        "queue_mutation": False,
    }

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
        "live_write": False,
        "queue_mutation": False,
        "can_import_to_queue": False,
        "queue_import_allowed": False,
        "can_apply_live": False,
        "live_apply_allowed": False,
        "read_only_or_dry_run_only": True,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "mutates_mixer": False,
        "blockers": blockers,
        "evidence": evidence,
        "source_proposal_id": proposal.get("id"),
    }
    if requested_change:
        requested_change["gain_fader_preview_safety_summary"] = safety_summary
        output["requested_change"] = requested_change
    if requested_changes:
        for item in requested_changes:
            item["gain_fader_preview_safety_summary"] = safety_summary
        output["requested_changes"] = requested_changes
    operator_payload = {
        **proposal,
        "evidence": evidence,
        "live_write": False,
        "queue_mutation": False,
        "can_apply_live": False,
        "can_import_to_queue": False,
        "queue_import_allowed": False,
        "live_apply_allowed": False,
        "read_only_or_dry_run_only": True,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "mutates_mixer": False,
    }
    output["operator_correction"] = build_operator_correction_payload(
        proposal=operator_payload,
        method_id=method_id,
        target=output["target"],
        requested_change=requested_change,
        requested_changes=requested_changes,
        blockers=blockers,
    )
    return output


_COMPACT_OPERATOR_VALUE_OMITTED = object()


def _compact_quarantined_operator_value(
    value: Any,
    *,
    depth: int = 0,
    remaining_nodes: Optional[List[int]] = None,
) -> Any:
    """Copy a small native-JSON request without coercing hostile values.

    Quarantined write responses are explanatory no-send receipts.  They must
    not embed the multi-megabyte proposal review graph, but retaining a small
    operator-authored request is useful for diagnosing which direct write was
    refused.  Unsupported, non-finite, oversized, or deeply nested values are
    omitted instead of making the entire safety receipt unserialisable.
    """

    if remaining_nodes is None:
        remaining_nodes = [256]
    if remaining_nodes[0] <= 0 or depth > 6:
        return _COMPACT_OPERATOR_VALUE_OMITTED
    remaining_nodes[0] -= 1
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        return (
            value
            if value.bit_length() <= 256
            else _COMPACT_OPERATOR_VALUE_OMITTED
        )
    if type(value) is float:
        return value if math.isfinite(value) else _COMPACT_OPERATOR_VALUE_OMITTED
    if type(value) is str:
        try:
            encoded = value.encode("utf-8")
        except UnicodeEncodeError:
            return _COMPACT_OPERATOR_VALUE_OMITTED
        return value if len(encoded) <= 2_048 else _COMPACT_OPERATOR_VALUE_OMITTED
    if type(value) is dict:
        if len(value) > 64:
            return _COMPACT_OPERATOR_VALUE_OMITTED
        output: Dict[str, Any] = {}
        for key, child in value.items():
            if type(key) is not str:
                continue
            try:
                key_bytes = key.encode("utf-8")
            except UnicodeEncodeError:
                continue
            if len(key_bytes) > 128:
                continue
            compact = _compact_quarantined_operator_value(
                child,
                depth=depth + 1,
                remaining_nodes=remaining_nodes,
            )
            if compact is not _COMPACT_OPERATOR_VALUE_OMITTED:
                output[key] = compact
        return output
    if type(value) in {list, tuple}:
        if len(value) > 64:
            return _COMPACT_OPERATOR_VALUE_OMITTED
        output = []
        for child in value:
            compact = _compact_quarantined_operator_value(
                child,
                depth=depth + 1,
                remaining_nodes=remaining_nodes,
            )
            if compact is not _COMPACT_OPERATOR_VALUE_OMITTED:
                output.append(compact)
        return output
    return _COMPACT_OPERATOR_VALUE_OMITTED


def compact_quarantined_operator_mapping(value: Any) -> Dict[str, Any]:
    """Return the bounded native-JSON subset of one quarantine mapping."""

    compact = _compact_quarantined_operator_value(value)
    return compact if type(compact) is dict else {}


def build_quarantined_operator_correction_payload(
    *,
    method_id: str,
    target: Any,
    proposed_action: Dict[str, Any],
    blockers: Iterable[str],
) -> Dict[str, Any]:
    """Build a bounded, explicitly non-authoritative quarantine explanation."""

    inputs = _operator_contract_inputs(
        method_id=method_id,
        requested_change_value_type=None,
        value_type=None,
        target_kind=None,
        old_value_source=None,
    )
    profile = OPERATOR_CORRECTION_CONTRACTS_BY_VALUE_TYPE.get(
        inputs["value_type"],
        DEFAULT_OPERATOR_CORRECTION_CONTRACT,
    )
    compact_action = compact_quarantined_operator_mapping(proposed_action)
    compact_blockers: List[str] = []
    for value in blockers:
        compact = _compact_quarantined_operator_value(value)
        if type(compact) is str and compact:
            compact_blockers.append(compact)
        if len(compact_blockers) >= 16:
            break
    compact_target = _compact_quarantined_operator_value(target)
    if compact_target is _COMPACT_OPERATOR_VALUE_OMITTED:
        compact_target = None
    future_readback = {
        "required_before_live_promotion": True,
        "old_value_source": inputs["old_value_source"],
        "target_kind": inputs["target_kind"],
        "readback_targets": list(profile["future_wing_readback_targets"]),
        "blocked_live_actions": list(profile["blocked_live_actions"]),
        "read_only_or_dry_run_only": True,
        "live_apply_allowed": False,
        "queue_import_allowed": False,
        "live_authority_granted": False,
        "operator_review_required": True,
        "hardware_proof_required": True,
        "blocks_live_promotion": True,
        "blocks_queue_import": True,
    }
    return {
        "contract_schema": "automixer.quarantined-operator-correction.v1",
        "source_evidence": {
            "method_id": method_id,
            "source": "wing_deployment_boundary",
            "target": compact_target,
            "no_console_send_proven": True,
        },
        "proposed_correction": compact_action,
        "safety_blockers": compact_blockers,
        "future_wing_readback_contract": future_readback,
        "denial_reason": (
            "real_wing_hardware_readback_rollback_and_operator_review_required"
        ),
        "operator_next_step": (
            "capture_real_wing_readback_and_rollback_proof_before_live_promotion"
        ),
        "read_only_or_dry_run_only": True,
        "live_apply_allowed": False,
        "queue_import_allowed": False,
        "can_apply_live": False,
        "can_import_to_queue": False,
        "live_authority_granted": False,
        "mutates_mixer": False,
        "operator_review_required": True,
        "future_wing_readback_required": True,
        "blocks_live_promotion": True,
        "blocks_queue_import": True,
    }


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
        _sync_operator_correction_safety_blockers(output)
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
        operator_visible_evidence = _operator_visible_source_evidence(evidence)
        operator_visible_action = _operator_visible_source_evidence(proposed_action)
        raw_input_authority_claims = [
            *_raw_input_authority_claims(evidence, "evidence"),
            *_raw_input_authority_claims(proposed_action, "proposed_action"),
        ]
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
                "proposed_action": operator_visible_action,
                "evidence": {
                    **operator_visible_evidence,
                    "live_executor": False,
                    "requested_change_allowed": False,
                    "queue_mutation": False,
                    "raw_input_authority_claims": raw_input_authority_claims,
                },
                "raw_input_authority_claims": raw_input_authority_claims,
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
        operator_visible_evidence = _operator_visible_source_evidence(evidence)
        operator_visible_action = _operator_visible_source_evidence(proposed_action)
        raw_input_authority_claims = [
            *_raw_input_authority_claims(evidence, "evidence"),
            *_raw_input_authority_claims(proposed_action, "proposed_action"),
        ]
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
                "proposed_action": operator_visible_action,
                "evidence": {
                    **operator_visible_evidence,
                    "live_executor": False,
                    "requested_change_allowed": False,
                    "queue_mutation": False,
                    "model_mutation": False,
                    "memory_mutation": False,
                    "downstream_priority_mutation": False,
                    "raw_input_authority_claims": raw_input_authority_claims,
                },
                "raw_input_authority_claims": raw_input_authority_claims,
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
                "can_apply_live": False,
                "queue_import_allowed": False,
                "live_apply_allowed": False,
                "read_only_or_dry_run_only": True,
                "live_authority_granted": False,
                "full_autonomous_ready": False,
                "readiness_closure_allowed": False,
                "can_authorize_live": False,
                "can_close_readiness": False,
                "mutates_mixer": False,
                "operator_review_required": True,
                "blocks_live_promotion": True,
                "blocks_queue_import": True,
                "future_wing_readback_required": True,
                "denial_reason": reason,
                "blockers": [reason],
                "evidence": {
                    "blocked_reason": reason,
                    "promotion_next": method.promotion_next,
                    "live_executor": False,
                    "requested_change_allowed": False,
                    "queue_mutation": False,
                    "live_write": False,
                    "read_only_or_dry_run_only": True,
                    "live_authority_granted": False,
                    "operator_review_required": True,
                    "future_wing_readback_required": True,
                    "blocks_live_promotion": True,
                    "blocks_queue_import": True,
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
        target_kind = str(contract["target_kind"])
        old_value_source = str(contract["old_value_source"])
        operator_contract = _operator_correction_contract(
            value_type,
            target_kind,
            old_value_source,
            contract["safety_evidence"],
        )
        operator_required_field_flags = {
            f"{field}_required": bool(
                operator_contract.get(f"{field}_required", True)
            )
            for field in OPERATOR_CORRECTION_REQUIRED_FIELDS
        }
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
                "objective_family": operator_contract["objective_family"],
                "value_type": value_type,
                "target_kind": target_kind,
                "old_value_source": old_value_source,
                "bounded_delta_unit": operator_contract["bounded_delta_unit"],
                "bounded_delta_policy": operator_contract["bounded_delta_policy"],
                "title": f"{method.label}: proposal contract only",
                "reason": method.summary,
                "risk": "blocked" if method.status in BLOCKED_METHOD_STATUSES else "medium",
                "target": target_kind,
                "queue_action": "not_queueable_until_promoted",
                "live_write": False,
                "queue_mutation": False,
                "can_import_to_queue": False,
                "can_apply_live": False,
                "queue_import_allowed": False,
                "live_apply_allowed": False,
                "read_only_or_dry_run_only": True,
                "live_authority_granted": False,
                "readiness_closure_allowed": False,
                "mutates_mixer": False,
                "operator_review_required": True,
                "blocks_live_promotion": True,
                "blocks_queue_import": True,
                "future_wing_readback_required": True,
                "blockers": blockers,
                "operator_correction_required_fields": list(
                    OPERATOR_CORRECTION_REQUIRED_FIELDS
                ),
                "operator_correction_required_field_flags": (
                    operator_required_field_flags
                ),
                "source_evidence_required": True,
                "source_evidence_details_required": True,
                "measured_or_audible_problem_required": True,
                "proposed_correction_required": True,
                "confidence_required": True,
                "confidence_review_required": True,
                "confidence_review_details_required": True,
                "operator_proposal_review_gate_required": True,
                "bounded_delta_required": True,
                "bounded_delta_details_required": True,
                "safety_blockers_required": True,
                "effective_safety_blockers_required": True,
                "safety_blocker_details_required": True,
                "why_safe_details_required": True,
                "why_blocked_details_required": True,
                "headroom_feedback_true_peak_checks_required": True,
                "headroom_feedback_true_peak_check_details_required": True,
                "operator_safety_review_matrix_required": True,
                "operator_readable_explanation_required": True,
                "operator_explanation_packet_required": True,
                "operator_evidence_packet_required": True,
                "rollback_expectation_required": True,
                "rollback_readback_contract_required": True,
                "rollback_readback_details_required": True,
                "future_live_promotion_requirements_required": True,
                "future_live_requirement_details_required": True,
                "future_wing_readback_contract_required": True,
                "live_authority_boundary_required": True,
                "operator_correction_contract": operator_contract,
                "future_wing_readback_contract": operator_contract[
                    "future_wing_readback_contract"
                ],
                "live_authority_boundary": operator_contract[
                    "live_authority_boundary"
                ],
                "future_live_promotion_requirements": operator_contract[
                    "future_live_promotion_requirements"
                ],
                "operator_evidence_packet_contract": operator_contract[
                    "operator_evidence_packet"
                ],
                "operator_review_record_template_contract": operator_contract[
                    "operator_review_record_template"
                ],
                "proposal_contract": {
                    "value_type": value_type,
                    "target_kind": target_kind,
                    "apply_state": "proposal_only",
                    "required_fields": _unique_fields(
                        BASE_PROPOSAL_REQUIRED_FIELDS,
                        OPERATOR_CORRECTION_REQUIRED_FIELDS,
                    ),
                    "operator_correction_required_fields": list(
                        OPERATOR_CORRECTION_REQUIRED_FIELDS
                    ),
                    "old_value_source": old_value_source,
                    "safety_evidence_required": list(contract["safety_evidence"]),
                    "required_approval": "operator_supervised",
                    "readback_required": True,
                    "rollback_required": True,
                    "read_only_or_dry_run_only": True,
                    "live_apply_allowed": False,
                    "queue_import_allowed": False,
                    "live_authority_granted": False,
                    "readiness_closure_allowed": False,
                    "allowed_live_write_kinds": list(LIVE_ALLOWED_WRITE_KINDS),
                    "promotion_next": promotion_next,
                    "operator_correction_contract": operator_contract,
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


def _operator_review_record_field_detail_summary(
    required_fields: Iterable[Any],
) -> List[str]:
    details: List[str] = []
    for field in required_fields:
        field_name = str(field)
        source, field_type = OPERATOR_REVIEW_RECORD_FIELD_PROFILES.get(
            field_name,
            ("operator_review_record_template", "text"),
        )
        details.append(
            (
                f"{field_name}:source={source};type={field_type};"
                f"action=fill_operator_review_record_field:{field_name}"
            )
        )
    return details


def _operator_review_checklist_detail_summary(
    checklist_items: Iterable[Any],
) -> List[str]:
    details: List[str] = []
    for item in checklist_items:
        item_name = str(item)
        if item_name.startswith("missing_required_evidence:"):
            source = "required_evidence_status"
            action_type = "collect_missing_required_evidence"
            acknowledgement_fields = "required_confirmations_acknowledgement"
        elif item_name.startswith("confidence_blocker:"):
            source = "confidence_review"
            action_type = "resolve_confidence_blocker"
            acknowledgement_fields = "confidence_acknowledgement"
        else:
            source, action_type, acknowledgement_fields = (
                OPERATOR_REVIEW_CHECKLIST_DETAIL_PROFILES.get(
                    item_name,
                    (
                        "operator_review_checklist",
                        "operator_review_check",
                        "required_confirmations_acknowledgement",
                    ),
                )
            )
        details.append(
            (
                f"{item_name}:source={source};action_type={action_type};"
                f"acknowledgement_fields={acknowledgement_fields};"
                f"action=review_operator_checklist_item:{item_name}"
            )
        )
    return details


def _required_confirmation_detail_summary(
    confirmation_items: Iterable[Any],
) -> List[str]:
    details: List[str] = []
    for item in confirmation_items:
        item_name = str(item)
        source, action_type, acknowledgement_fields = (
            REQUIRED_CONFIRMATION_DETAIL_PROFILES.get(
                item_name,
                (
                    "what_operator_must_confirm",
                    "confirm_required_operator_condition",
                    "required_confirmations_acknowledgement",
                ),
            )
        )
        details.append(
            (
                f"{item_name}:source={source};action_type={action_type};"
                f"acknowledgement_fields={acknowledgement_fields};"
                f"action=confirm_operator_required_condition:{item_name}"
            )
        )
    return details


def _operator_acceptance_gate_detail_summary(
    confirmation_items: Iterable[Any],
) -> List[str]:
    details: List[str] = []
    for item in confirmation_items:
        item_name = str(item)
        source, action_type, acknowledgement_fields = (
            REQUIRED_CONFIRMATION_DETAIL_PROFILES.get(
                item_name,
                (
                    "what_operator_must_confirm",
                    "confirm_required_operator_condition",
                    "required_confirmations_acknowledgement",
                ),
            )
        )
        details.append(
            (
                f"{item_name}:source={source};action_type={action_type};"
                f"acknowledgement_fields={acknowledgement_fields};"
                f"action=confirm_operator_required_condition:{item_name};"
                "acceptance_scope=offline_proposal_only;required=true;"
                "required_before_operator_acceptance=true;"
                "required_before_live_promotion=true;"
                "required_before_queue_import=true;"
                "live_apply_allowed=false;queue_import_allowed=false;"
                "readiness_closure_allowed=false;"
                "requires_real_wing_proof_before_live=true"
            )
        )
    return details


def _operator_review_decision_option_detail_summary(
    decision_options: Iterable[Any],
) -> List[str]:
    details: List[str] = []
    for option in decision_options:
        if isinstance(option, dict):
            decision = str(option.get("decision") or "")
            effect = str(option.get("effect") or "")
        else:
            decision = str(option)
            effect = ""
        if not decision:
            continue
        default_effect, action_type, operator_action, blocks_offline = (
            OPERATOR_REVIEW_DECISION_OPTION_DETAIL_PROFILES.get(
                decision,
                (
                    "operator_review_decision",
                    "operator_review_decision",
                    f"record_operator_decision:{decision}",
                    True,
                ),
            )
        )
        details.append(
            (
                f"{decision}:effect={effect or default_effect};"
                f"action_type={action_type};"
                f"operator_action={operator_action};"
                f"blocks_offline_acceptance={str(bool(blocks_offline)).lower()};"
                "live_apply_allowed=false;queue_import_allowed=false;"
                "requires_real_wing_proof_before_live=true"
            )
        )
    return sorted(details)


def _operator_next_action_detail_summary(
    *,
    operator_next_step: Any,
    operator_evidence_required_fields: Iterable[Any],
    operator_review_record_required_fields: Iterable[Any],
    future_wing_readback_targets: Iterable[Any],
    future_live_required_proofs: Iterable[Any],
    blocked_live_actions: Iterable[Any],
) -> List[str]:
    def _joined(values: Iterable[Any]) -> str:
        items = sorted({str(item) for item in values if str(item)})
        return ",".join(items) if items else "none"

    def _detail(
        action: str,
        *,
        source: str,
        action_type: str,
        status: str,
        required_before_operator_acceptance: bool,
        fields: Iterable[Any] = (),
        readback_targets: Iterable[Any] = (),
        proofs: Iterable[Any] = (),
        blocked_actions: Iterable[Any] = (),
    ) -> str:
        return (
            f"{action}:source={source};"
            f"action_type={action_type};"
            f"operator_action={action};"
            f"status={status};"
            f"fields={_joined(fields)};"
            f"readback_targets={_joined(readback_targets)};"
            f"proofs={_joined(proofs)};"
            f"blocked_live_actions={_joined(blocked_actions)};"
            "required_before_operator_acceptance="
            f"{str(bool(required_before_operator_acceptance)).lower()};"
            "required_before_live_promotion=true;"
            "required_before_queue_import=true;"
            "live_apply_allowed=false;"
            "queue_import_allowed=false;"
            "requires_real_wing_proof_before_live=true"
        )

    details = [
        _detail(
            "review_operator_evidence_packet",
            source="operator_evidence_packet",
            action_type="review_operator_evidence_packet",
            status="operator_review_required",
            required_before_operator_acceptance=True,
            fields=operator_evidence_required_fields,
        ),
        _detail(
            "record_operator_decision_before_promotion",
            source="operator_review_record_template",
            action_type="record_operator_review_decision",
            status="operator_review_required",
            required_before_operator_acceptance=True,
            fields=operator_review_record_required_fields,
        ),
        _detail(
            "collect_future_live_proofs_before_live_promotion",
            source="future_live_promotion_requirements",
            action_type="collect_future_live_proofs",
            status="pending_future_live_proof",
            required_before_operator_acceptance=False,
            proofs=future_live_required_proofs,
        ),
        _detail(
            "do_not_queue_or_apply_live",
            source="live_authority_boundary",
            action_type="hold_live_and_queue_authority",
            status="live_authority_denied_until_real_wing_proof",
            required_before_operator_acceptance=True,
            blocked_actions=blocked_live_actions,
        ),
    ]
    if operator_next_step:
        details.append(
            _detail(
                str(operator_next_step),
                source="future_wing_readback_contract",
                action_type="capture_future_wing_readback",
                status="pending_future_wing_readback",
                required_before_operator_acceptance=False,
                readback_targets=future_wing_readback_targets,
            )
        )
    return sorted(dict.fromkeys(details))


def _future_live_requirement_detail_summary(
    *,
    future_live_required_proofs: Iterable[Any],
    future_wing_readback_targets: Iterable[Any],
    blocked_live_actions: Iterable[Any],
) -> List[str]:
    def _joined(values: Iterable[Any]) -> str:
        items = sorted({str(item) for item in values if str(item)})
        return ",".join(items) if items else "none"

    blocked = list(blocked_live_actions)
    details: List[str] = []

    for proof in sorted({str(item) for item in future_live_required_proofs if str(item)}):
        action = f"collect_future_live_required_proof:{proof}"
        details.append(
            (
                f"{proof}:source=future_live_promotion_requirements;"
                "requirement_type=future_live_required_proof;"
                f"proof={proof};"
                "readback_target=none;"
                f"operator_action={action};"
                "status=pending_real_wing_proof;"
                "required_before_operator_acceptance=false;"
                "required_before_live_promotion=true;"
                "required_before_queue_import=true;"
                f"blocked_live_actions={_joined(blocked)};"
                "live_apply_allowed=false;"
                "queue_import_allowed=false;"
                "requires_real_wing_proof_before_live=true"
            )
        )

    for target in sorted({str(item) for item in future_wing_readback_targets if str(item)}):
        action = f"verify_future_wing_readback_target:{target}"
        details.append(
            (
                f"{target}:source=future_wing_readback_contract;"
                "requirement_type=future_wing_readback_target;"
                "proof=none;"
                f"readback_target={target};"
                f"operator_action={action};"
                "status=pending_future_wing_readback;"
                "required_before_operator_acceptance=false;"
                "required_before_live_promotion=true;"
                "required_before_queue_import=true;"
                f"blocked_live_actions={_joined(blocked)};"
                "live_apply_allowed=false;"
                "queue_import_allowed=false;"
                "requires_real_wing_proof_before_live=true"
            )
        )

    return sorted(details)


def _safety_blocker_detail_summary(
    *,
    required_safety_checks: Iterable[Any],
    blocked_live_actions: Iterable[Any],
    missing_required_operator_surfaces: Iterable[Any],
    denial_reason: Any,
) -> List[str]:
    reason = str(denial_reason or "real_wing_hardware_proof_required")
    details: List[str] = []

    for check in sorted({str(item) for item in required_safety_checks if str(item)}):
        details.append(
            (
                f"{check}:source=required_safety_checks;"
                "blocker_type=required_safety_check;"
                f"safety_check={check};"
                "blocked_live_action=none;"
                "missing_operator_surface=none;"
                f"operator_action=verify_required_safety_check:{check};"
                "status=operator_review_required;"
                f"denial_reason={reason};"
                "required_before_operator_acceptance=true;"
                "required_before_live_promotion=true;"
                "required_before_queue_import=true;"
                "live_apply_allowed=false;"
                "queue_import_allowed=false;"
                "requires_real_wing_proof_before_live=true"
            )
        )

    for action in sorted({str(item) for item in blocked_live_actions if str(item)}):
        details.append(
            (
                f"{action}:source=live_authority_boundary;"
                "blocker_type=blocked_live_action;"
                "safety_check=none;"
                f"blocked_live_action={action};"
                "missing_operator_surface=none;"
                f"operator_action=keep_live_action_blocked:{action};"
                "status=live_authority_denied_until_real_wing_proof;"
                f"denial_reason={reason};"
                "required_before_operator_acceptance=true;"
                "required_before_live_promotion=true;"
                "required_before_queue_import=true;"
                "live_apply_allowed=false;"
                "queue_import_allowed=false;"
                "requires_real_wing_proof_before_live=true"
            )
        )

    for surface in sorted(
        {str(item) for item in missing_required_operator_surfaces if str(item)}
    ):
        details.append(
            (
                f"{surface}:source=operator_required_surfaces;"
                "blocker_type=missing_operator_surface;"
                "safety_check=none;"
                "blocked_live_action=none;"
                f"missing_operator_surface={surface};"
                f"operator_action=complete_operator_surface:{surface};"
                "status=operator_surface_missing;"
                f"denial_reason={reason};"
                "required_before_operator_acceptance=true;"
                "required_before_live_promotion=true;"
                "required_before_queue_import=true;"
                "live_apply_allowed=false;"
                "queue_import_allowed=false;"
                "requires_real_wing_proof_before_live=true"
            )
        )

    return sorted(details)


def _headroom_feedback_true_peak_check_detail_summary(
    *,
    required_safety_checks: Iterable[Any],
    denial_reason: Any,
) -> List[str]:
    def _check_type(check: str) -> str:
        normalized = check.strip().lower().replace("-", "_")
        if "true_peak" in normalized or "truepeak" in normalized:
            return "true_peak"
        if "feedback" in normalized:
            return "feedback"
        if (
            "headroom" in normalized
            or "ceiling" in normalized
            or "level_plane" in normalized
        ):
            return "headroom"
        return "related_safety_check"

    reason = str(denial_reason or "real_wing_hardware_proof_required")
    details: List[str] = []
    for check in sorted({str(item) for item in required_safety_checks if str(item)}):
        check_type = _check_type(check)
        details.append(
            (
                f"{check}:source=headroom_feedback_true_peak_checks.required_checks;"
                f"check_type={check_type};"
                f"safety_check={check};"
                f"operator_action=review_headroom_feedback_true_peak_check:{check};"
                "status=operator_review_required;"
                f"denial_reason={reason};"
                "required_before_operator_acceptance=true;"
                "required_before_live_promotion=true;"
                "required_before_queue_import=true;"
                "live_apply_allowed=false;"
                "queue_import_allowed=false;"
                "requires_real_wing_proof_before_live=true"
            )
        )
    return sorted(details)


def _evidence_problem_correction_chain_detail_summary(
    *,
    source_evidence_required: bool,
    source_evidence_details_required: bool,
    measured_or_audible_problem_required: bool,
    proposed_correction_required: bool,
    confidence_required: bool,
    confidence_review_required: bool,
    bounded_delta_required: bool,
    bounded_delta_details_required: bool,
    denial_reason: Any,
) -> List[str]:
    reason = str(denial_reason or "real_wing_hardware_proof_required")

    def _detail(
        name: str,
        *,
        chain_step: str,
        source_fields: Iterable[str],
        operator_action: str,
        required: bool,
    ) -> str:
        fields = ",".join(str(field) for field in source_fields)
        return (
            f"{name}:source=operator_review_contract.{name};"
            f"chain_step={chain_step};"
            f"source_fields={fields};"
            f"operator_action={operator_action};"
            "review_status=required_from_offline_proposal_payload;"
            f"required={str(required).lower()};"
            f"denial_reason={reason};"
            "required_before_operator_acceptance=true;"
            "required_before_live_promotion=true;"
            "required_before_queue_import=true;"
            "live_apply_allowed=false;"
            "queue_import_allowed=false;"
            "requires_real_wing_proof_before_live=true"
        )

    details = [
        _detail(
            "source_evidence",
            chain_step="1_source_evidence",
            source_fields=["source_evidence"],
            operator_action=(
                "confirm_source_evidence_matches_current_offline_material"
            ),
            required=source_evidence_required,
        ),
        _detail(
            "source_evidence_details",
            chain_step="2_source_evidence_details",
            source_fields=["source_evidence_details"],
            operator_action="review_source_evidence_details_before_acceptance",
            required=source_evidence_details_required,
        ),
        _detail(
            "measured_or_audible_problem",
            chain_step="3_measured_or_audible_problem",
            source_fields=["measured_or_audible_problem"],
            operator_action="confirm_problem_statement_matches_source_evidence",
            required=measured_or_audible_problem_required,
        ),
        _detail(
            "proposed_correction",
            chain_step="4_proposed_correction",
            source_fields=["proposed_correction"],
            operator_action="confirm_proposed_correction_matches_the_problem",
            required=proposed_correction_required,
        ),
        _detail(
            "confidence",
            chain_step="5_confidence",
            source_fields=[
                "confidence",
                "confidence_review",
                "confidence_review_details",
            ],
            operator_action="review_confidence_and_confidence_blockers",
            required=confidence_required and confidence_review_required,
        ),
        _detail(
            "bounded_delta",
            chain_step="6_bounded_delta",
            source_fields=[
                "bounded_delta",
                "bounded_delta_details",
                "bounded_delta_unit",
                "bounded_delta_policy",
            ],
            operator_action="review_bounded_delta_and_required_evidence",
            required=bounded_delta_required and bounded_delta_details_required,
        ),
    ]
    return sorted(details)


def _goal_requirement_detail_summary(
    *,
    source_evidence_required: bool,
    source_evidence_details_required: bool,
    measured_or_audible_problem_required: bool,
    proposed_correction_required: bool,
    confidence_required: bool,
    confidence_review_required: bool,
    bounded_delta_required: bool,
    bounded_delta_details_required: bool,
    safety_blockers_required: bool,
    safety_blocker_details_required: bool,
    required_safety_checks: Iterable[Any],
    headroom_feedback_true_peak_checks_required: bool,
    headroom_feedback_true_peak_check_details_required: bool,
    operator_readable_explanation_required: bool,
    operator_explanation_packet_required: bool,
    rollback_expectation: Any,
    rollback_readback_contract_required: bool,
    rollback_readback_details_required: bool,
    future_wing_readback_targets: Iterable[Any],
    future_wing_readback_contract_summary: Iterable[Any],
    test_evidence_references: Iterable[Any],
    denial_reason: Any,
) -> List[str]:
    reason = str(denial_reason or "real_wing_hardware_proof_required")

    def _covered(value: Any) -> str:
        return "covered" if bool(value) else "missing"

    def _has_items(values: Iterable[Any]) -> bool:
        return any(str(item) for item in values)

    def _detail(
        requirement: str,
        *,
        source_fields: Iterable[str],
        operator_action: str,
        covered: bool,
    ) -> str:
        fields = ",".join(str(field) for field in source_fields)
        return (
            f"{requirement}:source=correction_method_goal_requirement;"
            f"source_fields={fields};"
            f"operator_action={operator_action};"
            f"coverage_status={_covered(covered)};"
            "required=true;"
            "acceptance_scope=offline_proposal_only;"
            "required_before_operator_acceptance=true;"
            "required_before_live_promotion=true;"
            "required_before_queue_import=true;"
            f"denial_reason={reason};"
            "live_apply_allowed=false;"
            "queue_import_allowed=false;"
            "readiness_closure_allowed=false;"
            "requires_real_wing_proof_before_live=true"
        )

    details = [
        _detail(
            "source_evidence",
            source_fields=["source_evidence", "source_evidence_details"],
            operator_action="confirm_source_evidence_before_acceptance",
            covered=source_evidence_required and source_evidence_details_required,
        ),
        _detail(
            "measured_or_audible_problem",
            source_fields=["measured_or_audible_problem"],
            operator_action="confirm_problem_statement_before_acceptance",
            covered=measured_or_audible_problem_required,
        ),
        _detail(
            "proposed_correction",
            source_fields=["proposed_correction"],
            operator_action="confirm_proposed_correction_before_acceptance",
            covered=proposed_correction_required,
        ),
        _detail(
            "confidence",
            source_fields=[
                "confidence",
                "confidence_review",
                "confidence_review_details",
            ],
            operator_action="review_confidence_before_acceptance",
            covered=confidence_required and confidence_review_required,
        ),
        _detail(
            "bounded_delta",
            source_fields=[
                "bounded_delta",
                "bounded_delta_details",
                "bounded_delta_unit",
                "bounded_delta_policy",
            ],
            operator_action="review_bounded_delta_before_acceptance",
            covered=bounded_delta_required and bounded_delta_details_required,
        ),
        _detail(
            "safety_blockers",
            source_fields=[
                "safety_blockers",
                "effective_safety_blockers",
                "safety_blocker_details",
                "required_safety_checks",
            ],
            operator_action="review_safety_blockers_before_acceptance",
            covered=(
                safety_blockers_required
                and safety_blocker_details_required
                and _has_items(required_safety_checks)
            ),
        ),
        _detail(
            "headroom_feedback_true_peak_checks",
            source_fields=[
                "headroom_feedback_true_peak_checks",
                "headroom_feedback_true_peak_check_details",
            ],
            operator_action=(
                "verify_headroom_feedback_true_peak_checks_before_acceptance"
            ),
            covered=(
                headroom_feedback_true_peak_checks_required
                and headroom_feedback_true_peak_check_details_required
            ),
        ),
        _detail(
            "operator_readable_explanation",
            source_fields=[
                "operator_readable_explanation",
                "operator_explanation_packet",
            ],
            operator_action="confirm_operator_explanation_before_acceptance",
            covered=(
                operator_readable_explanation_required
                and operator_explanation_packet_required
            ),
        ),
        _detail(
            "rollback_expectation",
            source_fields=[
                "rollback_expectation",
                "rollback_readback_contract",
                "rollback_readback_details",
            ],
            operator_action="confirm_rollback_expectation_before_acceptance",
            covered=(
                bool(rollback_expectation)
                and rollback_readback_contract_required
                and rollback_readback_details_required
            ),
        ),
        _detail(
            "future_wing_readback_contract",
            source_fields=[
                "future_wing_readback_contract",
                "future_wing_readback_contract_summary",
                "future_wing_readback_targets",
            ],
            operator_action=(
                "verify_future_wing_readback_contract_before_live_promotion"
            ),
            covered=(
                _has_items(future_wing_readback_targets)
                and _has_items(future_wing_readback_contract_summary)
            ),
        ),
        _detail(
            "test_evidence",
            source_fields=[
                "test_evidence_references",
                "operator_review_contract_tests",
                "export_verification_tests",
                "no_wing_release_gate_tests",
            ],
            operator_action="verify_offline_tests_before_acceptance",
            covered=_has_items(test_evidence_references),
        ),
    ]
    return sorted(details)


METHOD_SPECIFIC_REVIEW_PROFILES: Dict[str, Dict[str, str]] = {
    "gain_trim": {
        "problem_focus": "overload_weak_signal_headroom",
        "correction_focus": "safe_gain_down_or_trim",
        "safety_focus": "headroom_true_peak_feedback_ayaic_level_plane",
    },
    "gain_trim_and_fader_balance": {
        "problem_focus": "gain_structure_and_balance_conflict",
        "correction_focus": "bounded_gain_or_fader_balance_proposal",
        "safety_focus": "headroom_sum_bus_fader_ceiling_ayaic_level_plane",
    },
    "fader_balance": {
        "problem_focus": "channel_group_vocal_instrument_balance",
        "correction_focus": "small_bounded_fader_or_group_balance",
        "safety_focus": "fader_ceiling_sum_bus_headroom_ayaic_level_plane",
    },
    "eq": {
        "problem_focus": "hpf_mud_harshness_masking",
        "correction_focus": "corrective_eq_cut_or_hpf_proposal",
        "safety_focus": "feedback_manual_band_phase_sum_bus",
    },
    "compressor_dynamics": {
        "problem_focus": "dynamic_range_gr_threshold_ratio_attack_release",
        "correction_focus": "compressor_threshold_ratio_attack_release_proposal",
        "safety_focus": "gain_reduction_makeup_gain_true_peak_headroom",
    },
    "gate_expander": {
        "problem_focus": "noise_bleed_chatter_source_identity",
        "correction_focus": "gate_or_expander_threshold_hysteresis_proposal",
        "safety_focus": "chatter_guard_tails_attack_speech_intelligibility",
    },
    "gate_expander_deesser": {
        "problem_focus": "noise_bleed_sibilance_tails",
        "correction_focus": "gate_expander_deesser_bounded_attenuation",
        "safety_focus": "source_identity_hysteresis_speech_tails_attack_true_peak",
    },
    "fx_sends": {
        "problem_focus": "reverb_delay_send_return_intelligibility",
        "correction_focus": "bounded_fx_send_return_level_proposal",
        "safety_focus": "return_level_monitor_safety_feedback_intelligibility",
    },
    "pan_stereo": {
        "problem_focus": "stereo_placement_mono_phase_risk",
        "correction_focus": "bounded_pan_proposal",
        "safety_focus": "mono_compatibility_phase_source_identity",
    },
    "routing_groups_scenes": {
        "problem_focus": "routing_group_scene_inventory_gap",
        "correction_focus": "read_only_map_or_inventory_only",
        "safety_focus": "no_routing_scene_mute_dca_matrix_writes",
    },
    "master_output": {
        "problem_focus": "master_true_peak_pa_feed_limiter_risk",
        "correction_focus": "offline_master_output_recommendation",
        "safety_focus": "true_peak_pa_feed_limiter_dynamics_proof_reference_feed_exclusion",
    },
}
DEFAULT_METHOD_SPECIFIC_REVIEW_PROFILE = {
    "problem_focus": "supporting_context_or_calibration_gap",
    "correction_focus": "proposal_layer_context_only",
    "safety_focus": "read_only_no_live_authority",
}


def _method_specific_review_detail_summary(
    *,
    method_id: Any,
    objective_family: Any,
    priority: Any,
    priority_label: Any,
    required_evidence: Iterable[Any],
    required_safety_checks: Iterable[Any],
    bounded_delta_unit: Any,
    bounded_delta_policy: Any,
    rollback_expectation: Any,
    future_wing_readback_targets: Iterable[Any],
    blocked_live_actions: Iterable[Any],
    denial_reason: Any,
) -> List[str]:
    family = str(objective_family or "supporting_context")
    priority_value = "none" if priority is None else str(priority)
    profile = METHOD_SPECIFIC_REVIEW_PROFILES.get(
        family,
        DEFAULT_METHOD_SPECIFIC_REVIEW_PROFILE,
    )

    def _joined(values: Iterable[Any]) -> str:
        items = sorted({str(item) for item in values if str(item)})
        return ",".join(items) if items else "none"

    def _detail(name: str, focus_value: str, *, source_fields: Iterable[str]) -> str:
        fields = ",".join(str(field) for field in source_fields)
        return (
            f"{name}:source=correction_method_goal.method_specific_review;"
            f"method_id={method_id};"
            f"objective_family={family};"
            f"priority={priority_value};"
            f"priority_label={priority_label};"
            f"focus={focus_value};"
            f"source_fields={fields};"
            f"required_evidence={_joined(required_evidence)};"
            f"required_safety_checks={_joined(required_safety_checks)};"
            f"bounded_delta={bounded_delta_unit}:{bounded_delta_policy};"
            f"rollback_expectation={rollback_expectation};"
            f"future_wing_readback_targets={_joined(future_wing_readback_targets)};"
            f"blocked_live_actions={_joined(blocked_live_actions)};"
            f"operator_action=review_method_specific_{name}:{family};"
            "acceptance_scope=offline_proposal_only;"
            "required_before_operator_acceptance=true;"
            "required_before_live_promotion=true;"
            "required_before_queue_import=true;"
            f"denial_reason={denial_reason or 'real_wing_hardware_proof_required'};"
            "live_apply_allowed=false;"
            "queue_import_allowed=false;"
            "readiness_closure_allowed=false;"
            "requires_real_wing_proof_before_live=true"
        )

    return sorted(
        [
            _detail(
                "method_problem_focus",
                profile["problem_focus"],
                source_fields=[
                    "source_evidence",
                    "measured_or_audible_problem",
                    "operator_evidence_packet",
                ],
            ),
            _detail(
                "method_correction_focus",
                profile["correction_focus"],
                source_fields=[
                    "proposed_correction",
                    "bounded_delta",
                    "operator_explanation_packet",
                ],
            ),
            _detail(
                "method_safety_focus",
                profile["safety_focus"],
                source_fields=[
                    "safety_blockers",
                    "headroom_feedback_true_peak_checks",
                    "required_safety_checks",
                ],
            ),
            _detail(
                "method_future_live_boundary",
                "future_wing_readback_and_real_hardware_proof_required",
                source_fields=[
                    "future_wing_readback_contract",
                    "future_live_promotion_requirements",
                    "live_authority_boundary",
                ],
            ),
        ]
    )


def _operator_actionable_explanation_detail_summary(
    *,
    method_id: Any,
    objective_family: Any,
    priority: Any,
    priority_label: Any,
    required_evidence: Iterable[Any],
    required_safety_checks: Iterable[Any],
    bounded_delta_unit: Any,
    bounded_delta_policy: Any,
    rollback_expectation: Any,
    future_wing_readback_targets: Iterable[Any],
    blocked_live_actions: Iterable[Any],
    denial_reason: Any,
) -> List[str]:
    family = str(objective_family or "supporting_context")
    priority_value = "none" if priority is None else str(priority)
    profile = METHOD_SPECIFIC_REVIEW_PROFILES.get(
        family,
        DEFAULT_METHOD_SPECIFIC_REVIEW_PROFILE,
    )

    def _joined(values: Iterable[Any]) -> str:
        items = sorted({str(item) for item in values if str(item)})
        return ",".join(items) if items else "none"

    def _detail(
        name: str,
        explanation: str,
        *,
        source_fields: Iterable[str],
        operator_action: str,
    ) -> str:
        fields = ",".join(str(field) for field in source_fields)
        return (
            f"{name}:source=operator_actionable_explanation;"
            f"method_id={method_id};"
            f"objective_family={family};"
            f"priority={priority_value};"
            f"priority_label={priority_label};"
            f"explanation={explanation};"
            f"source_fields={fields};"
            f"required_evidence={_joined(required_evidence)};"
            f"required_safety_checks={_joined(required_safety_checks)};"
            f"bounded_delta={bounded_delta_unit}:{bounded_delta_policy};"
            f"rollback_expectation={rollback_expectation};"
            f"future_wing_readback_targets={_joined(future_wing_readback_targets)};"
            f"blocked_live_actions={_joined(blocked_live_actions)};"
            f"operator_action={operator_action};"
            "acceptance_scope=offline_proposal_only;"
            "required_before_operator_acceptance=true;"
            "required_before_live_promotion=true;"
            "required_before_queue_import=true;"
            f"denial_reason={denial_reason or 'real_wing_hardware_proof_required'};"
            "live_apply_allowed=false;"
            "queue_import_allowed=false;"
            "readiness_closure_allowed=false;"
            "requires_real_wing_proof_before_live=true"
        )

    return sorted(
        [
            _detail(
                "what_is_wrong",
                f"review_{profile['problem_focus']}_from_source_evidence",
                source_fields=[
                    "source_evidence",
                    "source_evidence_details",
                    "measured_or_audible_problem",
                ],
                operator_action="confirm_what_is_wrong_before_acceptance",
            ),
            _detail(
                "what_is_proposed",
                f"review_{profile['correction_focus']}_from_proposed_correction",
                source_fields=[
                    "proposed_correction",
                    "bounded_delta",
                    "operator_readable_explanation",
                ],
                operator_action="confirm_what_is_proposed_before_acceptance",
            ),
            _detail(
                "why_safe_or_blocked",
                f"offline_only_until_{profile['safety_focus']}_and_real_wing_proof",
                source_fields=[
                    "why_safe_summary",
                    "why_blocked_summary",
                    "safety_blocker_details",
                ],
                operator_action="review_why_safe_or_blocked_before_acceptance",
            ),
            _detail(
                "what_is_needed_before_live",
                "collect_operator_acceptance_real_wing_readback_rollback_and_emergency_proof",
                source_fields=[
                    "future_wing_readback_contract",
                    "future_live_promotion_requirements",
                    "operator_acceptance_gate_details",
                ],
                operator_action="collect_future_live_proofs_before_live_promotion",
            ),
        ]
    )


def build_correction_method_goal_coverage_report(
    *, pipeline_id: str = MAIN_PIPELINE_ID
) -> Dict[str, Any]:
    """Summarize proposal-layer coverage for the requested correction methods.

    This report is an audit surface only. It cannot grant readiness or live
    authority; it exists so operator-facing and release checks can see whether
    each requested correction family has proposal-only contract coverage.
    """

    contracts = build_proposal_only_method_contracts(pipeline_id=pipeline_id)
    required_flags = [f"{field}_required" for field in OPERATOR_CORRECTION_REQUIRED_FIELDS]
    required_operator_surfaces = list(
        CORRECTION_METHOD_GOAL_REQUIRED_OPERATOR_SURFACES
    )

    def _operator_surface_required(
        contract: Dict[str, Any],
        surface: Dict[str, str],
    ) -> bool:
        operator_contract = contract.get("operator_correction_contract") or {}
        surface_name = str(surface["surface"])
        flag = str(surface["flag"])
        if operator_contract.get(flag) is True or contract.get(flag) is True:
            return True
        if surface_name == "rollback_expectation":
            return bool(operator_contract.get("rollback_expectation"))
        if surface_name == "future_wing_readback_contract":
            return bool(
                contract.get("future_wing_readback_contract")
                or operator_contract.get("future_wing_readback_contract")
            ) and (
                operator_contract.get("future_wing_readback_required") is True
                or contract.get("future_wing_readback_contract_required") is True
            )
        return False

    contracts_by_family: Dict[str, List[Dict[str, Any]]] = {}
    for contract in contracts:
        family = str(contract.get("objective_family") or "")
        contracts_by_family.setdefault(family, []).append(contract)
    priority_group_by_family = {
        str(family): group
        for group in CORRECTION_METHOD_GOAL_PRIORITY_GROUPS
        for family in group["objective_families"]
    }

    method_rows: List[Dict[str, Any]] = []
    incomplete_method_ids: List[str] = []
    missing_operator_surface_method_ids: List[str] = []
    missing_operator_surface_details: List[Dict[str, Any]] = []
    live_authority_method_ids: List[str] = []
    queueable_method_ids: List[str] = []
    live_apply_method_ids: List[str] = []

    def _summary_value(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, list):
            return ",".join(str(item) for item in value) if value else "none"
        if value in (None, ""):
            return "none"
        return str(value)

    def _summary_items(mapping: Dict[str, Any], keys: Iterable[str]) -> List[str]:
        return [
            f"{key}={_summary_value(mapping.get(key))}"
            for key in keys
        ]

    def _coverage_fail_closed_authority_fields() -> Dict[str, Any]:
        return {
            "read_only_or_dry_run_only": True,
            "operator_review_required": True,
            "future_wing_readback_required": True,
            "hardware_proof_required_before_live": True,
            "blocks_live_promotion": True,
            "blocks_queue_import": True,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "readiness_closure_allowed": False,
            "live_apply_allowed": False,
            "queue_import_allowed": False,
            "can_apply_live": False,
            "can_import_to_queue": False,
            "can_authorize_live": False,
            "can_close_readiness": False,
            "queue_action": "not_queueable_until_promoted",
            "contacts_wing": False,
            "contacts_backend": False,
            "mutates_mixer": False,
        }

    for contract in contracts:
        method_id = str(contract.get("method_id") or "")
        objective_family = str(contract.get("objective_family") or "")
        priority_group = priority_group_by_family.get(objective_family)
        operator_contract = contract.get("operator_correction_contract") or {}
        proposal_contract = contract.get("proposal_contract") or {}
        future_wing_readback_contract = (
            contract.get("future_wing_readback_contract")
            or operator_contract.get("future_wing_readback_contract")
            or {}
        )
        live_authority_boundary = (
            contract.get("live_authority_boundary")
            or operator_contract.get("live_authority_boundary")
            or {}
        )
        future_live_requirements = (
            contract.get("future_live_promotion_requirements")
            or operator_contract.get("future_live_promotion_requirements")
            or {}
        )
        operator_review_record_template = (
            operator_contract.get("operator_review_record_template") or {}
        )
        flags = contract.get("operator_correction_required_field_flags")
        flags = flags if isinstance(flags, dict) else {}
        missing_flags = [
            flag
            for flag in required_flags
            if flags.get(flag) is not True
        ]
        operator_surface_flags = {
            str(surface["surface"]): _operator_surface_required(contract, surface)
            for surface in required_operator_surfaces
        }
        missing_operator_surfaces = sorted(
            surface
            for surface, required in operator_surface_flags.items()
            if required is not True
        )
        if missing_flags:
            incomplete_method_ids.append(method_id)
        if missing_operator_surfaces:
            missing_operator_surface_method_ids.append(method_id)
            missing_operator_surface_details.append(
                {
                    "method_id": method_id,
                    "objective_family": contract.get("objective_family"),
                    "missing_required_operator_surfaces": missing_operator_surfaces,
                }
            )
        if contract.get("live_authority_granted") is not False:
            live_authority_method_ids.append(method_id)
        if contract.get("queue_import_allowed") is not False:
            queueable_method_ids.append(method_id)
        if contract.get("live_apply_allowed") is not False:
            live_apply_method_ids.append(method_id)

        method_rows.append(
            {
                "method_id": method_id,
                "method_label": contract.get("method_label"),
                "objective_family": objective_family,
                "priority": (
                    priority_group["priority"]
                    if priority_group is not None
                    else None
                ),
                "priority_label": (
                    priority_group["label"]
                    if priority_group is not None
                    else "Supporting / Context"
                ),
                "priority_review_scope": (
                    "required_priority"
                    if priority_group is not None
                    else "supporting_context"
                ),
                "domain": contract.get("domain"),
                "status": contract.get("status"),
                "value_type": contract.get("value_type"),
                "target_kind": contract.get("target_kind"),
                "source_contract_read_only_or_dry_run_only": (
                    contract.get("read_only_or_dry_run_only") is True
                ),
                "source_contract_live_authority_granted_claimed": (
                    contract.get("live_authority_granted") is True
                ),
                "source_contract_live_apply_allowed_claimed": (
                    contract.get("live_apply_allowed") is True
                ),
                "source_contract_queue_import_allowed_claimed": (
                    contract.get("queue_import_allowed") is True
                ),
                **_coverage_fail_closed_authority_fields(),
                "required_field_count": len(required_flags),
                "missing_required_flags": missing_flags,
                "safety_evidence_required": list(
                    proposal_contract.get("safety_evidence_required")
                    or []
                ),
                "applicable_safety_checks": list(
                    operator_contract.get("applicable_safety_checks") or []
                ),
                "operator_required_surfaces": [
                    str(surface["surface"])
                    for surface in required_operator_surfaces
                ],
                "operator_required_surface_flags": operator_surface_flags,
                "missing_required_operator_surfaces": missing_operator_surfaces,
                "operator_payload_required_fields": list(
                    contract.get("operator_correction_required_fields") or []
                ),
                "operator_evidence_required_fields": list(
                    operator_contract.get("required_evidence_fields") or []
                ),
                "source_evidence_required": operator_contract.get(
                    "source_evidence_required"
                )
                is True,
                "source_evidence_details_required": operator_contract.get(
                    "source_evidence_details_required"
                )
                is True,
                "measured_or_audible_problem_required": operator_contract.get(
                    "measured_or_audible_problem_required"
                )
                is True,
                "proposed_correction_required": operator_contract.get(
                    "proposed_correction_required"
                )
                is True,
                "confidence_required": operator_contract.get(
                    "confidence_required"
                )
                is True,
                "confidence_review_required": operator_contract.get(
                    "confidence_review_required"
                )
                is True,
                "bounded_delta_required": operator_contract.get(
                    "bounded_delta_required"
                )
                is True,
                "bounded_delta_details_required": operator_contract.get(
                    "bounded_delta_details_required"
                )
                is True,
                "bounded_delta_unit": contract.get("bounded_delta_unit"),
                "bounded_delta_policy": contract.get("bounded_delta_policy"),
                "evidence_problem_correction_chain_details": (
                    _evidence_problem_correction_chain_detail_summary(
                        source_evidence_required=(
                            operator_contract.get("source_evidence_required") is True
                        ),
                        source_evidence_details_required=(
                            operator_contract.get(
                                "source_evidence_details_required"
                            )
                            is True
                        ),
                        measured_or_audible_problem_required=(
                            operator_contract.get(
                                "measured_or_audible_problem_required"
                            )
                            is True
                        ),
                        proposed_correction_required=(
                            operator_contract.get("proposed_correction_required")
                            is True
                        ),
                        confidence_required=(
                            operator_contract.get("confidence_required") is True
                        ),
                        confidence_review_required=(
                            operator_contract.get("confidence_review_required")
                            is True
                        ),
                        bounded_delta_required=(
                            operator_contract.get("bounded_delta_required") is True
                        ),
                        bounded_delta_details_required=(
                            operator_contract.get("bounded_delta_details_required")
                            is True
                        ),
                        denial_reason=(
                            operator_contract.get("denial_reason")
                            or live_authority_boundary.get("denial_reason")
                        ),
                    )
                ),
                "safety_blockers_required": operator_contract.get(
                    "safety_blockers_required"
                )
                is True,
                "safety_blocker_details_required": operator_contract.get(
                    "safety_blocker_details_required"
                )
                is True,
                "headroom_feedback_true_peak_checks_required": (
                    operator_contract.get(
                        "headroom_feedback_true_peak_checks_required"
                    )
                    is True
                ),
                "headroom_feedback_true_peak_check_details_required": (
                    operator_contract.get(
                        "headroom_feedback_true_peak_check_details_required"
                    )
                    is True
                ),
                "operator_readable_explanation_required": (
                    operator_contract.get(
                        "operator_readable_explanation_required"
                    )
                    is True
                ),
                "operator_explanation_packet_required": operator_contract.get(
                    "operator_explanation_packet_required"
                )
                is True,
                "operator_evidence_packet_required": operator_contract.get(
                    "operator_evidence_packet_required"
                )
                is True,
                "rollback_expectation": operator_contract.get(
                    "rollback_expectation"
                ),
                "rollback_readback_contract_required": operator_contract.get(
                    "rollback_readback_contract_required"
                )
                is True,
                "rollback_readback_details_required": operator_contract.get(
                    "rollback_readback_details_required"
                )
                is True,
                "future_live_required_proofs": list(
                    future_live_requirements.get("required_proofs") or []
                ),
                "future_wing_readback_targets": list(
                    future_wing_readback_contract.get("readback_targets")
                    or []
                ),
                "future_wing_readback_contract_summary": _summary_items(
                    future_wing_readback_contract,
                    (
                        "target_kind",
                        "old_value_source",
                        "readback_targets",
                        "required_evidence_fields",
                        "operator_next_step",
                        "denial_reason",
                    ),
                ),
                "blocked_live_actions": list(
                    live_authority_boundary.get("blocked_live_actions")
                    or []
                ),
                "safety_blocker_details": _safety_blocker_detail_summary(
                    required_safety_checks=(
                        operator_contract.get("applicable_safety_checks") or []
                    ),
                    blocked_live_actions=(
                        live_authority_boundary.get("blocked_live_actions")
                        or []
                    ),
                    missing_required_operator_surfaces=missing_operator_surfaces,
                    denial_reason=(
                        operator_contract.get("denial_reason")
                        or live_authority_boundary.get("denial_reason")
                    ),
                ),
                "headroom_feedback_true_peak_check_details": (
                    _headroom_feedback_true_peak_check_detail_summary(
                        required_safety_checks=(
                            operator_contract.get("applicable_safety_checks") or []
                        ),
                        denial_reason=(
                            operator_contract.get("denial_reason")
                            or live_authority_boundary.get("denial_reason")
                        ),
                    )
                ),
                "future_live_requirement_details": (
                    _future_live_requirement_detail_summary(
                        future_live_required_proofs=(
                            future_live_requirements.get("required_proofs") or []
                        ),
                        future_wing_readback_targets=(
                            future_wing_readback_contract.get("readback_targets")
                            or []
                        ),
                        blocked_live_actions=(
                            live_authority_boundary.get("blocked_live_actions")
                            or []
                        ),
                    )
                ),
                "live_authority_boundary_summary": _summary_items(
                    live_authority_boundary,
                    (
                        "allowed_current_live_write_kinds",
                        "can_authorize_live",
                        "denial_reason",
                        "live_apply_allowed",
                        "live_authority_granted",
                        "queue_import_allowed",
                        "readiness_closure_allowed",
                    ),
                ),
                "operator_review_checklist": list(
                    operator_contract.get("operator_review_checklist") or []
                ),
                "operator_review_checklist_details": (
                    _operator_review_checklist_detail_summary(
                        operator_contract.get("operator_review_checklist") or []
                    )
                ),
                "operator_review_decisions": [
                    option.get("decision")
                    for option in (
                        operator_contract.get("operator_review_decision_options")
                        or []
                    )
                ],
                "operator_review_decision_option_details": (
                    _operator_review_decision_option_detail_summary(
                        operator_contract.get("operator_review_decision_options")
                        or []
                    )
                ),
                "operator_review_record_required_fields": list(
                    operator_review_record_template.get("required_record_fields")
                    or []
                ),
                "operator_review_record_field_details": (
                    _operator_review_record_field_detail_summary(
                        operator_review_record_template.get("required_record_fields")
                        or []
                    )
                ),
                "operator_next_action_details": (
                    _operator_next_action_detail_summary(
                        operator_next_step=(
                            future_wing_readback_contract.get("operator_next_step")
                            or contract.get("operator_next_step")
                        ),
                        operator_evidence_required_fields=operator_contract.get(
                            "required_evidence_fields"
                        )
                        or [],
                        operator_review_record_required_fields=(
                            operator_review_record_template.get(
                                "required_record_fields"
                            )
                            or []
                        ),
                        future_wing_readback_targets=(
                            future_wing_readback_contract.get("readback_targets")
                            or []
                        ),
                        future_live_required_proofs=(
                            future_live_requirements.get("required_proofs") or []
                        ),
                        blocked_live_actions=(
                            live_authority_boundary.get("blocked_live_actions")
                            or []
                        ),
                    )
                ),
                "denial_reason": (
                    operator_contract.get("denial_reason")
                    or live_authority_boundary.get("denial_reason")
                ),
                "operator_next_step": (
                    future_wing_readback_contract.get("operator_next_step")
                    or contract.get("operator_next_step")
                ),
            }
        )

    family_rows: List[Dict[str, Any]] = []
    for family in CORRECTION_METHOD_GOAL_REQUIRED_OBJECTIVE_FAMILIES:
        family_contracts = contracts_by_family.get(family, [])
        family_method_ids = sorted(
            str(contract.get("method_id") or "")
            for contract in family_contracts
        )
        family_missing_flags = sorted(
            {
                flag
                for contract in family_contracts
                for flag in required_flags
                if (
                    contract.get("operator_correction_required_field_flags")
                    or {}
                ).get(flag)
                is not True
            }
        )
        family_missing_operator_surfaces = sorted(
            {
                str(surface["surface"])
                for contract in family_contracts
                for surface in required_operator_surfaces
                if not _operator_surface_required(contract, surface)
            }
        )
        family_rows.append(
            {
                "objective_family": family,
                "method_count": len(family_contracts),
                "method_ids": family_method_ids,
                "covered": bool(family_contracts),
                "missing_required_flags": family_missing_flags,
                "missing_required_operator_surfaces": (
                    family_missing_operator_surfaces
                ),
                "source_family_read_only_or_dry_run_only": all(
                    contract.get("read_only_or_dry_run_only") is True
                    for contract in family_contracts
                )
                if family_contracts
                else False,
                "source_family_live_authority_granted_claimed": any(
                    contract.get("live_authority_granted") is True
                    for contract in family_contracts
                ),
                "source_family_live_apply_allowed_claimed": any(
                    contract.get("live_apply_allowed") is True
                    for contract in family_contracts
                ),
                "source_family_queue_import_allowed_claimed": any(
                    contract.get("queue_import_allowed") is True
                    for contract in family_contracts
                ),
                **_coverage_fail_closed_authority_fields(),
            }
        )

    missing_families = [
        row["objective_family"]
        for row in family_rows
        if not row["covered"]
    ]
    incomplete_field_flag_families = [
        row["objective_family"]
        for row in family_rows
        if row["missing_required_flags"]
    ]
    incomplete_operator_surface_families = [
        row["objective_family"]
        for row in family_rows
        if row["missing_required_operator_surfaces"]
    ]
    incomplete_families = sorted(
        set(incomplete_field_flag_families)
        | set(incomplete_operator_surface_families)
    )
    priority_group_rows: List[Dict[str, Any]] = []
    for group in CORRECTION_METHOD_GOAL_PRIORITY_GROUPS:
        group_families = list(group["objective_families"])
        group_method_rows = [
            row
            for row in method_rows
            if row["objective_family"] in group_families
        ]
        group_method_ids = sorted(
            str(row["method_id"]) for row in group_method_rows
        )
        group_missing_operator_surface_method_ids = sorted(
            str(row["method_id"])
            for row in group_method_rows
            if row["missing_required_operator_surfaces"]
        )
        group_live_authority_method_ids = sorted(
            str(row["method_id"])
            for row in group_method_rows
            if row["source_contract_live_authority_granted_claimed"] is True
        )
        group_live_apply_method_ids = sorted(
            str(row["method_id"])
            for row in group_method_rows
            if row["source_contract_live_apply_allowed_claimed"] is True
        )
        group_queueable_method_ids = sorted(
            str(row["method_id"])
            for row in group_method_rows
            if row["source_contract_queue_import_allowed_claimed"] is True
        )
        priority_group_rows.append(
            {
                "priority": group["priority"],
                "label": group["label"],
                "objective_families": group_families,
                "covered": all(
                    family not in missing_families
                    and family not in incomplete_families
                    for family in group_families
                )
                and not group_missing_operator_surface_method_ids
                and not group_live_authority_method_ids
                and not group_live_apply_method_ids
                and not group_queueable_method_ids,
                "method_count": len(group_method_rows),
                "method_ids": group_method_ids,
                "applicable_safety_checks": sorted(
                    {
                        str(check)
                        for row in group_method_rows
                        for check in row["applicable_safety_checks"]
                    }
                ),
                "bounded_delta_policies": sorted(
                    {
                        str(row["bounded_delta_policy"])
                        for row in group_method_rows
                        if row["bounded_delta_policy"]
                    }
                ),
                "future_wing_readback_targets": sorted(
                    {
                        str(target)
                        for row in group_method_rows
                        for target in row["future_wing_readback_targets"]
                    }
                ),
                "future_live_required_proofs": sorted(
                    {
                        str(proof)
                        for row in group_method_rows
                        for proof in row["future_live_required_proofs"]
                    }
                ),
                "blocked_live_actions": sorted(
                    {
                        str(action)
                        for row in group_method_rows
                        for action in row["blocked_live_actions"]
                    }
                ),
                "missing_required_operator_surface_method_ids": (
                    group_missing_operator_surface_method_ids
                ),
                "live_authority_method_ids": group_live_authority_method_ids,
                "live_apply_method_ids": group_live_apply_method_ids,
                "queueable_method_ids": group_queueable_method_ids,
                "read_only_or_dry_run_only": all(
                    row["read_only_or_dry_run_only"] is True
                    for row in group_method_rows
                )
                if group_method_rows
                else False,
                **_coverage_fail_closed_authority_fields(),
                "operator_next_step": (
                    "review_priority_group_offline_methods_and_collect_real_"
                    "wing_readback_before_live_promotion"
                ),
            }
        )

    first_priority_review_summary = next(
        row for row in priority_group_rows if row["priority"] == 1
    )
    second_priority_review_summary = next(
        row for row in priority_group_rows if row["priority"] == 2
    )
    third_priority_review_summary = next(
        row for row in priority_group_rows if row["priority"] == 3
    )
    fourth_priority_review_summary = next(
        row for row in priority_group_rows if row["priority"] == 4
    )
    fifth_priority_review_summary = next(
        row for row in priority_group_rows if row["priority"] == 5
    )
    sixth_priority_review_summary = next(
        row for row in priority_group_rows if row["priority"] == 6
    )
    seventh_priority_review_summary = next(
        row for row in priority_group_rows if row["priority"] == 7
    )
    blocked_by = []
    if missing_families:
        blocked_by.append("missing_required_objective_family_contracts")
    if incomplete_field_flag_families or incomplete_method_ids:
        blocked_by.append("incomplete_operator_correction_required_flags")
    if missing_operator_surface_method_ids:
        blocked_by.append("missing_required_operator_surfaces")
    if live_authority_method_ids:
        blocked_by.append("proposal_contract_claims_live_authority")
    if live_apply_method_ids or queueable_method_ids:
        blocked_by.append("proposal_contract_claims_apply_or_queue")

    covered = not blocked_by
    return {
        "schema": "automixer.correction_method_goal_coverage.v1",
        "pipeline_id": pipeline_id,
        "status": (
            "proposal_layer_contracts_ready_operator_review_required"
            if covered
            else "proposal_layer_contracts_incomplete"
        ),
        "covered": covered,
        **_coverage_fail_closed_authority_fields(),
        "operator_required_surfaces": [
            str(surface["surface"])
            for surface in required_operator_surfaces
        ],
        "required_objective_families": list(
            CORRECTION_METHOD_GOAL_REQUIRED_OBJECTIVE_FAMILIES
        ),
        "priority_groups": [
            {
                "priority": row["priority"],
                "label": row["label"],
                "objective_families": list(row["objective_families"]),
                "covered": row["covered"],
            }
            for row in priority_group_rows
        ],
        "priority_group_rows": priority_group_rows,
        "first_priority_review_summary": first_priority_review_summary,
        "second_priority_review_summary": second_priority_review_summary,
        "third_priority_review_summary": third_priority_review_summary,
        "fourth_priority_review_summary": fourth_priority_review_summary,
        "fifth_priority_review_summary": fifth_priority_review_summary,
        "sixth_priority_review_summary": sixth_priority_review_summary,
        "seventh_priority_review_summary": seventh_priority_review_summary,
        "family_rows": family_rows,
        "method_rows": method_rows,
        "summary": {
            "required_objective_family_count": len(
                CORRECTION_METHOD_GOAL_REQUIRED_OBJECTIVE_FAMILIES
            ),
            "covered_objective_family_count": sum(
                1 for row in family_rows if row["covered"]
            ),
            "contract_method_count": len(method_rows),
            "missing_objective_families": missing_families,
            "incomplete_objective_families": incomplete_families,
            "incomplete_field_flag_objective_families": (
                incomplete_field_flag_families
            ),
            "incomplete_operator_surface_objective_families": (
                incomplete_operator_surface_families
            ),
            "incomplete_method_ids": sorted(incomplete_method_ids),
            "missing_required_operator_surface_method_ids": sorted(
                missing_operator_surface_method_ids
            ),
            "missing_required_operator_surface_details": sorted(
                missing_operator_surface_details,
                key=lambda item: str(item["method_id"]),
            ),
            "first_priority_method_count": first_priority_review_summary[
                "method_count"
            ],
            "first_priority_method_ids": list(
                first_priority_review_summary["method_ids"]
            ),
            "first_priority_applicable_safety_checks": list(
                first_priority_review_summary["applicable_safety_checks"]
            ),
            "first_priority_future_wing_readback_target_count": len(
                first_priority_review_summary["future_wing_readback_targets"]
            ),
            "first_priority_blocked_live_action_count": len(
                first_priority_review_summary["blocked_live_actions"]
            ),
            "second_priority_method_count": second_priority_review_summary[
                "method_count"
            ],
            "second_priority_method_ids": list(
                second_priority_review_summary["method_ids"]
            ),
            "second_priority_applicable_safety_checks": list(
                second_priority_review_summary["applicable_safety_checks"]
            ),
            "second_priority_future_wing_readback_target_count": len(
                second_priority_review_summary["future_wing_readback_targets"]
            ),
            "second_priority_blocked_live_action_count": len(
                second_priority_review_summary["blocked_live_actions"]
            ),
            "third_priority_method_count": third_priority_review_summary[
                "method_count"
            ],
            "third_priority_method_ids": list(
                third_priority_review_summary["method_ids"]
            ),
            "third_priority_applicable_safety_checks": list(
                third_priority_review_summary["applicable_safety_checks"]
            ),
            "third_priority_future_wing_readback_target_count": len(
                third_priority_review_summary["future_wing_readback_targets"]
            ),
            "third_priority_blocked_live_action_count": len(
                third_priority_review_summary["blocked_live_actions"]
            ),
            "fourth_priority_method_count": fourth_priority_review_summary[
                "method_count"
            ],
            "fourth_priority_method_ids": list(
                fourth_priority_review_summary["method_ids"]
            ),
            "fourth_priority_applicable_safety_checks": list(
                fourth_priority_review_summary["applicable_safety_checks"]
            ),
            "fourth_priority_future_wing_readback_target_count": len(
                fourth_priority_review_summary["future_wing_readback_targets"]
            ),
            "fourth_priority_blocked_live_action_count": len(
                fourth_priority_review_summary["blocked_live_actions"]
            ),
            "fifth_priority_method_count": fifth_priority_review_summary[
                "method_count"
            ],
            "fifth_priority_method_ids": list(
                fifth_priority_review_summary["method_ids"]
            ),
            "fifth_priority_applicable_safety_checks": list(
                fifth_priority_review_summary["applicable_safety_checks"]
            ),
            "fifth_priority_future_wing_readback_target_count": len(
                fifth_priority_review_summary["future_wing_readback_targets"]
            ),
            "fifth_priority_blocked_live_action_count": len(
                fifth_priority_review_summary["blocked_live_actions"]
            ),
            "sixth_priority_method_count": sixth_priority_review_summary[
                "method_count"
            ],
            "sixth_priority_method_ids": list(
                sixth_priority_review_summary["method_ids"]
            ),
            "sixth_priority_applicable_safety_checks": list(
                sixth_priority_review_summary["applicable_safety_checks"]
            ),
            "sixth_priority_future_wing_readback_target_count": len(
                sixth_priority_review_summary["future_wing_readback_targets"]
            ),
            "sixth_priority_blocked_live_action_count": len(
                sixth_priority_review_summary["blocked_live_actions"]
            ),
            "seventh_priority_method_count": seventh_priority_review_summary[
                "method_count"
            ],
            "seventh_priority_method_ids": list(
                seventh_priority_review_summary["method_ids"]
            ),
            "seventh_priority_applicable_safety_checks": list(
                seventh_priority_review_summary["applicable_safety_checks"]
            ),
            "seventh_priority_future_wing_readback_target_count": len(
                seventh_priority_review_summary["future_wing_readback_targets"]
            ),
            "seventh_priority_blocked_live_action_count": len(
                seventh_priority_review_summary["blocked_live_actions"]
            ),
            "live_authority_method_ids": sorted(live_authority_method_ids),
            "live_apply_method_ids": sorted(live_apply_method_ids),
            "queueable_method_ids": sorted(queueable_method_ids),
            "blocked_by": blocked_by,
        },
        "denial_reason": "proposal_layer_coverage_cannot_replace_real_wing_proof",
        "operator_next_step": "review_offline_proposals_and_capture_real_wing_readback_before_live_promotion",
    }


def build_correction_method_goal_operator_review_packet(
    *, pipeline_id: str = MAIN_PIPELINE_ID
) -> Dict[str, Any]:
    """Build an operator-facing review packet for offline correction methods.

    The packet is intentionally derived from proposal-only contracts. It gives
    a sound engineer the checklist for reviewing each method family, but it
    cannot queue proposals, apply mixer changes, close readiness or replace
    future real WING readback evidence.
    """

    coverage = build_correction_method_goal_coverage_report(pipeline_id=pipeline_id)
    required_families = set(coverage["required_objective_families"])
    blocked_by = list(coverage["summary"]["blocked_by"])

    def _operator_review_contract(
        *,
        source_fields: List[str],
        required: bool,
        operator_action: str,
    ) -> Dict[str, Any]:
        return {
            "source_fields": list(source_fields),
            "required": bool(required),
            "review_status": "required_from_offline_proposal_payload",
            "operator_action": operator_action,
        }

    def _packet_fail_closed_authority_fields() -> Dict[str, Any]:
        return {
            "read_only_or_dry_run_only": True,
            "operator_review_required": True,
            "future_wing_readback_required": True,
            "hardware_proof_required_before_live": True,
            "blocks_live_promotion": True,
            "blocks_queue_import": True,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "readiness_closure_allowed": False,
            "live_apply_allowed": False,
            "queue_import_allowed": False,
            "can_apply_live": False,
            "can_import_to_queue": False,
            "can_authorize_live": False,
            "can_close_readiness": False,
            "queue_action": "not_queueable_until_promoted",
            "contacts_wing": False,
            "contacts_backend": False,
            "mutates_mixer": False,
        }

    method_primary_test_references = {
        "agentic_method_foundation": ("tests/test_agentic_method_foundation.py",),
        "arrangement_role_proposal": ("tests/test_arrangement_role_proposal.py",),
        "auto_compressor": ("tests/test_auto_compressor_fail_closed.py",),
        "auto_compressor_cf": ("tests/test_auto_compressor_fail_closed.py",),
        "eq_v3": ("tests/test_eq_v3.py",),
        "comp_v3": ("tests/test_comp_v3.py",),
        "balance_v3": ("tests/test_balance_v3.py",),
        "gain_v3": ("tests/test_gain_v3.py",),
        "phase_v3": ("tests/test_phase_v3.py",),
        "pan_v3": ("tests/test_pan_v3.py",),
        "fx_v3": ("tests/test_fx_v3.py",),
        "pa_v3": ("tests/test_pa_v3.py",),
        "auto_eq": ("tests/test_auto_eq_fail_closed.py",),
        "auto_fader_v1": ("tests/test_auto_fader_level_plane.py",),
        "auto_fx": ("tests/test_auto_fx.py",),
        "auto_gate_caig": ("tests/test_gate.py",),
        "auto_panner": ("tests/test_auto_panner.py",),
        "auto_soundcheck_engine": ("tests/test_auto_soundcheck_engine.py",),
        "bus_group_balance_proposal": ("tests/test_bus_group_balance_proposal.py",),
        "compressor_dynamics_proposal": (
            "tests/test_compressor_dynamics_proposal.py",
        ),
        "context_aware_gate_agent": ("tests/test_agentic_method_foundation.py",),
        "corrective_eq_proposal": ("tests/test_corrective_eq_proposal.py",),
        "cross_adaptive_eq": ("tests/test_cross_adaptive_eq.py",),
        "decision_evidence_packet_proposal": (
            "tests/test_decision_evidence_packet_proposal.py",
        ),
        "deesser_vocal_sibilance": ("tests/test_deesser_vocal_sibilance.py",),
        "frequency_separation_proposal": (
            "tests/test_frequency_separation_proposal.py",
        ),
        "fx_direct_apply": ("tests/test_auto_fx.py",),
        "fx_send_proposal": ("tests/test_fx_send_proposal.py",),
        "live_shared_mix": ("tests/test_live_shared_mix.py",),
        "long_term_memory_learning_proposal": (
            "tests/test_long_term_memory_learning_proposal.py",
        ),
        "lufs_gain_staging": ("tests/test_input_trim_autogain.py",),
        "master_bus_optimization_proposal": (
            "tests/test_master_bus_optimization_proposal.py",
        ),
        "mix_variant_profile_proposal": (
            "tests/test_mix_variant_profile_proposal.py",
        ),
        "mute_management_proposal": ("tests/test_mute_management_proposal.py",),
        "pan_spatial_proposal": ("tests/test_pan_spatial_proposal.py",),
        "performance_adaptation_proposal": (
            "tests/test_performance_adaptation_proposal.py",
        ),
        "phase_alignment": ("tests/test_phase_alignment_fail_closed.py",),
        "phase_aware_eq_guard": ("tests/test_phase_aware_eq_guard.py",),
        "routing_management_proposal": (
            "tests/test_routing_management_proposal.py",
        ),
        "routing_snapshot_reset_writes": (
            "tests/test_routing_management_proposal.py",
        ),
        "scene_preset_management_proposal": (
            "tests/test_scene_preset_management_proposal.py",
        ),
        "source_identity_confidence_proposal": (
            "tests/test_source_identity_confidence_proposal.py",
        ),
        "source_identity_replay_calibration": (
            "tests/test_source_identity_replay_calibration.py",
        ),
    }

    def _test_evidence_references(method_id: str) -> List[str]:
        references = [
            *method_primary_test_references.get(method_id, ()),
            (
                "tests/test_processing_proposal_contracts.py::"
                "test_correction_method_goal_operator_review_packet_is_readonly_and_actionable"
            ),
            (
                "tests/test_processing_proposal_contracts.py::"
                "test_correction_method_goal_operator_review_export_writes_readonly_artifacts"
            ),
            (
                "tests/test_no_wing_release_check.py::"
                "test_verify_no_wing_artifact_manifest_passes_written_bundle"
            ),
        ]
        deduped: List[str] = []
        for reference in references:
            if reference not in deduped:
                deduped.append(reference)
        return deduped

    operator_problem_review_summary = [
        "source_evidence=required_from_offline_proposal_payload",
        "source_evidence_details=operator_must_review_before_acceptance",
        "measured_or_audible_problem=required_operator_visible_problem_statement",
        "proposed_correction=required_operator_visible_correction_statement",
        "operator_action=confirm_problem_statement_matches_source_evidence_and_correction",
        "live_boundary=offline_only_until_real_wing_readback_and_rollback_proof",
    ]

    def _csv(values: Any) -> str:
        return ",".join(str(value) for value in (values or []))

    def _operator_method_review_focus(row: Dict[str, Any]) -> List[str]:
        priority = row.get("priority")
        priority_value = "none" if priority is None else str(priority)
        return [
            f"method_id={row['method_id']}",
            f"priority={priority_value}",
            f"priority_label={row['priority_label']}",
            f"objective_family={row['objective_family']}",
            f"required_evidence={_csv(row['safety_evidence_required'])}",
            f"required_safety_checks={_csv(row['applicable_safety_checks'])}",
            (
                "bounded_delta="
                f"{row['bounded_delta_unit']}:{row['bounded_delta_policy']}"
            ),
            f"rollback_expectation={row['rollback_expectation']}",
            (
                "future_wing_readback_targets="
                f"{_csv(row['future_wing_readback_targets'])}"
            ),
            f"blocked_live_actions={_csv(row['blocked_live_actions'])}",
            "live_boundary=offline_only_no_queue_no_live_apply_until_real_wing_proof",
        ]

    def _operator_why_safe_summary(row: Dict[str, Any]) -> List[str]:
        return [
            "read_only_or_dry_run_only=true",
            "live_apply_allowed=false",
            "queue_import_allowed=false",
            "operator_review_required=true",
            (
                "bounded_delta="
                f"{row['bounded_delta_unit']}:{row['bounded_delta_policy']}"
            ),
            f"required_safety_checks={_csv(row['applicable_safety_checks'])}",
        ]

    def _operator_why_blocked_summary(row: Dict[str, Any]) -> List[str]:
        return [
            f"denial_reason={row['denial_reason']}",
            "future_wing_readback_required=true",
            "hardware_proof_required_before_live=true",
            "readiness_closure_allowed=false",
            f"blocked_live_actions={_csv(row['blocked_live_actions'])}",
            f"future_live_required_proofs={_csv(row['future_live_required_proofs'])}",
        ]

    method_cards: List[Dict[str, Any]] = []
    for row in coverage["method_rows"]:
        method_label = row.get("method_label") or row["method_id"]
        method_cards.append(
            {
                "method_id": row["method_id"],
                "method_label": method_label,
                "objective_family": row["objective_family"],
                "priority": row["priority"],
                "priority_label": row["priority_label"],
                "priority_review_scope": row["priority_review_scope"],
                "domain": row["domain"],
                "status": row["status"],
                "review_status": "operator_review_required",
                "review_scope": "offline_proposal_only",
                "operator_question": (
                    f"Review {method_label} as an offline proposal only; "
                    "do not queue or apply it live."
                ),
                "source_evidence": _operator_review_contract(
                    source_fields=["source_evidence"],
                    required=row["source_evidence_required"],
                    operator_action=(
                        "confirm_source_evidence_matches_current_offline_material"
                    ),
                ),
                "source_evidence_details": _operator_review_contract(
                    source_fields=["source_evidence_details"],
                    required=row["source_evidence_details_required"],
                    operator_action=(
                        "review_source_evidence_details_before_acceptance"
                    ),
                ),
                "measured_or_audible_problem": _operator_review_contract(
                    source_fields=["measured_or_audible_problem"],
                    required=row["measured_or_audible_problem_required"],
                    operator_action=(
                        "confirm_problem_statement_matches_source_evidence"
                    ),
                ),
                "proposed_correction": _operator_review_contract(
                    source_fields=["proposed_correction"],
                    required=row["proposed_correction_required"],
                    operator_action=(
                        "confirm_proposed_correction_matches_the_problem"
                    ),
                ),
                "confidence": _operator_review_contract(
                    source_fields=[
                        "confidence",
                        "confidence_review",
                        "confidence_review_details",
                    ],
                    required=(
                        row["confidence_required"]
                        and row["confidence_review_required"]
                    ),
                    operator_action=(
                        "review_confidence_and_confidence_blockers"
                    ),
                ),
                "bounded_delta": _operator_review_contract(
                    source_fields=[
                        "bounded_delta",
                        "bounded_delta_details",
                        "bounded_delta_unit",
                        "bounded_delta_policy",
                    ],
                    required=(
                        row["bounded_delta_required"]
                        and row["bounded_delta_details_required"]
                    ),
                    operator_action=(
                        "review_bounded_delta_and_required_evidence"
                    ),
                ),
                "safety_blockers": _operator_review_contract(
                    source_fields=[
                        "safety_blockers",
                        "effective_safety_blockers",
                        "safety_blocker_details",
                    ],
                    required=(
                        row["safety_blockers_required"]
                        and row["safety_blocker_details_required"]
                    ),
                    operator_action=(
                        "review_safety_blockers_before_acceptance"
                    ),
                ),
                "headroom_feedback_true_peak_checks": (
                    _operator_review_contract(
                        source_fields=[
                            "headroom_feedback_true_peak_checks",
                            "headroom_feedback_true_peak_check_details",
                        ],
                        required=row[
                            "headroom_feedback_true_peak_checks_required"
                        ],
                        operator_action=(
                            "verify_applicable_headroom_feedback_true_peak_checks"
                        ),
                    )
                ),
                "operator_readable_explanation": _operator_review_contract(
                    source_fields=[
                        "operator_readable_explanation",
                        "operator_explanation_packet",
                    ],
                    required=row["operator_readable_explanation_required"],
                    operator_action=(
                        "confirm_operator_explanation_is_clear_before_acceptance"
                    ),
                ),
                "rollback_readback_contract": _operator_review_contract(
                    source_fields=[
                        "rollback_expectation",
                        "rollback_readback_contract",
                        "rollback_readback_details",
                    ],
                    required=row["rollback_readback_contract_required"],
                    operator_action=(
                        "confirm_rollback_readback_contract_before_live_promotion"
                    ),
                ),
                "future_wing_readback_contract": _operator_review_contract(
                    source_fields=[
                        "future_wing_readback_contract",
                        "future_wing_readback_contract_summary",
                        "future_wing_readback_targets",
                    ],
                    required=bool(
                        row["future_wing_readback_targets"]
                        and row["future_wing_readback_contract_summary"]
                    ),
                    operator_action=(
                        "verify_future_wing_readback_contract_before_live_promotion"
                    ),
                ),
                "test_evidence": _operator_review_contract(
                    source_fields=[
                        "test_evidence_references",
                        "operator_review_contract_tests",
                        "export_verification_tests",
                        "no_wing_release_gate_tests",
                    ],
                    required=True,
                    operator_action=(
                        "verify_offline_method_tests_pass_before_acceptance"
                    ),
                ),
                "test_evidence_references": _test_evidence_references(
                    str(row["method_id"])
                ),
                "goal_requirement_details": _goal_requirement_detail_summary(
                    source_evidence_required=row["source_evidence_required"],
                    source_evidence_details_required=(
                        row["source_evidence_details_required"]
                    ),
                    measured_or_audible_problem_required=(
                        row["measured_or_audible_problem_required"]
                    ),
                    proposed_correction_required=(
                        row["proposed_correction_required"]
                    ),
                    confidence_required=row["confidence_required"],
                    confidence_review_required=row["confidence_review_required"],
                    bounded_delta_required=row["bounded_delta_required"],
                    bounded_delta_details_required=(
                        row["bounded_delta_details_required"]
                    ),
                    safety_blockers_required=row["safety_blockers_required"],
                    safety_blocker_details_required=(
                        row["safety_blocker_details_required"]
                    ),
                    required_safety_checks=row["applicable_safety_checks"],
                    headroom_feedback_true_peak_checks_required=row[
                        "headroom_feedback_true_peak_checks_required"
                    ],
                    headroom_feedback_true_peak_check_details_required=row[
                        "headroom_feedback_true_peak_check_details_required"
                    ],
                    operator_readable_explanation_required=row[
                        "operator_readable_explanation_required"
                    ],
                    operator_explanation_packet_required=row[
                        "operator_explanation_packet_required"
                    ],
                    rollback_expectation=row["rollback_expectation"],
                    rollback_readback_contract_required=row[
                        "rollback_readback_contract_required"
                    ],
                    rollback_readback_details_required=row[
                        "rollback_readback_details_required"
                    ],
                    future_wing_readback_targets=row[
                        "future_wing_readback_targets"
                    ],
                    future_wing_readback_contract_summary=row[
                        "future_wing_readback_contract_summary"
                    ],
                    test_evidence_references=_test_evidence_references(
                        str(row["method_id"])
                    ),
                    denial_reason=row["denial_reason"],
                ),
                "method_specific_review_details": (
                    _method_specific_review_detail_summary(
                        method_id=row["method_id"],
                        objective_family=row["objective_family"],
                        priority=row["priority"],
                        priority_label=row["priority_label"],
                        required_evidence=row["safety_evidence_required"],
                        required_safety_checks=row["applicable_safety_checks"],
                        bounded_delta_unit=row["bounded_delta_unit"],
                        bounded_delta_policy=row["bounded_delta_policy"],
                        rollback_expectation=row["rollback_expectation"],
                        future_wing_readback_targets=row[
                            "future_wing_readback_targets"
                        ],
                        blocked_live_actions=row["blocked_live_actions"],
                        denial_reason=row["denial_reason"],
                    )
                ),
                "operator_actionable_explanation_details": (
                    _operator_actionable_explanation_detail_summary(
                        method_id=row["method_id"],
                        objective_family=row["objective_family"],
                        priority=row["priority"],
                        priority_label=row["priority_label"],
                        required_evidence=row["safety_evidence_required"],
                        required_safety_checks=row["applicable_safety_checks"],
                        bounded_delta_unit=row["bounded_delta_unit"],
                        bounded_delta_policy=row["bounded_delta_policy"],
                        rollback_expectation=row["rollback_expectation"],
                        future_wing_readback_targets=row[
                            "future_wing_readback_targets"
                        ],
                        blocked_live_actions=row["blocked_live_actions"],
                        denial_reason=row["denial_reason"],
                    )
                ),
                "operator_problem_review_summary": list(
                    operator_problem_review_summary
                ),
                "operator_method_review_focus": _operator_method_review_focus(
                    row
                ),
                "why_safe_summary": _operator_why_safe_summary(row),
                "why_blocked_summary": _operator_why_blocked_summary(row),
                "what_operator_must_confirm": list(
                    CORRECTION_METHOD_REQUIRED_CONFIRMATIONS
                ),
                "required_confirmation_details": (
                    _required_confirmation_detail_summary(
                        CORRECTION_METHOD_REQUIRED_CONFIRMATIONS
                    )
                ),
                "operator_acceptance_gate_details": (
                    _operator_acceptance_gate_detail_summary(
                        CORRECTION_METHOD_REQUIRED_CONFIRMATIONS
                    )
                ),
                "required_operator_surfaces": list(
                    row["operator_required_surfaces"]
                ),
                "missing_required_operator_surfaces": list(
                    row["missing_required_operator_surfaces"]
                ),
                "operator_payload_required_fields": list(
                    row["operator_payload_required_fields"]
                ),
                "operator_evidence_required_fields": list(
                    row["operator_evidence_required_fields"]
                ),
                "required_evidence": list(row["safety_evidence_required"]),
                "required_safety_checks": list(row["applicable_safety_checks"]),
                "bounded_delta_unit": row["bounded_delta_unit"],
                "bounded_delta_policy": row["bounded_delta_policy"],
                "evidence_problem_correction_chain_details": list(
                    row["evidence_problem_correction_chain_details"]
                ),
                "rollback_expectation": row["rollback_expectation"],
                "future_wing_readback_targets": list(
                    row["future_wing_readback_targets"]
                ),
                "future_wing_readback_contract_summary": list(
                    row["future_wing_readback_contract_summary"]
                ),
                "future_live_required_proofs": list(
                    row["future_live_required_proofs"]
                ),
                "future_live_requirement_details": list(
                    row["future_live_requirement_details"]
                ),
                "blocked_live_actions": list(row["blocked_live_actions"]),
                "safety_blocker_details": list(row["safety_blocker_details"]),
                "headroom_feedback_true_peak_check_details": list(
                    row["headroom_feedback_true_peak_check_details"]
                ),
                "live_authority_boundary_summary": list(
                    row["live_authority_boundary_summary"]
                ),
                "operator_review_checklist": list(row["operator_review_checklist"]),
                "operator_review_checklist_details": list(
                    row["operator_review_checklist_details"]
                ),
                "operator_review_decisions": list(row["operator_review_decisions"]),
                "operator_review_decision_option_details": list(
                    row["operator_review_decision_option_details"]
                ),
                "operator_review_record_required_fields": list(
                    row["operator_review_record_required_fields"]
                ),
                "operator_review_record_field_details": list(
                    row["operator_review_record_field_details"]
                ),
                "operator_next_action_details": list(
                    row["operator_next_action_details"]
                ),
                "operator_next_step": row["operator_next_step"],
                "operator_visible_explanation_contract": {
                    "source_evidence_required": row["source_evidence_required"],
                    "problem_statement_required": row[
                        "measured_or_audible_problem_required"
                    ],
                    "proposed_correction_required": row[
                        "proposed_correction_required"
                    ],
                    "confidence_required": row["confidence_required"],
                    "bounded_delta_required": row["bounded_delta_required"],
                    "safety_blockers_required": row["safety_blockers_required"],
                    "safety_checks_required": row[
                        "headroom_feedback_true_peak_checks_required"
                    ],
                    "operator_explanation_required": row[
                        "operator_readable_explanation_required"
                    ],
                    "rollback_required": bool(row["rollback_expectation"]),
                    "future_wing_readback_required": bool(
                        row["future_wing_readback_targets"]
                    ),
                    "test_evidence_required": True,
                },
                **_packet_fail_closed_authority_fields(),
                "denial_reason": row["denial_reason"],
            }
        )

    cards_by_family: Dict[str, List[Dict[str, Any]]] = {}
    for card in method_cards:
        cards_by_family.setdefault(str(card["objective_family"]), []).append(card)

    operator_review_contract_fields = (
        "source_evidence",
        "source_evidence_details",
        "measured_or_audible_problem",
        "proposed_correction",
        "confidence",
        "bounded_delta",
        "safety_blockers",
        "headroom_feedback_true_peak_checks",
        "operator_readable_explanation",
        "rollback_readback_contract",
        "future_wing_readback_contract",
        "test_evidence",
    )

    priority_sections: List[Dict[str, Any]] = []
    for group in coverage["priority_groups"]:
        section_cards = [
            card
            for family in group["objective_families"]
            for card in cards_by_family.get(str(family), [])
        ]
        def _section_items(field: str) -> List[str]:
            return sorted(
                {
                    str(item)
                    for card in section_cards
                    for item in card.get(field, [])
                }
            )

        def _section_contract_items(field: str) -> List[str]:
            values: List[str] = []
            for card in section_cards:
                contract = card.get(field)
                if not isinstance(contract, dict):
                    continue
                source_fields = contract.get("source_fields")
                if isinstance(source_fields, list):
                    source_field_value = ",".join(
                        str(item) for item in source_fields
                    )
                else:
                    source_field_value = ""
                values.extend(
                    [
                        f"operator_action={contract.get('operator_action')}",
                        f"required={str(contract.get('required')).lower()}",
                        f"review_status={contract.get('review_status')}",
                        f"source_fields={source_field_value}",
                    ]
                )
            return sorted(set(values))

        def _section_explanation_contract_items() -> List[str]:
            values: List[str] = []
            for card in section_cards:
                contract = card.get("operator_visible_explanation_contract")
                if not isinstance(contract, dict):
                    continue
                values.extend(
                    f"{key}={str(value).lower()}"
                    for key, value in sorted(contract.items())
                )
            return sorted(set(values))

        missing_surface_method_ids = sorted(
            str(card["method_id"])
            for card in section_cards
            if card.get("missing_required_operator_surfaces")
        )
        live_authority_method_ids = sorted(
            str(card["method_id"])
            for card in section_cards
            if card.get("live_authority_granted")
        )
        live_apply_method_ids = sorted(
            str(card["method_id"])
            for card in section_cards
            if card.get("live_apply_allowed")
        )
        queueable_method_ids = sorted(
            str(card["method_id"])
            for card in section_cards
            if card.get("queue_import_allowed")
        )
        priority_sections.append(
            {
                "priority": group["priority"],
                "label": group["label"],
                "objective_families": list(group["objective_families"]),
                "covered": group["covered"],
                "method_count": len(section_cards),
                "review_status": (
                    "operator_review_required"
                    if section_cards
                    else "missing_proposal_contract"
                ),
                "methods": section_cards,
                "required_evidence": _section_items("required_evidence"),
                "required_safety_checks": _section_items("required_safety_checks"),
                "test_evidence_references": _section_items(
                    "test_evidence_references"
                ),
                **{
                    field: _section_contract_items(field)
                    for field in operator_review_contract_fields
                },
                "operator_payload_required_fields": _section_items(
                    "operator_payload_required_fields"
                ),
                "operator_evidence_required_fields": _section_items(
                    "operator_evidence_required_fields"
                ),
                "operator_problem_review_summary": _section_items(
                    "operator_problem_review_summary"
                ),
                "operator_method_review_focus": _section_items(
                    "operator_method_review_focus"
                ),
                "why_safe_summary": _section_items("why_safe_summary"),
                "why_blocked_summary": _section_items("why_blocked_summary"),
                "goal_requirement_details": _section_items(
                    "goal_requirement_details"
                ),
                "method_specific_review_details": _section_items(
                    "method_specific_review_details"
                ),
                "operator_actionable_explanation_details": _section_items(
                    "operator_actionable_explanation_details"
                ),
                "evidence_problem_correction_chain_details": _section_items(
                    "evidence_problem_correction_chain_details"
                ),
                "operator_visible_explanation_contract": (
                    _section_explanation_contract_items()
                ),
                "bounded_delta_policies": sorted(
                    {
                        str(card["bounded_delta_policy"])
                        for card in section_cards
                        if card.get("bounded_delta_policy")
                    }
                ),
                "rollback_expectations": sorted(
                    {
                        str(card["rollback_expectation"])
                        for card in section_cards
                        if card.get("rollback_expectation")
                    }
                ),
                "future_wing_readback_targets": _section_items(
                    "future_wing_readback_targets"
                ),
                "future_wing_readback_contract_summary": _section_items(
                    "future_wing_readback_contract_summary"
                ),
                "future_live_required_proofs": _section_items(
                    "future_live_required_proofs"
                ),
                "future_live_requirement_details": _section_items(
                    "future_live_requirement_details"
                ),
                "blocked_live_actions": _section_items("blocked_live_actions"),
                "safety_blocker_details": _section_items(
                    "safety_blocker_details"
                ),
                "headroom_feedback_true_peak_check_details": _section_items(
                    "headroom_feedback_true_peak_check_details"
                ),
                "live_authority_boundary_summary": _section_items(
                    "live_authority_boundary_summary"
                ),
            "operator_review_record_required_fields": _section_items(
                "operator_review_record_required_fields"
            ),
            "required_confirmation_details": _section_items(
                "required_confirmation_details"
            ),
            "operator_acceptance_gate_details": _section_items(
                "operator_acceptance_gate_details"
            ),
            "operator_review_decision_option_details": _section_items(
                "operator_review_decision_option_details"
            ),
            "operator_review_checklist_details": _section_items(
                "operator_review_checklist_details"
            ),
                "operator_review_record_field_details": _section_items(
                    "operator_review_record_field_details"
                ),
                "operator_next_action_details": _section_items(
                    "operator_next_action_details"
                ),
                "missing_required_operator_surface_method_ids": (
                    missing_surface_method_ids
                ),
                "live_authority_method_ids": live_authority_method_ids,
                "live_apply_method_ids": live_apply_method_ids,
                "queueable_method_ids": queueable_method_ids,
                "operator_next_step": (
                    "review_priority_group_offline_methods_and_collect_real_wing_"
                    "readback_before_live_promotion"
                ),
                **_packet_fail_closed_authority_fields(),
            }
        )

    supporting_method_cards = [
        card
        for card in method_cards
        if card["objective_family"] not in required_families
    ]
    operator_review_contract_coverage: Dict[str, Dict[str, Any]] = {}
    for field in operator_review_contract_fields:
        method_contracts = [
            card.get(field)
            for card in method_cards
            if isinstance(card.get(field), dict)
        ]
        required_method_contracts = [
            contract
            for contract in method_contracts
            if contract.get("required") is True
        ]
        priority_sections_with_contract = [
            section for section in priority_sections if section.get(field)
        ]
        operator_review_contract_coverage[field] = {
            "method_card_count": len(method_contracts),
            "required_method_card_count": len(required_method_contracts),
            "priority_section_count": len(priority_sections_with_contract),
            "priority_section_contract_item_count": sum(
                len(section.get(field) or []) for section in priority_sections
            ),
            "covered": (
                len(method_contracts) == len(method_cards)
                and len(required_method_contracts) == len(method_cards)
                and len(priority_sections_with_contract) == len(priority_sections)
            ),
        }

    return {
        "schema": "automixer.correction_method_goal_operator_review_packet.v1",
        "pipeline_id": pipeline_id,
        "status": (
            "operator_review_packet_ready_proposal_only"
            if coverage["covered"]
            else "operator_review_packet_incomplete"
        ),
        "coverage_schema": coverage["schema"],
        "coverage_status": coverage["status"],
        "covered": coverage["covered"],
        **_packet_fail_closed_authority_fields(),
        "required_operator_surfaces": list(
            coverage["operator_required_surfaces"]
        ),
        "required_objective_families": list(
            coverage["required_objective_families"]
        ),
        "priority_sections": priority_sections,
        "first_priority_review_summary": dict(
            coverage["first_priority_review_summary"]
        ),
        "second_priority_review_summary": dict(
            coverage["second_priority_review_summary"]
        ),
        "third_priority_review_summary": dict(
            coverage["third_priority_review_summary"]
        ),
        "fourth_priority_review_summary": dict(
            coverage["fourth_priority_review_summary"]
        ),
        "fifth_priority_review_summary": dict(
            coverage["fifth_priority_review_summary"]
        ),
        "sixth_priority_review_summary": dict(
            coverage["sixth_priority_review_summary"]
        ),
        "seventh_priority_review_summary": dict(
            coverage["seventh_priority_review_summary"]
        ),
        "supporting_method_cards": supporting_method_cards,
        "method_review_cards": method_cards,
        "summary": {
            "priority_section_count": len(priority_sections),
            "required_objective_family_count": coverage["summary"][
                "required_objective_family_count"
            ],
            "covered_objective_family_count": coverage["summary"][
                "covered_objective_family_count"
            ],
            "method_review_card_count": len(method_cards),
            "supporting_method_review_card_count": len(supporting_method_cards),
            "operator_review_contract_coverage": (
                operator_review_contract_coverage
            ),
            "missing_required_operator_surface_method_ids": coverage["summary"][
                "missing_required_operator_surface_method_ids"
            ],
            "first_priority_method_count": coverage["summary"][
                "first_priority_method_count"
            ],
            "first_priority_method_ids": list(
                coverage["summary"]["first_priority_method_ids"]
            ),
            "first_priority_applicable_safety_checks": list(
                coverage["summary"]["first_priority_applicable_safety_checks"]
            ),
            "first_priority_future_wing_readback_target_count": coverage[
                "summary"
            ]["first_priority_future_wing_readback_target_count"],
            "first_priority_blocked_live_action_count": coverage["summary"][
                "first_priority_blocked_live_action_count"
            ],
            "second_priority_method_count": coverage["summary"][
                "second_priority_method_count"
            ],
            "second_priority_method_ids": list(
                coverage["summary"]["second_priority_method_ids"]
            ),
            "second_priority_applicable_safety_checks": list(
                coverage["summary"]["second_priority_applicable_safety_checks"]
            ),
            "second_priority_future_wing_readback_target_count": coverage[
                "summary"
            ]["second_priority_future_wing_readback_target_count"],
            "second_priority_blocked_live_action_count": coverage["summary"][
                "second_priority_blocked_live_action_count"
            ],
            "third_priority_method_count": coverage["summary"][
                "third_priority_method_count"
            ],
            "third_priority_method_ids": list(
                coverage["summary"]["third_priority_method_ids"]
            ),
            "third_priority_applicable_safety_checks": list(
                coverage["summary"]["third_priority_applicable_safety_checks"]
            ),
            "third_priority_future_wing_readback_target_count": coverage[
                "summary"
            ]["third_priority_future_wing_readback_target_count"],
            "third_priority_blocked_live_action_count": coverage["summary"][
                "third_priority_blocked_live_action_count"
            ],
            "fourth_priority_method_count": coverage["summary"][
                "fourth_priority_method_count"
            ],
            "fourth_priority_method_ids": list(
                coverage["summary"]["fourth_priority_method_ids"]
            ),
            "fourth_priority_applicable_safety_checks": list(
                coverage["summary"]["fourth_priority_applicable_safety_checks"]
            ),
            "fourth_priority_future_wing_readback_target_count": coverage[
                "summary"
            ]["fourth_priority_future_wing_readback_target_count"],
            "fourth_priority_blocked_live_action_count": coverage["summary"][
                "fourth_priority_blocked_live_action_count"
            ],
            "fifth_priority_method_count": coverage["summary"][
                "fifth_priority_method_count"
            ],
            "fifth_priority_method_ids": list(
                coverage["summary"]["fifth_priority_method_ids"]
            ),
            "fifth_priority_applicable_safety_checks": list(
                coverage["summary"]["fifth_priority_applicable_safety_checks"]
            ),
            "fifth_priority_future_wing_readback_target_count": coverage[
                "summary"
            ]["fifth_priority_future_wing_readback_target_count"],
            "fifth_priority_blocked_live_action_count": coverage["summary"][
                "fifth_priority_blocked_live_action_count"
            ],
            "sixth_priority_method_count": coverage["summary"][
                "sixth_priority_method_count"
            ],
            "sixth_priority_method_ids": list(
                coverage["summary"]["sixth_priority_method_ids"]
            ),
            "sixth_priority_applicable_safety_checks": list(
                coverage["summary"]["sixth_priority_applicable_safety_checks"]
            ),
            "sixth_priority_future_wing_readback_target_count": coverage[
                "summary"
            ]["sixth_priority_future_wing_readback_target_count"],
            "sixth_priority_blocked_live_action_count": coverage["summary"][
                "sixth_priority_blocked_live_action_count"
            ],
            "seventh_priority_method_count": coverage["summary"][
                "seventh_priority_method_count"
            ],
            "seventh_priority_method_ids": list(
                coverage["summary"]["seventh_priority_method_ids"]
            ),
            "seventh_priority_applicable_safety_checks": list(
                coverage["summary"]["seventh_priority_applicable_safety_checks"]
            ),
            "seventh_priority_future_wing_readback_target_count": coverage[
                "summary"
            ]["seventh_priority_future_wing_readback_target_count"],
            "seventh_priority_blocked_live_action_count": coverage["summary"][
                "seventh_priority_blocked_live_action_count"
            ],
            "live_authority_method_ids": coverage["summary"][
                "live_authority_method_ids"
            ],
            "live_apply_method_ids": coverage["summary"]["live_apply_method_ids"],
            "queueable_method_ids": coverage["summary"]["queueable_method_ids"],
            "blocked_by": blocked_by,
        },
        "denial_reason": "operator_review_packet_cannot_replace_real_wing_proof",
        "operator_next_step": "review_offline_method_cards_and_collect_real_wing_readback_before_live_promotion",
    }


def render_correction_method_goal_operator_review_markdown(
    *, pipeline_id: str = MAIN_PIPELINE_ID
) -> str:
    """Render the correction-method review packet as operator-readable Markdown.

    This is a presentation layer over the proposal-only packet. It does not
    contact WING/backend, enqueue proposals, mutate mixer state, or grant live
    authority.
    """

    packet = build_correction_method_goal_operator_review_packet(
        pipeline_id=pipeline_id
    )

    def _code(value: Any) -> str:
        text = "none" if value in (None, "") else str(value)
        escaped = text.replace("`", "\\`")
        return f"`{escaped}`"

    def _bool(value: Any) -> str:
        return _code("true" if value is True else "false")

    def _items(values: Any) -> str:
        if not values:
            return "`none`"
        return ", ".join(_code(value) for value in values)

    def _section_items(section: Dict[str, Any], key: str) -> List[str]:
        return sorted(
            {
                str(value)
                for card in section.get("methods", [])
                for value in (card.get(key) or [])
            }
        )

    def _authority_lines(payload: Dict[str, Any]) -> List[str]:
        return [
            (
                "- read_only_or_dry_run_only: "
                f"{_bool(payload.get('read_only_or_dry_run_only'))}"
            ),
            (
                "- operator_review_required: "
                f"{_bool(payload.get('operator_review_required'))}"
            ),
            (
                "- future_wing_readback_required: "
                f"{_bool(payload.get('future_wing_readback_required'))}"
            ),
            (
                "- hardware_proof_required_before_live: "
                f"{_bool(payload.get('hardware_proof_required_before_live'))}"
            ),
            (
                "- blocks_live_promotion: "
                f"{_bool(payload.get('blocks_live_promotion'))}"
            ),
            (
                "- blocks_queue_import: "
                f"{_bool(payload.get('blocks_queue_import'))}"
            ),
            (
                "- live_authority_granted: "
                f"{_bool(payload.get('live_authority_granted'))}"
            ),
            (
                "- full_autonomous_ready: "
                f"{_bool(payload.get('full_autonomous_ready'))}"
            ),
            (
                "- readiness_closure_allowed: "
                f"{_bool(payload.get('readiness_closure_allowed'))}"
            ),
            f"- live_apply_allowed: {_bool(payload.get('live_apply_allowed'))}",
            f"- queue_import_allowed: {_bool(payload.get('queue_import_allowed'))}",
            f"- can_apply_live: {_bool(payload.get('can_apply_live'))}",
            f"- can_import_to_queue: {_bool(payload.get('can_import_to_queue'))}",
            f"- can_authorize_live: {_bool(payload.get('can_authorize_live'))}",
            f"- can_close_readiness: {_bool(payload.get('can_close_readiness'))}",
            f"- queue_action: {_code(payload.get('queue_action'))}",
            f"- contacts_wing: {_bool(payload.get('contacts_wing'))}",
            f"- contacts_backend: {_bool(payload.get('contacts_backend'))}",
            f"- mutates_mixer: {_bool(payload.get('mutates_mixer'))}",
        ]

    lines = [
        "# Automixer Correction Method Operator Review Packet",
        "",
        f"- schema: {_code(packet['schema'])}",
        f"- status: {_code(packet['status'])}",
        f"- coverage_status: {_code(packet['coverage_status'])}",
        *_authority_lines(packet),
        f"- denial_reason: {_code(packet['denial_reason'])}",
        "",
        "## Operator Next Step",
        str(packet["operator_next_step"]),
        "",
        "## Summary",
    ]

    for key in (
        "priority_section_count",
        "required_objective_family_count",
        "covered_objective_family_count",
        "method_review_card_count",
        "supporting_method_review_card_count",
        "missing_required_operator_surface_method_ids",
        "first_priority_method_count",
        "first_priority_method_ids",
        "first_priority_applicable_safety_checks",
        "first_priority_future_wing_readback_target_count",
        "first_priority_blocked_live_action_count",
        "second_priority_method_count",
        "second_priority_method_ids",
        "second_priority_applicable_safety_checks",
        "second_priority_future_wing_readback_target_count",
        "second_priority_blocked_live_action_count",
        "third_priority_method_count",
        "third_priority_method_ids",
        "third_priority_applicable_safety_checks",
        "third_priority_future_wing_readback_target_count",
        "third_priority_blocked_live_action_count",
        "fourth_priority_method_count",
        "fourth_priority_method_ids",
        "fourth_priority_applicable_safety_checks",
        "fourth_priority_future_wing_readback_target_count",
        "fourth_priority_blocked_live_action_count",
        "fifth_priority_method_count",
        "fifth_priority_method_ids",
        "fifth_priority_applicable_safety_checks",
        "fifth_priority_future_wing_readback_target_count",
        "fifth_priority_blocked_live_action_count",
        "sixth_priority_method_count",
        "sixth_priority_method_ids",
        "sixth_priority_applicable_safety_checks",
        "sixth_priority_future_wing_readback_target_count",
        "sixth_priority_blocked_live_action_count",
        "seventh_priority_method_count",
        "seventh_priority_method_ids",
        "seventh_priority_applicable_safety_checks",
        "seventh_priority_future_wing_readback_target_count",
        "seventh_priority_blocked_live_action_count",
        "live_authority_method_ids",
        "live_apply_method_ids",
        "queueable_method_ids",
        "blocked_by",
    ):
        lines.append(f"- {key}: {_code(packet['summary'][key])}")

    lines.extend(
        [
            "",
            "## Operator Decision Workflow",
            (
                "- completion_requires: "
                "`review_packet_json`, `operator_review_markdown`, "
                "`verification_report_passed`, "
                "`explicit_offline_decision_per_priority_section`, "
                "`explicit_offline_decision_per_method`, "
                "`future_real_wing_readback_before_live_promotion`"
            ),
            "- workflow_scope: `offline_proposal_only`",
            (
                "- required_priority_operator_decision: "
                "`one_offline_decision_per_priority_section`"
            ),
            "- required_operator_decisions: `one_offline_decision_per_method`",
            *_authority_lines(packet),
        ]
    )
    for section in packet["priority_sections"]:
        lines.extend(
            [
                "",
                f"### Decision Section {section['priority']}: {section['label']}",
                "- review_scope: `offline_proposal_only`",
                "- review_status: `operator_review_pending`",
                (
                    "- required_priority_operator_decision: "
                    "`one_offline_decision_per_priority_section`"
                ),
                "- required_operator_decisions: `one_offline_decision_per_method`",
                f"- method_ids: {_items(card['method_id'] for card in section['methods'])}",
                (
                    "- allowed_decisions: "
                    f"{_items(_section_items(section, 'operator_review_decisions'))}"
                ),
                f"- required_evidence: {_items(section['required_evidence'])}",
                (
                    "- required_safety_checks: "
                    f"{_items(section['required_safety_checks'])}"
                ),
                (
                    "- test_evidence_references: "
                    f"{_items(section['test_evidence_references'])}"
                ),
                (
                    "- expected_evidence_packet_reference: "
                    f"{_code('packet.json#priority_sections/' + str(section['priority']))}"
                ),
                (
                    "- expected_source_material_reference: "
                    f"{_code('packet.json#priority_sections/' + str(section['priority']) + '/source_evidence')}"
                ),
                (
                    "- expected_test_evidence_references: "
                    f"{_items(section['test_evidence_references'])}"
                ),
                (
                    "- expected_verification_report_reference: "
                    f"{_code('verification-report.json')}"
                ),
                (
                    "- source_evidence: "
                    f"{_items(section['source_evidence'])}"
                ),
                (
                    "- source_evidence_details: "
                    f"{_items(section['source_evidence_details'])}"
                ),
                (
                    "- measured_or_audible_problem: "
                    f"{_items(section['measured_or_audible_problem'])}"
                ),
                (
                    "- proposed_correction: "
                    f"{_items(section['proposed_correction'])}"
                ),
                (
                    "- goal_requirement_details: "
                    f"{_items(section['goal_requirement_details'])}"
                ),
                (
                    "- method_specific_review_details: "
                    f"{_items(section['method_specific_review_details'])}"
                ),
                (
                    "- operator_actionable_explanation_details: "
                    f"{_items(section['operator_actionable_explanation_details'])}"
                ),
                (
                    "- evidence_problem_correction_chain_details: "
                    f"{_items(section['evidence_problem_correction_chain_details'])}"
                ),
                f"- confidence: {_items(section['confidence'])}",
                (
                    "- bounded_delta_contract: "
                    f"{_items(section['bounded_delta'])}"
                ),
                f"- safety_blockers: {_items(section['safety_blockers'])}",
                (
                    "- safety_blocker_details: "
                    f"{_items(section['safety_blocker_details'])}"
                ),
                (
                    "- headroom_feedback_true_peak_checks: "
                    f"{_items(section['headroom_feedback_true_peak_checks'])}"
                ),
                (
                    "- headroom_feedback_true_peak_check_details: "
                    f"{_items(section['headroom_feedback_true_peak_check_details'])}"
                ),
                (
                    "- operator_readable_explanation: "
                    f"{_items(section['operator_readable_explanation'])}"
                ),
                (
                    "- operator_visible_explanation_contract: "
                    f"{_items(section['operator_visible_explanation_contract'])}"
                ),
                (
                    "- operator_payload_required_fields: "
                    f"{_items(section['operator_payload_required_fields'])}"
                ),
                (
                    "- operator_evidence_required_fields: "
                    f"{_items(section['operator_evidence_required_fields'])}"
                ),
                (
                    "- operator_problem_review_summary: "
                    f"{_items(section['operator_problem_review_summary'])}"
                ),
                (
                    "- operator_method_review_focus: "
                    f"{_items(section['operator_method_review_focus'])}"
                ),
                (
                    "- why_safe_summary: "
                    f"{_items(section['why_safe_summary'])}"
                ),
                (
                    "- why_blocked_summary: "
                    f"{_items(section['why_blocked_summary'])}"
                ),
                (
                    "- bounded_delta_policies: "
                    f"{_items(section['bounded_delta_policies'])}"
                ),
                (
                    "- rollback_expectations: "
                    f"{_items(section['rollback_expectations'])}"
                ),
                (
                    "- rollback_readback_contract: "
                    f"{_items(section['rollback_readback_contract'])}"
                ),
                (
                    "- future_wing_readback_targets: "
                    f"{_items(section['future_wing_readback_targets'])}"
                ),
                (
                    "- future_wing_readback_contract: "
                    f"{_items(section['future_wing_readback_contract'])}"
                ),
                f"- test_evidence: {_items(section['test_evidence'])}",
                (
                    "- future_wing_readback_contract_summary: "
                    f"{_items(section['future_wing_readback_contract_summary'])}"
                ),
                (
                    "- future_live_required_proofs: "
                    f"{_items(section['future_live_required_proofs'])}"
                ),
                (
                    "- future_live_requirement_details: "
                    f"{_items(section['future_live_requirement_details'])}"
                ),
                (
                    "- blocked_live_actions: "
                    f"{_items(section['blocked_live_actions'])}"
                ),
                (
                    "- live_authority_boundary_summary: "
                    f"{_items(section['live_authority_boundary_summary'])}"
                ),
                (
                    "- operator_review_record_required_fields: "
                    f"{_items(section['operator_review_record_required_fields'])}"
                ),
                (
                    "- required_confirmation_details: "
                    f"{_items(section['required_confirmation_details'])}"
                ),
                (
                    "- operator_acceptance_gate_details: "
                    f"{_items(section['operator_acceptance_gate_details'])}"
                ),
                (
                    "- operator_review_decision_option_details: "
                    f"{_items(section['operator_review_decision_option_details'])}"
                ),
                (
                    "- operator_review_checklist_details: "
                    f"{_items(section['operator_review_checklist_details'])}"
                ),
                (
                    "- operator_review_record_field_details: "
                    f"{_items(section['operator_review_record_field_details'])}"
                ),
                (
                    "- operator_next_action_details: "
                    f"{_items(section['operator_next_action_details'])}"
                ),
                (
                    "- missing_required_operator_surface_method_ids: "
                    f"{_items(section['missing_required_operator_surface_method_ids'])}"
                ),
                (
                    "- live_authority_method_ids: "
                    f"{_items(section['live_authority_method_ids'])}"
                ),
                f"- live_apply_method_ids: {_items(section['live_apply_method_ids'])}",
                f"- queueable_method_ids: {_items(section['queueable_method_ids'])}",
                f"- operator_next_step: {_code(section['operator_next_step'])}",
                *_authority_lines(section),
            ]
        )

    first_priority = packet["first_priority_review_summary"]
    lines.extend(
        [
            "",
            "## First Priority Review",
            f"- label: {_code(first_priority['label'])}",
            f"- objective_families: {_items(first_priority['objective_families'])}",
            f"- method_ids: {_items(first_priority['method_ids'])}",
            (
                "- applicable_safety_checks: "
                f"{_items(first_priority['applicable_safety_checks'])}"
            ),
            (
                "- future_wing_readback_targets: "
                f"{_items(first_priority['future_wing_readback_targets'])}"
            ),
            (
                "- blocked_live_actions: "
                f"{_items(first_priority['blocked_live_actions'])}"
            ),
            f"- read_only_or_dry_run_only: {_bool(first_priority['read_only_or_dry_run_only'])}",
            f"- operator_review_required: {_bool(first_priority['operator_review_required'])}",
            f"- future_wing_readback_required: {_bool(first_priority['future_wing_readback_required'])}",
            f"- hardware_proof_required_before_live: {_bool(first_priority['hardware_proof_required_before_live'])}",
            f"- blocks_live_promotion: {_bool(first_priority['blocks_live_promotion'])}",
            f"- blocks_queue_import: {_bool(first_priority['blocks_queue_import'])}",
            f"- live_authority_granted: {_bool(first_priority['live_authority_granted'])}",
            f"- full_autonomous_ready: {_bool(first_priority['full_autonomous_ready'])}",
            f"- readiness_closure_allowed: {_bool(first_priority['readiness_closure_allowed'])}",
            f"- live_apply_allowed: {_bool(first_priority['live_apply_allowed'])}",
            f"- queue_import_allowed: {_bool(first_priority['queue_import_allowed'])}",
            f"- can_apply_live: {_bool(first_priority['can_apply_live'])}",
            f"- can_import_to_queue: {_bool(first_priority['can_import_to_queue'])}",
            f"- can_authorize_live: {_bool(first_priority['can_authorize_live'])}",
            f"- can_close_readiness: {_bool(first_priority['can_close_readiness'])}",
            f"- queue_action: {_code(first_priority['queue_action'])}",
            f"- contacts_wing: {_bool(first_priority['contacts_wing'])}",
            f"- contacts_backend: {_bool(first_priority['contacts_backend'])}",
            f"- mutates_mixer: {_bool(first_priority['mutates_mixer'])}",
            f"- operator_next_step: {_code(first_priority['operator_next_step'])}",
        ]
    )

    second_priority = packet["second_priority_review_summary"]
    lines.extend(
        [
            "",
            "## Second Priority Review",
            f"- label: {_code(second_priority['label'])}",
            f"- objective_families: {_items(second_priority['objective_families'])}",
            f"- method_ids: {_items(second_priority['method_ids'])}",
            (
                "- applicable_safety_checks: "
                f"{_items(second_priority['applicable_safety_checks'])}"
            ),
            (
                "- future_wing_readback_targets: "
                f"{_items(second_priority['future_wing_readback_targets'])}"
            ),
            (
                "- blocked_live_actions: "
                f"{_items(second_priority['blocked_live_actions'])}"
            ),
            f"- read_only_or_dry_run_only: {_bool(second_priority['read_only_or_dry_run_only'])}",
            f"- operator_review_required: {_bool(second_priority['operator_review_required'])}",
            f"- future_wing_readback_required: {_bool(second_priority['future_wing_readback_required'])}",
            f"- hardware_proof_required_before_live: {_bool(second_priority['hardware_proof_required_before_live'])}",
            f"- blocks_live_promotion: {_bool(second_priority['blocks_live_promotion'])}",
            f"- blocks_queue_import: {_bool(second_priority['blocks_queue_import'])}",
            f"- live_authority_granted: {_bool(second_priority['live_authority_granted'])}",
            f"- full_autonomous_ready: {_bool(second_priority['full_autonomous_ready'])}",
            f"- readiness_closure_allowed: {_bool(second_priority['readiness_closure_allowed'])}",
            f"- live_apply_allowed: {_bool(second_priority['live_apply_allowed'])}",
            f"- queue_import_allowed: {_bool(second_priority['queue_import_allowed'])}",
            f"- can_apply_live: {_bool(second_priority['can_apply_live'])}",
            f"- can_import_to_queue: {_bool(second_priority['can_import_to_queue'])}",
            f"- can_authorize_live: {_bool(second_priority['can_authorize_live'])}",
            f"- can_close_readiness: {_bool(second_priority['can_close_readiness'])}",
            f"- queue_action: {_code(second_priority['queue_action'])}",
            f"- contacts_wing: {_bool(second_priority['contacts_wing'])}",
            f"- contacts_backend: {_bool(second_priority['contacts_backend'])}",
            f"- mutates_mixer: {_bool(second_priority['mutates_mixer'])}",
            f"- operator_next_step: {_code(second_priority['operator_next_step'])}",
        ]
    )

    third_priority = packet["third_priority_review_summary"]
    lines.extend(
        [
            "",
            "## Third Priority Review",
            f"- label: {_code(third_priority['label'])}",
            f"- objective_families: {_items(third_priority['objective_families'])}",
            f"- method_ids: {_items(third_priority['method_ids'])}",
            (
                "- applicable_safety_checks: "
                f"{_items(third_priority['applicable_safety_checks'])}"
            ),
            (
                "- future_wing_readback_targets: "
                f"{_items(third_priority['future_wing_readback_targets'])}"
            ),
            (
                "- blocked_live_actions: "
                f"{_items(third_priority['blocked_live_actions'])}"
            ),
            f"- read_only_or_dry_run_only: {_bool(third_priority['read_only_or_dry_run_only'])}",
            f"- operator_review_required: {_bool(third_priority['operator_review_required'])}",
            f"- future_wing_readback_required: {_bool(third_priority['future_wing_readback_required'])}",
            f"- hardware_proof_required_before_live: {_bool(third_priority['hardware_proof_required_before_live'])}",
            f"- blocks_live_promotion: {_bool(third_priority['blocks_live_promotion'])}",
            f"- blocks_queue_import: {_bool(third_priority['blocks_queue_import'])}",
            f"- live_authority_granted: {_bool(third_priority['live_authority_granted'])}",
            f"- full_autonomous_ready: {_bool(third_priority['full_autonomous_ready'])}",
            f"- readiness_closure_allowed: {_bool(third_priority['readiness_closure_allowed'])}",
            f"- live_apply_allowed: {_bool(third_priority['live_apply_allowed'])}",
            f"- queue_import_allowed: {_bool(third_priority['queue_import_allowed'])}",
            f"- can_apply_live: {_bool(third_priority['can_apply_live'])}",
            f"- can_import_to_queue: {_bool(third_priority['can_import_to_queue'])}",
            f"- can_authorize_live: {_bool(third_priority['can_authorize_live'])}",
            f"- can_close_readiness: {_bool(third_priority['can_close_readiness'])}",
            f"- queue_action: {_code(third_priority['queue_action'])}",
            f"- contacts_wing: {_bool(third_priority['contacts_wing'])}",
            f"- contacts_backend: {_bool(third_priority['contacts_backend'])}",
            f"- mutates_mixer: {_bool(third_priority['mutates_mixer'])}",
            f"- operator_next_step: {_code(third_priority['operator_next_step'])}",
        ]
    )

    fourth_priority = packet["fourth_priority_review_summary"]
    lines.extend(
        [
            "",
            "## Fourth Priority Review",
            f"- label: {_code(fourth_priority['label'])}",
            f"- objective_families: {_items(fourth_priority['objective_families'])}",
            f"- method_ids: {_items(fourth_priority['method_ids'])}",
            (
                "- applicable_safety_checks: "
                f"{_items(fourth_priority['applicable_safety_checks'])}"
            ),
            (
                "- future_wing_readback_targets: "
                f"{_items(fourth_priority['future_wing_readback_targets'])}"
            ),
            (
                "- blocked_live_actions: "
                f"{_items(fourth_priority['blocked_live_actions'])}"
            ),
            f"- read_only_or_dry_run_only: {_bool(fourth_priority['read_only_or_dry_run_only'])}",
            f"- operator_review_required: {_bool(fourth_priority['operator_review_required'])}",
            f"- future_wing_readback_required: {_bool(fourth_priority['future_wing_readback_required'])}",
            f"- hardware_proof_required_before_live: {_bool(fourth_priority['hardware_proof_required_before_live'])}",
            f"- blocks_live_promotion: {_bool(fourth_priority['blocks_live_promotion'])}",
            f"- blocks_queue_import: {_bool(fourth_priority['blocks_queue_import'])}",
            f"- live_authority_granted: {_bool(fourth_priority['live_authority_granted'])}",
            f"- full_autonomous_ready: {_bool(fourth_priority['full_autonomous_ready'])}",
            f"- readiness_closure_allowed: {_bool(fourth_priority['readiness_closure_allowed'])}",
            f"- live_apply_allowed: {_bool(fourth_priority['live_apply_allowed'])}",
            f"- queue_import_allowed: {_bool(fourth_priority['queue_import_allowed'])}",
            f"- can_apply_live: {_bool(fourth_priority['can_apply_live'])}",
            f"- can_import_to_queue: {_bool(fourth_priority['can_import_to_queue'])}",
            f"- can_authorize_live: {_bool(fourth_priority['can_authorize_live'])}",
            f"- can_close_readiness: {_bool(fourth_priority['can_close_readiness'])}",
            f"- queue_action: {_code(fourth_priority['queue_action'])}",
            f"- contacts_wing: {_bool(fourth_priority['contacts_wing'])}",
            f"- contacts_backend: {_bool(fourth_priority['contacts_backend'])}",
            f"- mutates_mixer: {_bool(fourth_priority['mutates_mixer'])}",
            f"- operator_next_step: {_code(fourth_priority['operator_next_step'])}",
        ]
    )

    fifth_priority = packet["fifth_priority_review_summary"]
    lines.extend(
        [
            "",
            "## Fifth Priority Review",
            f"- label: {_code(fifth_priority['label'])}",
            f"- objective_families: {_items(fifth_priority['objective_families'])}",
            f"- method_ids: {_items(fifth_priority['method_ids'])}",
            (
                "- applicable_safety_checks: "
                f"{_items(fifth_priority['applicable_safety_checks'])}"
            ),
            (
                "- future_wing_readback_targets: "
                f"{_items(fifth_priority['future_wing_readback_targets'])}"
            ),
            (
                "- blocked_live_actions: "
                f"{_items(fifth_priority['blocked_live_actions'])}"
            ),
            f"- read_only_or_dry_run_only: {_bool(fifth_priority['read_only_or_dry_run_only'])}",
            f"- operator_review_required: {_bool(fifth_priority['operator_review_required'])}",
            f"- future_wing_readback_required: {_bool(fifth_priority['future_wing_readback_required'])}",
            f"- hardware_proof_required_before_live: {_bool(fifth_priority['hardware_proof_required_before_live'])}",
            f"- blocks_live_promotion: {_bool(fifth_priority['blocks_live_promotion'])}",
            f"- blocks_queue_import: {_bool(fifth_priority['blocks_queue_import'])}",
            f"- live_authority_granted: {_bool(fifth_priority['live_authority_granted'])}",
            f"- full_autonomous_ready: {_bool(fifth_priority['full_autonomous_ready'])}",
            f"- readiness_closure_allowed: {_bool(fifth_priority['readiness_closure_allowed'])}",
            f"- live_apply_allowed: {_bool(fifth_priority['live_apply_allowed'])}",
            f"- queue_import_allowed: {_bool(fifth_priority['queue_import_allowed'])}",
            f"- can_apply_live: {_bool(fifth_priority['can_apply_live'])}",
            f"- can_import_to_queue: {_bool(fifth_priority['can_import_to_queue'])}",
            f"- can_authorize_live: {_bool(fifth_priority['can_authorize_live'])}",
            f"- can_close_readiness: {_bool(fifth_priority['can_close_readiness'])}",
            f"- queue_action: {_code(fifth_priority['queue_action'])}",
            f"- contacts_wing: {_bool(fifth_priority['contacts_wing'])}",
            f"- contacts_backend: {_bool(fifth_priority['contacts_backend'])}",
            f"- mutates_mixer: {_bool(fifth_priority['mutates_mixer'])}",
            f"- operator_next_step: {_code(fifth_priority['operator_next_step'])}",
        ]
    )

    sixth_priority = packet["sixth_priority_review_summary"]
    lines.extend(
        [
            "",
            "## Sixth Priority Review",
            f"- label: {_code(sixth_priority['label'])}",
            f"- objective_families: {_items(sixth_priority['objective_families'])}",
            f"- method_ids: {_items(sixth_priority['method_ids'])}",
            (
                "- applicable_safety_checks: "
                f"{_items(sixth_priority['applicable_safety_checks'])}"
            ),
            (
                "- future_wing_readback_targets: "
                f"{_items(sixth_priority['future_wing_readback_targets'])}"
            ),
            (
                "- blocked_live_actions: "
                f"{_items(sixth_priority['blocked_live_actions'])}"
            ),
            f"- read_only_or_dry_run_only: {_bool(sixth_priority['read_only_or_dry_run_only'])}",
            f"- operator_review_required: {_bool(sixth_priority['operator_review_required'])}",
            f"- future_wing_readback_required: {_bool(sixth_priority['future_wing_readback_required'])}",
            f"- hardware_proof_required_before_live: {_bool(sixth_priority['hardware_proof_required_before_live'])}",
            f"- blocks_live_promotion: {_bool(sixth_priority['blocks_live_promotion'])}",
            f"- blocks_queue_import: {_bool(sixth_priority['blocks_queue_import'])}",
            f"- live_authority_granted: {_bool(sixth_priority['live_authority_granted'])}",
            f"- full_autonomous_ready: {_bool(sixth_priority['full_autonomous_ready'])}",
            f"- readiness_closure_allowed: {_bool(sixth_priority['readiness_closure_allowed'])}",
            f"- live_apply_allowed: {_bool(sixth_priority['live_apply_allowed'])}",
            f"- queue_import_allowed: {_bool(sixth_priority['queue_import_allowed'])}",
            f"- can_apply_live: {_bool(sixth_priority['can_apply_live'])}",
            f"- can_import_to_queue: {_bool(sixth_priority['can_import_to_queue'])}",
            f"- can_authorize_live: {_bool(sixth_priority['can_authorize_live'])}",
            f"- can_close_readiness: {_bool(sixth_priority['can_close_readiness'])}",
            f"- queue_action: {_code(sixth_priority['queue_action'])}",
            f"- contacts_wing: {_bool(sixth_priority['contacts_wing'])}",
            f"- contacts_backend: {_bool(sixth_priority['contacts_backend'])}",
            f"- mutates_mixer: {_bool(sixth_priority['mutates_mixer'])}",
            f"- operator_next_step: {_code(sixth_priority['operator_next_step'])}",
        ]
    )

    seventh_priority = packet["seventh_priority_review_summary"]
    lines.extend(
        [
            "",
            "## Seventh Priority Review",
            f"- label: {_code(seventh_priority['label'])}",
            f"- objective_families: {_items(seventh_priority['objective_families'])}",
            f"- method_ids: {_items(seventh_priority['method_ids'])}",
            (
                "- applicable_safety_checks: "
                f"{_items(seventh_priority['applicable_safety_checks'])}"
            ),
            (
                "- future_wing_readback_targets: "
                f"{_items(seventh_priority['future_wing_readback_targets'])}"
            ),
            (
                "- blocked_live_actions: "
                f"{_items(seventh_priority['blocked_live_actions'])}"
            ),
            f"- read_only_or_dry_run_only: {_bool(seventh_priority['read_only_or_dry_run_only'])}",
            f"- operator_review_required: {_bool(seventh_priority['operator_review_required'])}",
            f"- future_wing_readback_required: {_bool(seventh_priority['future_wing_readback_required'])}",
            f"- hardware_proof_required_before_live: {_bool(seventh_priority['hardware_proof_required_before_live'])}",
            f"- blocks_live_promotion: {_bool(seventh_priority['blocks_live_promotion'])}",
            f"- blocks_queue_import: {_bool(seventh_priority['blocks_queue_import'])}",
            f"- live_authority_granted: {_bool(seventh_priority['live_authority_granted'])}",
            f"- full_autonomous_ready: {_bool(seventh_priority['full_autonomous_ready'])}",
            f"- readiness_closure_allowed: {_bool(seventh_priority['readiness_closure_allowed'])}",
            f"- live_apply_allowed: {_bool(seventh_priority['live_apply_allowed'])}",
            f"- queue_import_allowed: {_bool(seventh_priority['queue_import_allowed'])}",
            f"- can_apply_live: {_bool(seventh_priority['can_apply_live'])}",
            f"- can_import_to_queue: {_bool(seventh_priority['can_import_to_queue'])}",
            f"- can_authorize_live: {_bool(seventh_priority['can_authorize_live'])}",
            f"- can_close_readiness: {_bool(seventh_priority['can_close_readiness'])}",
            f"- queue_action: {_code(seventh_priority['queue_action'])}",
            f"- contacts_wing: {_bool(seventh_priority['contacts_wing'])}",
            f"- contacts_backend: {_bool(seventh_priority['contacts_backend'])}",
            f"- mutates_mixer: {_bool(seventh_priority['mutates_mixer'])}",
            f"- operator_next_step: {_code(seventh_priority['operator_next_step'])}",
        ]
    )

    lines.extend(["", "## Required Operator Surfaces"])
    lines.extend(
        f"- {_code(surface)}" for surface in packet["required_operator_surfaces"]
    )

    def _explanation_contract_items(card: Dict[str, Any]) -> List[str]:
        contract = card.get("operator_visible_explanation_contract")
        if not isinstance(contract, dict):
            return []
        return [
            f"{key}={str(value).lower()}"
            for key, value in sorted(contract.items())
        ]

    def _operator_review_contract_items(card: Dict[str, Any], field: str) -> List[str]:
        contract = card.get(field)
        if not isinstance(contract, dict):
            return []
        source_fields = contract.get("source_fields")
        if isinstance(source_fields, list):
            source_field_value = ",".join(str(item) for item in source_fields)
        else:
            source_field_value = ""
        return [
            f"operator_action={contract.get('operator_action')}",
            f"required={str(contract.get('required')).lower()}",
            f"review_status={contract.get('review_status')}",
            f"source_fields={source_field_value}",
        ]

    def _append_card(card: Dict[str, Any]) -> None:
        lines.extend(
            [
                "",
                f"#### {card['method_label']} ({_code(card['method_id'])})",
                f"- review_scope: {_code(card['review_scope'])}",
                f"- status: {_code(card['status'])}",
                f"- review_status: {_code(card['review_status'])}",
                f"- domain: {_code(card['domain'])}",
                f"- objective_family: {_code(card['objective_family'])}",
                f"- priority: {_code(card.get('priority'))}",
                f"- priority_label: {_code(card['priority_label'])}",
                f"- priority_review_scope: {_code(card['priority_review_scope'])}",
                f"- operator_question: {_code(card['operator_question'])}",
                f"- denial_reason: {_code(card['denial_reason'])}",
                (
                    "- source_evidence: "
                    f"{_items(_operator_review_contract_items(card, 'source_evidence'))}"
                ),
                (
                    "- source_evidence_details: "
                    f"{_items(_operator_review_contract_items(card, 'source_evidence_details'))}"
                ),
                (
                    "- measured_or_audible_problem: "
                    f"{_items(_operator_review_contract_items(card, 'measured_or_audible_problem'))}"
                ),
                (
                    "- proposed_correction: "
                    f"{_items(_operator_review_contract_items(card, 'proposed_correction'))}"
                ),
                (
                    "- goal_requirement_details: "
                    f"{_items(card['goal_requirement_details'])}"
                ),
                (
                    "- method_specific_review_details: "
                    f"{_items(card['method_specific_review_details'])}"
                ),
                (
                    "- operator_actionable_explanation_details: "
                    f"{_items(card['operator_actionable_explanation_details'])}"
                ),
                (
                    "- evidence_problem_correction_chain_details: "
                    f"{_items(card['evidence_problem_correction_chain_details'])}"
                ),
                (
                    "- operator_problem_review_summary: "
                    f"{_items(card['operator_problem_review_summary'])}"
                ),
                (
                    "- operator_method_review_focus: "
                    f"{_items(card['operator_method_review_focus'])}"
                ),
                (
                    "- why_safe_summary: "
                    f"{_items(card['why_safe_summary'])}"
                ),
                (
                    "- why_blocked_summary: "
                    f"{_items(card['why_blocked_summary'])}"
                ),
                (
                    "- confidence: "
                    f"{_items(_operator_review_contract_items(card, 'confidence'))}"
                ),
                (
                    "- bounded_delta_contract: "
                    f"{_items(_operator_review_contract_items(card, 'bounded_delta'))}"
                ),
                (
                    "- safety_blockers: "
                    f"{_items(_operator_review_contract_items(card, 'safety_blockers'))}"
                ),
                (
                    "- safety_blocker_details: "
                    f"{_items(card['safety_blocker_details'])}"
                ),
                (
                    "- headroom_feedback_true_peak_checks: "
                    f"{_items(_operator_review_contract_items(card, 'headroom_feedback_true_peak_checks'))}"
                ),
                (
                    "- headroom_feedback_true_peak_check_details: "
                    f"{_items(card['headroom_feedback_true_peak_check_details'])}"
                ),
                (
                    "- operator_readable_explanation: "
                    f"{_items(_operator_review_contract_items(card, 'operator_readable_explanation'))}"
                ),
                f"- required_evidence: {_items(card['required_evidence'])}",
                f"- required_safety_checks: {_items(card['required_safety_checks'])}",
                (
                    "- required_operator_surfaces: "
                    f"{_items(card['required_operator_surfaces'])}"
                ),
                (
                    "- operator_payload_required_fields: "
                    f"{_items(card['operator_payload_required_fields'])}"
                ),
                (
                    "- operator_evidence_required_fields: "
                    f"{_items(card['operator_evidence_required_fields'])}"
                ),
                f"- must_confirm: {_items(card['what_operator_must_confirm'])}",
                (
                    "- required_confirmation_details: "
                    f"{_items(card['required_confirmation_details'])}"
                ),
                (
                    "- operator_acceptance_gate_details: "
                    f"{_items(card['operator_acceptance_gate_details'])}"
                ),
                (
                    "- operator_review_decision_option_details: "
                    f"{_items(card['operator_review_decision_option_details'])}"
                ),
                (
                    "- operator_review_checklist: "
                    f"{_items(card['operator_review_checklist'])}"
                ),
                (
                    "- operator_review_checklist_details: "
                    f"{_items(card['operator_review_checklist_details'])}"
                ),
                (
                    "- operator_review_record_required_fields: "
                    f"{_items(card['operator_review_record_required_fields'])}"
                ),
                (
                    "- operator_review_record_field_details: "
                    f"{_items(card['operator_review_record_field_details'])}"
                ),
                (
                    "- operator_next_action_details: "
                    f"{_items(card['operator_next_action_details'])}"
                ),
                (
                    f"- bounded_delta: {_code(card['bounded_delta_unit'])}; "
                    f"{_code(card['bounded_delta_policy'])}"
                ),
                (
                    "- operator_visible_explanation_contract: "
                    f"{_items(_explanation_contract_items(card))}"
                ),
                f"- rollback_expectation: {_code(card['rollback_expectation'])}",
                (
                    "- rollback_readback_contract: "
                    f"{_items(_operator_review_contract_items(card, 'rollback_readback_contract'))}"
                ),
                (
                    "- future_wing_readback_targets: "
                    f"{_items(card['future_wing_readback_targets'])}"
                ),
                (
                    "- future_wing_readback_contract: "
                    f"{_items(_operator_review_contract_items(card, 'future_wing_readback_contract'))}"
                ),
                (
                    "- test_evidence: "
                    f"{_items(_operator_review_contract_items(card, 'test_evidence'))}"
                ),
                (
                    "- test_evidence_references: "
                    f"{_items(card['test_evidence_references'])}"
                ),
                (
                    "- expected_evidence_packet_reference: "
                    f"{_code('packet.json#method_review_cards/' + str(card['method_id']))}"
                ),
                (
                    "- expected_source_material_reference: "
                    f"{_code('packet.json#method_review_cards/' + str(card['method_id']) + '/source_evidence')}"
                ),
                (
                    "- expected_test_evidence_references: "
                    f"{_items(card['test_evidence_references'])}"
                ),
                (
                    "- expected_verification_report_reference: "
                    f"{_code('verification-report.json')}"
                ),
                (
                    "- future_wing_readback_contract_summary: "
                    f"{_items(card['future_wing_readback_contract_summary'])}"
                ),
                (
                    "- future_live_required_proofs: "
                    f"{_items(card['future_live_required_proofs'])}"
                ),
                (
                    "- future_live_requirement_details: "
                    f"{_items(card['future_live_requirement_details'])}"
                ),
                f"- blocked_live_actions: {_items(card['blocked_live_actions'])}",
                (
                    "- live_authority_boundary_summary: "
                    f"{_items(card['live_authority_boundary_summary'])}"
                ),
                (
                    "- operator_review_decisions: "
                    f"{_items(card['operator_review_decisions'])}"
                ),
                f"- operator_next_step: {_code(card['operator_next_step'])}",
                *_authority_lines(card),
                (
                    "- authority: "
                    f"live_authority_granted={_bool(card['live_authority_granted'])}; "
                    f"live_apply_allowed={_bool(card['live_apply_allowed'])}; "
                    f"queue_import_allowed={_bool(card['queue_import_allowed'])}; "
                    f"contacts_wing={_bool(card['contacts_wing'])}; "
                    f"contacts_backend={_bool(card['contacts_backend'])}; "
                    f"mutates_mixer={_bool(card['mutates_mixer'])}"
                ),
            ]
        )

    lines.extend(["", "## Priority Sections"])
    for section in packet["priority_sections"]:
        lines.extend(
            [
                "",
                f"### {section['priority']}. {section['label']}",
                f"- objective_families: {_items(section['objective_families'])}",
                f"- method_count: {_code(section['method_count'])}",
                f"- review_status: {_code(section['review_status'])}",
                *_authority_lines(section),
            ]
        )
        for card in section["methods"]:
            _append_card(card)

    lines.extend(["", "## Supporting Proposal Methods"])
    for card in packet["supporting_method_cards"]:
        _append_card(card)

    return "\n".join(lines).rstrip() + "\n"
