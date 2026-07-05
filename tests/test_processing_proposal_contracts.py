import hashlib
import json

from backend.processing_proposal_adapters import (
    adapt_agentic_foundation_gain_fader_recommendations,
    adapt_operator_analysis_gain_fader_proposals,
    adapt_safe_gain_suggestions,
    build_correction_method_goal_coverage_report,
    build_correction_method_goal_operator_review_packet,
    build_operator_correction_payload,
    build_proposal_only_method_contracts,
    build_registry_blockers,
    render_correction_method_goal_operator_review_markdown,
)
from backend.processing_registry import iter_processing_methods
from tools.export_correction_method_operator_review_packet import (
    export_correction_method_operator_review_packet,
)
from tools.audit_correction_method_goal import (
    audit_correction_method_goal_packet,
    main as audit_correction_method_goal_main,
    render_correction_method_goal_audit_markdown,
)
from tools.verify_correction_method_operator_review_export import (
    ALLOWED_BOUNDED_DELTA_POLICIES,
    ALLOWED_BOUNDED_DELTA_UNITS,
    ALLOWED_BLOCKED_LIVE_ACTIONS,
    ALLOWED_FUTURE_WING_READBACK_TARGETS,
    ALLOWED_REQUIRED_SAFETY_CHECKS,
    ALLOWED_ROLLBACK_EXPECTATIONS,
    REQUIRED_CONFIDENCE_ACKNOWLEDGEMENT,
    REQUIRED_FUTURE_PROOFS,
    REQUIRED_FUTURE_WING_READBACK_ACKNOWLEDGEMENT,
    REQUIRED_FUTURE_WING_READBACK_CONTRACT_SUMMARY_KEYS,
    REQUIRED_FUTURE_WING_READBACK_DENIAL_REASON,
    REQUIRED_FUTURE_WING_READBACK_OPERATOR_NEXT_STEP,
    REQUIRED_BOUNDED_DELTA_ACKNOWLEDGEMENT,
    REQUIRED_HEADROOM_FEEDBACK_TRUE_PEAK_ACKNOWLEDGEMENT,
    REQUIRED_LIVE_AUTHORITY_BOUNDARY_SUMMARY,
    REQUIRED_METHOD_CARD_CONFIRMATIONS,
    REQUIRED_MEASURED_PROBLEM_ACKNOWLEDGEMENT,
    REQUIRED_OFFLINE_ACCEPTANCE_BOUNDARY_ACKNOWLEDGEMENT,
    REQUIRED_OFFLINE_REVIEW_DECISIONS,
    REQUIRED_OPERATOR_METHOD_REVIEW_FOCUS_KEYS,
    REQUIRED_OPERATOR_PROBLEM_REVIEW_SUMMARY,
    REQUIRED_OPERATOR_CONFIRMATION_ACKNOWLEDGEMENT,
    REQUIRED_OPERATOR_EXPLANATION_ACKNOWLEDGEMENT,
    REQUIRED_OPERATOR_REVIEW_RECORD_FIELDS,
    REQUIRED_OPERATOR_WHY_BLOCKED_SUMMARY_KEYS,
    REQUIRED_OPERATOR_WHY_SAFE_SUMMARY_KEYS,
    REQUIRED_PROPOSED_CORRECTION_ACKNOWLEDGEMENT,
    REQUIRED_ROLLBACK_READBACK_ACKNOWLEDGEMENT,
    REQUIRED_SAFETY_BLOCKERS_ACKNOWLEDGEMENT,
    REQUIRED_SOURCE_EVIDENCE_ACKNOWLEDGEMENT,
    REQUIRED_TEST_EVIDENCE_ACKNOWLEDGEMENT,
    REQUIRED_VERIFICATION_REPORT_ACKNOWLEDGEMENT,
    _expected_evidence_problem_correction_chain_details,
    _expected_future_live_requirement_details,
    _expected_goal_requirement_details,
    _expected_headroom_feedback_true_peak_check_details,
    _expected_method_specific_review_details,
    _expected_operator_actionable_explanation_details,
    _expected_operator_acceptance_gate_details,
    _expected_safety_blocker_details,
    _expected_operator_review_checklist_details,
    _expected_operator_review_decision_option_details,
    _expected_operator_next_action_details,
    _expected_required_confirmation_details,
    _expected_operator_review_record_field_details,
    validate_correction_method_operator_decision_record,
    verify_correction_method_operator_review_export,
    write_correction_method_operator_decision_validation_report,
    write_correction_method_operator_review_verification_report,
)


CORRECTION_PROPOSAL_DOMAINS = {
    "adaptation",
    "arrangement",
    "decision_evidence",
    "gain_fader",
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
    "memory_learning",
    "mix_variant",
    "source_identity",
}


OBJECTIVE_FAMILY_METHODS = {
    "gain_trim": ("lufs_gain_staging",),
    "gain_trim_and_fader_balance": (
        "agentic_method_foundation",
        "auto_soundcheck_engine",
    ),
    "source_identity": ("source_identity_confidence_proposal",),
    "fader_balance": (
        "auto_fader_v1",
        "bus_group_balance_proposal",
        "live_shared_mix",
    ),
    "eq": (
        "eq_v3",
        "auto_eq",
        "corrective_eq_proposal",
        "cross_adaptive_eq",
        "frequency_separation_proposal",
    ),
    "compressor_dynamics": (
        "comp_v3",
        "auto_compressor",
        "auto_compressor_cf",
        "compressor_dynamics_proposal",
    ),
    "gate_expander": ("auto_gate_caig", "context_aware_gate_agent"),
    "gate_expander_deesser": ("deesser_vocal_sibilance",),
    "fx_sends": ("auto_fx", "fx_send_proposal"),
    "pan_stereo": ("auto_panner", "pan_spatial_proposal"),
    "phase_alignment": ("phase_alignment", "phase_aware_eq_guard"),
    "routing_groups_scenes": (
        "routing_management_proposal",
        "scene_preset_management_proposal",
        "mute_management_proposal",
    ),
    "master_output": ("master_bus_optimization_proposal",),
    "source_identity_calibration": ("source_identity_replay_calibration",),
    "mix_variant_profile": ("mix_variant_profile_proposal",),
    "performance_adaptation": ("performance_adaptation_proposal",),
    "arrangement_role": ("arrangement_role_proposal",),
    "decision_evidence_packet": ("decision_evidence_packet_proposal",),
    "proposal_only_method": ("smart_dsp_strategy_engine",),
    "long_term_memory_learning": ("long_term_memory_learning_proposal",),
}

QUARANTINED_WRITE_SURFACE_METHODS = (
    "routing_snapshot_reset_writes",
    "fx_direct_apply",
)

EXPECTED_METHOD_REVIEW_CARD_COUNT = 38

REQUIRED_OPERATOR_FIELDS = (
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
)

OPERATOR_IDENTITY_PROFILE_FIELDS = (
    "method_id",
    "source",
    "target",
    "value_type",
    "target_kind",
    "old_value_source",
    "objective_family",
    "bounded_delta_unit",
    "bounded_delta_policy",
)

OPERATOR_CORRECTION_CONTEXT_PROFILE_FIELDS = {
    "correction_method_id": "method_id",
    "correction_source": "source",
    "correction_target": "target",
    "correction_value_type": "value_type",
    "correction_target_kind": "target_kind",
    "correction_old_value_source": "old_value_source",
    "correction_objective_family": "objective_family",
    "correction_bounded_delta_unit": "bounded_delta_unit",
    "correction_bounded_delta_policy": "bounded_delta_policy",
}

OPERATOR_REVIEW_CONTRACT_FIELDS = (
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
REQUIRED_OPERATOR_VISIBLE_CONTRACT_FLAGS = (
    "source_evidence_required",
    "problem_statement_required",
    "proposed_correction_required",
    "confidence_required",
    "bounded_delta_required",
    "safety_blockers_required",
    "safety_checks_required",
    "operator_explanation_required",
    "rollback_required",
    "future_wing_readback_required",
    "test_evidence_required",
)
FULL_FUTURE_LIVE_REQUIRED_PROOFS = (
    "real_wing_readonly_connection_artifact",
    "physical_console_readback_for_old_and_proposed_values",
    "operator_review_acceptance_record",
    "rollback_readback_proof",
    "emergency_stop_proof",
    "connection_loss_timeout_fail_closed_proof",
    "safety_kernel_trace_for_target_action",
)
FULL_FUTURE_WING_READBACK_CONTRACT_SUMMARY_KEYS = (
    "target_kind",
    "old_value_source",
    "readback_targets",
    "required_evidence_fields",
    "operator_next_step",
    "denial_reason",
)
FULL_FUTURE_WING_READBACK_OPERATOR_NEXT_STEP = (
    "capture_real_wing_readback_and_rollback_proof_before_live_promotion"
)
FULL_FUTURE_WING_READBACK_DENIAL_REASON = (
    "real_wing_hardware_readback_rollback_and_operator_review_required"
)
FULL_REQUIRED_SAFETY_CHECKS = (
    "active_performance_guard",
    "active_signal_guard",
    "arrangement_context",
    "arrangement_role",
    "ayaic_level_plane",
    "bleed",
    "chatter_guard",
    "classifier_confidence",
    "context_freshness",
    "cooldown",
    "delay_readback",
    "destructive_scope",
    "downstream_rejection",
    "dry_run",
    "eq_hold_policy",
    "fader_ceiling",
    "feedback",
    "future_wing_readback",
    "gain_reduction",
    "headroom",
    "hysteresis",
    "insert_bypass_state",
    "intelligibility",
    "inter_channel_context",
    "limiter_or_dynamics_proof",
    "main_bus_headroom",
    "makeup_gain",
    "manual_band",
    "manual_override",
    "manual_overrides",
    "masking",
    "monitor_safety",
    "mono_compatibility",
    "offline_replay",
    "operator_acceptance",
    "operator_outcomes",
    "pa_feed_safety",
    "parameter_bounds",
    "phase",
    "phase_correlation",
    "polarity_readback",
    "read_only_inventory",
    "read_only_map",
    "readback",
    "reference_feed_exclusion",
    "replay_match",
    "replay_outcome",
    "return_level",
    "role_persistence",
    "rollback",
    "safety_blockers",
    "safety_regressions",
    "source_identity",
    "speech_intelligibility",
    "style_clone",
    "sum_bus",
    "tails_and_attack",
    "true_peak",
)
FULL_BOUNDED_DELTA_UNITS = (
    "db",
    "db_and_processor_params",
    "db_hz_lane",
    "db_hz_q",
    "delay_ms_or_polarity",
    "evidence_packet",
    "frequency_threshold_range_db",
    "fx_insert_or_rack_parameter",
    "genre_task_profile",
    "memory_learning_record",
    "method_specific",
    "mute_state_diff",
    "pan_percent",
    "phase_guard_policy",
    "role_label",
    "route_diff",
    "scoped_scene_diff",
    "send_return_db_and_fx_params",
    "source_label",
    "threshold_range_hysteresis_timing_db_ms",
    "threshold_ratio_attack_release_makeup_db",
    "trust_state",
)
FULL_BOUNDED_DELTA_POLICIES = (
    "bounded_cut_or_small_boost_with_protected_band_policy",
    "bounded_pan_delta_with_mono_phase_check_for_non_center_moves",
    "bounded_send_with_return_level_monitor_and_feedback_limits",
    "bounded_sibilance_attenuation_only",
    "bounded_threshold_ratio_timing_and_makeup_gain_with_gr_estimate",
    "chatter_safe_expander_or_gate_delta_only",
    "gain_down_is_bounded; gain_up_requires_operator_review_and_explicit_future_proof",
    "offline_static_trim_eq_limiter_or_dynamics_recommendation_only",
    "operator_accepted_offline_learning_record_only_with_replay_correlation_and_no_safety_regressions",
    "operator_reviewed_arrangement_role_label_only_with_context_persistence_and_source_identity_trust",
    "operator_reviewed_evidence_packet_only_with_readback_rollback_headroom_and_feedback_context",
    "operator_reviewed_replay_calibration_only_with_accepted_rejected_outcomes_and_no_live_authority_context",
    "operator_reviewed_source_identity_label_only_with_classifier_confidence_and_replay_calibration_before_downstream_trust",
    "operator_selected_mix_variant_profile_only_with_level_plane_reference_and_no_named_artist_style_clone",
    "proposal_must_state_method_specific_bounds_before_operator_review",
    "persistent_level_drift_proposal_only_with_headroom_feedback_cooldown_and_operator_review",
    "preview_only_downward_gain_or_fader_delta",
    "proposal_only_delay_or_polarity_change_requires_correlation_readback_and_manual_confirmation",
    "read_only_fx_parameter_contract_requires_insert_bypass_return_headroom_and_rollback_readback",
    "read_only_inventory_and_scoped_diff_only",
    "read_only_map_and_non_queueable_route_diff_only",
    "read_only_mute_state_proposal_requires_operator_confirmation_and_readback",
    "read_only_phase_guard_can_hold_eq_proposals_until_multichannel_phase_context_is_fresh",
    "small_bus_group_dca_matrix_delta_with_ayaic_and_sum_bus_limits",
    "small_fader_delta_with_fader_ceiling_and_headroom_limits",
    "subtractive_or_mirror_eq_only_until_sum_bus_readback_exists",
)
FULL_FUTURE_WING_READBACK_TARGETS = (
    "accepted_learning_record",
    "accepted_source_identity_label",
    "assignment_map",
    "attack",
    "band_frequency",
    "band_gain",
    "band_id",
    "band_q",
    "bus_assignment",
    "bus_fader",
    "channel_delay",
    "channel_fader",
    "channel_gain",
    "channel_input_trim",
    "channel_level",
    "channel_mute",
    "channel_pan",
    "classifier_evidence",
    "correlation_meter",
    "current_arrangement_role",
    "current_mix_profile",
    "current_state_readback",
    "dca_fader",
    "decision_evidence_packet",
    "decision_trace",
    "deesser_band",
    "downstream_trust_state",
    "dyn_on",
    "dynamic_eq_band",
    "eq_band",
    "eq_on",
    "eq_proposal_hold_state",
    "filter_type",
    "full_console_state",
    "fx_return_level",
    "fx_slot_parameter",
    "fx_slot_parameters",
    "gate_on",
    "group_fader",
    "hold",
    "hysteresis",
    "insert_bypass",
    "level_plane_targets",
    "limiter",
    "main_bus_processing",
    "makeup_gain",
    "matrix_fader",
    "matrix_or_pa_feed",
    "matrix_send",
    "memory_rollback_plan",
    "monitor_send",
    "mono_phase_check",
    "mute_group",
    "offline_replay_summary",
    "old_value",
    "operator_acceptance_record",
    "operator_accepted_source_label",
    "operator_accepted_source_role",
    "operator_review_record",
    "operator_selected_profile",
    "proposed_value",
    "phase_measurements",
    "polarity_invert",
    "preamp_gain",
    "preset_inventory",
    "range",
    "ratio",
    "reference_feed",
    "release",
    "replay_outcome",
    "replay_outcome_summary",
    "rollback_delay_polarity",
    "rollback_eq_hold",
    "rollback_fader",
    "rollback_fx_parameter",
    "rollback_gain",
    "rollback_gain_or_fader",
    "rollback_level_or_fader",
    "rollback_mute_state",
    "rollback_plan",
    "rollback_profile",
    "rollback_value",
    "rollback_route",
    "rollback_snapshot",
    "routing_map",
    "scene_inventory",
    "send_level",
    "stem_spectrum",
    "stereo_pair_link",
    "sum_bus",
    "target_parameter",
    "threshold",
)
FULL_ROLLBACK_EXPECTATIONS = (
    "discard_unaccepted_learning_record_or_restore_previous_memory_snapshot_after_operator_review",
    "discard_unaccepted_source_identity_label_or_restore_previous_operator_accepted_label",
    "discard_unaccepted_trust_state_or_restore_previous_operator_accepted_source_identity_calibration",
    "operator-reviewed full-state snapshot required before any future scene or preset action",
    "operator-reviewed route snapshot required before any future manual rollback",
    "release_or_restore_held_eq_proposals_after_fresh_phase_context_review",
    "restore_or_discard_unaccepted_decision_packet_without_mutating_mixer_or_memory",
    "restore_previous_bus_group_dca_matrix_fader_after_physical_readback",
    "restore_previous_delay_and_polarity_after_physical_readback",
    "restore_previous_dynamics_or_dynamic_eq_values_after_readback",
    "restore_previous_dynamics_values_or_bypass_state_after_physical_readback",
    "restore_previous_eq_band_and_stem_lane_plan_after_readback",
    "restore_previous_eq_band_or_leave_feedback_manual_band_unchanged",
    "restore_previous_fader_value_after_physical_readback",
    "restore_previous_fx_parameter_or_insert_bypass_state_after_physical_readback",
    "restore_previous_gain_or_fader_value_after_supervised_readback",
    "restore_previous_gate_or_expander_values_or_bypass_after_readback",
    "restore_previous_input_trim_or_a_safer_lower_value_after_physical_readback",
    "restore_previous_level_or_fader_state_after_physical_readback",
    "restore_previous_master_processing_or_hold_for_operator_manual_rollback",
    "restore_previous_mute_state_after_physical_readback_or_operator_manual_action",
    "restore_previous_operator_accepted_arrangement_role_or_hold_for_manual_review",
    "restore_previous_operator_selected_mix_profile_or_hold_for_manual_review",
    "restore_previous_readback_value_or_hold_for_operator_manual_rollback",
    "restore_previous_pan_value_after_physical_readback",
    "restore_previous_send_and_return_values_after_physical_readback",
)
FULL_BLOCKED_LIVE_ACTIONS = (
    "bus_assignment_write",
    "bus_group_dca_matrix_write",
    "compressor_write",
    "decision_packet_apply",
    "deesser_write",
    "delay_write",
    "downstream_priority_apply",
    "downstream_source_specific_processing_apply",
    "downstream_trust_apply",
    "dynamic_eq_write",
    "eq_band_write",
    "eq_on_write",
    "eq_write_from_stale_phase_context",
    "expander_write",
    "fader_above_zero_dbfs",
    "frequency_split_live_write",
    "fx_parameter_write",
    "fx_rack_write",
    "fx_send_write",
    "gain_up_without_operator_opt_in",
    "gate_write",
    "input_trim_write",
    "insert_bypass_write",
    "limiter_write",
    "linked_eq_write",
    "live_apply",
    "live_console_write_from_identity",
    "live_console_write_from_learning",
    "live_console_write_from_packet",
    "live_preview_apply",
    "makeup_gain_write",
    "master_bus_write",
    "matrix_pa_feed_write",
    "matrix_write",
    "memory_write",
    "mix_profile_apply",
    "model_update",
    "monitor_send_write",
    "mute_group_write",
    "mute_write",
    "pan_write",
    "performance_adaptation_live_write",
    "polarity_write",
    "preamp_gain_write",
    "preset_load",
    "preset_save",
    "queue_import",
    "role_memory_mutation",
    "routing_write",
    "scene_load",
    "scene_save",
    "snapshot_recall",
    "source_identity_memory_write",
    "stereo_width_write",
    "trend_based_fader_write",
    "unmute_write",
    "unsupervised_arrangement_role_write",
    "unsupervised_fader_write",
    "unsupervised_level_adaptation",
    "unsupervised_style_clone",
    "upward_delta_without_operator_proof",
)
FULL_LIVE_AUTHORITY_BOUNDARY_SUMMARY = (
    "allowed_current_live_write_kinds=fader,gain",
    "can_authorize_live=false",
    "denial_reason=real_wing_hardware_readback_rollback_and_operator_review_required",
    "live_apply_allowed=false",
    "live_authority_granted=false",
    "queue_import_allowed=false",
    "readiness_closure_allowed=false",
)
FULL_OPERATOR_WHY_SAFE_SUMMARY_KEYS = (
    "read_only_or_dry_run_only",
    "live_apply_allowed",
    "queue_import_allowed",
    "operator_review_required",
    "bounded_delta",
    "required_safety_checks",
)
FULL_OPERATOR_WHY_BLOCKED_SUMMARY_KEYS = (
    "denial_reason",
    "future_wing_readback_required",
    "hardware_proof_required_before_live",
    "readiness_closure_allowed",
    "blocked_live_actions",
    "future_live_required_proofs",
)
FULL_OFFLINE_REVIEW_DECISIONS = (
    "request_more_evidence",
    "accept_as_offline_proposal",
    "reject_proposal",
    "defer_until_real_wing_proof",
)
FULL_METHOD_CARD_CONFIRMATIONS = (
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
FULL_OPERATOR_REVIEW_RECORD_FIELDS = (
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
)
FULL_OPERATOR_REVIEW_RECORD_FIELD_DETAILS = tuple(
    _expected_operator_review_record_field_details(
        list(FULL_OPERATOR_REVIEW_RECORD_FIELDS)
    )
)


def _contracts_by_method():
    return {
        item["method_id"]: item
        for item in build_proposal_only_method_contracts()
    }


def _tamper_live_authority_boundary_summary(surface):
    surface["live_authority_boundary_summary"] = [
        item
        for item in surface["live_authority_boundary_summary"]
        if item != "can_authorize_live=false"
    ]
    surface["live_authority_boundary_summary"].extend(
        [
            "live_authority_granted=true",
            "live_authority_granted=false",
        ]
    )


def _tamper_why_safe_summary(surface):
    surface["why_safe_summary"] = [
        item
        for item in surface["why_safe_summary"]
        if item != "live_apply_allowed=false"
    ]
    surface["why_safe_summary"].append("live_apply_allowed=true")


def _tamper_why_blocked_summary(surface):
    surface["why_blocked_summary"] = [
        item
        for item in surface["why_blocked_summary"]
        if item != "future_wing_readback_required=true"
    ]
    surface["why_blocked_summary"].append("future_wing_readback_required=false")


def _tamper_future_wing_readback_contract_summary(surface):
    duplicate_denial_reason = (
        f"denial_reason={FULL_FUTURE_WING_READBACK_DENIAL_REASON}"
    )
    tampered = []
    for item in surface["future_wing_readback_contract_summary"]:
        if item.startswith("required_evidence_fields="):
            continue
        if item.startswith("operator_next_step="):
            tampered.append("operator_next_step=apply_live_without_real_wing_proof")
            continue
        tampered.append(item)
    tampered.extend(
        [
            "live_authority_granted=true",
            duplicate_denial_reason,
            "denial_reason=operator_claims_live_ready_without_wing",
        ]
    )
    surface["future_wing_readback_contract_summary"] = tampered


def _tamper_required_safety_checks(surface):
    duplicate_check = (
        surface["required_safety_checks"][0]
        if surface["required_safety_checks"]
        else "headroom"
    )
    surface["required_safety_checks"] = [
        "authorize_live_without_safety_review",
        duplicate_check,
        duplicate_check,
    ]


def _tamper_bounded_delta_policies(surface):
    duplicate_policy = (
        surface["bounded_delta_policies"][0]
        if surface["bounded_delta_policies"]
        else "small_fader_delta_with_fader_ceiling_and_headroom_limits"
    )
    surface["bounded_delta_policies"] = [
        "unbounded_live_delta",
        duplicate_policy,
        duplicate_policy,
    ]


def _tamper_future_wing_readback_targets(surface):
    duplicate_target = (
        surface["future_wing_readback_targets"][0]
        if surface["future_wing_readback_targets"]
        else "channel_fader"
    )
    surface["future_wing_readback_targets"] = [
        "authorize_live_without_readback",
        duplicate_target,
        duplicate_target,
    ]


def _tamper_rollback_expectations(surface):
    duplicate_expectation = (
        surface["rollback_expectations"][0]
        if surface["rollback_expectations"]
        else "restore_previous_fader_value_after_physical_readback"
    )
    surface["rollback_expectations"] = [
        "no_rollback_required",
        duplicate_expectation,
        duplicate_expectation,
    ]


def _assert_correction_context_profile(row, correction):
    for (
        row_field,
        correction_field,
    ) in OPERATOR_CORRECTION_CONTEXT_PROFILE_FIELDS.items():
        assert row[row_field] == correction[correction_field]


def _assert_raw_input_evidence_boundary(boundary, *, surface, evidence_fields):
    assert boundary["surface"] == surface
    assert boundary["raw_input_evidence_only"] is True
    assert boundary["raw_input_cannot_grant_authority"] is True
    assert boundary["evidence_fields"] == sorted(str(field) for field in evidence_fields)
    assert boundary["read_only_or_dry_run_only"] is True
    assert boundary["live_apply_allowed"] is False
    assert boundary["queue_import_allowed"] is False
    assert boundary["live_authority_granted"] is False
    assert boundary["full_autonomous_ready"] is False
    assert boundary["readiness_closure_allowed"] is False
    assert boundary["can_authorize_live"] is False
    assert boundary["can_close_readiness"] is False
    assert boundary["mutates_mixer"] is False
    assert boundary["operator_review_required"] is True
    assert boundary["future_wing_readback_required"] is True
    assert boundary["blocks_live_promotion"] is True
    assert boundary["blocks_queue_import"] is True
    assert (
        boundary["denial_reason"]
        == "raw_input_evidence_requires_operator_review_and_future_wing_readback"
    )


def _assert_operator_review_contract(contract, *, source_fields):
    assert contract["source_fields"] == source_fields
    assert contract["required"] is True
    assert contract["review_status"] == "required_from_offline_proposal_payload"
    assert contract["operator_action"]


def _operator_review_contract_items(contract):
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


def _assert_required_evidence_status(status, *, completeness, evidence_gate):
    correction_context_profile = completeness.get("correction_context_profile") or {}
    assert status["status"] == completeness["status"]
    assert status["complete"] == completeness["complete"]
    assert status["required_fields"] == completeness["required"]
    assert status["provided_fields"] == completeness["provided"]
    assert status["missing_fields"] == evidence_gate["missing_required_evidence"]
    assert status["required_count"] == len(completeness["required"])
    assert status["provided_count"] == len(completeness["provided"])
    assert status["missing_count"] == len(evidence_gate["missing_required_evidence"])
    assert status["operator_action"] == evidence_gate["operator_action"]
    assert status["read_only_or_dry_run_only"] is True
    assert status["live_apply_allowed"] is False
    assert status["queue_import_allowed"] is False
    assert status["live_authority_granted"] is False
    assert status["full_autonomous_ready"] is False
    assert status["readiness_closure_allowed"] is False
    assert status["can_authorize_live"] is False
    assert status["can_close_readiness"] is False
    assert status["mutates_mixer"] is False
    assert status["operator_review_required"] is True
    assert status["future_wing_readback_required"] is True
    assert status["blocks_live_promotion"] is True
    assert status["blocks_queue_import"] is True
    assert status["denial_reason"] == "operator_review_and_future_wing_readback_required"
    for field, value in correction_context_profile.items():
        assert status[field] == value

    rows = status["rows"]
    actions = status["required_evidence_actions"]
    assert [row["field"] for row in rows] == completeness["required"]
    assert [action["field"] for action in actions] == completeness["required"]
    assert status["operator_next_actions"] == [
        action["action"] for action in actions
    ]
    missing = set(evidence_gate["missing_required_evidence"])
    provided = set(completeness["provided"])
    for row, action in zip(rows, actions):
        for field, value in correction_context_profile.items():
            assert row[field] == value
            assert action[field] == value
        is_missing = row["field"] in missing
        assert row["provided"] is (not is_missing)
        assert row["missing"] is is_missing
        assert row["status"] == (
            "missing_required_evidence"
            if is_missing
            else "provided_operator_review_required"
        )
        assert set(row["matched_provided_fields"]) <= provided
        if is_missing:
            assert row["operator_action"] == (
                "collect_missing_required_evidence_before_operator_review"
            )
        else:
            assert row["matched_provided_fields"]
            assert row["operator_action"] == (
                "operator_review_required_before_future_live_promotion"
            )
        assert row["read_only_or_dry_run_only"] is True
        assert row["live_apply_allowed"] is False
        assert row["queue_import_allowed"] is False
        assert row["live_authority_granted"] is False
        assert row["full_autonomous_ready"] is False
        assert row["readiness_closure_allowed"] is False
        assert row["can_authorize_live"] is False
        assert row["can_close_readiness"] is False
        assert row["mutates_mixer"] is False
        assert row["operator_review_required"] is True
        assert row["future_wing_readback_required"] is True
        assert row["blocks_live_promotion"] is True
        assert row["blocks_queue_import"] is True
        assert (
            row["denial_reason"]
            == "operator_review_and_future_wing_readback_required"
        )
        assert action["field"] == row["field"]
        assert action["status"] == row["status"]
        assert action["provided"] == row["provided"]
        assert action["missing"] == row["missing"]
        assert action["matched_provided_fields"] == row["matched_provided_fields"]
        assert action["required_before_operator_acceptance"] is is_missing
        assert action["required_before_live_promotion"] is True
        assert action["operator_action"] == row["operator_action"]
        if is_missing:
            assert action["action"] == f"collect_missing_required_evidence:{row['field']}"
            assert action["action_type"] == "collect_missing_required_evidence"
        else:
            assert action["action"] == f"review_required_evidence:{row['field']}"
            assert action["action_type"] == "review_provided_required_evidence"
        assert action["read_only_or_dry_run_only"] is True
        assert action["live_apply_allowed"] is False
        assert action["queue_import_allowed"] is False
        assert action["live_authority_granted"] is False
        assert action["full_autonomous_ready"] is False
        assert action["readiness_closure_allowed"] is False
        assert action["can_authorize_live"] is False
        assert action["can_close_readiness"] is False
        assert action["mutates_mixer"] is False
        assert action["operator_review_required"] is True
        assert action["future_wing_readback_required"] is True
        assert action["blocks_live_promotion"] is True
        assert action["blocks_queue_import"] is True
        assert (
            action["denial_reason"]
            == "operator_review_and_future_wing_readback_required"
        )


def _assert_source_evidence_details(correction):
    details = correction["source_evidence_details"]
    source_evidence = correction["source_evidence"]
    explanation_packet = correction["operator_explanation_packet"]
    evidence_packet = correction["operator_evidence_packet"]
    review_summary = correction["operator_review_summary"]

    assert details
    assert source_evidence["source_evidence_details"] == details
    assert explanation_packet["source_evidence_details"] == details
    assert explanation_packet["source_evidence_summary"][
        "source_evidence_details"
    ] == details
    assert evidence_packet["source_evidence_details"] == details
    assert review_summary["source_evidence_details"] == details

    required_fields = source_evidence["required_evidence_fields"]
    assert [row["field"] for row in details] == required_fields

    completeness = source_evidence["evidence_completeness"]
    evidence_gate = correction["evidence_gate"]
    missing = set(completeness["missing"])
    provided = set(completeness["provided"])

    for row in details:
        field = row["field"]
        is_missing = field in missing

        _assert_correction_context_profile(row, correction)
        for identity_field in OPERATOR_IDENTITY_PROFILE_FIELDS:
            assert row[identity_field] == correction[identity_field]
        assert row["required_evidence_field"] == field
        assert row["status"] == (
            "missing_required_evidence"
            if is_missing
            else "provided_operator_review_required"
        )
        assert row["provided"] is (not is_missing)
        assert row["missing"] is is_missing
        assert set(row["matched_provided_fields"]) <= provided
        assert row["matched_provided_fields_in_completeness"] == [
            field_name
            for field_name in row["matched_provided_fields"]
            if field_name in provided
        ]
        assert row["objective_family"] == correction["objective_family"]
        assert row["target_kind"] == correction["future_wing_readback_contract"][
            "target_kind"
        ]
        assert row["old_value_source"] == correction["future_wing_readback_contract"][
            "old_value_source"
        ]
        assert row["all_required_evidence_fields"] == required_fields
        assert row["provided_evidence_fields"] == completeness["provided"]
        assert row["missing_required_evidence"] == evidence_gate[
            "missing_required_evidence"
        ]
        assert row["evidence_gate_status"] == evidence_gate["status"]
        assert row["required_before_operator_acceptance"] is True
        assert row["required_before_live_promotion"] is True
        assert row["required_before_queue_import"] is True
        assert row["read_only_or_dry_run_only"] is True
        assert row["live_apply_allowed"] is False
        assert row["queue_import_allowed"] is False
        assert row["live_authority_granted"] is False
        assert row["full_autonomous_ready"] is False
        assert row["readiness_closure_allowed"] is False
        assert row["can_authorize_live"] is False
        assert row["can_close_readiness"] is False
        assert row["can_import_to_queue"] is False
        assert row["mutates_mixer"] is False
        assert row["operator_review_required"] is True
        assert row["future_wing_readback_required"] is True
        assert row["blocks_live_promotion"] is True
        assert row["blocks_queue_import"] is True
        assert row["denial_reason"] == correction["live_authority_boundary"][
            "denial_reason"
        ]
        assert row["operator_action"] in explanation_packet["operator_next_actions"]

        if is_missing:
            assert "missing" in row["source_locations"]
            assert row["source_value_available"] is False
            assert row["raw_evidence_values"] == {}
            assert row["structured_evidence_values"] == {}
            assert row["action_type"] == "collect_source_evidence"
            assert row["operator_action"] == f"collect_source_evidence:{field}"
        else:
            assert row["source_locations"]
            assert "missing" not in row["source_locations"]
            assert row["action_type"] == "review_source_evidence"
            assert row["operator_action"] == f"review_source_evidence:{field}"


def _assert_safety_blocker_details(correction):
    details = correction["safety_blocker_details"]
    effective_blockers = correction["effective_safety_blockers"]

    assert [row["blocker"] for row in details] == effective_blockers
    for row in details:
        _assert_correction_context_profile(row, correction)
        assert row["sources"]
        assert row["operator_actions"]
        assert row["operator_action"] == row["operator_actions"][0]
        assert row["read_only_or_dry_run_only"] is True
        assert row["live_apply_allowed"] is False
        assert row["queue_import_allowed"] is False
        assert row["live_authority_granted"] is False
        assert row["full_autonomous_ready"] is False
        assert row["readiness_closure_allowed"] is False
        assert row["can_authorize_live"] is False
        assert row["can_close_readiness"] is False
        assert row["can_import_to_queue"] is False
        assert row["mutates_mixer"] is False
        assert row["operator_review_required"] is True
        assert row["future_wing_readback_required"] is True
        assert row["blocks_live_promotion"] is True
        assert row["blocks_queue_import"] is True
        assert (
            row["denial_reason"]
            == "operator_review_and_future_wing_readback_required"
        )

        blocker = row["blocker"]
        if blocker in correction["safety_blockers"]:
            assert "input_safety_blocker" in row["sources"]
            assert f"review_safety_blocker:{blocker}" in row["operator_actions"]
        if blocker in correction["evidence_gate"]["blockers"]:
            assert "evidence_gate" in row["sources"]
            assert row["source_statuses"]["evidence_gate"] == correction[
                "evidence_gate"
            ]["status"]
            if blocker.startswith("required_evidence_missing:"):
                field = blocker.removeprefix("required_evidence_missing:")
                assert (
                    f"collect_missing_required_evidence:{field}"
                    in row["operator_actions"]
                )
            else:
                assert correction["evidence_gate"]["operator_action"] in row[
                    "operator_actions"
                ]
        if blocker in correction["confidence_review"]["blockers"]:
            assert "confidence_review" in row["sources"]
            assert row["source_statuses"]["confidence_review"] == correction[
                "confidence_review"
            ]["status"]
            assert f"resolve_confidence_blocker:{blocker}" in row[
                "operator_actions"
            ]
        if blocker in correction["operator_proposal_review_gate"]["blockers"]:
            assert "proposal_review_gate" in row["sources"]
            assert row["source_statuses"]["proposal_review_gate"] == correction[
                "operator_proposal_review_gate"
            ]["status"]
        if blocker in correction["operator_proposal_review_gate"][
            "live_authority_blockers"
        ]:
            assert "proposal_review_live_authority_boundary" in row["sources"]
            assert "do_not_queue_or_apply_live" in row["operator_actions"]
        if blocker in correction["live_authority_boundary"]["blockers"]:
            assert "live_authority_boundary" in row["sources"]
            assert row["source_statuses"]["live_authority_boundary"] == correction[
                "live_authority_boundary"
            ]["evidence_gate_status"]
            assert (
                "capture_real_wing_readback_and_rollback_proof_before_live_promotion"
                in row["operator_actions"]
            )


def _assert_why_safe_details(correction):
    details = correction["why_safe_details"]
    reasons = [row["reason"] for row in details]
    expected = [
        "read_only_or_dry_run_only",
        "live_apply_denied",
        "queue_import_denied",
        f"bounded_delta_policy:{correction['bounded_delta_policy']}",
        f"rollback_expectation:{correction['rollback_expectation']}",
    ]

    assert reasons == expected
    for row in details:
        _assert_correction_context_profile(row, correction)
        assert row["source"]
        assert row["explanation"]
        assert row["verified_by"]
        assert row["operator_action"]
        assert row["read_only_or_dry_run_only"] is True
        assert row["live_apply_allowed"] is False
        assert row["queue_import_allowed"] is False
        assert row["live_authority_granted"] is False
        assert row["full_autonomous_ready"] is False
        assert row["readiness_closure_allowed"] is False
        assert row["can_authorize_live"] is False
        assert row["can_close_readiness"] is False
        assert row["can_import_to_queue"] is False
        assert row["mutates_mixer"] is False
        assert row["operator_review_required"] is True
        assert row["future_wing_readback_required"] is True
        assert row["blocks_live_promotion"] is True
        assert row["blocks_queue_import"] is True
        assert (
            row["denial_reason"]
            == "operator_review_and_future_wing_readback_required"
        )

    by_reason = {row["reason"]: row for row in details}
    assert by_reason["read_only_or_dry_run_only"]["operator_action"] == (
        "review_operator_evidence_packet"
    )
    assert by_reason["live_apply_denied"]["blocked_live_actions"] == correction[
        "live_authority_boundary"
    ]["blocked_live_actions"]
    assert by_reason["live_apply_denied"]["operator_action"] == "do_not_apply_live"
    assert by_reason["queue_import_denied"]["operator_action"] == (
        "do_not_import_to_queue"
    )
    bounded_reason = f"bounded_delta_policy:{correction['bounded_delta_policy']}"
    assert by_reason[bounded_reason]["bounded_delta"] == correction["bounded_delta"]
    assert by_reason[bounded_reason]["bounded_delta_policy"] == correction[
        "bounded_delta_policy"
    ]
    assert by_reason[bounded_reason]["operator_action"] == (
        "review_bounded_delta_and_required_evidence"
    )
    rollback_reason = f"rollback_expectation:{correction['rollback_expectation']}"
    assert by_reason[rollback_reason]["rollback_expectation"] == correction[
        "rollback_expectation"
    ]
    assert by_reason[rollback_reason]["rollback_readback_contract"] == correction[
        "rollback_readback_contract"
    ]
    assert by_reason[rollback_reason]["operator_action"] == (
        "confirm_rollback_expectation"
    )


def _assert_why_blocked_details(correction):
    details = correction["why_blocked_details"]
    reasons = [row["reason"] for row in details]

    assert details
    assert set(correction["operator_review_summary"]["why_blocked"]) <= set(reasons)
    assert set(correction["operator_explanation_packet"]["why_blocked"]) <= set(
        reasons
    )
    assert set(correction["effective_safety_blockers"]) <= set(reasons)

    for row in details:
        _assert_correction_context_profile(row, correction)
        assert row["reason"]
        assert row["source"]
        assert row["sources"]
        assert row["explanation"]
        assert row["operator_actions"]
        assert row["operator_action"] == row["operator_actions"][0]
        assert row["read_only_or_dry_run_only"] is True
        assert row["live_apply_allowed"] is False
        assert row["queue_import_allowed"] is False
        assert row["live_authority_granted"] is False
        assert row["full_autonomous_ready"] is False
        assert row["readiness_closure_allowed"] is False
        assert row["can_authorize_live"] is False
        assert row["can_close_readiness"] is False
        assert row["can_import_to_queue"] is False
        assert row["mutates_mixer"] is False
        assert row["operator_review_required"] is True
        assert row["future_wing_readback_required"] is True
        assert row["blocks_live_promotion"] is True
        assert row["blocks_queue_import"] is True
        assert (
            row["denial_reason"]
            == "operator_review_and_future_wing_readback_required"
        )

        reason = row["reason"]
        if reason in correction["evidence_gate"]["blockers"]:
            assert "evidence_gate" in row["sources"]
            assert row["source_statuses"]["evidence_gate"] == correction[
                "evidence_gate"
            ]["status"]
        if reason in correction["confidence_review"]["blockers"]:
            assert "confidence_review" in row["sources"]
            assert row["source_statuses"]["confidence_review"] == correction[
                "confidence_review"
            ]["status"]
            assert f"resolve_confidence_blocker:{reason}" in row[
                "operator_actions"
            ]
        if reason.startswith("required_evidence_missing:"):
            field = reason.removeprefix("required_evidence_missing:")
            assert f"collect_missing_required_evidence:{field}" in row[
                "operator_actions"
            ]
        if reason.startswith("live_apply_denied"):
            assert "live_authority_boundary" in row["sources"]
            assert (
                "capture_real_wing_readback_and_rollback_proof_before_live_promotion"
                in row["operator_actions"]
            )
        if reason.startswith("queue_import_denied"):
            assert "operator_proposal_review_gate" in row["sources"]
            assert "record_operator_decision_before_promotion" in row[
                "operator_actions"
            ]


def _assert_future_live_requirement_details(correction):
    details = correction["future_live_requirement_details"]
    future_requirements = correction["future_live_promotion_requirements"]
    future_readback = correction["future_wing_readback_contract"]
    explanation_packet = correction["operator_explanation_packet"]
    evidence_packet = correction["operator_evidence_packet"]
    review_summary = correction["operator_review_summary"]

    assert details
    assert future_requirements["future_live_requirement_details"] == details
    assert evidence_packet["future_live_requirement_details"] == details
    assert explanation_packet["future_live_requirement_details"] == details
    assert explanation_packet["future_live_requirements"][
        "requirement_details"
    ] == details
    assert review_summary["future_live_requirement_details"] == details
    assert review_summary["future_live_requirements"]["requirement_details"] == details
    assert explanation_packet["future_live_proof_contract_summary"][
        "requirement_details"
    ] == details
    _assert_correction_context_profile(future_requirements, correction)
    _assert_correction_context_profile(future_readback, correction)
    for field in OPERATOR_IDENTITY_PROFILE_FIELDS:
        assert future_requirements[field] == correction[field]

    proof_rows = [
        row
        for row in details
        if row["requirement_type"] == "future_live_required_proof"
    ]
    readback_rows = [
        row
        for row in details
        if row["requirement_type"] == "future_wing_readback_target"
    ]
    assert [row["proof"] for row in proof_rows] == future_requirements[
        "required_proofs"
    ]
    assert [row["readback_target"] for row in readback_rows] == future_readback[
        "readback_targets"
    ]
    assert explanation_packet["future_wing_readback_summary"][
        "requirement_details"
    ] == readback_rows

    for row in details:
        _assert_correction_context_profile(row, correction)
        assert row["requirement"]
        assert row["operator_action"] == row["action"]
        assert row["operator_action"] in explanation_packet["operator_next_actions"]
        for field in OPERATOR_IDENTITY_PROFILE_FIELDS:
            assert row[field] == future_readback[field]
        assert row["target_kind"] == future_readback["target_kind"]
        assert row["old_value_source"] == future_readback["old_value_source"]
        assert row["blocked_live_actions"] == future_readback["blocked_live_actions"]
        assert row["live_authority_blockers"] == correction[
            "live_authority_boundary"
        ]["blockers"]
        assert row["required_before_live_promotion"] is True
        assert row["required_before_queue_import"] is True
        assert row["read_only_or_dry_run_only"] is True
        assert row["live_apply_allowed"] is False
        assert row["queue_import_allowed"] is False
        assert row["live_authority_granted"] is False
        assert row["full_autonomous_ready"] is False
        assert row["readiness_closure_allowed"] is False
        assert row["can_authorize_live"] is False
        assert row["can_close_readiness"] is False
        assert row["mutates_mixer"] is False
        assert row["operator_review_required"] is True
        assert row["operator_approval_required"] is True
        assert row["hardware_proof_required"] is True
        assert row["future_wing_readback_required"] is True
        assert row["blocks_live_promotion"] is True
        assert row["blocks_queue_import"] is True
        assert (
            row["denial_reason"]
            == correction["live_authority_boundary"]["denial_reason"]
        )

        if row["requirement_type"] == "future_live_required_proof":
            assert row["status"] == "pending_real_wing_proof"
            assert row["proof"] == row["requirement"]
            assert row["readback_target"] is None
            assert row["operator_action"] == (
                f"collect_future_live_required_proof:{row['proof']}"
            )
        else:
            assert row["requirement_type"] == "future_wing_readback_target"
            assert row["status"] == "pending_future_wing_readback"
            assert row["readback_target"] == row["requirement"]
            assert row["proof"] is None
            assert row["operator_action"] == (
                f"verify_future_wing_readback_target:{row['readback_target']}"
            )


def _assert_headroom_feedback_true_peak_check_details(correction):
    details = correction["headroom_feedback_true_peak_check_details"]
    checks = correction["headroom_feedback_true_peak_checks"]
    explanation_packet = correction["operator_explanation_packet"]
    evidence_packet = correction["operator_evidence_packet"]
    review_summary = correction["operator_review_summary"]
    source_evidence = correction["source_evidence"]

    assert details
    assert checks["headroom_feedback_true_peak_check_details"] == details
    assert explanation_packet[
        "headroom_feedback_true_peak_check_details"
    ] == details
    assert evidence_packet["headroom_feedback_true_peak_check_details"] == details
    assert review_summary["headroom_feedback_true_peak_check_details"] == details
    assert [row["check"] for row in details] == checks["required_checks"]

    required_evidence = checks["required_evidence_fields"]
    provided = set(correction["source_evidence"]["evidence_completeness"]["provided"])
    missing = set(correction["evidence_gate"]["missing_required_evidence"])
    evidence_checks = checks.get("evidence_checks") or {}

    for row in details:
        _assert_correction_context_profile(row, correction)
        assert row["safety_check"] == row["check"]
        assert row["method_id"] == source_evidence["method_id"]
        assert row["objective_family"] == correction["objective_family"]
        assert row["target_kind"] == correction["future_wing_readback_contract"][
            "target_kind"
        ]
        assert row["old_value_source"] == correction["future_wing_readback_contract"][
            "old_value_source"
        ]
        assert row["bounded_delta_policy"] == correction["bounded_delta_policy"]
        assert row["safety_check_source"] == (
            "operator_correction_contract.applicable_safety_checks"
        )
        assert row["method_safety_check"] == (
            f"{source_evidence['method_id']}:{row['check']}"
        )
        assert row["operator_action"] == row["action"]
        assert row["operator_action"] in explanation_packet["operator_next_actions"]
        assert row["all_contract_evidence_fields"] == required_evidence
        assert row["available_contract_evidence_fields"] == sorted(provided)
        assert row["missing_contract_evidence_fields"] == correction[
            "evidence_gate"
        ]["missing_required_evidence"]
        assert set(row["required_evidence_fields"]) <= set(required_evidence)
        assert set(row["provided_evidence_fields"]) <= provided
        assert set(row["missing_required_evidence"]) <= missing
        assert row["evidence_checks"] == {
            field: evidence_checks[field]
            for field in row["required_evidence_fields"]
            if field in evidence_checks
        }
        assert row["evidence_gate_status"] == correction["evidence_gate"]["status"]
        assert row["required_before_operator_acceptance"] is True
        assert row["required_before_live_promotion"] is True
        assert row["required_before_queue_import"] is True
        assert row["read_only_or_dry_run_only"] is True
        assert row["live_apply_allowed"] is False
        assert row["queue_import_allowed"] is False
        assert row["live_authority_granted"] is False
        assert row["full_autonomous_ready"] is False
        assert row["readiness_closure_allowed"] is False
        assert row["can_authorize_live"] is False
        assert row["can_close_readiness"] is False
        assert row["can_import_to_queue"] is False
        assert row["mutates_mixer"] is False
        assert row["operator_review_required"] is True
        assert row["future_wing_readback_required"] is True
        assert row["blocks_live_promotion"] is True
        assert row["blocks_queue_import"] is True
        assert row["denial_reason"] == correction["live_authority_boundary"][
            "denial_reason"
        ]
        assert "operator_safety_review_required" in row["why_blocked"]

        if row["missing_required_evidence"]:
            assert row["status"] == "missing_required_evidence"
            assert row["evidence_mapping_status"] == "direct"
            assert row["action_type"] == "collect_safety_check_evidence"
            assert row["operator_action"] == (
                f"collect_safety_check_evidence:{row['check']}"
            )
        elif row["required_evidence_fields"]:
            assert row["status"] == "evidence_available_operator_review_required"
            assert row["evidence_mapping_status"] == "direct"
            assert row["action_type"] == "review_safety_check_evidence"
            assert row["operator_action"] == (
                f"review_safety_check_evidence:{row['check']}"
            )
        else:
            assert row["status"] == (
                "operator_review_required_no_direct_evidence_mapping"
            )
            assert row["evidence_mapping_status"] == (
                "no_direct_required_evidence_field_match"
            )
            assert row["action_type"] == "review_safety_check_mapping"
            assert row["operator_action"] == (
                f"review_safety_check_mapping:{row['check']}"
            )


def _assert_rollback_readback_details(correction):
    details = correction["rollback_readback_details"]
    rollback_contract = correction["rollback_readback_contract"]
    future_requirements = correction["future_live_promotion_requirements"]
    future_readback = correction["future_wing_readback_contract"]
    explanation_packet = correction["operator_explanation_packet"]
    evidence_packet = correction["operator_evidence_packet"]
    review_summary = correction["operator_review_summary"]

    assert details
    assert explanation_packet["rollback_readback_details"] == details
    assert evidence_packet["rollback_readback_details"] == details
    assert review_summary["rollback_readback_details"] == details
    assert explanation_packet["future_wing_readback_summary"][
        "rollback_readback_details"
    ] == details
    assert explanation_packet["future_live_proof_contract_summary"][
        "rollback_readback_details"
    ] == details
    assert review_summary["future_live_requirements"][
        "rollback_readback_details"
    ] == details

    _assert_correction_context_profile(rollback_contract, correction)
    _assert_correction_context_profile(future_readback, correction)
    expectation_rows = [
        row for row in details if row["detail"] == "rollback_expectation"
    ]
    readback_rows = [
        row for row in details if row["detail"] == "rollback_readback_target"
    ]
    assert len(expectation_rows) == 1
    assert [row["readback_target"] for row in readback_rows] == future_readback[
        "readback_targets"
    ]
    assert [row["readback_target"] for row in readback_rows] == rollback_contract[
        "readback_targets"
    ]

    for row in details:
        _assert_correction_context_profile(row, correction)
        assert row["operator_action"] == row["action"]
        assert row["operator_action"] in explanation_packet["operator_next_actions"]
        assert row["rollback_expectation"] == correction["rollback_expectation"]
        for field in OPERATOR_IDENTITY_PROFILE_FIELDS:
            assert row[field] == rollback_contract[field]
        assert row["target_kind"] == rollback_contract["target_kind"]
        assert row["target_kind"] == future_readback["target_kind"]
        assert row["old_value_source"] == rollback_contract["old_value_source"]
        assert row["old_value_source"] == future_readback["old_value_source"]
        assert row["readback_targets"] == rollback_contract["readback_targets"]
        assert row["blocked_live_actions"] == rollback_contract["blocked_live_actions"]
        assert row["blocked_live_actions"] == future_readback["blocked_live_actions"]
        assert row["required_evidence_fields"] == rollback_contract[
            "required_evidence_fields"
        ]
        assert row["missing_required_evidence"] == rollback_contract[
            "missing_required_evidence"
        ]
        assert row["required_evidence_status"] == rollback_contract[
            "required_evidence_status"
        ]
        assert row["required_proofs"] == rollback_contract["required_proofs"]
        assert row["future_live_promotion_requires"] == future_requirements[
            "required_proofs"
        ]
        assert row["evidence_gate_status"] == future_requirements[
            "evidence_gate_status"
        ]
        assert row["required_before_operator_acceptance"] is True
        assert row["required_before_live_promotion"] is True
        assert row["required_before_queue_import"] is True
        assert row["rollback_required"] is True
        assert row["rollback_required_before_promotion"] is True
        assert row["readback_required"] is True
        assert row["readback_required_before_live_promotion"] is True
        assert row["operator_review_required"] is True
        assert row["operator_approval_required"] is True
        assert row["hardware_proof_required"] is True
        assert row["read_only_or_dry_run_only"] is True
        assert row["live_apply_allowed"] is False
        assert row["queue_import_allowed"] is False
        assert row["live_authority_granted"] is False
        assert row["full_autonomous_ready"] is False
        assert row["readiness_closure_allowed"] is False
        assert row["can_authorize_live"] is False
        assert row["can_close_readiness"] is False
        assert row["can_import_to_queue"] is False
        assert row["mutates_mixer"] is False
        assert row["future_wing_readback_required"] is True
        assert row["blocks_live_promotion"] is True
        assert row["blocks_queue_import"] is True
        assert row["denial_reason"] == correction["live_authority_boundary"][
            "denial_reason"
        ]
        assert set(rollback_contract["why_blocked"]) <= set(row["why_blocked"])
        assert set(correction["live_authority_boundary"]["blockers"]) <= set(
            row["why_blocked"]
        )

        if row["detail"] == "rollback_expectation":
            assert row["detail_type"] == "rollback_expectation"
            assert row["description"] == rollback_contract["description"]
            assert row["status"] == rollback_contract["required_evidence_status"][
                "status"
            ]
            assert row["readback_target"] is None
            assert row["operator_action"] == "confirm_rollback_expectation"
            assert row["action_type"] == "confirm_rollback_expectation"
        else:
            assert row["detail_type"] == "rollback_readback_target"
            assert row["status"] == "pending_future_wing_readback"
            assert row["readback_target"] in future_readback["readback_targets"]
            assert row["operator_action"] == (
                f"verify_rollback_readback_target:{row['readback_target']}"
            )
            assert row["action_type"] == "verify_rollback_readback_target"


def _assert_operator_next_action_details(correction):
    details = correction["operator_next_action_details"]
    explanation_packet = correction["operator_explanation_packet"]
    evidence_packet = correction["operator_evidence_packet"]
    review_summary = correction["operator_review_summary"]

    assert details
    assert [row["action"] for row in details] == explanation_packet[
        "operator_next_actions"
    ]
    assert explanation_packet["operator_next_action_details"] == details
    assert evidence_packet["operator_next_action_details"] == details
    assert review_summary["operator_next_action_details"] == details

    by_action = {row["action"]: row for row in details}
    assert len(by_action) == len(details)

    for row in details:
        _assert_correction_context_profile(row, correction)
        assert row["action"]
        assert row["operator_action"] == row["action"]
        assert row["source"]
        assert row["source"] in row["sources"]
        assert row["action_type"]
        assert row["status"]
        assert row["required_before_live_promotion"] is True
        assert row["required_before_queue_import"] is True
        assert row["read_only_or_dry_run_only"] is True
        assert row["live_apply_allowed"] is False
        assert row["queue_import_allowed"] is False
        assert row["live_authority_granted"] is False
        assert row["full_autonomous_ready"] is False
        assert row["readiness_closure_allowed"] is False
        assert row["can_authorize_live"] is False
        assert row["can_close_readiness"] is False
        assert row["can_import_to_queue"] is False
        assert row["mutates_mixer"] is False
        assert row["operator_review_required"] is True
        assert row["future_wing_readback_required"] is True
        assert row["blocks_live_promotion"] is True
        assert row["blocks_queue_import"] is True
        assert (
            row["denial_reason"]
            == correction["live_authority_boundary"]["denial_reason"]
        )

    for action in explanation_packet["required_evidence_actions"]:
        row = by_action[action["action"]]
        assert "required_evidence_status" in row["sources"]
        assert action["field"] in row["fields"]
        assert row["action_type"] == action["action_type"]
        assert row["required_before_operator_acceptance"] is action[
            "required_before_operator_acceptance"
        ]

    for detail in correction["source_evidence_details"]:
        row = by_action[detail["operator_action"]]
        assert "source_evidence_details" in row["sources"]
        assert detail["field"] in row["fields"]
        assert row["action_type"] == detail["action_type"]
        assert row["required_before_operator_acceptance"] is True

    for requirement in correction["future_live_requirement_details"]:
        row = by_action[requirement["operator_action"]]
        assert "future_live_requirement_details" in row["sources"]
        assert requirement["requirement"] in row["requirements"]
        assert requirement["requirement_type"] in row["requirement_types"]
        assert row["action_type"] == requirement["requirement_type"]
        if requirement["proof"] is not None:
            assert requirement["proof"] in row["proofs"]
        if requirement["readback_target"] is not None:
            assert requirement["readback_target"] in row["readback_targets"]

    for detail in correction["rollback_readback_details"]:
        row = by_action[detail["operator_action"]]
        assert "rollback_readback_details" in row["sources"]
        assert detail["detail"] in row["requirements"]
        assert detail["detail_type"] in row["requirement_types"]
        assert row["action_type"] == detail["action_type"]
        if detail["readback_target"] is not None:
            assert detail["readback_target"] in row["readback_targets"]

    for detail in correction["headroom_feedback_true_peak_check_details"]:
        row = by_action[detail["operator_action"]]
        assert "headroom_feedback_true_peak_check_details" in row["sources"]
        assert detail["check"] in row["requirements"]
        assert detail["evidence_mapping_status"] in row["requirement_types"]
        assert row["action_type"] == detail["action_type"]

    for detail in correction["safety_blocker_details"]:
        for action in detail["operator_actions"]:
            assert "safety_blocker_details" in by_action[action]["sources"]
    for detail in correction["why_blocked_details"]:
        for action in detail["operator_actions"]:
            assert "why_blocked_details" in by_action[action]["sources"]


def _assert_operator_review_checklist_details(correction):
    details = correction["operator_review_checklist_details"]
    explanation_packet = correction["operator_explanation_packet"]
    evidence_packet = correction["operator_evidence_packet"]
    review_summary = correction["operator_review_summary"]

    assert details
    assert [row["item"] for row in details] == correction[
        "operator_review_checklist"
    ]
    assert explanation_packet["operator_review_checklist_details"] == details
    assert evidence_packet["operator_review_checklist_details"] == details
    assert review_summary["operator_review_checklist_details"] == details

    by_item = {row["item"]: row for row in details}
    assert len(by_item) == len(details)

    for row in details:
        _assert_correction_context_profile(row, correction)
        assert row["item"]
        assert row["checklist_item"] == row["item"]
        assert row["source"]
        assert row["source"] in row["sources"]
        assert row["action_type"]
        assert row["status"]
        assert row["operator_action"]
        assert row["required_before_live_promotion"] is True
        assert row["required_before_queue_import"] is True
        assert row["read_only_or_dry_run_only"] is True
        assert row["live_apply_allowed"] is False
        assert row["queue_import_allowed"] is False
        assert row["live_authority_granted"] is False
        assert row["full_autonomous_ready"] is False
        assert row["readiness_closure_allowed"] is False
        assert row["can_authorize_live"] is False
        assert row["can_close_readiness"] is False
        assert row["can_import_to_queue"] is False
        assert row["mutates_mixer"] is False
        assert row["operator_review_required"] is True
        assert row["future_wing_readback_required"] is True
        assert row["blocks_live_promotion"] is True
        assert row["blocks_queue_import"] is True
        assert (
            row["denial_reason"]
            == correction["live_authority_boundary"]["denial_reason"]
        )
        for action in row["related_next_actions"]:
            assert action in explanation_packet["operator_next_actions"]

    assert by_item["verify_source_evidence_matches_current_offline_material"][
        "source"
    ] == "source_evidence"
    assert by_item["confirm_measured_or_audible_problem"]["source"] == (
        "measured_or_audible_problem"
    )
    assert by_item["inspect_proposed_correction_and_bounded_delta"][
        "action_type"
    ] == "inspect_proposed_correction"
    assert by_item["verify_required_safety_checks"]["source"] == (
        "operator_safety_review_matrix"
    )
    assert by_item["confirm_rollback_expectation"]["operator_action"] == (
        "confirm_rollback_expectation"
    )
    assert by_item["do_not_queue_or_apply_live"]["operator_action"] == (
        "do_not_queue_or_apply_live"
    )

    for field in correction["evidence_gate"]["missing_required_evidence"]:
        item = f"missing_required_evidence:{field}"
        row = by_item[item]
        assert row["field"] == field
        assert row["fields"] == [field]
        assert row["blocker"] == f"required_evidence_missing:{field}"
        assert row["operator_action"] == f"collect_missing_required_evidence:{field}"
        assert row["source"] == "required_evidence_status"

    for blocker in correction["confidence_review"]["blockers"]:
        item = f"confidence_blocker:{blocker}"
        row = by_item[item]
        assert row["blocker"] == blocker
        assert row["operator_action"] == f"resolve_confidence_blocker:{blocker}"
        assert row["source"] == "confidence_review"


def _assert_operator_review_decision_option_details(correction):
    details = correction["operator_review_decision_option_details"]
    options = correction["operator_review_decision_options"]
    explanation_packet = correction["operator_explanation_packet"]
    evidence_packet = correction["operator_evidence_packet"]
    review_summary = correction["operator_review_summary"]

    assert details
    assert [row["decision"] for row in details] == [
        option["decision"] for option in options
    ]
    assert explanation_packet["operator_review_decision_option_details"] == details
    assert evidence_packet["operator_review_decision_option_details"] == details
    assert review_summary["operator_review_decision_option_details"] == details

    by_decision = {row["decision"]: row for row in details}
    assert set(by_decision) == {
        "request_more_evidence",
        "accept_as_offline_proposal",
        "reject_proposal",
        "defer_until_real_wing_proof",
    }
    recommended = [
        row["decision"] for row in details if row["recommended"]
    ]
    assert recommended == [
        correction["operator_proposal_review_gate"]["recommended_decision"]
    ]

    for option, row in zip(options, details):
        _assert_correction_context_profile(option, correction)
        _assert_correction_context_profile(row, correction)
        for field in OPERATOR_IDENTITY_PROFILE_FIELDS:
            assert option[field] == correction[field]
            assert row[field] == correction[field]
        assert row["option"] == row["decision"]
        assert row["effect"] == option["effect"]
        assert row["recommended"] is option["recommended"]
        assert row["operator_action"] == f"record_operator_decision:{row['decision']}"
        assert row["records_operator_review_only"] is True
        assert row["decision_records_live_authority"] is False
        assert row["decision_records_queue_import"] is False
        assert row["requires_real_wing_proof_before_live"] is True
        assert row["required_before_live_promotion"] is True
        assert row["required_before_queue_import"] is True
        assert row["read_only_or_dry_run_only"] is True
        assert row["live_apply_allowed"] is False
        assert row["queue_import_allowed"] is False
        assert row["live_authority_granted"] is False
        assert row["full_autonomous_ready"] is False
        assert row["readiness_closure_allowed"] is False
        assert row["can_authorize_live"] is False
        assert row["can_close_readiness"] is False
        assert row["can_import_to_queue"] is False
        assert row["mutates_mixer"] is False
        assert row["operator_review_required"] is True
        assert row["future_wing_readback_required"] is True
        assert row["blocks_live_promotion"] is True
        assert row["blocks_queue_import"] is True
        assert (
            row["denial_reason"]
            == correction["live_authority_boundary"]["denial_reason"]
        )

    assert by_decision["request_more_evidence"]["action_type"] == (
        "request_more_evidence"
    )
    assert by_decision["request_more_evidence"]["blocks_offline_acceptance"] is True
    assert by_decision["accept_as_offline_proposal"]["action_type"] == (
        "accept_offline_proposal_only"
    )
    assert by_decision["accept_as_offline_proposal"][
        "offline_acceptance_allowed"
    ] is correction["operator_proposal_review_gate"]["offline_acceptance_allowed"]
    assert by_decision["accept_as_offline_proposal"][
        "blocks_offline_acceptance"
    ] is False
    assert by_decision["reject_proposal"]["action_type"] == (
        "reject_offline_proposal"
    )
    assert by_decision["defer_until_real_wing_proof"]["action_type"] == (
        "defer_until_real_wing_proof"
    )


def _assert_operator_review_record_field_details(correction):
    details = correction["operator_review_record_field_details"]
    template = correction["operator_review_record_template"]
    explanation_packet = correction["operator_explanation_packet"]
    evidence_packet = correction["operator_evidence_packet"]
    review_summary = correction["operator_review_summary"]

    _assert_correction_context_profile(template, correction)
    assert details
    assert [row["field"] for row in details] == template[
        "required_record_fields"
    ]
    assert [row["record_field"] for row in details] == template[
        "required_record_fields"
    ]
    assert explanation_packet["operator_review_record_field_details"] == details
    assert evidence_packet["operator_review_record_field_details"] == details
    assert review_summary["operator_review_record_field_details"] == details
    assert explanation_packet["operator_review_record_summary"][
        "field_details"
    ] == details

    expected_sources = {
        "reviewer": "operator_identity",
        "reviewed_at_utc": "operator_review_timestamp",
        "decision": "operator_review_decision_options",
        "rationale": "operator_rationale",
        "evidence_packet_reference": "operator_evidence_packet",
        "source_material_reference": "source_evidence",
        "source_evidence_acknowledgement": "source_evidence",
        "test_evidence_reference": "test_evidence",
        "test_evidence_acknowledgement": "test_evidence",
        "verification_report_reference": "operator_review_verification_report",
        "verification_report_acknowledgement": (
            "operator_review_verification_report"
        ),
        "measured_problem_acknowledgement": "measured_or_audible_problem",
        "proposed_correction_acknowledgement": "proposed_correction",
        "bounded_delta_acknowledgement": "bounded_delta",
        "confidence_acknowledgement": "confidence",
        "operator_explanation_acknowledgement": "operator_explanation_packet",
        "future_wing_readback_acknowledgement": "future_wing_readback_contract",
        "headroom_feedback_true_peak_acknowledgement": (
            "headroom_feedback_true_peak_checks"
        ),
        "rollback_readback_acknowledgement": "rollback_readback_contract",
        "safety_blockers_acknowledgement": "safety_blockers",
        "required_confirmations_acknowledgement": "what_operator_must_confirm",
        "offline_acceptance_boundary_acknowledgement": "live_authority_boundary",
        "operator_notes": "operator_notes",
    }
    by_field = {row["field"]: row for row in details}
    assert set(by_field) == set(template["required_record_fields"])

    for row in details:
        _assert_correction_context_profile(row, correction)
        field = row["field"]
        assert row["record_field"] == field
        assert row["field_required"] is True
        assert row["field_type"]
        assert row["record_field_type"] == row["field_type"]
        assert row["source"] == expected_sources[field]
        assert row["source"] in row["sources"]
        assert "operator_review_record_template" in row["sources"]
        assert row["operator_action"] == (
            f"fill_operator_review_record_field:{field}"
        )
        assert row["operator_action"] in explanation_packet["operator_next_actions"]
        assert row["required_before_operator_acceptance"] is True
        assert row["required_before_live_promotion"] is True
        assert row["required_before_queue_import"] is True
        assert row["records_operator_review_only"] is True
        assert row["decision_records_live_authority"] is False
        assert row["decision_records_queue_import"] is False
        assert row["requires_real_wing_proof_before_live"] is True
        assert row["read_only_or_dry_run_only"] is True
        assert row["live_apply_allowed"] is False
        assert row["queue_import_allowed"] is False
        assert row["live_authority_granted"] is False
        assert row["full_autonomous_ready"] is False
        assert row["readiness_closure_allowed"] is False
        assert row["can_authorize_live"] is False
        assert row["can_close_readiness"] is False
        assert row["can_import_to_queue"] is False
        assert row["mutates_mixer"] is False
        assert row["operator_review_required"] is True
        assert row["future_wing_readback_required"] is True
        assert row["blocks_live_promotion"] is True
        assert row["blocks_queue_import"] is True
        assert (
            row["denial_reason"]
            == correction["live_authority_boundary"]["denial_reason"]
        )

    decision_row = by_field["decision"]
    assert decision_row["field_type"] == "enum"
    assert decision_row["allowed_decisions"] == template["allowed_decisions"]
    assert decision_row["recommended_decision"] == template["recommended_decision"]
    assert decision_row["status"] == (
        "required_for_recommended_operator_decision_record"
    )
    for field, row in by_field.items():
        if field == "decision":
            continue
        assert row["allowed_decisions"] == []
        assert row["recommended_decision"] is None
        assert row["status"] == "required_before_operator_decision_record"


def _assert_confidence_review_details(correction):
    details = correction["confidence_review_details"]
    confidence_review = correction["confidence_review"]
    explanation_packet = correction["operator_explanation_packet"]
    evidence_packet = correction["operator_evidence_packet"]
    review_summary = correction["operator_review_summary"]

    _assert_correction_context_profile(confidence_review, correction)
    assert details
    assert details[0]["detail"] == "confidence_value_review"
    assert explanation_packet["confidence_review_details"] == details
    assert evidence_packet["confidence_review_details"] == details
    assert review_summary["confidence_review_details"] == details
    assert explanation_packet["confidence_review_summary"]["detail_rows"] == details

    expected_blockers = list(confidence_review["blockers"])
    blocker_rows = [row for row in details if row["detail"] == "confidence_blocker"]
    assert [row["blocker"] for row in blocker_rows] == expected_blockers
    assert len(details) == 1 + len(expected_blockers)

    value_row = details[0]
    assert value_row["field"] == "confidence"
    assert value_row["fields"] == ["confidence"]
    assert value_row["raw_confidence"] == confidence_review["raw_confidence"]
    assert value_row["confidence"] == confidence_review["confidence"]
    assert value_row["expected_range"] == confidence_review["expected_range"]
    assert value_row["low_confidence_threshold"] == confidence_review[
        "low_confidence_threshold"
    ]
    assert value_row["status"] == confidence_review["status"]
    assert value_row["operator_action"] == confidence_review["operator_action"]
    assert value_row["blockers"] == expected_blockers
    assert value_row["confidence_available"] is (
        confidence_review["confidence"] is not None
    )
    confidence_value = confidence_review["confidence"]
    confidence_in_range = (
        confidence_value is not None
        and confidence_review["expected_range"][0]
        <= confidence_value
        <= confidence_review["expected_range"][1]
    )
    assert value_row["confidence_in_expected_range"] is confidence_in_range
    assert value_row["low_confidence"] is (
        confidence_in_range
        and confidence_value < confidence_review["low_confidence_threshold"]
    )

    for row in details:
        _assert_correction_context_profile(row, correction)
        assert row["source"] == "confidence_review"
        assert row["sources"] == ["confidence_review"]
        assert row["operator_action"] in explanation_packet["operator_next_actions"]
        assert row["proposal_review_gate_status"] == correction[
            "operator_proposal_review_gate"
        ]["status"]
        assert row["offline_acceptance_allowed"] is correction[
            "operator_proposal_review_gate"
        ]["offline_acceptance_allowed"]
        assert row["recommended_decision"] == correction[
            "operator_proposal_review_gate"
        ]["recommended_decision"]
        assert row["required_before_operator_acceptance"] is True
        assert row["required_before_live_promotion"] is True
        assert row["required_before_queue_import"] is True
        assert row["read_only_or_dry_run_only"] is True
        assert row["live_apply_allowed"] is False
        assert row["queue_import_allowed"] is False
        assert row["live_authority_granted"] is False
        assert row["full_autonomous_ready"] is False
        assert row["readiness_closure_allowed"] is False
        assert row["can_authorize_live"] is False
        assert row["can_close_readiness"] is False
        assert row["can_import_to_queue"] is False
        assert row["mutates_mixer"] is False
        assert row["operator_review_required"] is True
        assert row["future_wing_readback_required"] is True
        assert row["blocks_live_promotion"] is True
        assert row["blocks_queue_import"] is True
        assert (
            row["denial_reason"]
            == correction["live_authority_boundary"]["denial_reason"]
        )

    for blocker, row in zip(expected_blockers, blocker_rows):
        assert row["action_type"] == "resolve_confidence_blocker"
        assert row["blockers"] == [blocker]
        assert row["operator_action"] == f"resolve_confidence_blocker:{blocker}"


def _assert_bounded_delta_details(correction):
    details = correction["bounded_delta_details"]
    bounded_delta = correction["bounded_delta"]
    explanation_packet = correction["operator_explanation_packet"]
    evidence_packet = correction["operator_evidence_packet"]
    review_summary = correction["operator_review_summary"]

    _assert_correction_context_profile(bounded_delta, correction)
    expected_deltas = bounded_delta.get("changes") or [bounded_delta]
    assert details
    assert len(details) == len(expected_deltas)
    assert explanation_packet["bounded_delta_details"] == details
    assert evidence_packet["bounded_delta_details"] == details
    assert review_summary["bounded_delta_details"] == details

    for index, (row, delta) in enumerate(zip(details, expected_deltas)):
        _assert_correction_context_profile(row, correction)
        assert row["detail"] == "bounded_delta_change"
        assert row["source"] == "bounded_delta"
        assert row["sources"] == ["bounded_delta"]
        assert row["change_index"] == index
        assert row["value_type"] == delta.get("value_type")
        assert row["delta_action_type"] == delta.get("action_type")
        assert row["target"] == delta.get("target")
        assert row["old_value"] == delta.get("old_value")
        assert row["proposed_value"] == delta.get("proposed_value")
        assert row["delta_db"] == delta.get("delta_db")
        assert row["parameter_deltas"] == delta.get("parameter_deltas", {})
        assert row["unit"] == delta.get("unit", bounded_delta.get("unit"))
        assert row["delta_unit"] == row["unit"]
        assert row["policy"] == delta.get("policy", bounded_delta.get("policy"))
        assert row["bounded_delta_policy"] == correction["bounded_delta_policy"]
        assert row["target_kind"] == correction["future_wing_readback_contract"][
            "target_kind"
        ]
        assert row["old_value_source"] == correction[
            "future_wing_readback_contract"
        ]["old_value_source"]
        assert row["readback_required"] is bool(
            delta.get("readback_required", True)
        )
        assert row["readback_required_before_live_promotion"] is True
        assert row["has_concrete_delta"] is (
            any(
                delta.get(key) is not None
                for key in ("delta_db", "old_value", "proposed_value")
            )
            or bool(delta.get("parameter_deltas", {}))
        )
        assert row["status"]
        assert row["action_type"] == "review_bounded_delta"
        assert row["operator_action"] == "review_bounded_delta_and_required_evidence"
        assert row["operator_action"] in explanation_packet["operator_next_actions"]
        assert row["safety_evidence"] == delta.get("safety_evidence", [])
        assert row["required_before_operator_acceptance"] is True
        assert row["required_before_live_promotion"] is True
        assert row["required_before_queue_import"] is True
        assert row["bounded_delta_required"] is True
        assert row["read_only_or_dry_run_only"] is True
        assert row["live_apply_allowed"] is False
        assert row["queue_import_allowed"] is False
        assert row["live_authority_granted"] is False
        assert row["full_autonomous_ready"] is False
        assert row["readiness_closure_allowed"] is False
        assert row["can_authorize_live"] is False
        assert row["can_close_readiness"] is False
        assert row["can_import_to_queue"] is False
        assert row["mutates_mixer"] is False
        assert row["operator_review_required"] is True
        assert row["future_wing_readback_required"] is True
        assert row["blocks_live_promotion"] is True
        assert row["blocks_queue_import"] is True
        assert (
            row["denial_reason"]
            == correction["live_authority_boundary"]["denial_reason"]
        )


def _assert_gain_fader_preview_denial(payload):
    assert payload["live_write"] is False
    assert payload["queue_mutation"] is False
    assert payload["can_import_to_queue"] is False
    assert payload["queue_import_allowed"] is False
    assert payload["can_apply_live"] is False
    assert payload["live_apply_allowed"] is False
    assert payload["read_only_or_dry_run_only"] is True
    assert payload["live_authority_granted"] is False
    assert payload["readiness_closure_allowed"] is False
    assert payload["mutates_mixer"] is False


def _assert_operator_correction_payload(proposal, objective_family):
    correction = proposal["operator_correction"]

    for field in REQUIRED_OPERATOR_FIELDS:
        assert field in correction

    assert correction["objective_family"] == objective_family
    assert correction["read_only_or_dry_run_only"] is True
    assert correction["live_apply_allowed"] is False
    assert correction["queue_import_allowed"] is False
    assert correction["can_apply_live"] is False
    assert correction["can_import_to_queue"] is False
    assert correction["live_authority_granted"] is False
    assert correction["full_autonomous_ready"] is False
    assert correction["readiness_closure_allowed"] is False
    assert correction["can_authorize_live"] is False
    assert correction["can_close_readiness"] is False
    assert correction["mutates_mixer"] is False
    assert correction["operator_review_required"] is True
    assert correction["future_wing_readback_required"] is True
    assert correction["blocks_live_promotion"] is True
    assert correction["blocks_queue_import"] is True
    assert correction["denial_reason"] == correction[
        "future_live_promotion_requirements"
    ]["blocked_until"]
    assert correction["operator_next_step"] == correction["evidence_gate"][
        "operator_action"
    ]
    effective_blockers = correction["effective_safety_blockers"]
    assert effective_blockers
    assert "live_authority_denied" in effective_blockers
    assert "real_wing_proof_required" in effective_blockers
    _assert_safety_blocker_details(correction)
    safety_blocker_details = correction["safety_blocker_details"]
    _assert_why_safe_details(correction)
    why_safe_details = correction["why_safe_details"]
    _assert_why_blocked_details(correction)
    why_blocked_details = correction["why_blocked_details"]
    future_readback_contract = correction["future_wing_readback_contract"]
    _assert_correction_context_profile(future_readback_contract, correction)
    assert future_readback_contract["required_before_live_promotion"] is True
    assert future_readback_contract["read_only_or_dry_run_only"] is True
    assert future_readback_contract["live_apply_allowed"] is False
    assert future_readback_contract["queue_import_allowed"] is False
    assert future_readback_contract["live_authority_granted"] is False
    assert future_readback_contract["full_autonomous_ready"] is False
    assert future_readback_contract["readiness_closure_allowed"] is False
    assert future_readback_contract["can_authorize_live"] is False
    assert future_readback_contract["can_close_readiness"] is False
    assert future_readback_contract["mutates_mixer"] is False
    assert future_readback_contract["operator_review_required"] is True
    assert future_readback_contract["operator_approval_required"] is True
    assert future_readback_contract["hardware_proof_required"] is True
    assert future_readback_contract["blocks_live_promotion"] is True
    assert future_readback_contract["blocks_queue_import"] is True
    assert future_readback_contract["future_wing_readback_required"] is True
    assert (
        future_readback_contract["denial_reason"]
        == "real_wing_hardware_readback_rollback_and_operator_review_required"
    )
    assert (
        future_readback_contract["operator_next_step"]
        == "capture_real_wing_readback_and_rollback_proof_before_live_promotion"
    )
    assert "live_apply_denied_until_real_wing_proof" in future_readback_contract[
        "why_blocked"
    ]
    assert "future_wing_readback_required" in future_readback_contract[
        "why_blocked"
    ]
    assert correction["headroom_feedback_true_peak_checks"][
        "readback_required_before_live_promotion"
    ] is True
    source_evidence = correction["source_evidence"]
    checks = correction["headroom_feedback_true_peak_checks"]
    _assert_correction_context_profile(source_evidence, correction)
    _assert_correction_context_profile(checks, correction)
    assert correction["method_id"] == source_evidence["method_id"]
    assert correction["source"] == source_evidence["source"]
    assert correction["target"] == source_evidence["target"]
    assert correction["value_type"] == source_evidence["contract_profile"][
        "value_type"
    ]
    assert correction["target_kind"] == source_evidence["contract_profile"][
        "target_kind"
    ]
    assert correction["old_value_source"] == source_evidence["contract_profile"][
        "old_value_source"
    ]
    assert correction["denial_reason"] == source_evidence["denial_reason"]
    assert correction["operator_next_step"] == source_evidence["operator_next_step"]
    assert correction["target_kind"] == future_readback_contract["target_kind"]
    assert correction["old_value_source"] == future_readback_contract[
        "old_value_source"
    ]
    for field in OPERATOR_IDENTITY_PROFILE_FIELDS:
        assert future_readback_contract[field] == correction[field]
    assert source_evidence["read_only_or_dry_run_only"] is True
    assert source_evidence["live_apply_allowed"] is False
    assert source_evidence["queue_import_allowed"] is False
    assert source_evidence["live_authority_granted"] is False
    assert source_evidence["full_autonomous_ready"] is False
    assert source_evidence["readiness_closure_allowed"] is False
    assert source_evidence["can_authorize_live"] is False
    assert source_evidence["can_close_readiness"] is False
    assert source_evidence["mutates_mixer"] is False
    assert source_evidence["operator_review_required"] is True
    assert source_evidence["blocks_live_promotion"] is True
    assert source_evidence["blocks_queue_import"] is True
    assert source_evidence["future_wing_readback_required"] is True
    _assert_raw_input_evidence_boundary(
        source_evidence["raw_input_evidence_boundary"],
        surface="source_evidence",
        evidence_fields=set(source_evidence["raw"]) | set(source_evidence["evidence"]),
    )
    assert source_evidence["denial_reason"] == correction[
        "future_live_promotion_requirements"
    ]["blocked_until"]
    assert source_evidence["operator_next_step"] == correction["evidence_gate"][
        "operator_action"
    ]
    assert source_evidence["future_live_promotion_requires"] == correction[
        "future_live_promotion_requirements"
    ]["required_proofs"]
    assert "live_apply_denied_until_real_wing_proof" in source_evidence[
        "why_blocked"
    ]
    assert "future_wing_readback_required" in source_evidence["why_blocked"]
    assert checks["read_only_or_dry_run_only"] is True
    assert checks["live_write"] is False
    assert checks["queue_mutation"] is False
    assert checks["live_apply_allowed"] is False
    assert checks["queue_import_allowed"] is False
    assert checks["live_authority_granted"] is False
    assert checks["full_autonomous_ready"] is False
    assert checks["readiness_closure_allowed"] is False
    assert checks["can_authorize_live"] is False
    assert checks["can_close_readiness"] is False
    assert checks["mutates_mixer"] is False
    assert checks["operator_review_required"] is True
    assert checks["blocks_live_promotion"] is True
    assert checks["blocks_queue_import"] is True
    assert checks["future_wing_readback_required"] is True
    assert checks["denial_reason"] == correction["future_live_promotion_requirements"][
        "blocked_until"
    ]
    assert checks["operator_next_step"] == correction["evidence_gate"][
        "operator_action"
    ]
    assert checks["future_live_promotion_requires"] == correction[
        "future_live_promotion_requirements"
    ]["required_proofs"]
    assert "live_apply_denied_until_real_wing_proof" in checks["why_blocked"]
    assert "future_wing_readback_required" in checks["why_blocked"]
    required_evidence = source_evidence["required_evidence_fields"]
    assert required_evidence
    assert checks["required_evidence_fields"] == required_evidence
    assert future_readback_contract["required_evidence_fields"] == required_evidence
    assert source_evidence["contract_profile"]["target_kind"] == future_readback_contract[
        "target_kind"
    ]
    rollback_contract = correction["rollback_readback_contract"]
    assert rollback_contract["description"] == correction["rollback_expectation"]
    assert rollback_contract["rollback_expectation"] == correction[
        "rollback_expectation"
    ]
    assert rollback_contract["target_kind"] == future_readback_contract["target_kind"]
    assert rollback_contract["old_value_source"] == future_readback_contract[
        "old_value_source"
    ]
    for field in OPERATOR_IDENTITY_PROFILE_FIELDS:
        assert rollback_contract[field] == correction[field]
    assert rollback_contract["readback_targets"] == future_readback_contract[
        "readback_targets"
    ]
    assert rollback_contract["blocked_live_actions"] == future_readback_contract[
        "blocked_live_actions"
    ]
    assert rollback_contract["required_evidence_fields"] == required_evidence
    assert rollback_contract["missing_required_evidence"] == correction[
        "evidence_gate"
    ]["missing_required_evidence"]
    assert rollback_contract["required_proofs"] == correction[
        "future_live_promotion_requirements"
    ]["required_proofs"]
    assert rollback_contract["rollback_required_before_promotion"] is True
    assert rollback_contract["readback_required_before_live_promotion"] is True
    assert rollback_contract["operator_review_required"] is True
    assert rollback_contract["operator_approval_required"] is True
    assert rollback_contract["hardware_proof_required"] is True
    assert rollback_contract["read_only_or_dry_run_only"] is True
    assert rollback_contract["live_apply_allowed"] is False
    assert rollback_contract["queue_import_allowed"] is False
    assert rollback_contract["live_authority_granted"] is False
    assert rollback_contract["full_autonomous_ready"] is False
    assert rollback_contract["readiness_closure_allowed"] is False
    assert rollback_contract["can_authorize_live"] is False
    assert rollback_contract["can_close_readiness"] is False
    assert rollback_contract["mutates_mixer"] is False
    assert rollback_contract["blocks_live_promotion"] is True
    assert rollback_contract["blocks_queue_import"] is True
    assert rollback_contract["future_wing_readback_required"] is True
    assert rollback_contract["denial_reason"] == correction[
        "future_live_promotion_requirements"
    ]["blocked_until"]
    assert (
        rollback_contract["operator_next_step"]
        == "capture_real_wing_readback_and_rollback_proof_before_live_promotion"
    )
    assert "rollback_readback_proof_required" in rollback_contract["why_blocked"]
    completeness = source_evidence["evidence_completeness"]
    assert completeness["required"] == required_evidence
    assert completeness["status"] in {"complete", "partial", "missing"}
    assert completeness["missing"] == [
        field for field in required_evidence if field in completeness["missing"]
    ]
    required_evidence_status = source_evidence["required_evidence_status"]
    _assert_required_evidence_status(
        required_evidence_status,
        completeness=completeness,
        evidence_gate=correction["evidence_gate"],
    )
    _assert_source_evidence_details(correction)
    assert rollback_contract["required_evidence_status"] == required_evidence_status
    assert checks["evidence_completeness"] == completeness
    assert checks["required_evidence_status"] == required_evidence_status
    assert source_evidence["evidence_gate"] == correction["evidence_gate"]
    assert checks["evidence_gate"] == correction["evidence_gate"]
    evidence_gate = correction["evidence_gate"]
    _assert_correction_context_profile(evidence_gate, correction)
    assert evidence_gate["read_only_or_dry_run_only"] is True
    assert evidence_gate["live_apply_allowed"] is False
    assert evidence_gate["queue_import_allowed"] is False
    assert evidence_gate["live_authority_granted"] is False
    assert evidence_gate["full_autonomous_ready"] is False
    assert evidence_gate["readiness_closure_allowed"] is False
    assert evidence_gate["can_authorize_live"] is False
    assert evidence_gate["can_close_readiness"] is False
    assert evidence_gate["mutates_mixer"] is False
    assert evidence_gate["operator_review_required"] is True
    assert evidence_gate["blocks_live_promotion"] is True
    assert evidence_gate["blocks_queue_import"] is True
    assert evidence_gate["future_wing_readback_required"] is True
    assert (
        evidence_gate["denial_reason"]
        == "operator_review_and_future_wing_readback_required"
    )
    _assert_correction_context_profile(
        correction["live_authority_boundary"],
        correction,
    )
    safety_matrix = correction["operator_safety_review_matrix"]
    _assert_correction_context_profile(safety_matrix, correction)
    assert safety_matrix["read_only_or_dry_run_only"] is True
    assert safety_matrix["live_apply_allowed"] is False
    assert safety_matrix["queue_import_allowed"] is False
    assert safety_matrix["live_authority_granted"] is False
    assert safety_matrix["full_autonomous_ready"] is False
    assert safety_matrix["readiness_closure_allowed"] is False
    assert safety_matrix["can_authorize_live"] is False
    assert safety_matrix["can_close_readiness"] is False
    assert safety_matrix["mutates_mixer"] is False
    assert safety_matrix["operator_review_required"] is True
    assert safety_matrix["future_wing_readback_required"] is True
    assert safety_matrix["blocks_live_promotion"] is True
    assert safety_matrix["blocks_queue_import"] is True
    assert (
        safety_matrix["denial_reason"]
        == "operator_review_and_future_wing_readback_required"
    )
    for field in OPERATOR_IDENTITY_PROFILE_FIELDS:
        assert safety_matrix[field] == correction[field]
    assert safety_matrix["evidence_gate_status"] == correction["evidence_gate"][
        "status"
    ]
    assert safety_matrix["operator_action"] == correction["evidence_gate"][
        "operator_action"
    ]
    assert safety_matrix["required_evidence_fields"] == required_evidence
    assert safety_matrix["provided_evidence_fields"] == completeness["provided"]
    assert safety_matrix["missing_required_evidence"] == correction[
        "evidence_gate"
    ]["missing_required_evidence"]
    assert safety_matrix["applicable_safety_checks"] == correction[
        "headroom_feedback_true_peak_checks"
    ]["required_checks"]
    assert safety_matrix["summary"]["check_count"] == len(
        safety_matrix["applicable_safety_checks"]
    )
    assert safety_matrix["summary"]["check_count"] == len(safety_matrix["checks"])
    assert safety_matrix["summary"]["checks_with_missing_evidence"] == sum(
        1
        for row in safety_matrix["checks"]
        if row["status"] == "missing_required_evidence"
    )
    assert safety_matrix["summary"]["checks_with_direct_evidence"] == sum(
        1 for row in safety_matrix["checks"] if row["required_evidence_fields"]
    )
    assert safety_matrix["summary"]["checks_requiring_mapping_review"] == sum(
        1 for row in safety_matrix["checks"] if row["mapping_review_required"]
    )
    for row in safety_matrix["checks"]:
        for field in OPERATOR_IDENTITY_PROFILE_FIELDS:
            assert row[field] == correction[field]
        assert row["check"] in safety_matrix["applicable_safety_checks"]
        assert row["status"] in {
            "evidence_available_operator_review_required",
            "missing_required_evidence",
            "operator_review_required_no_direct_evidence_mapping",
        }
        assert row["operator_action"] == correction["evidence_gate"][
            "operator_action"
        ]
        assert row["evidence_mapping_status"] in {
            "direct",
            "no_direct_required_evidence_field_match",
        }
        assert row["available_contract_evidence_fields"] == sorted(
            completeness["provided"]
        )
        assert row["all_contract_evidence_fields"] == required_evidence
        assert row["mapping_review_required"] == (
            row["evidence_mapping_status"]
            == "no_direct_required_evidence_field_match"
        )
        if row["mapping_review_required"]:
            assert row["required_evidence_fields"] == []
            assert "full evidence packet" in row["review_note"]
        else:
            assert row["required_evidence_fields"]
            assert "Direct required evidence" in row["review_note"]
        assert row["read_only_or_dry_run_only"] is True
        assert row["live_apply_allowed"] is False
        assert row["queue_import_allowed"] is False
        assert row["live_authority_granted"] is False
        assert row["full_autonomous_ready"] is False
        assert row["readiness_closure_allowed"] is False
        assert row["can_authorize_live"] is False
        assert row["can_close_readiness"] is False
        assert row["mutates_mixer"] is False
        assert row["operator_review_required"] is True
        assert row["future_wing_readback_required"] is True
        assert (
            row["denial_reason"]
            == "operator_review_and_future_wing_readback_required"
        )
        assert row["blocks_live_promotion"] is True
        assert row["blocks_queue_import"] is True
        assert set(row["required_evidence_fields"]) <= set(required_evidence)
        assert set(row["provided_evidence_fields"]) <= set(completeness["provided"])
        assert set(row["missing_required_evidence"]) <= set(
            correction["evidence_gate"]["missing_required_evidence"]
        )
    evidence_packet = correction["operator_evidence_packet"]
    _assert_correction_context_profile(evidence_packet, correction)
    assert evidence_packet["packet_type"] == "offline_operator_correction_review"
    assert evidence_packet["read_only_or_dry_run_only"] is True
    assert evidence_packet["live_apply_allowed"] is False
    assert evidence_packet["queue_import_allowed"] is False
    assert evidence_packet["source_evidence_required"] is True
    assert evidence_packet["source_evidence_details_required"] is True
    assert evidence_packet["problem_statement_required"] is True
    assert evidence_packet["proposed_correction_required"] is True
    assert evidence_packet["confidence_required"] is True
    assert evidence_packet["confidence_review_required"] is True
    assert evidence_packet["confidence_review_details_required"] is True
    assert evidence_packet["operator_proposal_review_gate_required"] is True
    assert evidence_packet["bounded_delta_required"] is True
    assert evidence_packet["bounded_delta_details_required"] is True
    assert evidence_packet["safety_blockers_required"] is True
    assert evidence_packet["safety_checks_required"] is True
    assert evidence_packet["headroom_feedback_true_peak_checks_required"] is True
    assert (
        evidence_packet["headroom_feedback_true_peak_check_details_required"]
        is True
    )
    assert evidence_packet["operator_safety_review_matrix_required"] is True
    assert evidence_packet["operator_readable_explanation_required"] is True
    assert evidence_packet["operator_explanation_packet_required"] is True
    assert evidence_packet["operator_next_action_details_required"] is True
    assert evidence_packet["rollback_expectation_required"] is True
    assert evidence_packet["rollback_readback_contract_required"] is True
    assert evidence_packet["rollback_readback_details_required"] is True
    assert evidence_packet["future_live_promotion_requirements_required"] is True
    assert evidence_packet["future_live_requirement_details_required"] is True
    assert evidence_packet["future_wing_readback_contract_required"] is True
    assert evidence_packet["live_authority_boundary_required"] is True
    assert evidence_packet["operator_review_required"] is True
    assert evidence_packet["operator_review_record_template_required"] is True
    assert evidence_packet["operator_review_record_field_details_required"] is True
    assert evidence_packet["operator_review_checklist_details_required"] is True
    assert evidence_packet["operator_review_decision_option_details_required"] is True
    assert evidence_packet["method_id"] == correction["source_evidence"]["method_id"]
    assert evidence_packet["objective_family"] == objective_family
    assert evidence_packet["source"] == correction["source_evidence"]["source"]
    assert evidence_packet["target"] == correction["source_evidence"]["target"]
    assert evidence_packet["value_type"] == correction["value_type"]
    assert evidence_packet["target_kind"] == correction["target_kind"]
    assert evidence_packet["old_value_source"] == correction["old_value_source"]
    assert evidence_packet["bounded_delta_unit"] == correction["bounded_delta_unit"]
    assert evidence_packet["bounded_delta_policy"] == correction["bounded_delta_policy"]
    assert evidence_packet["evidence_completeness"] == completeness
    assert evidence_packet["required_evidence_status"] == required_evidence_status
    assert evidence_packet["required_evidence_actions"] == required_evidence_status[
        "required_evidence_actions"
    ]
    assert evidence_packet["evidence_gate_status"] == correction["evidence_gate"][
        "status"
    ]
    assert evidence_packet["missing_required_evidence"] == correction[
        "evidence_gate"
    ]["missing_required_evidence"]
    assert evidence_packet["required_evidence_fields"] == required_evidence
    assert evidence_packet["provided_evidence_fields"] == completeness["provided"]
    assert evidence_packet["operator_action"] == correction["evidence_gate"][
        "operator_action"
    ]
    review_summary = correction["operator_review_summary"]
    _assert_correction_context_profile(review_summary, correction)
    measured_problem = correction["measured_or_audible_problem"]
    _assert_correction_context_profile(measured_problem, correction)
    assert evidence_packet["problem_statement"] == measured_problem["summary"]
    assert measured_problem["source"] == correction["source_evidence"]["source"]
    assert measured_problem["method_id"] == correction["source_evidence"]["method_id"]
    assert measured_problem["target"] == correction["source_evidence"]["target"]
    assert measured_problem["value_type"] == correction["value_type"]
    assert measured_problem["objective_family"] == objective_family
    assert measured_problem["target_kind"] == correction["source_evidence"][
        "contract_profile"
    ]["target_kind"]
    assert measured_problem["old_value_source"] == correction["old_value_source"]
    assert measured_problem["bounded_delta_unit"] == correction["bounded_delta_unit"]
    assert measured_problem["bounded_delta_policy"] == correction[
        "bounded_delta_policy"
    ]
    assert measured_problem["evidence_completeness_status"] == completeness["status"]
    assert measured_problem["evidence_complete"] == completeness["complete"]
    assert measured_problem["required_evidence_fields"] == required_evidence
    assert measured_problem["provided_evidence_fields"] == completeness["provided"]
    assert measured_problem["missing_required_evidence"] == correction[
        "evidence_gate"
    ]["missing_required_evidence"]
    assert measured_problem["required_evidence_status"] == required_evidence_status
    assert measured_problem["required_evidence_actions"] == required_evidence_status[
        "required_evidence_actions"
    ]
    assert measured_problem["evidence_gate_status"] == correction["evidence_gate"][
        "status"
    ]
    assert measured_problem["operator_action"] == correction["evidence_gate"][
        "operator_action"
    ]
    assert measured_problem["problem_statement_requires_operator_review"] is True
    assert measured_problem["problem_statement_can_authorize_live"] is False
    assert measured_problem["read_only_or_dry_run_only"] is True
    assert measured_problem["live_apply_allowed"] is False
    assert measured_problem["queue_import_allowed"] is False
    assert measured_problem["live_authority_granted"] is False
    assert measured_problem["full_autonomous_ready"] is False
    assert measured_problem["readiness_closure_allowed"] is False
    assert measured_problem["can_authorize_live"] is False
    assert measured_problem["can_close_readiness"] is False
    assert measured_problem["mutates_mixer"] is False
    assert measured_problem["operator_review_required"] is True
    assert measured_problem["future_wing_readback_required"] is True
    assert measured_problem["blocks_live_promotion"] is True
    assert measured_problem["blocks_queue_import"] is True
    assert measured_problem["denial_reason"] == correction[
        "future_live_promotion_requirements"
    ]["blocked_until"]
    assert "problem_statement_is_offline_evidence_only" in measured_problem[
        "why_blocked"
    ]
    proposed_correction = correction["proposed_correction"]
    _assert_correction_context_profile(proposed_correction, correction)
    assert proposed_correction["source"] == correction["source"]
    assert proposed_correction["target"] == correction["target"]
    assert proposed_correction["contract_value_type"] == correction["value_type"]
    assert proposed_correction["contract_target_kind"] == correction["target_kind"]
    assert proposed_correction["contract_old_value_source"] == correction[
        "old_value_source"
    ]
    assert proposed_correction["proposal_review_status"] == (
        "proposed_correction_requires_operator_review"
    )
    assert proposed_correction["bounded_delta"] == correction["bounded_delta"]
    assert proposed_correction["bounded_delta_policy"] == correction[
        "bounded_delta_policy"
    ]
    assert proposed_correction["bounded_delta_unit"] == correction[
        "bounded_delta_unit"
    ]
    assert proposed_correction["rollback_expectation"] == correction[
        "rollback_expectation"
    ]
    assert proposed_correction["rollback_readback_contract"] == correction[
        "rollback_readback_contract"
    ]
    assert proposed_correction["required_evidence_fields"] == required_evidence
    assert proposed_correction["provided_evidence_fields"] == completeness["provided"]
    assert proposed_correction["missing_required_evidence"] == correction[
        "evidence_gate"
    ]["missing_required_evidence"]
    assert proposed_correction["required_evidence_status"] == required_evidence_status
    assert proposed_correction["required_evidence_actions"] == required_evidence_status[
        "required_evidence_actions"
    ]
    assert proposed_correction["evidence_gate_status"] == correction[
        "evidence_gate"
    ]["status"]
    assert proposed_correction["operator_action"] == correction["evidence_gate"][
        "operator_action"
    ]
    assert proposed_correction["readback_required_before_live_promotion"] is True
    assert proposed_correction["future_wing_readback_contract"] == correction[
        "future_wing_readback_contract"
    ]
    assert proposed_correction["future_live_promotion_requires"] == correction[
        "future_live_promotion_requirements"
    ]["required_proofs"]
    assert proposed_correction["read_only_or_dry_run_only"] is True
    assert proposed_correction["live_apply_allowed"] is False
    assert proposed_correction["queue_import_allowed"] is False
    assert proposed_correction["can_apply_live"] is False
    assert proposed_correction["can_import_to_queue"] is False
    assert proposed_correction["live_authority_granted"] is False
    assert proposed_correction["full_autonomous_ready"] is False
    assert proposed_correction["readiness_closure_allowed"] is False
    assert proposed_correction["can_authorize_live"] is False
    assert proposed_correction["can_close_readiness"] is False
    assert proposed_correction["mutates_mixer"] is False
    assert proposed_correction["operator_review_required"] is True
    assert proposed_correction["blocks_live_promotion"] is True
    assert proposed_correction["blocks_queue_import"] is True
    assert proposed_correction["future_wing_readback_required"] is True
    assert proposed_correction["denial_reason"] == correction[
        "future_live_promotion_requirements"
    ]["blocked_until"]
    assert proposed_correction["operator_next_step"] == (
        "review_proposed_correction_bounded_delta_and_required_evidence"
    )
    assert "proposed_correction_requires_operator_review" in proposed_correction[
        "why_blocked"
    ]
    assert evidence_packet["proposed_correction_summary"] == review_summary[
        "proposed_correction"
    ]
    assert evidence_packet["confidence"] == correction["confidence"]
    confidence_review = correction["confidence_review"]
    assert confidence_review["raw_confidence"] == correction["confidence"]
    assert confidence_review["status"] in {
        "missing_confidence",
        "invalid_confidence_range",
        "low_confidence_operator_review_required",
        "confidence_available_operator_review_required",
    }
    assert confidence_review["expected_range"] == [0.0, 1.0]
    assert confidence_review["low_confidence_threshold"] == 0.5
    assert confidence_review["read_only_or_dry_run_only"] is True
    assert confidence_review["live_apply_allowed"] is False
    assert confidence_review["queue_import_allowed"] is False
    assert confidence_review["live_authority_granted"] is False
    assert confidence_review["full_autonomous_ready"] is False
    assert confidence_review["readiness_closure_allowed"] is False
    assert confidence_review["can_authorize_live"] is False
    assert confidence_review["can_close_readiness"] is False
    assert confidence_review["mutates_mixer"] is False
    assert confidence_review["blocks_live_promotion"] is True
    assert confidence_review["blocks_queue_import"] is True
    assert confidence_review["operator_review_required"] is True
    assert confidence_review["future_wing_readback_required"] is True
    assert evidence_packet["confidence_review"] == confidence_review
    _assert_confidence_review_details(correction)
    review_gate = correction["operator_proposal_review_gate"]
    _assert_correction_context_profile(review_gate, correction)
    assert review_gate["status"] in {
        "blocked_until_required_evidence_complete",
        "blocked_until_confidence_available",
        "blocked_until_confidence_normalized",
        "low_confidence_requires_operator_recheck",
        "offline_proposal_review_ready",
    }
    assert review_gate["recommended_decision"] in {
        "request_more_evidence",
        "accept_as_offline_proposal",
    }
    assert review_gate["evidence_gate_status"] == correction["evidence_gate"][
        "status"
    ]
    assert review_gate["confidence_review_status"] == confidence_review["status"]
    assert review_gate["missing_required_evidence"] == correction["evidence_gate"][
        "missing_required_evidence"
    ]
    assert review_gate["confidence_blockers"] == confidence_review["blockers"]
    assert review_gate["operator_review_required"] is True
    assert review_gate["requires_operator_decision_record"] is True
    assert review_gate["read_only_or_dry_run_only"] is True
    assert review_gate["live_apply_allowed"] is False
    assert review_gate["queue_import_allowed"] is False
    assert review_gate["live_authority_granted"] is False
    assert review_gate["full_autonomous_ready"] is False
    assert review_gate["readiness_closure_allowed"] is False
    assert review_gate["blocks_live_promotion"] is True
    assert review_gate["blocks_queue_import"] is True
    assert review_gate["can_authorize_live"] is False
    assert review_gate["can_close_readiness"] is False
    assert review_gate["can_import_to_queue"] is False
    assert review_gate["mutates_mixer"] is False
    assert review_gate["future_wing_readback_required"] is True
    assert (
        review_gate["denial_reason"]
        == "operator_review_and_future_wing_readback_required"
    )
    for field in OPERATOR_IDENTITY_PROFILE_FIELDS:
        assert review_gate[field] == correction[field]
    assert review_gate["effective_safety_blockers"] == effective_blockers
    assert review_gate["safety_blocker_details"] == safety_blocker_details
    assert "live_authority_denied" in review_gate["live_authority_blockers"]
    assert evidence_packet["operator_proposal_review_gate"] == review_gate
    bounded_delta = correction["bounded_delta"]
    bounded_delta_contracts = [bounded_delta]
    bounded_delta_contracts.extend(
        change
        for change in bounded_delta.get("changes", [])
        if isinstance(change, dict)
    )
    for delta_contract in bounded_delta_contracts:
        assert delta_contract["read_only_or_dry_run_only"] is True
        assert delta_contract["live_apply_allowed"] is False
        assert delta_contract["queue_import_allowed"] is False
        assert delta_contract["live_authority_granted"] is False
        assert delta_contract["full_autonomous_ready"] is False
        assert delta_contract["readiness_closure_allowed"] is False
        assert delta_contract["can_authorize_live"] is False
        assert delta_contract["can_close_readiness"] is False
        assert delta_contract["mutates_mixer"] is False
        assert delta_contract["operator_review_required"] is True
        assert delta_contract["blocks_live_promotion"] is True
        assert delta_contract["blocks_queue_import"] is True
        assert delta_contract["future_wing_readback_required"] is True
        assert delta_contract["denial_reason"] == correction[
            "future_live_promotion_requirements"
        ]["blocked_until"]
        assert (
            delta_contract["operator_next_step"]
            == "review_bounded_delta_and_required_evidence"
        )
        assert delta_contract["future_live_promotion_requires"] == correction[
            "future_live_promotion_requirements"
        ]["required_proofs"]
        assert "live_apply_denied_until_real_wing_proof" in delta_contract[
            "why_blocked"
        ]
        assert "future_wing_readback_required" in delta_contract["why_blocked"]
    _assert_bounded_delta_details(correction)
    assert evidence_packet["bounded_delta"] == correction["bounded_delta"]
    assert evidence_packet["bounded_delta_policy"] == correction["bounded_delta_policy"]
    assert evidence_packet["safety_blockers"] == correction["safety_blockers"]
    assert evidence_packet["effective_safety_blockers"] == effective_blockers
    assert evidence_packet["safety_blocker_details"] == safety_blocker_details
    assert evidence_packet["why_safe_details"] == why_safe_details
    assert evidence_packet["why_blocked_details"] == why_blocked_details
    assert evidence_packet["headroom_feedback_true_peak_checks"] == correction[
        "headroom_feedback_true_peak_checks"
    ]
    assert evidence_packet["operator_safety_review_matrix"] == safety_matrix
    assert evidence_packet["operator_readable_explanation"] == correction[
        "operator_readable_explanation"
    ]
    explanation_packet = correction["operator_explanation_packet"]
    assert explanation_packet["packet_type"] == (
        "operator_readable_correction_explanation"
    )
    _assert_correction_context_profile(explanation_packet, correction)
    assert explanation_packet["audience"] == "sound_engineer"
    assert explanation_packet["mode"] == "offline_dry_run_proposal_only"
    for summary_key in (
        "source_evidence_summary",
        "confidence_review_summary",
        "safety_review_summary",
        "future_live_requirements",
        "live_authority_boundary_summary",
        "future_wing_readback_summary",
        "future_live_proof_contract_summary",
        "operator_review_record_summary",
    ):
        _assert_correction_context_profile(
            explanation_packet[summary_key],
            correction,
        )
    for packet in (explanation_packet, review_summary):
        assert packet["method_id"] == correction["method_id"]
        assert packet["source"] == correction["source"]
        assert packet["target"] == correction["target"]
        assert packet["value_type"] == correction["value_type"]
        assert packet["target_kind"] == correction["target_kind"]
        assert packet["old_value_source"] == correction["old_value_source"]
        assert packet["objective_family"] == correction["objective_family"]
        assert packet["bounded_delta_unit"] == correction["bounded_delta_unit"]
        assert packet["bounded_delta_policy"] == correction["bounded_delta_policy"]
    assert explanation_packet["problem"] == review_summary["problem"]
    assert explanation_packet["proposed_correction"] == review_summary[
        "proposed_correction"
    ]
    source_summary = explanation_packet["source_evidence_summary"]
    source_evidence = correction["source_evidence"]
    assert source_summary["source"] == source_evidence["source"]
    assert source_summary["method_id"] == source_evidence["method_id"]
    assert source_summary["target"] == source_evidence["target"]
    assert source_summary["value_type"] == correction["value_type"]
    assert source_summary["objective_family"] == objective_family
    assert source_summary["target_kind"] == source_evidence["contract_profile"][
        "target_kind"
    ]
    assert source_summary["old_value_source"] == source_evidence[
        "contract_profile"
    ]["old_value_source"]
    assert source_summary["required_evidence_fields"] == required_evidence
    assert source_summary["provided_evidence_fields"] == completeness["provided"]
    assert source_summary["missing_required_evidence"] == correction[
        "evidence_gate"
    ]["missing_required_evidence"]
    assert source_summary["evidence_completeness_status"] == completeness["status"]
    assert source_summary["evidence_complete"] == completeness["complete"]
    assert source_summary["evidence_gate_status"] == correction["evidence_gate"][
        "status"
    ]
    assert source_summary["required_evidence_status"] == required_evidence_status
    assert source_summary["operator_action"] == correction["evidence_gate"][
        "operator_action"
    ]
    assert source_summary["raw_evidence_fields"] == sorted(
        source_evidence["raw"].keys()
    )
    assert source_summary["structured_evidence_fields"] == sorted(
        source_evidence["evidence"].keys()
    )
    _assert_raw_input_evidence_boundary(
        source_summary["raw_input_evidence_boundary"],
        surface="source_evidence",
        evidence_fields=set(source_evidence["raw"]) | set(source_evidence["evidence"]),
    )
    assert source_summary["read_only_or_dry_run_only"] is True
    assert source_summary["live_apply_allowed"] is False
    assert source_summary["queue_import_allowed"] is False
    assert explanation_packet["bounded_delta"] == correction["bounded_delta"]
    assert explanation_packet["bounded_delta_policy"] == correction[
        "bounded_delta_policy"
    ]
    assert explanation_packet["evidence_status"] == correction["evidence_gate"][
        "status"
    ]
    assert explanation_packet["required_evidence_status"] == required_evidence_status
    assert explanation_packet["required_evidence_actions"] == required_evidence_status[
        "required_evidence_actions"
    ]
    for action in required_evidence_status["operator_next_actions"]:
        assert action in explanation_packet["operator_next_actions"]
    assert explanation_packet["missing_required_evidence"] == correction[
        "evidence_gate"
    ]["missing_required_evidence"]
    assert explanation_packet["confidence_status"] == confidence_review["status"]
    assert explanation_packet["confidence_blockers"] == confidence_review["blockers"]
    confidence_summary = explanation_packet["confidence_review_summary"]
    assert confidence_summary["raw_confidence"] == confidence_review["raw_confidence"]
    assert confidence_summary["confidence"] == confidence_review["confidence"]
    assert confidence_summary["status"] == confidence_review["status"]
    assert confidence_summary["expected_range"] == confidence_review["expected_range"]
    assert confidence_summary["low_confidence_threshold"] == confidence_review[
        "low_confidence_threshold"
    ]
    assert confidence_summary["blockers"] == confidence_review["blockers"]
    assert confidence_summary["operator_action"] == confidence_review[
        "operator_action"
    ]
    assert confidence_summary["operator_review_required"] is True
    assert confidence_summary["blocks_live_promotion"] is True
    assert confidence_summary["blocks_queue_import"] is True
    assert confidence_summary["read_only_or_dry_run_only"] is True
    assert confidence_summary["live_apply_allowed"] is False
    assert confidence_summary["queue_import_allowed"] is False
    assert confidence_summary["live_authority_granted"] is False
    assert confidence_summary["full_autonomous_ready"] is False
    assert confidence_summary["readiness_closure_allowed"] is False
    assert confidence_summary["can_authorize_live"] is False
    assert confidence_summary["can_close_readiness"] is False
    assert confidence_summary["mutates_mixer"] is False
    assert confidence_summary["future_wing_readback_required"] is True
    assert explanation_packet["proposal_review_gate_status"] == review_gate["status"]
    assert explanation_packet["offline_acceptance_allowed"] == review_gate[
        "offline_acceptance_allowed"
    ]
    assert explanation_packet["recommended_decision"] == review_gate[
        "recommended_decision"
    ]
    assert explanation_packet["safety_checks"] == correction[
        "headroom_feedback_true_peak_checks"
    ]["required_checks"]
    assert explanation_packet["safety_review_status"] == safety_matrix[
        "evidence_gate_status"
    ]
    assert explanation_packet["safety_review_operator_action"] == safety_matrix[
        "operator_action"
    ]
    assert explanation_packet["safety_review_summary"] == safety_matrix["summary"]
    assert explanation_packet["safety_review_checks"] == safety_matrix["checks"]
    assert explanation_packet["missing_safety_evidence"] == safety_matrix[
        "missing_required_evidence"
    ]
    assert explanation_packet["safety_blockers"] == correction["safety_blockers"]
    assert explanation_packet["effective_safety_blockers"] == effective_blockers
    assert explanation_packet["safety_blocker_details"] == safety_blocker_details
    assert explanation_packet["why_safe_details"] == why_safe_details
    assert [row["reason"] for row in why_safe_details] == explanation_packet[
        "why_safe"
    ]
    assert explanation_packet["why_blocked_details"] == why_blocked_details
    for detail in why_blocked_details:
        for action in detail["operator_actions"]:
            assert action in explanation_packet["operator_next_actions"]
    for detail in safety_blocker_details:
        for action in detail["operator_actions"]:
            assert action in explanation_packet["operator_next_actions"]
    assert explanation_packet["rollback_expectation"] == correction[
        "rollback_expectation"
    ]
    assert explanation_packet["rollback_readback_contract"] == rollback_contract
    assert "read_only_or_dry_run_only" in explanation_packet["why_safe"]
    assert "live_apply_denied_until_real_wing_proof" in explanation_packet[
        "why_blocked"
    ]
    assert "do_not_queue_or_apply_live" in explanation_packet[
        "operator_next_actions"
    ]
    assert "confirm_rollback_expectation" in explanation_packet[
        "operator_next_actions"
    ]
    assert explanation_packet["future_live_requirements"][
        "required_before_live_promotion"
    ] is True
    assert explanation_packet["future_live_requirements"][
        "live_authority_granted"
    ] is False
    assert explanation_packet["future_live_requirements"][
        "readiness_closure_allowed"
    ] is False
    assert explanation_packet["future_live_requirements"][
        "read_only_or_dry_run_only"
    ] is True
    assert explanation_packet["future_live_requirements"][
        "live_apply_allowed"
    ] is False
    assert explanation_packet["future_live_requirements"][
        "queue_import_allowed"
    ] is False
    assert explanation_packet["future_live_requirements"][
        "full_autonomous_ready"
    ] is False
    assert explanation_packet["future_live_requirements"][
        "can_authorize_live"
    ] is False
    assert explanation_packet["future_live_requirements"][
        "can_close_readiness"
    ] is False
    assert explanation_packet["future_live_requirements"]["mutates_mixer"] is False
    assert explanation_packet["future_live_requirements"][
        "operator_review_required"
    ] is True
    assert explanation_packet["future_live_requirements"][
        "future_wing_readback_required"
    ] is True
    assert explanation_packet["future_live_requirements"][
        "blocks_live_promotion"
    ] is True
    assert explanation_packet["future_live_requirements"][
        "blocks_queue_import"
    ] is True
    assert explanation_packet["future_live_requirements"][
        "denial_reason"
    ] == correction["live_authority_boundary"]["denial_reason"]
    assert explanation_packet["future_live_requirements"][
        "wing_proof_required"
    ] is True
    assert explanation_packet["future_live_requirements"][
        "readback_targets"
    ] == correction["future_wing_readback_contract"]["readback_targets"]
    for target in explanation_packet["future_live_requirements"][
        "readback_targets"
    ]:
        assert f"verify_future_wing_readback_target:{target}" in explanation_packet[
            "operator_next_actions"
        ]
    assert explanation_packet["future_live_requirements"][
        "blocked_live_actions"
    ] == correction["future_wing_readback_contract"]["blocked_live_actions"]
    assert explanation_packet["future_live_requirements"][
        "required_evidence_status"
    ] == required_evidence_status
    assert {
        "real_wing_readonly_connection_artifact",
        "physical_console_readback_for_old_and_proposed_values",
    } <= set(explanation_packet["future_live_requirements"]["required_proofs"])
    for proof in explanation_packet["future_live_requirements"]["required_proofs"]:
        assert f"collect_future_live_required_proof:{proof}" in explanation_packet[
            "operator_next_actions"
        ]
    readback_summary = explanation_packet["future_wing_readback_summary"]
    future_readback = correction["future_wing_readback_contract"]
    assert readback_summary["required_before_live_promotion"] is True
    assert readback_summary["target_kind"] == future_readback["target_kind"]
    assert readback_summary["old_value_source"] == future_readback[
        "old_value_source"
    ]
    assert readback_summary["readback_targets"] == future_readback[
        "readback_targets"
    ]
    assert readback_summary["required_evidence_fields"] == future_readback[
        "required_evidence_fields"
    ]
    assert readback_summary["missing_required_evidence"] == correction[
        "evidence_gate"
    ]["missing_required_evidence"]
    assert readback_summary["required_evidence_status"] == required_evidence_status
    assert readback_summary["blocked_live_actions"] == future_readback[
        "blocked_live_actions"
    ]
    assert readback_summary["evidence_gate_status"] == correction["evidence_gate"][
        "status"
    ]
    assert readback_summary["operator_action"] == correction["evidence_gate"][
        "operator_action"
    ]
    assert readback_summary["read_only_or_dry_run_only"] is True
    assert readback_summary["live_apply_allowed"] is False
    assert readback_summary["queue_import_allowed"] is False
    assert readback_summary["live_authority_granted"] is False
    assert readback_summary["full_autonomous_ready"] is False
    assert readback_summary["readiness_closure_allowed"] is False
    assert readback_summary["can_authorize_live"] is False
    assert readback_summary["can_close_readiness"] is False
    assert readback_summary["mutates_mixer"] is False
    assert readback_summary["operator_review_required"] is True
    assert readback_summary["future_wing_readback_required"] is True
    assert readback_summary["blocks_live_promotion"] is True
    assert readback_summary["blocks_queue_import"] is True
    assert (
        readback_summary["denial_reason"]
        == correction["live_authority_boundary"]["denial_reason"]
    )
    live_proof_summary = explanation_packet["future_live_proof_contract_summary"]
    assert live_proof_summary["required_before_live_promotion"] is True
    assert live_proof_summary["bounded_delta_policy"] == correction[
        "bounded_delta_policy"
    ]
    assert live_proof_summary["bounded_delta"] == correction["bounded_delta"]
    assert live_proof_summary["rollback_expectation"] == correction[
        "rollback_expectation"
    ]
    assert live_proof_summary["rollback_readback_contract"] == rollback_contract
    assert live_proof_summary["target_kind"] == future_readback["target_kind"]
    assert live_proof_summary["old_value_source"] == future_readback[
        "old_value_source"
    ]
    assert live_proof_summary["readback_targets"] == future_readback[
        "readback_targets"
    ]
    assert live_proof_summary["required_proofs"] == correction[
        "future_live_promotion_requirements"
    ]["required_proofs"]
    assert live_proof_summary["required_evidence_fields"] == future_readback[
        "required_evidence_fields"
    ]
    assert live_proof_summary["missing_required_evidence"] == correction[
        "evidence_gate"
    ]["missing_required_evidence"]
    assert live_proof_summary["required_evidence_status"] == required_evidence_status
    assert live_proof_summary["safety_blockers"] == correction["safety_blockers"]
    assert live_proof_summary["effective_safety_blockers"] == effective_blockers
    assert live_proof_summary["safety_blocker_details"] == safety_blocker_details
    assert live_proof_summary["blocked_live_actions"] == future_readback[
        "blocked_live_actions"
    ]
    assert live_proof_summary["evidence_gate_status"] == correction[
        "evidence_gate"
    ]["status"]
    assert live_proof_summary["confidence_review_status"] == confidence_review[
        "status"
    ]
    assert live_proof_summary["operator_review_gate_status"] == review_gate["status"]
    assert live_proof_summary["operator_action"] == review_gate["operator_action"]
    assert live_proof_summary["read_only_or_dry_run_only"] is True
    assert live_proof_summary["live_apply_allowed"] is False
    assert live_proof_summary["queue_import_allowed"] is False
    assert live_proof_summary["live_authority_granted"] is False
    assert live_proof_summary["full_autonomous_ready"] is False
    assert live_proof_summary["readiness_closure_allowed"] is False
    assert live_proof_summary["can_authorize_live"] is False
    assert live_proof_summary["can_close_readiness"] is False
    assert live_proof_summary["mutates_mixer"] is False
    assert live_proof_summary["operator_review_required"] is True
    assert live_proof_summary["future_wing_readback_required"] is True
    assert live_proof_summary["blocks_live_promotion"] is True
    assert live_proof_summary["blocks_queue_import"] is True
    assert (
        live_proof_summary["denial_reason"]
        == correction["live_authority_boundary"]["denial_reason"]
    )
    assert "real_wing_proof_required" in live_proof_summary[
        "live_authority_blockers"
    ]
    assert explanation_packet["read_only_or_dry_run_only"] is True
    assert explanation_packet["live_apply_allowed"] is False
    assert explanation_packet["queue_import_allowed"] is False
    assert explanation_packet["live_authority_granted"] is False
    assert explanation_packet["full_autonomous_ready"] is False
    assert explanation_packet["readiness_closure_allowed"] is False
    assert explanation_packet["can_authorize_live"] is False
    assert explanation_packet["can_close_readiness"] is False
    assert explanation_packet["mutates_mixer"] is False
    assert explanation_packet["operator_review_required"] is True
    assert explanation_packet["future_wing_readback_required"] is True
    assert explanation_packet["blocks_live_promotion"] is True
    assert explanation_packet["blocks_queue_import"] is True
    assert (
        explanation_packet["denial_reason"]
        == correction["live_authority_boundary"]["denial_reason"]
    )
    assert evidence_packet["operator_explanation_packet"] == explanation_packet
    assert evidence_packet["rollback_expectation"] == correction["rollback_expectation"]
    assert evidence_packet["rollback_readback_contract"] == rollback_contract
    for blocker in correction["safety_blockers"]:
        assert blocker in evidence_packet["safety_blockers"]
        assert blocker in review_summary["why_blocked"]
        assert f"review_safety_blocker:{blocker}" in explanation_packet[
            "operator_next_actions"
        ]
    assert review_summary["effective_safety_blockers"] == effective_blockers
    assert review_summary["safety_blocker_details"] == safety_blocker_details
    assert review_summary["why_safe_details"] == why_safe_details
    assert review_summary["why_blocked_details"] == why_blocked_details
    assert set(effective_blockers) <= set(review_summary["why_blocked"])
    assert review_summary["problem"] == correction["measured_or_audible_problem"][
        "summary"
    ]
    assert "read_only_or_dry_run_only" in review_summary["why_safe"]
    assert "live_apply_denied" in review_summary["why_safe"]
    assert "queue_import_denied" in review_summary["why_safe"]
    assert (
        f"bounded_delta_policy:{correction['bounded_delta_policy']}"
        in review_summary["why_safe"]
    )
    assert "live_apply_denied_until_future_wing_readback" in review_summary[
        "why_blocked"
    ]
    assert review_summary["operator_action"] == correction[
        "operator_proposal_review_gate"
    ][
        "operator_action"
    ]
    assert review_summary["proposal_review_gate_status"] == review_gate["status"]
    assert review_summary["offline_acceptance_allowed"] == review_gate[
        "offline_acceptance_allowed"
    ]
    assert review_summary["recommended_decision"] == review_gate[
        "recommended_decision"
    ]
    assert review_summary["confidence_review_status"] == confidence_review["status"]
    assert review_summary["confidence_blockers"] == confidence_review["blockers"]
    assert review_summary["rollback_expectation"] == correction[
        "rollback_expectation"
    ]
    assert review_summary["rollback_readback_contract"] == rollback_contract
    assert review_summary["read_only_or_dry_run_only"] is True
    assert review_summary["live_apply_allowed"] is False
    assert review_summary["queue_import_allowed"] is False
    assert review_summary["live_authority_granted"] is False
    assert review_summary["full_autonomous_ready"] is False
    assert review_summary["readiness_closure_allowed"] is False
    assert review_summary["can_authorize_live"] is False
    assert review_summary["can_close_readiness"] is False
    assert review_summary["mutates_mixer"] is False
    assert review_summary["operator_review_required"] is True
    assert review_summary["blocks_live_promotion"] is True
    assert review_summary["blocks_queue_import"] is True
    assert review_summary["future_wing_readback_required"] is True
    assert review_summary["future_live_requirements"][
        "readback_required_before_live_promotion"
    ] is True
    _assert_correction_context_profile(
        review_summary["future_live_requirements"],
        correction,
    )
    assert review_summary["future_live_requirements"][
        "read_only_or_dry_run_only"
    ] is True
    assert review_summary["future_live_requirements"]["live_apply_allowed"] is False
    assert review_summary["future_live_requirements"]["queue_import_allowed"] is False
    assert review_summary["future_live_requirements"][
        "live_authority_granted"
    ] is False
    assert review_summary["future_live_requirements"][
        "full_autonomous_ready"
    ] is False
    assert review_summary["future_live_requirements"][
        "readiness_closure_allowed"
    ] is False
    assert review_summary["future_live_requirements"]["can_authorize_live"] is False
    assert review_summary["future_live_requirements"]["can_close_readiness"] is False
    assert review_summary["future_live_requirements"]["mutates_mixer"] is False
    assert review_summary["future_live_requirements"]["operator_review_required"] is True
    assert review_summary["future_live_requirements"]["blocks_live_promotion"] is True
    assert review_summary["future_live_requirements"]["blocks_queue_import"] is True
    assert (
        review_summary["future_live_requirements"]["future_wing_readback_required"]
        is True
    )
    assert review_summary["future_live_requirements"]["wing_proof_required"] is True
    assert review_summary["future_live_requirements"][
        "blocked_live_actions"
    ] == correction["future_wing_readback_contract"]["blocked_live_actions"]
    assert review_summary["future_live_requirements"][
        "required_evidence_status"
    ] == required_evidence_status
    future_requirements = correction["future_live_promotion_requirements"]
    assert future_requirements["required_before_live_promotion"] is True
    assert future_requirements["read_only_or_dry_run_only"] is True
    assert future_requirements["live_authority_granted"] is False
    assert future_requirements["full_autonomous_ready"] is False
    assert future_requirements["readiness_closure_allowed"] is False
    assert future_requirements["live_apply_allowed"] is False
    assert future_requirements["queue_import_allowed"] is False
    assert future_requirements["can_authorize_live"] is False
    assert future_requirements["can_close_readiness"] is False
    assert future_requirements["mutates_mixer"] is False
    assert future_requirements["operator_approval_required"] is True
    assert future_requirements["hardware_proof_required"] is True
    assert future_requirements["target_kind"] == correction[
        "future_wing_readback_contract"
    ]["target_kind"]
    assert future_requirements["old_value_source"] == correction[
        "future_wing_readback_contract"
    ]["old_value_source"]
    for field in OPERATOR_IDENTITY_PROFILE_FIELDS:
        assert future_requirements[field] == correction[field]
    assert future_requirements["readback_targets"] == correction[
        "future_wing_readback_contract"
    ]["readback_targets"]
    assert future_requirements["blocked_live_actions"] == correction[
        "future_wing_readback_contract"
    ]["blocked_live_actions"]
    assert future_requirements["required_evidence_fields"] == required_evidence
    assert future_requirements["missing_required_evidence"] == correction[
        "evidence_gate"
    ]["missing_required_evidence"]
    assert future_requirements["required_evidence_status"] == required_evidence_status
    assert future_requirements["evidence_gate_status"] == correction["evidence_gate"][
        "status"
    ]
    assert future_requirements["operator_action"] == correction["evidence_gate"][
        "operator_action"
    ]
    assert {
        "real_wing_readonly_connection_artifact",
        "physical_console_readback_for_old_and_proposed_values",
        "operator_review_acceptance_record",
        "rollback_readback_proof",
        "emergency_stop_proof",
        "connection_loss_timeout_fail_closed_proof",
        "safety_kernel_trace_for_target_action",
    } <= set(future_requirements["required_proofs"])
    _assert_future_live_requirement_details(correction)
    _assert_rollback_readback_details(correction)
    _assert_headroom_feedback_true_peak_check_details(correction)
    _assert_operator_next_action_details(correction)
    _assert_operator_review_checklist_details(correction)
    _assert_operator_review_decision_option_details(correction)
    _assert_operator_review_record_field_details(correction)
    authority_boundary = correction["live_authority_boundary"]
    assert authority_boundary["live_authority_granted"] is False
    assert authority_boundary["full_autonomous_ready"] is False
    assert authority_boundary["readiness_closure_allowed"] is False
    assert authority_boundary["wing_proof_required"] is True
    assert authority_boundary["hardware_proof_required"] is True
    assert authority_boundary["operator_review_required"] is True
    assert authority_boundary["allowed_current_live_write_kinds"] == ["fader", "gain"]
    assert authority_boundary["read_only_or_dry_run_only"] is True
    assert authority_boundary["live_apply_allowed"] is False
    assert authority_boundary["queue_import_allowed"] is False
    assert authority_boundary["can_authorize_live"] is False
    assert authority_boundary["can_close_readiness"] is False
    assert authority_boundary["mutates_mixer"] is False
    assert authority_boundary["blocks_live_promotion"] is True
    assert authority_boundary["blocks_queue_import"] is True
    assert authority_boundary["future_wing_readback_required"] is True
    assert authority_boundary["blocked_live_actions"] == correction[
        "future_wing_readback_contract"
    ]["blocked_live_actions"]
    assert authority_boundary["evidence_gate_status"] == correction["evidence_gate"][
        "status"
    ]
    for field in OPERATOR_IDENTITY_PROFILE_FIELDS:
        assert authority_boundary[field] == correction[field]
    assert "live_authority_denied" in authority_boundary["blockers"]
    authority_summary = explanation_packet["live_authority_boundary_summary"]
    assert authority_summary["live_authority_granted"] is False
    assert authority_summary["full_autonomous_ready"] is False
    assert authority_summary["readiness_closure_allowed"] is False
    assert authority_summary["wing_proof_required"] is True
    assert authority_summary["hardware_proof_required"] is True
    assert authority_summary["operator_review_required"] is True
    assert authority_summary["readback_required_before_live_promotion"] is True
    assert authority_summary["allowed_current_live_write_kinds"] == authority_boundary[
        "allowed_current_live_write_kinds"
    ]
    assert authority_summary["blocked_live_actions"] == authority_boundary[
        "blocked_live_actions"
    ]
    assert authority_summary["blockers"] == authority_boundary["blockers"]
    assert authority_summary["denial_reason"] == authority_boundary["denial_reason"]
    assert authority_summary["evidence_gate_status"] == authority_boundary[
        "evidence_gate_status"
    ]
    assert authority_summary["live_apply_allowed"] is False
    assert authority_summary["queue_import_allowed"] is False
    assert authority_summary["read_only_or_dry_run_only"] is True
    assert authority_summary["mutates_mixer"] is False
    assert authority_summary["future_wing_readback_required"] is True
    assert authority_summary["blocks_live_promotion"] is True
    assert authority_summary["blocks_queue_import"] is True
    assert authority_summary["can_authorize_live"] is False
    assert authority_summary["can_close_readiness"] is False
    assert evidence_packet["live_authority_boundary"] == authority_boundary
    assert evidence_packet["future_wing_readback_contract"] == correction[
        "future_wing_readback_contract"
    ]
    assert evidence_packet["future_live_promotion_requirements"] == future_requirements
    assert evidence_packet["blocked_live_actions"] == correction[
        "future_wing_readback_contract"
    ]["blocked_live_actions"]
    assert evidence_packet["live_authority_blockers"] == authority_boundary["blockers"]
    assert evidence_packet["live_authority_granted"] is False
    assert evidence_packet["full_autonomous_ready"] is False
    assert evidence_packet["readiness_closure_allowed"] is False
    assert evidence_packet["can_authorize_live"] is False
    assert evidence_packet["can_close_readiness"] is False
    assert evidence_packet["mutates_mixer"] is False
    assert evidence_packet["future_wing_readback_required"] is True
    assert evidence_packet["blocks_live_promotion"] is True
    assert evidence_packet["blocks_queue_import"] is True
    review_checklist = correction["operator_review_checklist"]
    assert "verify_source_evidence_matches_current_offline_material" in review_checklist
    assert "confirm_measured_or_audible_problem" in review_checklist
    assert "inspect_proposed_correction_and_bounded_delta" in review_checklist
    assert "verify_required_safety_checks" in review_checklist
    assert "confirm_rollback_expectation" in review_checklist
    assert "do_not_queue_or_apply_live" in review_checklist
    assert (
        "capture_real_wing_readback_and_rollback_proof_before_live_promotion"
        in review_checklist
    )
    assert explanation_packet["operator_review_checklist"] == review_checklist
    assert "do_not_queue_or_apply_live" in explanation_packet[
        "operator_review_checklist"
    ]
    review_options = correction["operator_review_decision_options"]
    decisions = {option["decision"] for option in review_options}
    assert {
        "request_more_evidence",
        "accept_as_offline_proposal",
        "reject_proposal",
        "defer_until_real_wing_proof",
    } <= decisions
    recommended_options = [
        option["decision"] for option in review_options if option["recommended"]
    ]
    assert recommended_options == [review_gate["recommended_decision"]]
    review_record = correction["operator_review_record_template"]
    assert review_record["record_type"] == "offline_operator_correction_review_record"
    assert review_record["read_only_or_dry_run_only"] is True
    assert review_record["live_apply_allowed"] is False
    assert review_record["queue_import_allowed"] is False
    assert review_record["live_authority_granted"] is False
    assert review_record["full_autonomous_ready"] is False
    assert review_record["readiness_closure_allowed"] is False
    assert review_record["can_authorize_live"] is False
    assert review_record["can_close_readiness"] is False
    assert review_record["mutates_mixer"] is False
    assert review_record["operator_review_required"] is True
    assert review_record["future_wing_readback_required"] is True
    assert review_record["future_live_promotion_requires_real_wing_proof"] is True
    assert review_record["method_id"] == correction["source_evidence"]["method_id"]
    assert review_record["objective_family"] == objective_family
    assert review_record["source"] == correction["source_evidence"]["source"]
    assert review_record["target"] == correction["source_evidence"]["target"]
    for field in OPERATOR_IDENTITY_PROFILE_FIELDS:
        assert review_record[field] == correction[field]
    assert review_record["evidence_gate_status"] == correction["evidence_gate"][
        "status"
    ]
    assert review_record["operator_proposal_review_gate_status"] == review_gate[
        "status"
    ]
    assert review_record["offline_acceptance_allowed"] == review_gate[
        "offline_acceptance_allowed"
    ]
    assert review_record["confidence_review_status"] == confidence_review["status"]
    assert review_record["missing_required_evidence"] == correction[
        "evidence_gate"
    ]["missing_required_evidence"]
    assert review_record["operator_action"] == correction[
        "operator_proposal_review_gate"
    ][
        "operator_action"
    ]
    assert review_record["decision_options"] == [
        option["decision"] for option in review_options
    ]
    assert review_record["recommended_decision"] in review_record["decision_options"]
    assert review_record["recommended_decision"] == next(
        option["decision"] for option in review_options if option["recommended"]
    )
    assert set(review_record["required_record_fields"]) == {
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
    }
    assert set(review_record["allowed_decisions"]) == decisions
    assert review_record["decision_records_live_authority"] is False
    assert review_record["decision_records_queue_import"] is False
    assert (
        review_record["denial_reason"]
        == "operator_review_and_future_wing_readback_required"
    )
    assert explanation_packet["operator_review_decision_options"] == review_options
    explanation_record_summary = explanation_packet["operator_review_record_summary"]
    assert explanation_record_summary["record_type"] == review_record["record_type"]
    assert explanation_record_summary["required_record_fields"] == review_record[
        "required_record_fields"
    ]
    assert explanation_record_summary["allowed_decisions"] == review_record[
        "allowed_decisions"
    ]
    assert explanation_record_summary["recommended_decision"] == review_record[
        "recommended_decision"
    ]
    assert explanation_record_summary["operator_action"] == review_record[
        "operator_action"
    ]
    assert explanation_record_summary["evidence_gate_status"] == review_record[
        "evidence_gate_status"
    ]
    assert explanation_record_summary["missing_required_evidence"] == review_record[
        "missing_required_evidence"
    ]
    assert explanation_record_summary["offline_acceptance_allowed"] == review_record[
        "offline_acceptance_allowed"
    ]
    assert explanation_record_summary["operator_review_required"] is True
    assert explanation_record_summary["decision_records_live_authority"] is False
    assert explanation_record_summary["decision_records_queue_import"] is False
    assert explanation_record_summary["read_only_or_dry_run_only"] is True
    assert explanation_record_summary["live_apply_allowed"] is False
    assert explanation_record_summary["queue_import_allowed"] is False
    assert explanation_record_summary["live_authority_granted"] is False
    assert explanation_record_summary["full_autonomous_ready"] is False
    assert explanation_record_summary["readiness_closure_allowed"] is False
    assert explanation_record_summary["can_authorize_live"] is False
    assert explanation_record_summary["can_close_readiness"] is False
    assert explanation_record_summary["mutates_mixer"] is False
    assert explanation_record_summary["future_wing_readback_required"] is True
    assert explanation_record_summary["blocks_live_promotion"] is True
    assert explanation_record_summary["blocks_queue_import"] is True
    assert explanation_record_summary["denial_reason"] == review_record[
        "denial_reason"
    ]
    assert review_record["blocks_live_promotion"] is True
    assert review_record["blocks_queue_import"] is True
    assert evidence_packet["operator_review_record_template"] == review_record
    for option in review_options:
        for field in OPERATOR_IDENTITY_PROFILE_FIELDS:
            assert option[field] == correction[field]
        assert option["live_apply_allowed"] is False
        assert option["queue_import_allowed"] is False
        assert option["read_only_or_dry_run_only"] is True
        assert option["live_authority_granted"] is False
        assert option["full_autonomous_ready"] is False
        assert option["readiness_closure_allowed"] is False
        assert option["can_authorize_live"] is False
        assert option["can_close_readiness"] is False
        assert option["mutates_mixer"] is False
        assert option["operator_review_required"] is True
        assert option["future_wing_readback_required"] is True
        assert option["blocks_live_promotion"] is True
        assert option["blocks_queue_import"] is True
        assert (
            option["denial_reason"]
            == "operator_review_and_future_wing_readback_required"
        )
        assert option["requires_real_wing_proof_before_live"] is True
    assert correction["evidence_gate"]["blocks_live_promotion"] is True
    assert correction["evidence_gate"]["blocks_queue_import"] is True
    assert correction["evidence_gate"]["live_apply_allowed"] is False
    assert correction["evidence_gate"]["queue_import_allowed"] is False


def test_registry_blockers_are_explicitly_fail_closed_operator_surfaces():
    blockers = build_registry_blockers()
    assert blockers

    for row in blockers:
        assert row["queue_action"] == "not_queueable"
        assert row["can_import_to_queue"] is False
        assert row["can_apply_live"] is False
        assert row["queue_import_allowed"] is False
        assert row["live_apply_allowed"] is False
        assert row["read_only_or_dry_run_only"] is True
        assert row["live_authority_granted"] is False
        assert row["full_autonomous_ready"] is False
        assert row["readiness_closure_allowed"] is False
        assert row["can_authorize_live"] is False
        assert row["can_close_readiness"] is False
        assert row["mutates_mixer"] is False
        assert row["operator_review_required"] is True
        assert row["blocks_live_promotion"] is True
        assert row["blocks_queue_import"] is True
        assert row["future_wing_readback_required"] is True
        assert row["denial_reason"] in row["blockers"]
        assert row["evidence"]["live_executor"] is False
        assert row["evidence"]["requested_change_allowed"] is False
        assert row["evidence"]["queue_mutation"] is False
        assert row["evidence"]["live_write"] is False
        assert row["evidence"]["read_only_or_dry_run_only"] is True
        assert row["evidence"]["live_authority_granted"] is False
        assert row["evidence"]["operator_review_required"] is True
        assert row["evidence"]["future_wing_readback_required"] is True
        assert row["evidence"]["blocks_live_promotion"] is True
        assert row["evidence"]["blocks_queue_import"] is True


def test_user_named_correction_method_families_have_operator_contracts():
    contracts = _contracts_by_method()
    missing = sorted(
        {
            method_id
            for method_ids in OBJECTIVE_FAMILY_METHODS.values()
            for method_id in method_ids
            if method_id not in contracts
        }
    )

    assert missing == []

    for objective_family, method_ids in OBJECTIVE_FAMILY_METHODS.items():
        for method_id in method_ids:
            contract = contracts[method_id]
            proposal_contract = contract["proposal_contract"]
            operator_contract = proposal_contract["operator_correction_contract"]

            assert contract["operator_correction_contract"] == operator_contract
            assert contract["objective_family"] == objective_family
            assert contract["value_type"] == operator_contract["value_type"]
            assert contract["target_kind"] == operator_contract["target_kind"]
            assert contract["old_value_source"] == operator_contract["old_value_source"]
            assert contract["bounded_delta_unit"] == operator_contract[
                "bounded_delta_unit"
            ]
            assert contract["bounded_delta_policy"] == operator_contract[
                "bounded_delta_policy"
            ]
            assert contract["future_wing_readback_contract"] == operator_contract[
                "future_wing_readback_contract"
            ]
            assert contract["live_authority_boundary"] == operator_contract[
                "live_authority_boundary"
            ]
            assert contract["future_live_promotion_requirements"] == operator_contract[
                "future_live_promotion_requirements"
            ]
            assert contract["operator_evidence_packet_contract"] == operator_contract[
                "operator_evidence_packet"
            ]
            assert contract["operator_review_record_template_contract"] == operator_contract[
                "operator_review_record_template"
            ]
            assert contract["operator_correction_required_fields"] == list(
                REQUIRED_OPERATOR_FIELDS
            )
            assert set(contract["operator_correction_required_field_flags"]) == {
                f"{field}_required" for field in REQUIRED_OPERATOR_FIELDS
            }
            for field in REQUIRED_OPERATOR_FIELDS:
                assert contract["operator_correction_required_field_flags"][
                    f"{field}_required"
                ] is True
            for required_flag in (
                "source_evidence_required",
                "source_evidence_details_required",
                "measured_or_audible_problem_required",
                "proposed_correction_required",
                "confidence_required",
                "confidence_review_required",
                "confidence_review_details_required",
                "operator_proposal_review_gate_required",
                "bounded_delta_required",
                "bounded_delta_details_required",
                "safety_blockers_required",
                "effective_safety_blockers_required",
                "safety_blocker_details_required",
                "why_safe_details_required",
                "why_blocked_details_required",
                "headroom_feedback_true_peak_checks_required",
                "headroom_feedback_true_peak_check_details_required",
                "operator_safety_review_matrix_required",
                "operator_readable_explanation_required",
                "operator_explanation_packet_required",
                "operator_evidence_packet_required",
                "rollback_expectation_required",
                "rollback_readback_contract_required",
                "rollback_readback_details_required",
                "future_live_promotion_requirements_required",
                "future_live_requirement_details_required",
                "future_wing_readback_contract_required",
                "live_authority_boundary_required",
            ):
                assert contract[required_flag] is True

            assert contract["can_import_to_queue"] is False
            assert contract["can_apply_live"] is False
            assert contract["queue_import_allowed"] is False
            assert contract["live_apply_allowed"] is False
            assert contract["read_only_or_dry_run_only"] is True
            assert contract["live_authority_granted"] is False
            assert contract["readiness_closure_allowed"] is False
            assert contract["mutates_mixer"] is False
            assert contract["operator_review_required"] is True
            assert contract["blocks_live_promotion"] is True
            assert contract["blocks_queue_import"] is True
            assert contract["future_wing_readback_required"] is True
            assert proposal_contract["live_apply_allowed"] is False
            assert proposal_contract["queue_import_allowed"] is False
            assert proposal_contract["read_only_or_dry_run_only"] is True
            assert proposal_contract["live_authority_granted"] is False
            assert proposal_contract["readiness_closure_allowed"] is False
            assert proposal_contract["readback_required"] is True
            assert proposal_contract["rollback_required"] is True
            assert proposal_contract["operator_correction_required_fields"] == list(
                REQUIRED_OPERATOR_FIELDS
            )
            for field in REQUIRED_OPERATOR_FIELDS:
                assert field in proposal_contract["required_fields"]
            assert proposal_contract["safety_evidence_required"]

            assert operator_contract["method_id_required"] is True
            assert operator_contract["source_required"] is True
            assert operator_contract["target_required"] is True
            assert operator_contract["value_type_required"] is True
            assert operator_contract["target_kind_required"] is True
            assert operator_contract["old_value_source_required"] is True
            assert operator_contract["objective_family_required"] is True
            assert operator_contract["source_evidence_required"] is True
            assert operator_contract["bounded_delta_unit_required"] is True
            assert operator_contract["bounded_delta_policy_required"] is True
            assert operator_contract["source_evidence_details_required"] is True
            assert operator_contract["measured_or_audible_problem_required"] is True
            assert operator_contract["proposed_correction_required"] is True
            assert operator_contract["confidence_required"] is True
            assert operator_contract["confidence_review_required"] is True
            assert (
                operator_contract["confidence_review_details_required"]
                is True
            )
            assert operator_contract[
                "operator_proposal_review_gate_required"
            ] is True
            assert operator_contract["bounded_delta_required"] is True
            assert operator_contract["bounded_delta_details_required"] is True
            assert operator_contract["safety_blockers_required"] is True
            assert operator_contract["safety_blocker_details_required"] is True
            assert operator_contract["why_safe_details_required"] is True
            assert operator_contract["why_blocked_details_required"] is True
            assert operator_contract["operator_safety_review_matrix_required"] is True
            assert (
                operator_contract[
                    "headroom_feedback_true_peak_check_details_required"
                ]
                is True
            )
            assert operator_contract["operator_readable_explanation_required"] is True
            assert operator_contract["operator_explanation_packet_required"] is True
            assert operator_contract["operator_next_action_details_required"] is True
            assert operator_contract["operator_evidence_packet_required"] is True
            assert operator_contract["operator_review_summary_required"] is True
            assert operator_contract["operator_review_checklist_required"] is True
            assert (
                operator_contract["operator_review_checklist_details_required"]
                is True
            )
            assert operator_contract[
                "operator_review_decision_options_required"
            ] is True
            assert (
                operator_contract["operator_review_decision_option_details_required"]
                is True
            )
            assert operator_contract["operator_review_record_template_required"] is True
            assert (
                operator_contract["operator_review_record_field_details_required"]
                is True
            )
            assert operator_contract[
                "future_live_promotion_requirements_required"
            ] is True
            assert operator_contract["future_live_requirement_details_required"] is True
            assert operator_contract["live_authority_boundary_required"] is True
            assert operator_contract["denial_reason_required"] is True
            assert operator_contract["operator_next_step_required"] is True
            assert operator_contract["rollback_readback_contract_required"] is True
            assert operator_contract["rollback_readback_details_required"] is True
            assert operator_contract["read_only_or_dry_run_only"] is True
            assert operator_contract["queue_import_allowed"] is False
            assert operator_contract["live_apply_allowed"] is False
            assert operator_contract["objective_family"] == objective_family
            assert operator_contract["required_evidence_fields"] == proposal_contract[
                "safety_evidence_required"
            ]

            readback_contract = operator_contract["future_wing_readback_contract"]
            assert readback_contract["required_before_live_promotion"] is True
            assert readback_contract["old_value_source"] == proposal_contract["old_value_source"]
            assert readback_contract["target_kind"] == proposal_contract["target_kind"]
            assert readback_contract["readback_targets"]
            assert readback_contract["blocked_live_actions"]
            assert readback_contract["required_evidence_fields"] == proposal_contract[
                "safety_evidence_required"
            ]
            authority_boundary = operator_contract["live_authority_boundary"]
            assert authority_boundary["live_authority_granted"] is False
            assert authority_boundary["full_autonomous_ready"] is False
            assert authority_boundary["readiness_closure_allowed"] is False
            assert authority_boundary["wing_proof_required"] is True
            assert authority_boundary["hardware_proof_required"] is True
            assert authority_boundary["allowed_current_live_write_kinds"] == [
                "fader",
                "gain",
            ]
            assert authority_boundary["read_only_or_dry_run_only"] is True
            assert authority_boundary["can_authorize_live"] is False
            assert authority_boundary["can_close_readiness"] is False
            assert authority_boundary["mutates_mixer"] is False
            assert authority_boundary["blocks_live_promotion"] is True
            assert authority_boundary["blocks_queue_import"] is True
            assert authority_boundary["future_wing_readback_required"] is True
            assert authority_boundary["blocked_live_actions"] == readback_contract[
                "blocked_live_actions"
            ]
            future_requirements_contract = operator_contract[
                "future_live_promotion_requirements"
            ]
            assert future_requirements_contract[
                "required_before_live_promotion"
            ] is True
            assert future_requirements_contract["read_only_or_dry_run_only"] is True
            assert future_requirements_contract["live_authority_granted"] is False
            assert future_requirements_contract["full_autonomous_ready"] is False
            assert future_requirements_contract["readiness_closure_allowed"] is False
            assert future_requirements_contract["can_authorize_live"] is False
            assert future_requirements_contract["can_close_readiness"] is False
            assert future_requirements_contract["mutates_mixer"] is False
            assert future_requirements_contract["live_apply_allowed"] is False
            assert future_requirements_contract["queue_import_allowed"] is False
            assert future_requirements_contract["operator_approval_required"] is True
            assert future_requirements_contract["hardware_proof_required"] is True
            assert {
                "real_wing_readonly_connection_artifact",
                "physical_console_readback_for_old_and_proposed_values",
                "operator_review_acceptance_record",
                "rollback_readback_proof",
                "emergency_stop_proof",
                "connection_loss_timeout_fail_closed_proof",
                "safety_kernel_trace_for_target_action",
            } <= set(future_requirements_contract["required_proofs"])
            evidence_packet_contract = operator_contract["operator_evidence_packet"]
            assert evidence_packet_contract["packet_type"] == (
                "offline_operator_correction_review"
            )
            assert evidence_packet_contract["read_only_or_dry_run_only"] is True
            assert evidence_packet_contract["live_apply_allowed"] is False
            assert evidence_packet_contract["queue_import_allowed"] is False
            assert evidence_packet_contract["source_evidence_required"] is True
            assert evidence_packet_contract["source_evidence_details_required"] is True
            assert evidence_packet_contract["problem_statement_required"] is True
            assert evidence_packet_contract["proposed_correction_required"] is True
            assert evidence_packet_contract["confidence_required"] is True
            assert evidence_packet_contract["confidence_review_required"] is True
            assert (
                evidence_packet_contract["confidence_review_details_required"]
                is True
            )
            assert evidence_packet_contract[
                "operator_proposal_review_gate_required"
            ] is True
            assert evidence_packet_contract["bounded_delta_required"] is True
            assert (
                evidence_packet_contract["bounded_delta_details_required"]
                is True
            )
            assert evidence_packet_contract["safety_blockers_required"] is True
            assert (
                evidence_packet_contract["safety_blocker_details_required"] is True
            )
            assert evidence_packet_contract["why_safe_details_required"] is True
            assert evidence_packet_contract["why_blocked_details_required"] is True
            assert evidence_packet_contract["safety_checks_required"] is True
            assert evidence_packet_contract[
                "headroom_feedback_true_peak_checks_required"
            ] is True
            assert (
                evidence_packet_contract[
                    "headroom_feedback_true_peak_check_details_required"
                ]
                is True
            )
            assert evidence_packet_contract[
                "operator_safety_review_matrix_required"
            ] is True
            assert evidence_packet_contract[
                "operator_readable_explanation_required"
            ] is True
            assert evidence_packet_contract[
                "operator_explanation_packet_required"
            ] is True
            assert (
                evidence_packet_contract["operator_next_action_details_required"]
                is True
            )
            assert evidence_packet_contract["rollback_expectation_required"] is True
            assert (
                evidence_packet_contract["rollback_readback_contract_required"] is True
            )
            assert (
                evidence_packet_contract["rollback_readback_details_required"] is True
            )
            assert evidence_packet_contract[
                "future_live_promotion_requirements_required"
            ] is True
            assert (
                evidence_packet_contract["future_live_requirement_details_required"]
                is True
            )
            assert evidence_packet_contract[
                "future_wing_readback_contract_required"
            ] is True
            assert evidence_packet_contract["live_authority_boundary_required"] is True
            assert evidence_packet_contract["operator_review_required"] is True
            assert evidence_packet_contract[
                "operator_review_record_template_required"
            ] is True
            assert (
                evidence_packet_contract[
                    "operator_review_record_field_details_required"
                ]
                is True
            )
            assert (
                evidence_packet_contract["operator_review_checklist_details_required"]
                is True
            )
            assert evidence_packet_contract["test_evidence_required"] is True
            assert (
                evidence_packet_contract[
                    "operator_review_decision_option_details_required"
                ]
                is True
            )
            review_checklist = operator_contract["operator_review_checklist"]
            assert "do_not_queue_or_apply_live" in review_checklist
            assert (
                "capture_real_wing_readback_and_rollback_proof_before_live_promotion"
                in review_checklist
            )
            review_options = operator_contract["operator_review_decision_options"]
            assert {
                "request_more_evidence",
                "accept_as_offline_proposal",
                "reject_proposal",
                "defer_until_real_wing_proof",
            } <= {option["decision"] for option in review_options}
            for option in review_options:
                assert option["live_apply_allowed"] is False
                assert option["queue_import_allowed"] is False
            review_record_contract = operator_contract[
                "operator_review_record_template"
            ]
            assert review_record_contract["record_type"] == (
                "offline_operator_correction_review_record"
            )
            assert review_record_contract["read_only_or_dry_run_only"] is True
            assert review_record_contract["live_apply_allowed"] is False
            assert review_record_contract["queue_import_allowed"] is False
            assert review_record_contract["can_authorize_live"] is False
            assert review_record_contract["can_close_readiness"] is False
            assert review_record_contract["operator_review_required"] is True
            assert review_record_contract[
                "future_live_promotion_requires_real_wing_proof"
            ] is True
            assert set(review_record_contract["allowed_decisions"]) == {
                "request_more_evidence",
                "accept_as_offline_proposal",
                "reject_proposal",
                "defer_until_real_wing_proof",
            }


def test_correction_method_goal_coverage_report_tracks_required_families_without_live_authority():
    report = build_correction_method_goal_coverage_report()

    def _assert_coverage_row_fail_closed(row):
        assert row["read_only_or_dry_run_only"] is True
        assert row["operator_review_required"] is True
        assert row["future_wing_readback_required"] is True
        assert row["hardware_proof_required_before_live"] is True
        assert row["blocks_live_promotion"] is True
        assert row["blocks_queue_import"] is True
        assert row["live_authority_granted"] is False
        assert row["full_autonomous_ready"] is False
        assert row["readiness_closure_allowed"] is False
        assert row["live_apply_allowed"] is False
        assert row["queue_import_allowed"] is False
        assert row["can_apply_live"] is False
        assert row["can_import_to_queue"] is False
        assert row["can_authorize_live"] is False
        assert row["can_close_readiness"] is False
        assert row["queue_action"] == "not_queueable_until_promoted"
        assert row["contacts_wing"] is False
        assert row["contacts_backend"] is False
        assert row["mutates_mixer"] is False

    assert report["schema"] == "automixer.correction_method_goal_coverage.v1"
    assert report["status"] == (
        "proposal_layer_contracts_ready_operator_review_required"
    )
    assert report["covered"] is True
    _assert_coverage_row_fail_closed(report)

    expected_operator_surfaces = {
        "source_evidence",
        "source_evidence_details",
        "measured_or_audible_problem",
        "proposed_correction",
        "confidence",
        "confidence_review",
        "bounded_delta",
        "bounded_delta_details",
        "safety_blockers",
        "safety_blocker_details",
        "headroom_feedback_true_peak_checks",
        "headroom_feedback_true_peak_check_details",
        "operator_safety_review_matrix",
        "operator_readable_explanation",
        "operator_explanation_packet",
        "operator_evidence_packet",
        "rollback_expectation",
        "rollback_readback_contract",
        "rollback_readback_details",
        "future_wing_readback_contract",
        "future_live_promotion_requirements",
        "live_authority_boundary",
    }
    assert set(report["operator_required_surfaces"]) == expected_operator_surfaces

    expected_required_families = {
        "gain_trim",
        "gain_trim_and_fader_balance",
        "fader_balance",
        "eq",
        "compressor_dynamics",
        "gate_expander",
        "gate_expander_deesser",
        "fx_sends",
        "pan_stereo",
        "routing_groups_scenes",
        "master_output",
    }
    expected_priority_by_family = {
        "gain_trim": (1, "Gain / Trim and Fader / Balance"),
        "gain_trim_and_fader_balance": (1, "Gain / Trim and Fader / Balance"),
        "fader_balance": (1, "Gain / Trim and Fader / Balance"),
        "eq": (2, "EQ and Compressor"),
        "compressor_dynamics": (2, "EQ and Compressor"),
        "gate_expander": (3, "Gate / Expander / De-esser"),
        "gate_expander_deesser": (3, "Gate / Expander / De-esser"),
        "fx_sends": (4, "FX / Sends"),
        "pan_stereo": (5, "Pan / Stereo"),
        "routing_groups_scenes": (6, "Routing / Groups / Scenes"),
        "master_output": (7, "Master / Output"),
    }
    assert set(report["required_objective_families"]) == expected_required_families
    assert len(report["priority_groups"]) == 7
    assert all(group["covered"] is True for group in report["priority_groups"])
    assert len(report["priority_group_rows"]) == 7
    for row in report["priority_group_rows"]:
        _assert_coverage_row_fail_closed(row)

    first_priority = report["first_priority_review_summary"]
    assert first_priority["priority"] == 1
    assert first_priority["label"] == "Gain / Trim and Fader / Balance"
    assert first_priority["objective_families"] == [
        "gain_trim",
        "gain_trim_and_fader_balance",
        "fader_balance",
    ]
    assert first_priority["covered"] is True
    assert {
        "lufs_gain_staging",
        "auto_soundcheck_engine",
        "auto_fader_v1",
        "live_shared_mix",
        "bus_group_balance_proposal",
    } <= set(first_priority["method_ids"])
    assert {
        "headroom",
        "feedback",
        "true_peak",
        "ayaic_level_plane",
        "fader_ceiling",
        "bleed",
    } <= set(first_priority["applicable_safety_checks"])
    assert first_priority["future_wing_readback_targets"]
    assert first_priority["blocked_live_actions"]
    assert first_priority["missing_required_operator_surface_method_ids"] == []
    assert first_priority["live_authority_method_ids"] == []
    assert first_priority["live_apply_method_ids"] == []
    assert first_priority["queueable_method_ids"] == []
    assert first_priority["read_only_or_dry_run_only"] is True
    assert first_priority["operator_review_required"] is True
    assert first_priority["future_wing_readback_required"] is True
    assert first_priority["hardware_proof_required_before_live"] is True
    assert first_priority["live_authority_granted"] is False
    assert first_priority["readiness_closure_allowed"] is False
    assert first_priority["live_apply_allowed"] is False
    assert first_priority["queue_import_allowed"] is False
    assert first_priority["contacts_wing"] is False
    assert first_priority["contacts_backend"] is False
    assert first_priority["mutates_mixer"] is False

    second_priority = report["second_priority_review_summary"]
    assert second_priority["priority"] == 2
    assert second_priority["label"] == "EQ and Compressor"
    assert second_priority["objective_families"] == [
        "eq",
        "compressor_dynamics",
    ]
    assert second_priority["covered"] is True
    assert {
        "auto_eq",
        "corrective_eq_proposal",
        "cross_adaptive_eq",
        "frequency_separation_proposal",
        "auto_compressor",
        "auto_compressor_cf",
        "compressor_dynamics_proposal",
    } <= set(second_priority["method_ids"])
    assert {
        "headroom",
        "feedback",
        "true_peak",
        "manual_band",
        "phase",
        "gain_reduction",
        "makeup_gain",
    } <= set(second_priority["applicable_safety_checks"])
    assert second_priority["future_wing_readback_targets"]
    assert second_priority["blocked_live_actions"]
    assert second_priority["missing_required_operator_surface_method_ids"] == []
    assert second_priority["live_authority_method_ids"] == []
    assert second_priority["live_apply_method_ids"] == []
    assert second_priority["queueable_method_ids"] == []
    assert second_priority["read_only_or_dry_run_only"] is True
    assert second_priority["operator_review_required"] is True
    assert second_priority["future_wing_readback_required"] is True
    assert second_priority["hardware_proof_required_before_live"] is True
    assert second_priority["live_authority_granted"] is False
    assert second_priority["readiness_closure_allowed"] is False
    assert second_priority["live_apply_allowed"] is False
    assert second_priority["queue_import_allowed"] is False
    assert second_priority["contacts_wing"] is False
    assert second_priority["contacts_backend"] is False
    assert second_priority["mutates_mixer"] is False

    third_priority = report["third_priority_review_summary"]
    assert third_priority["priority"] == 3
    assert third_priority["label"] == "Gate / Expander / De-esser"
    assert third_priority["objective_families"] == [
        "gate_expander",
        "gate_expander_deesser",
    ]
    assert third_priority["covered"] is True
    assert {
        "auto_gate_caig",
        "context_aware_gate_agent",
        "deesser_vocal_sibilance",
    } <= set(third_priority["method_ids"])
    assert {
        "source_identity",
        "chatter_guard",
        "hysteresis",
        "tails_and_attack",
        "speech_intelligibility",
        "true_peak",
    } <= set(third_priority["applicable_safety_checks"])
    assert {"gate_write", "deesser_write"} <= set(
        third_priority["blocked_live_actions"]
    )
    assert third_priority["future_wing_readback_targets"]
    assert third_priority["blocked_live_actions"]
    assert third_priority["missing_required_operator_surface_method_ids"] == []
    assert third_priority["live_authority_method_ids"] == []
    assert third_priority["live_apply_method_ids"] == []
    assert third_priority["queueable_method_ids"] == []
    assert third_priority["read_only_or_dry_run_only"] is True
    assert third_priority["operator_review_required"] is True
    assert third_priority["future_wing_readback_required"] is True
    assert third_priority["hardware_proof_required_before_live"] is True
    assert third_priority["live_authority_granted"] is False
    assert third_priority["readiness_closure_allowed"] is False
    assert third_priority["live_apply_allowed"] is False
    assert third_priority["queue_import_allowed"] is False
    assert third_priority["contacts_wing"] is False
    assert third_priority["contacts_backend"] is False
    assert third_priority["mutates_mixer"] is False

    fourth_priority = report["fourth_priority_review_summary"]
    assert fourth_priority["priority"] == 4
    assert fourth_priority["label"] == "FX / Sends"
    assert fourth_priority["objective_families"] == ["fx_sends"]
    assert fourth_priority["covered"] is True
    assert {
        "auto_fx",
        "fx_direct_apply",
        "fx_send_proposal",
    } <= set(fourth_priority["method_ids"])
    assert {
        "headroom",
        "feedback",
        "return_level",
        "monitor_safety",
        "intelligibility",
        "parameter_bounds",
    } <= set(fourth_priority["applicable_safety_checks"])
    assert {"fx_send_write", "fx_parameter_write"} <= set(
        fourth_priority["blocked_live_actions"]
    )
    assert fourth_priority["future_wing_readback_targets"]
    assert fourth_priority["blocked_live_actions"]
    assert fourth_priority["missing_required_operator_surface_method_ids"] == []
    assert fourth_priority["live_authority_method_ids"] == []
    assert fourth_priority["live_apply_method_ids"] == []
    assert fourth_priority["queueable_method_ids"] == []
    assert fourth_priority["read_only_or_dry_run_only"] is True
    assert fourth_priority["operator_review_required"] is True
    assert fourth_priority["future_wing_readback_required"] is True
    assert fourth_priority["hardware_proof_required_before_live"] is True
    assert fourth_priority["live_authority_granted"] is False
    assert fourth_priority["readiness_closure_allowed"] is False
    assert fourth_priority["live_apply_allowed"] is False
    assert fourth_priority["queue_import_allowed"] is False
    assert fourth_priority["contacts_wing"] is False
    assert fourth_priority["contacts_backend"] is False
    assert fourth_priority["mutates_mixer"] is False

    fifth_priority = report["fifth_priority_review_summary"]
    assert fifth_priority["priority"] == 5
    assert fifth_priority["label"] == "Pan / Stereo"
    assert fifth_priority["objective_families"] == ["pan_stereo"]
    assert fifth_priority["covered"] is True
    assert {
        "auto_panner",
        "pan_spatial_proposal",
    } <= set(fifth_priority["method_ids"])
    assert {
        "mono_compatibility",
        "phase",
        "source_identity",
    } <= set(fifth_priority["applicable_safety_checks"])
    assert {"pan_write", "stereo_width_write"} <= set(
        fifth_priority["blocked_live_actions"]
    )
    assert fifth_priority["future_wing_readback_targets"]
    assert fifth_priority["blocked_live_actions"]
    assert fifth_priority["missing_required_operator_surface_method_ids"] == []
    assert fifth_priority["live_authority_method_ids"] == []
    assert fifth_priority["live_apply_method_ids"] == []
    assert fifth_priority["queueable_method_ids"] == []
    assert fifth_priority["read_only_or_dry_run_only"] is True
    assert fifth_priority["operator_review_required"] is True
    assert fifth_priority["future_wing_readback_required"] is True
    assert fifth_priority["hardware_proof_required_before_live"] is True
    assert fifth_priority["live_authority_granted"] is False
    assert fifth_priority["readiness_closure_allowed"] is False
    assert fifth_priority["live_apply_allowed"] is False
    assert fifth_priority["queue_import_allowed"] is False
    assert fifth_priority["contacts_wing"] is False
    assert fifth_priority["contacts_backend"] is False
    assert fifth_priority["mutates_mixer"] is False

    sixth_priority = report["sixth_priority_review_summary"]
    assert sixth_priority["priority"] == 6
    assert sixth_priority["label"] == "Routing / Groups / Scenes"
    assert sixth_priority["objective_families"] == ["routing_groups_scenes"]
    assert sixth_priority["covered"] is True
    assert {
        "mute_management_proposal",
        "routing_management_proposal",
        "routing_snapshot_reset_writes",
        "scene_preset_management_proposal",
    } <= set(sixth_priority["method_ids"])
    assert {
        "active_performance_guard",
        "active_signal_guard",
        "destructive_scope",
        "feedback",
        "read_only_inventory",
        "read_only_map",
        "source_identity",
    } <= set(sixth_priority["applicable_safety_checks"])
    assert {
        "bus_assignment_write",
        "matrix_write",
        "mute_group_write",
        "mute_write",
        "preset_load",
        "preset_save",
        "routing_write",
        "scene_load",
        "scene_save",
        "unmute_write",
    } <= set(sixth_priority["blocked_live_actions"])
    assert {
        "bus_assignment",
        "channel_mute",
        "full_console_state",
        "routing_map",
        "scene_inventory",
    } <= set(sixth_priority["future_wing_readback_targets"])
    assert sixth_priority["missing_required_operator_surface_method_ids"] == []
    assert sixth_priority["live_authority_method_ids"] == []
    assert sixth_priority["live_apply_method_ids"] == []
    assert sixth_priority["queueable_method_ids"] == []
    assert sixth_priority["read_only_or_dry_run_only"] is True
    assert sixth_priority["operator_review_required"] is True
    assert sixth_priority["future_wing_readback_required"] is True
    assert sixth_priority["hardware_proof_required_before_live"] is True
    assert sixth_priority["live_authority_granted"] is False
    assert sixth_priority["readiness_closure_allowed"] is False
    assert sixth_priority["live_apply_allowed"] is False
    assert sixth_priority["queue_import_allowed"] is False
    assert sixth_priority["contacts_wing"] is False
    assert sixth_priority["contacts_backend"] is False
    assert sixth_priority["mutates_mixer"] is False

    seventh_priority = report["seventh_priority_review_summary"]
    assert seventh_priority["priority"] == 7
    assert seventh_priority["label"] == "Master / Output"
    assert seventh_priority["objective_families"] == ["master_output"]
    assert seventh_priority["covered"] is True
    assert {"master_bus_optimization_proposal"} <= set(
        seventh_priority["method_ids"]
    )
    assert {
        "headroom",
        "limiter_or_dynamics_proof",
        "pa_feed_safety",
        "reference_feed_exclusion",
        "true_peak",
    } <= set(seventh_priority["applicable_safety_checks"])
    assert {
        "limiter_write",
        "master_bus_write",
        "matrix_pa_feed_write",
    } <= set(seventh_priority["blocked_live_actions"])
    assert {
        "limiter",
        "main_bus_processing",
        "matrix_or_pa_feed",
        "reference_feed",
    } <= set(seventh_priority["future_wing_readback_targets"])
    assert seventh_priority["missing_required_operator_surface_method_ids"] == []
    assert seventh_priority["live_authority_method_ids"] == []
    assert seventh_priority["live_apply_method_ids"] == []
    assert seventh_priority["queueable_method_ids"] == []
    assert seventh_priority["read_only_or_dry_run_only"] is True
    assert seventh_priority["operator_review_required"] is True
    assert seventh_priority["future_wing_readback_required"] is True
    assert seventh_priority["hardware_proof_required_before_live"] is True
    assert seventh_priority["live_authority_granted"] is False
    assert seventh_priority["readiness_closure_allowed"] is False
    assert seventh_priority["live_apply_allowed"] is False
    assert seventh_priority["queue_import_allowed"] is False
    assert seventh_priority["contacts_wing"] is False
    assert seventh_priority["contacts_backend"] is False
    assert seventh_priority["mutates_mixer"] is False

    summary = report["summary"]
    assert summary["missing_objective_families"] == []
    assert summary["incomplete_objective_families"] == []
    assert summary["incomplete_field_flag_objective_families"] == []
    assert summary["incomplete_operator_surface_objective_families"] == []
    assert summary["incomplete_method_ids"] == []
    assert summary["missing_required_operator_surface_method_ids"] == []
    assert summary["missing_required_operator_surface_details"] == []
    assert summary["first_priority_method_count"] == first_priority["method_count"]
    assert summary["first_priority_method_ids"] == first_priority["method_ids"]
    assert summary["first_priority_applicable_safety_checks"] == first_priority[
        "applicable_safety_checks"
    ]
    assert summary["first_priority_future_wing_readback_target_count"] == len(
        first_priority["future_wing_readback_targets"]
    )
    assert summary["first_priority_blocked_live_action_count"] == len(
        first_priority["blocked_live_actions"]
    )
    assert summary["second_priority_method_count"] == second_priority["method_count"]
    assert summary["second_priority_method_ids"] == second_priority["method_ids"]
    assert summary["second_priority_applicable_safety_checks"] == second_priority[
        "applicable_safety_checks"
    ]
    assert summary["second_priority_future_wing_readback_target_count"] == len(
        second_priority["future_wing_readback_targets"]
    )
    assert summary["second_priority_blocked_live_action_count"] == len(
        second_priority["blocked_live_actions"]
    )
    assert summary["third_priority_method_count"] == third_priority["method_count"]
    assert summary["third_priority_method_ids"] == third_priority["method_ids"]
    assert summary["third_priority_applicable_safety_checks"] == third_priority[
        "applicable_safety_checks"
    ]
    assert summary["third_priority_future_wing_readback_target_count"] == len(
        third_priority["future_wing_readback_targets"]
    )
    assert summary["third_priority_blocked_live_action_count"] == len(
        third_priority["blocked_live_actions"]
    )
    assert summary["fourth_priority_method_count"] == fourth_priority["method_count"]
    assert summary["fourth_priority_method_ids"] == fourth_priority["method_ids"]
    assert summary["fourth_priority_applicable_safety_checks"] == fourth_priority[
        "applicable_safety_checks"
    ]
    assert summary["fourth_priority_future_wing_readback_target_count"] == len(
        fourth_priority["future_wing_readback_targets"]
    )
    assert summary["fourth_priority_blocked_live_action_count"] == len(
        fourth_priority["blocked_live_actions"]
    )
    assert summary["fifth_priority_method_count"] == fifth_priority["method_count"]
    assert summary["fifth_priority_method_ids"] == fifth_priority["method_ids"]
    assert summary["fifth_priority_applicable_safety_checks"] == fifth_priority[
        "applicable_safety_checks"
    ]
    assert summary["fifth_priority_future_wing_readback_target_count"] == len(
        fifth_priority["future_wing_readback_targets"]
    )
    assert summary["fifth_priority_blocked_live_action_count"] == len(
        fifth_priority["blocked_live_actions"]
    )
    assert summary["sixth_priority_method_count"] == sixth_priority["method_count"]
    assert summary["sixth_priority_method_ids"] == sixth_priority["method_ids"]
    assert summary["sixth_priority_applicable_safety_checks"] == sixth_priority[
        "applicable_safety_checks"
    ]
    assert summary["sixth_priority_future_wing_readback_target_count"] == len(
        sixth_priority["future_wing_readback_targets"]
    )
    assert summary["sixth_priority_blocked_live_action_count"] == len(
        sixth_priority["blocked_live_actions"]
    )
    assert summary["seventh_priority_method_count"] == seventh_priority[
        "method_count"
    ]
    assert summary["seventh_priority_method_ids"] == seventh_priority["method_ids"]
    assert summary["seventh_priority_applicable_safety_checks"] == (
        seventh_priority["applicable_safety_checks"]
    )
    assert summary["seventh_priority_future_wing_readback_target_count"] == len(
        seventh_priority["future_wing_readback_targets"]
    )
    assert summary["seventh_priority_blocked_live_action_count"] == len(
        seventh_priority["blocked_live_actions"]
    )
    assert summary["live_authority_method_ids"] == []
    assert summary["live_apply_method_ids"] == []
    assert summary["queueable_method_ids"] == []
    assert summary["blocked_by"] == []
    assert (
        summary["covered_objective_family_count"]
        == summary["required_objective_family_count"]
    )

    families = {row["objective_family"]: row for row in report["family_rows"]}
    for family in expected_required_families:
        row = families[family]
        assert row["covered"] is True
        assert row["method_count"] >= 1
        assert row["missing_required_flags"] == []
        assert row["missing_required_operator_surfaces"] == []
        _assert_coverage_row_fail_closed(row)
        assert row["source_family_read_only_or_dry_run_only"] is True
        assert row["source_family_live_authority_granted_claimed"] is False
        assert row["source_family_live_apply_allowed_claimed"] is False
        assert row["source_family_queue_import_allowed_claimed"] is False

    reported_method_ids = {row["method_id"] for row in report["method_rows"]}
    for family, method_ids in OBJECTIVE_FAMILY_METHODS.items():
        if family in expected_required_families:
            assert set(method_ids) <= reported_method_ids

    for row in report["method_rows"]:
        if row["objective_family"] not in expected_required_families:
            continue
        expected_priority, expected_priority_label = expected_priority_by_family[
            row["objective_family"]
        ]
        assert row["priority"] == expected_priority
        assert row["priority_label"] == expected_priority_label
        assert row["priority_review_scope"] == "required_priority"
        assert row["missing_required_flags"] == []
        assert row["missing_required_operator_surfaces"] == []
        assert set(row["operator_required_surfaces"]) == expected_operator_surfaces
        assert all(row["operator_required_surface_flags"].values())
        _assert_coverage_row_fail_closed(row)
        assert row["source_contract_read_only_or_dry_run_only"] is True
        assert row["source_contract_live_authority_granted_claimed"] is False
        assert row["source_contract_live_apply_allowed_claimed"] is False
        assert row["source_contract_queue_import_allowed_claimed"] is False
        assert row["safety_evidence_required"]
        assert row["applicable_safety_checks"]
        assert row["source_evidence_required"] is True
        assert row["source_evidence_details_required"] is True
        assert row["measured_or_audible_problem_required"] is True
        assert row["proposed_correction_required"] is True
        assert row["confidence_required"] is True
        assert row["confidence_review_required"] is True
        assert row["bounded_delta_required"] is True
        assert row["bounded_delta_details_required"] is True
        assert row["bounded_delta_unit"]
        assert row["bounded_delta_policy"]
        assert row["safety_blockers_required"] is True
        assert row["safety_blocker_details_required"] is True
        assert row["headroom_feedback_true_peak_checks_required"] is True
        assert row["headroom_feedback_true_peak_check_details_required"] is True
        assert row["operator_readable_explanation_required"] is True
        assert row["operator_explanation_packet_required"] is True
        assert row["operator_evidence_packet_required"] is True
        assert row["rollback_expectation"]
        assert row["rollback_readback_contract_required"] is True
        assert row["rollback_readback_details_required"] is True
        assert row["future_live_required_proofs"]
        assert row["future_wing_readback_targets"]
        assert row["blocked_live_actions"]
        assert "do_not_queue_or_apply_live" in row["operator_review_checklist"]
        assert {
            "request_more_evidence",
            "accept_as_offline_proposal",
            "reject_proposal",
            "defer_until_real_wing_proof",
        } <= set(row["operator_review_decisions"])
        assert row["denial_reason"]


def test_correction_method_goal_operator_review_packet_is_readonly_and_actionable():
    packet = build_correction_method_goal_operator_review_packet()

    assert (
        packet["schema"]
        == "automixer.correction_method_goal_operator_review_packet.v1"
    )
    assert packet["coverage_schema"] == "automixer.correction_method_goal_coverage.v1"
    assert packet["status"] == "operator_review_packet_ready_proposal_only"
    assert packet["coverage_status"] == (
        "proposal_layer_contracts_ready_operator_review_required"
    )
    assert packet["covered"] is True
    assert packet["read_only_or_dry_run_only"] is True
    assert packet["live_authority_granted"] is False
    assert packet["readiness_closure_allowed"] is False
    assert packet["live_apply_allowed"] is False
    assert packet["queue_import_allowed"] is False
    assert packet["contacts_wing"] is False
    assert packet["contacts_backend"] is False
    assert packet["mutates_mixer"] is False
    assert packet["operator_review_required"] is True
    assert packet["future_wing_readback_required"] is True
    assert packet["hardware_proof_required_before_live"] is True
    assert packet["blocks_live_promotion"] is True
    assert packet["blocks_queue_import"] is True
    assert packet["full_autonomous_ready"] is False
    assert packet["can_apply_live"] is False
    assert packet["can_import_to_queue"] is False
    assert packet["can_authorize_live"] is False
    assert packet["can_close_readiness"] is False
    assert packet["queue_action"] == "not_queueable_until_promoted"
    assert packet["denial_reason"] == (
        "operator_review_packet_cannot_replace_real_wing_proof"
    )

    summary = packet["summary"]
    assert summary["priority_section_count"] == 7
    assert (
        summary["covered_objective_family_count"]
        == summary["required_objective_family_count"]
    )
    assert summary["missing_required_operator_surface_method_ids"] == []
    assert summary["live_authority_method_ids"] == []
    assert summary["live_apply_method_ids"] == []
    assert summary["queueable_method_ids"] == []
    assert summary["blocked_by"] == []
    assert summary["method_review_card_count"] == len(packet["method_review_cards"])
    assert summary["supporting_method_review_card_count"] == len(
        packet["supporting_method_cards"]
    )
    contract_coverage = summary["operator_review_contract_coverage"]
    assert set(contract_coverage) == set(OPERATOR_REVIEW_CONTRACT_FIELDS)
    for field, row in contract_coverage.items():
        assert field in OPERATOR_REVIEW_CONTRACT_FIELDS
        assert row["method_card_count"] == len(packet["method_review_cards"])
        assert row["required_method_card_count"] == len(
            packet["method_review_cards"]
        )
        assert row["priority_section_count"] == len(packet["priority_sections"])
        assert row["priority_section_contract_item_count"] > 0
        assert row["covered"] is True
    first_priority = packet["first_priority_review_summary"]
    assert first_priority["priority"] == 1
    assert first_priority["label"] == "Gain / Trim and Fader / Balance"
    assert {
        "lufs_gain_staging",
        "auto_soundcheck_engine",
        "auto_fader_v1",
        "live_shared_mix",
        "bus_group_balance_proposal",
    } <= set(first_priority["method_ids"])
    assert {
        "headroom",
        "feedback",
        "true_peak",
        "ayaic_level_plane",
        "fader_ceiling",
        "bleed",
    } <= set(first_priority["applicable_safety_checks"])
    assert first_priority["read_only_or_dry_run_only"] is True
    assert first_priority["operator_review_required"] is True
    assert first_priority["future_wing_readback_required"] is True
    assert first_priority["live_authority_granted"] is False
    assert first_priority["readiness_closure_allowed"] is False
    assert first_priority["live_apply_allowed"] is False
    assert first_priority["queue_import_allowed"] is False
    assert summary["first_priority_method_count"] == first_priority["method_count"]
    assert summary["first_priority_method_ids"] == first_priority["method_ids"]
    assert summary["first_priority_applicable_safety_checks"] == first_priority[
        "applicable_safety_checks"
    ]
    assert summary["first_priority_future_wing_readback_target_count"] == len(
        first_priority["future_wing_readback_targets"]
    )
    assert summary["first_priority_blocked_live_action_count"] == len(
        first_priority["blocked_live_actions"]
    )
    second_priority = packet["second_priority_review_summary"]
    assert second_priority["priority"] == 2
    assert second_priority["label"] == "EQ and Compressor"
    assert second_priority["objective_families"] == [
        "eq",
        "compressor_dynamics",
    ]
    assert {
        "auto_eq",
        "corrective_eq_proposal",
        "cross_adaptive_eq",
        "frequency_separation_proposal",
        "auto_compressor",
        "auto_compressor_cf",
        "compressor_dynamics_proposal",
    } <= set(second_priority["method_ids"])
    assert {
        "headroom",
        "feedback",
        "true_peak",
        "manual_band",
        "phase",
        "gain_reduction",
        "makeup_gain",
    } <= set(second_priority["applicable_safety_checks"])
    assert second_priority["future_wing_readback_targets"]
    assert second_priority["blocked_live_actions"]
    assert second_priority["read_only_or_dry_run_only"] is True
    assert second_priority["operator_review_required"] is True
    assert second_priority["future_wing_readback_required"] is True
    assert second_priority["hardware_proof_required_before_live"] is True
    assert second_priority["live_authority_granted"] is False
    assert second_priority["readiness_closure_allowed"] is False
    assert second_priority["live_apply_allowed"] is False
    assert second_priority["queue_import_allowed"] is False
    assert second_priority["contacts_wing"] is False
    assert second_priority["contacts_backend"] is False
    assert second_priority["mutates_mixer"] is False
    assert summary["second_priority_method_count"] == second_priority["method_count"]
    assert summary["second_priority_method_ids"] == second_priority["method_ids"]
    assert summary["second_priority_applicable_safety_checks"] == second_priority[
        "applicable_safety_checks"
    ]
    assert summary["second_priority_future_wing_readback_target_count"] == len(
        second_priority["future_wing_readback_targets"]
    )
    assert summary["second_priority_blocked_live_action_count"] == len(
        second_priority["blocked_live_actions"]
    )
    third_priority = packet["third_priority_review_summary"]
    assert third_priority["priority"] == 3
    assert third_priority["label"] == "Gate / Expander / De-esser"
    assert third_priority["objective_families"] == [
        "gate_expander",
        "gate_expander_deesser",
    ]
    assert {
        "auto_gate_caig",
        "context_aware_gate_agent",
        "deesser_vocal_sibilance",
    } <= set(third_priority["method_ids"])
    assert {
        "source_identity",
        "chatter_guard",
        "hysteresis",
        "tails_and_attack",
        "speech_intelligibility",
        "true_peak",
    } <= set(third_priority["applicable_safety_checks"])
    assert {"gate_write", "deesser_write"} <= set(
        third_priority["blocked_live_actions"]
    )
    assert third_priority["future_wing_readback_targets"]
    assert third_priority["blocked_live_actions"]
    assert third_priority["read_only_or_dry_run_only"] is True
    assert third_priority["operator_review_required"] is True
    assert third_priority["future_wing_readback_required"] is True
    assert third_priority["hardware_proof_required_before_live"] is True
    assert third_priority["live_authority_granted"] is False
    assert third_priority["readiness_closure_allowed"] is False
    assert third_priority["live_apply_allowed"] is False
    assert third_priority["queue_import_allowed"] is False
    assert third_priority["contacts_wing"] is False
    assert third_priority["contacts_backend"] is False
    assert third_priority["mutates_mixer"] is False
    assert summary["third_priority_method_count"] == third_priority["method_count"]
    assert summary["third_priority_method_ids"] == third_priority["method_ids"]
    assert summary["third_priority_applicable_safety_checks"] == third_priority[
        "applicable_safety_checks"
    ]
    assert summary["third_priority_future_wing_readback_target_count"] == len(
        third_priority["future_wing_readback_targets"]
    )
    assert summary["third_priority_blocked_live_action_count"] == len(
        third_priority["blocked_live_actions"]
    )
    fourth_priority = packet["fourth_priority_review_summary"]
    assert fourth_priority["priority"] == 4
    assert fourth_priority["label"] == "FX / Sends"
    assert fourth_priority["objective_families"] == ["fx_sends"]
    assert {
        "auto_fx",
        "fx_direct_apply",
        "fx_send_proposal",
    } <= set(fourth_priority["method_ids"])
    assert {
        "headroom",
        "feedback",
        "return_level",
        "monitor_safety",
        "intelligibility",
        "parameter_bounds",
    } <= set(fourth_priority["applicable_safety_checks"])
    assert {"fx_send_write", "fx_parameter_write"} <= set(
        fourth_priority["blocked_live_actions"]
    )
    assert fourth_priority["future_wing_readback_targets"]
    assert fourth_priority["blocked_live_actions"]
    assert fourth_priority["read_only_or_dry_run_only"] is True
    assert fourth_priority["operator_review_required"] is True
    assert fourth_priority["future_wing_readback_required"] is True
    assert fourth_priority["hardware_proof_required_before_live"] is True
    assert fourth_priority["live_authority_granted"] is False
    assert fourth_priority["readiness_closure_allowed"] is False
    assert fourth_priority["live_apply_allowed"] is False
    assert fourth_priority["queue_import_allowed"] is False
    assert fourth_priority["contacts_wing"] is False
    assert fourth_priority["contacts_backend"] is False
    assert fourth_priority["mutates_mixer"] is False
    assert summary["fourth_priority_method_count"] == fourth_priority["method_count"]
    assert summary["fourth_priority_method_ids"] == fourth_priority["method_ids"]
    assert summary["fourth_priority_applicable_safety_checks"] == fourth_priority[
        "applicable_safety_checks"
    ]
    assert summary["fourth_priority_future_wing_readback_target_count"] == len(
        fourth_priority["future_wing_readback_targets"]
    )
    assert summary["fourth_priority_blocked_live_action_count"] == len(
        fourth_priority["blocked_live_actions"]
    )
    fifth_priority = packet["fifth_priority_review_summary"]
    assert fifth_priority["priority"] == 5
    assert fifth_priority["label"] == "Pan / Stereo"
    assert fifth_priority["objective_families"] == ["pan_stereo"]
    assert {
        "auto_panner",
        "pan_spatial_proposal",
    } <= set(fifth_priority["method_ids"])
    assert {
        "mono_compatibility",
        "phase",
        "source_identity",
    } <= set(fifth_priority["applicable_safety_checks"])
    assert {"pan_write", "stereo_width_write"} <= set(
        fifth_priority["blocked_live_actions"]
    )
    assert fifth_priority["future_wing_readback_targets"]
    assert fifth_priority["blocked_live_actions"]
    assert fifth_priority["read_only_or_dry_run_only"] is True
    assert fifth_priority["operator_review_required"] is True
    assert fifth_priority["future_wing_readback_required"] is True
    assert fifth_priority["hardware_proof_required_before_live"] is True
    assert fifth_priority["live_authority_granted"] is False
    assert fifth_priority["readiness_closure_allowed"] is False
    assert fifth_priority["live_apply_allowed"] is False
    assert fifth_priority["queue_import_allowed"] is False
    assert fifth_priority["contacts_wing"] is False
    assert fifth_priority["contacts_backend"] is False
    assert fifth_priority["mutates_mixer"] is False
    assert summary["fifth_priority_method_count"] == fifth_priority["method_count"]
    assert summary["fifth_priority_method_ids"] == fifth_priority["method_ids"]
    assert summary["fifth_priority_applicable_safety_checks"] == fifth_priority[
        "applicable_safety_checks"
    ]
    assert summary["fifth_priority_future_wing_readback_target_count"] == len(
        fifth_priority["future_wing_readback_targets"]
    )
    assert summary["fifth_priority_blocked_live_action_count"] == len(
        fifth_priority["blocked_live_actions"]
    )
    sixth_priority = packet["sixth_priority_review_summary"]
    assert sixth_priority["priority"] == 6
    assert sixth_priority["label"] == "Routing / Groups / Scenes"
    assert sixth_priority["objective_families"] == ["routing_groups_scenes"]
    assert {
        "mute_management_proposal",
        "routing_management_proposal",
        "routing_snapshot_reset_writes",
        "scene_preset_management_proposal",
    } <= set(sixth_priority["method_ids"])
    assert {
        "active_performance_guard",
        "active_signal_guard",
        "destructive_scope",
        "feedback",
        "read_only_inventory",
        "read_only_map",
        "source_identity",
    } <= set(sixth_priority["applicable_safety_checks"])
    assert {
        "bus_assignment_write",
        "matrix_write",
        "mute_group_write",
        "mute_write",
        "preset_load",
        "preset_save",
        "routing_write",
        "scene_load",
        "scene_save",
        "unmute_write",
    } <= set(sixth_priority["blocked_live_actions"])
    assert sixth_priority["future_wing_readback_targets"]
    assert sixth_priority["blocked_live_actions"]
    assert sixth_priority["read_only_or_dry_run_only"] is True
    assert sixth_priority["operator_review_required"] is True
    assert sixth_priority["future_wing_readback_required"] is True
    assert sixth_priority["hardware_proof_required_before_live"] is True
    assert sixth_priority["live_authority_granted"] is False
    assert sixth_priority["readiness_closure_allowed"] is False
    assert sixth_priority["live_apply_allowed"] is False
    assert sixth_priority["queue_import_allowed"] is False
    assert sixth_priority["contacts_wing"] is False
    assert sixth_priority["contacts_backend"] is False
    assert sixth_priority["mutates_mixer"] is False
    assert summary["sixth_priority_method_count"] == sixth_priority["method_count"]
    assert summary["sixth_priority_method_ids"] == sixth_priority["method_ids"]
    assert summary["sixth_priority_applicable_safety_checks"] == sixth_priority[
        "applicable_safety_checks"
    ]
    assert summary["sixth_priority_future_wing_readback_target_count"] == len(
        sixth_priority["future_wing_readback_targets"]
    )
    assert summary["sixth_priority_blocked_live_action_count"] == len(
        sixth_priority["blocked_live_actions"]
    )
    seventh_priority = packet["seventh_priority_review_summary"]
    assert seventh_priority["priority"] == 7
    assert seventh_priority["label"] == "Master / Output"
    assert seventh_priority["objective_families"] == ["master_output"]
    assert {"master_bus_optimization_proposal"} <= set(
        seventh_priority["method_ids"]
    )
    assert {
        "headroom",
        "limiter_or_dynamics_proof",
        "pa_feed_safety",
        "reference_feed_exclusion",
        "true_peak",
    } <= set(seventh_priority["applicable_safety_checks"])
    assert {
        "limiter_write",
        "master_bus_write",
        "matrix_pa_feed_write",
    } <= set(seventh_priority["blocked_live_actions"])
    assert seventh_priority["future_wing_readback_targets"]
    assert seventh_priority["blocked_live_actions"]
    assert seventh_priority["read_only_or_dry_run_only"] is True
    assert seventh_priority["operator_review_required"] is True
    assert seventh_priority["future_wing_readback_required"] is True
    assert seventh_priority["hardware_proof_required_before_live"] is True
    assert seventh_priority["live_authority_granted"] is False
    assert seventh_priority["readiness_closure_allowed"] is False
    assert seventh_priority["live_apply_allowed"] is False
    assert seventh_priority["queue_import_allowed"] is False
    assert seventh_priority["contacts_wing"] is False
    assert seventh_priority["contacts_backend"] is False
    assert seventh_priority["mutates_mixer"] is False
    assert summary["seventh_priority_method_count"] == seventh_priority[
        "method_count"
    ]
    assert summary["seventh_priority_method_ids"] == seventh_priority["method_ids"]
    assert summary["seventh_priority_applicable_safety_checks"] == (
        seventh_priority["applicable_safety_checks"]
    )
    assert summary["seventh_priority_future_wing_readback_target_count"] == len(
        seventh_priority["future_wing_readback_targets"]
    )
    assert summary["seventh_priority_blocked_live_action_count"] == len(
        seventh_priority["blocked_live_actions"]
    )

    expected_required_families = {
        "gain_trim",
        "gain_trim_and_fader_balance",
        "fader_balance",
        "eq",
        "compressor_dynamics",
        "gate_expander",
        "gate_expander_deesser",
        "fx_sends",
        "pan_stereo",
        "routing_groups_scenes",
        "master_output",
    }
    assert set(packet["required_objective_families"]) == expected_required_families

    section_families = set()
    section_method_ids = set()
    expected_priority_by_method = {}
    for section in packet["priority_sections"]:
        assert section["covered"] is True
        assert section["method_count"] >= 1
        assert section["review_status"] == "operator_review_required"
        assert section["read_only_or_dry_run_only"] is True
        assert section["live_authority_granted"] is False
        assert section["readiness_closure_allowed"] is False
        assert section["live_apply_allowed"] is False
        assert section["queue_import_allowed"] is False
        assert section["operator_review_required"] is True
        assert section["future_wing_readback_required"] is True
        assert section["hardware_proof_required_before_live"] is True
        assert section["blocks_live_promotion"] is True
        assert section["blocks_queue_import"] is True
        assert section["full_autonomous_ready"] is False
        assert section["can_apply_live"] is False
        assert section["can_import_to_queue"] is False
        assert section["can_authorize_live"] is False
        assert section["can_close_readiness"] is False
        assert section["queue_action"] == "not_queueable_until_promoted"
        for field in OPERATOR_REVIEW_CONTRACT_FIELDS:
            assert section[field]
            expected_items = sorted(
                {
                    str(item)
                    for card in section["methods"]
                    for item in _operator_review_contract_items(card[field])
                }
            )
            assert sorted(section[field]) == expected_items
        section_families.update(section["objective_families"])
        for card in section["methods"]:
            method_id = card["method_id"]
            assert card["priority"] == section["priority"]
            assert card["priority_label"] == section["label"]
            assert card["priority_review_scope"] == "required_priority"
            expected_priority_by_method[method_id] = (
                section["priority"],
                section["label"],
            )
            section_method_ids.add(method_id)
    assert section_families == expected_required_families

    all_cards = {card["method_id"]: card for card in packet["method_review_cards"]}
    assert ALLOWED_BLOCKED_LIVE_ACTIONS == FULL_BLOCKED_LIVE_ACTIONS
    assert ALLOWED_REQUIRED_SAFETY_CHECKS == FULL_REQUIRED_SAFETY_CHECKS
    assert ALLOWED_BOUNDED_DELTA_UNITS == FULL_BOUNDED_DELTA_UNITS
    assert ALLOWED_BOUNDED_DELTA_POLICIES == FULL_BOUNDED_DELTA_POLICIES
    assert (
        ALLOWED_FUTURE_WING_READBACK_TARGETS
        == FULL_FUTURE_WING_READBACK_TARGETS
    )
    assert ALLOWED_ROLLBACK_EXPECTATIONS == FULL_ROLLBACK_EXPECTATIONS
    generated_blocked_live_actions = {
        action
        for card in packet["method_review_cards"]
        for action in card["blocked_live_actions"]
    }
    assert generated_blocked_live_actions <= set(FULL_BLOCKED_LIVE_ACTIONS)
    assert set(FULL_BLOCKED_LIVE_ACTIONS) - generated_blocked_live_actions == set()
    generated_required_safety_checks = {
        check
        for card in packet["method_review_cards"]
        for check in card["required_safety_checks"]
    }
    assert generated_required_safety_checks <= set(FULL_REQUIRED_SAFETY_CHECKS)
    assert generated_required_safety_checks == set(FULL_REQUIRED_SAFETY_CHECKS)
    generated_bounded_delta_units = {
        card["bounded_delta_unit"] for card in packet["method_review_cards"]
    }
    assert generated_bounded_delta_units <= set(FULL_BOUNDED_DELTA_UNITS)
    assert generated_bounded_delta_units == set(FULL_BOUNDED_DELTA_UNITS)
    generated_bounded_delta_policies = {
        card["bounded_delta_policy"] for card in packet["method_review_cards"]
    }
    assert generated_bounded_delta_policies <= set(FULL_BOUNDED_DELTA_POLICIES)
    assert generated_bounded_delta_policies == set(FULL_BOUNDED_DELTA_POLICIES)
    generated_future_wing_readback_targets = {
        target
        for card in packet["method_review_cards"]
        for target in card["future_wing_readback_targets"]
    }
    assert generated_future_wing_readback_targets <= set(
        FULL_FUTURE_WING_READBACK_TARGETS
    )
    assert generated_future_wing_readback_targets == set(
        FULL_FUTURE_WING_READBACK_TARGETS
    )
    generated_rollback_expectations = {
        card["rollback_expectation"] for card in packet["method_review_cards"]
    }
    assert generated_rollback_expectations <= set(FULL_ROLLBACK_EXPECTATIONS)
    assert generated_rollback_expectations == set(FULL_ROLLBACK_EXPECTATIONS)
    for family, method_ids in OBJECTIVE_FAMILY_METHODS.items():
        if family in expected_required_families:
            assert set(method_ids) <= section_method_ids
            assert set(method_ids) <= set(all_cards)

    for card in packet["method_review_cards"]:
        assert card["review_status"] == "operator_review_required"
        assert card["review_scope"] == "offline_proposal_only"
        if card["method_id"] in expected_priority_by_method:
            expected_priority, expected_priority_label = expected_priority_by_method[
                card["method_id"]
            ]
            assert card["priority"] == expected_priority
            assert card["priority_label"] == expected_priority_label
            assert card["priority_review_scope"] == "required_priority"
        else:
            assert card["priority"] is None
            assert card["priority_label"] == "Supporting / Context"
            assert card["priority_review_scope"] == "supporting_context"
        assert "do not queue or apply it live" in card["operator_question"]
        assert REQUIRED_METHOD_CARD_CONFIRMATIONS == FULL_METHOD_CARD_CONFIRMATIONS
        assert tuple(card["what_operator_must_confirm"]) == (
            FULL_METHOD_CARD_CONFIRMATIONS
        )
        assert card["required_operator_surfaces"] == packet[
            "required_operator_surfaces"
        ]
        assert card["missing_required_operator_surfaces"] == []
        assert set(REQUIRED_OPERATOR_FIELDS) <= set(
            card["operator_payload_required_fields"]
        )
        assert card["operator_evidence_required_fields"]
        assert card["required_evidence"]
        assert card["required_safety_checks"]
        assert set(card["required_safety_checks"]) <= set(
            ALLOWED_REQUIRED_SAFETY_CHECKS
        )
        assert len(card["required_safety_checks"]) == len(
            set(card["required_safety_checks"])
        )
        _assert_operator_review_contract(
            card["source_evidence"],
            source_fields=["source_evidence"],
        )
        _assert_operator_review_contract(
            card["source_evidence_details"],
            source_fields=["source_evidence_details"],
        )
        _assert_operator_review_contract(
            card["measured_or_audible_problem"],
            source_fields=["measured_or_audible_problem"],
        )
        _assert_operator_review_contract(
            card["proposed_correction"],
            source_fields=["proposed_correction"],
        )
        _assert_operator_review_contract(
            card["confidence"],
            source_fields=[
                "confidence",
                "confidence_review",
                "confidence_review_details",
            ],
        )
        _assert_operator_review_contract(
            card["bounded_delta"],
            source_fields=[
                "bounded_delta",
                "bounded_delta_details",
                "bounded_delta_unit",
                "bounded_delta_policy",
            ],
        )
        _assert_operator_review_contract(
            card["safety_blockers"],
            source_fields=[
                "safety_blockers",
                "effective_safety_blockers",
                "safety_blocker_details",
            ],
        )
        _assert_operator_review_contract(
            card["headroom_feedback_true_peak_checks"],
            source_fields=[
                "headroom_feedback_true_peak_checks",
                "headroom_feedback_true_peak_check_details",
            ],
        )
        _assert_operator_review_contract(
            card["operator_readable_explanation"],
            source_fields=[
                "operator_readable_explanation",
                "operator_explanation_packet",
            ],
        )
        _assert_operator_review_contract(
            card["rollback_readback_contract"],
            source_fields=[
                "rollback_expectation",
                "rollback_readback_contract",
                "rollback_readback_details",
            ],
        )
        _assert_operator_review_contract(
            card["future_wing_readback_contract"],
            source_fields=[
                "future_wing_readback_contract",
                "future_wing_readback_contract_summary",
                "future_wing_readback_targets",
            ],
        )
        _assert_operator_review_contract(
            card["test_evidence"],
            source_fields=[
                "test_evidence_references",
                "operator_review_contract_tests",
                "export_verification_tests",
                "no_wing_release_gate_tests",
            ],
        )
        assert card["test_evidence_references"]
        assert any(
            reference.startswith("tests/test_")
            for reference in card["test_evidence_references"]
        )
        assert (
            "tests/test_processing_proposal_contracts.py::"
            "test_correction_method_goal_operator_review_packet_is_readonly_and_actionable"
        ) in card["test_evidence_references"]
        assert tuple(card["what_operator_must_confirm"]) == (
            FULL_METHOD_CARD_CONFIRMATIONS
        )
        assert sorted(card["required_confirmation_details"]) == sorted(
            _expected_required_confirmation_details(
                card["what_operator_must_confirm"]
            )
        )
        assert sorted(card["operator_acceptance_gate_details"]) == sorted(
            _expected_operator_acceptance_gate_details(
                card["what_operator_must_confirm"]
            )
        )
        assert len(card["operator_acceptance_gate_details"]) == len(
            card["what_operator_must_confirm"]
        )
        assert all(
            "acceptance_scope=offline_proposal_only" in item
            and "required=true" in item
            and "required_before_operator_acceptance=true" in item
            and "required_before_live_promotion=true" in item
            and "required_before_queue_import=true" in item
            and "live_apply_allowed=false" in item
            and "queue_import_allowed=false" in item
            and "readiness_closure_allowed=false" in item
            and "requires_real_wing_proof_before_live=true" in item
            for item in card["operator_acceptance_gate_details"]
        )
        assert any(
            item.startswith(
                "confidence_and_confidence_blockers_are_reviewed:source=confidence"
            )
            for item in card["required_confirmation_details"]
        )
        assert sorted(card["operator_review_decision_option_details"]) == sorted(
            _expected_operator_review_decision_option_details(
                card["operator_review_decisions"]
            )
        )
        assert any(
            item.startswith(
                "accept_as_offline_proposal:"
                "effect=records_operator_review_for_offline_plan_only"
            )
            for item in card["operator_review_decision_option_details"]
        )
        assert all(
            "live_apply_allowed=false" in item
            and "queue_import_allowed=false" in item
            for item in card["operator_review_decision_option_details"]
        )
        assert tuple(card["operator_problem_review_summary"]) == (
            REQUIRED_OPERATOR_PROBLEM_REVIEW_SUMMARY
        )
        focus = card["operator_method_review_focus"]
        assert tuple(item.split("=", 1)[0] for item in focus) == (
            REQUIRED_OPERATOR_METHOD_REVIEW_FOCUS_KEYS
        )
        assert f"method_id={card['method_id']}" in focus
        priority = "none" if card["priority"] is None else str(card["priority"])
        assert f"priority={priority}" in focus
        assert f"priority_label={card['priority_label']}" in focus
        assert f"objective_family={card['objective_family']}" in focus
        assert f"required_evidence={','.join(card['required_evidence'])}" in focus
        assert (
            f"required_safety_checks={','.join(card['required_safety_checks'])}"
            in focus
        )
        assert (
            "bounded_delta="
            f"{card['bounded_delta_unit']}:{card['bounded_delta_policy']}"
        ) in focus
        assert f"rollback_expectation={card['rollback_expectation']}" in focus
        assert (
            "future_wing_readback_targets="
            f"{','.join(card['future_wing_readback_targets'])}"
        ) in focus
        assert f"blocked_live_actions={','.join(card['blocked_live_actions'])}" in focus
        assert (
            "live_boundary=offline_only_no_queue_no_live_apply_until_real_wing_proof"
            in focus
        )
        assert (
            REQUIRED_OPERATOR_WHY_SAFE_SUMMARY_KEYS
            == FULL_OPERATOR_WHY_SAFE_SUMMARY_KEYS
        )
        why_safe_summary = card["why_safe_summary"]
        assert tuple(item.split("=", 1)[0] for item in why_safe_summary) == (
            FULL_OPERATOR_WHY_SAFE_SUMMARY_KEYS
        )
        assert "read_only_or_dry_run_only=true" in why_safe_summary
        assert "live_apply_allowed=false" in why_safe_summary
        assert "queue_import_allowed=false" in why_safe_summary
        assert "operator_review_required=true" in why_safe_summary
        assert (
            "bounded_delta="
            f"{card['bounded_delta_unit']}:{card['bounded_delta_policy']}"
        ) in why_safe_summary
        assert (
            f"required_safety_checks={','.join(card['required_safety_checks'])}"
            in why_safe_summary
        )
        assert (
            REQUIRED_OPERATOR_WHY_BLOCKED_SUMMARY_KEYS
            == FULL_OPERATOR_WHY_BLOCKED_SUMMARY_KEYS
        )
        why_blocked_summary = card["why_blocked_summary"]
        assert tuple(item.split("=", 1)[0] for item in why_blocked_summary) == (
            FULL_OPERATOR_WHY_BLOCKED_SUMMARY_KEYS
        )
        assert f"denial_reason={card['denial_reason']}" in why_blocked_summary
        assert "future_wing_readback_required=true" in why_blocked_summary
        assert "hardware_proof_required_before_live=true" in why_blocked_summary
        assert "readiness_closure_allowed=false" in why_blocked_summary
        assert (
            f"blocked_live_actions={','.join(card['blocked_live_actions'])}"
            in why_blocked_summary
        )
        assert (
            "future_live_required_proofs="
            f"{','.join(card['future_live_required_proofs'])}"
        ) in why_blocked_summary
        assert sorted(card["goal_requirement_details"]) == (
            _expected_goal_requirement_details(card)
        )
        goal_requirement_detail_text = ",".join(card["goal_requirement_details"])
        for requirement in (
            "source_evidence",
            "measured_or_audible_problem",
            "proposed_correction",
            "confidence",
            "bounded_delta",
            "safety_blockers",
            "headroom_feedback_true_peak_checks",
            "operator_readable_explanation",
            "rollback_expectation",
            "future_wing_readback_contract",
            "test_evidence",
        ):
            assert f"{requirement}:source=correction_method_goal_requirement" in (
                goal_requirement_detail_text
            )
        assert all(
            "coverage_status=covered" in item
            and "acceptance_scope=offline_proposal_only" in item
            and "live_apply_allowed=false" in item
            and "queue_import_allowed=false" in item
            and "readiness_closure_allowed=false" in item
            and "requires_real_wing_proof_before_live=true" in item
            for item in card["goal_requirement_details"]
        )
        assert sorted(card["method_specific_review_details"]) == (
            _expected_method_specific_review_details(card)
        )
        method_specific_review_text = ",".join(
            card["method_specific_review_details"]
        )
        for focus in (
            "method_problem_focus",
            "method_correction_focus",
            "method_safety_focus",
            "method_future_live_boundary",
        ):
            assert (
                f"{focus}:source=correction_method_goal.method_specific_review"
                in method_specific_review_text
            )
        assert all(
            "acceptance_scope=offline_proposal_only" in item
            and "live_apply_allowed=false" in item
            and "queue_import_allowed=false" in item
            and "readiness_closure_allowed=false" in item
            and "requires_real_wing_proof_before_live=true" in item
            for item in card["method_specific_review_details"]
        )
        assert sorted(card["operator_actionable_explanation_details"]) == (
            _expected_operator_actionable_explanation_details(card)
        )
        actionable_explanation_text = ",".join(
            card["operator_actionable_explanation_details"]
        )
        for explanation_step in (
            "what_is_wrong",
            "what_is_proposed",
            "why_safe_or_blocked",
            "what_is_needed_before_live",
        ):
            assert (
                f"{explanation_step}:source=operator_actionable_explanation"
                in actionable_explanation_text
            )
        assert all(
            "acceptance_scope=offline_proposal_only" in item
            and "required_before_operator_acceptance=true" in item
            and "live_apply_allowed=false" in item
            and "queue_import_allowed=false" in item
            and "readiness_closure_allowed=false" in item
            and "requires_real_wing_proof_before_live=true" in item
            for item in card["operator_actionable_explanation_details"]
        )
        assert sorted(card["evidence_problem_correction_chain_details"]) == (
            _expected_evidence_problem_correction_chain_details(card)
        )
        chain_detail_text = ",".join(
            card["evidence_problem_correction_chain_details"]
        )
        for chain_step in (
            "1_source_evidence",
            "2_source_evidence_details",
            "3_measured_or_audible_problem",
            "4_proposed_correction",
            "5_confidence",
            "6_bounded_delta",
        ):
            assert f"chain_step={chain_step}" in chain_detail_text
        assert all(
            "live_apply_allowed=false" in item
            and "queue_import_allowed=false" in item
            and "requires_real_wing_proof_before_live=true" in item
            for item in card["evidence_problem_correction_chain_details"]
        )
        assert card["bounded_delta_unit"]
        assert card["bounded_delta_unit"] in ALLOWED_BOUNDED_DELTA_UNITS
        assert card["bounded_delta_policy"]
        assert card["bounded_delta_policy"] in ALLOWED_BOUNDED_DELTA_POLICIES
        assert card["rollback_expectation"]
        assert card["rollback_expectation"] in ALLOWED_ROLLBACK_EXPECTATIONS
        assert card["future_wing_readback_targets"]
        assert set(card["future_wing_readback_targets"]) <= set(
            ALLOWED_FUTURE_WING_READBACK_TARGETS
        )
        assert len(card["future_wing_readback_targets"]) == len(
            set(card["future_wing_readback_targets"])
        )
        assert card["future_wing_readback_contract_summary"]
        assert (
            REQUIRED_FUTURE_WING_READBACK_CONTRACT_SUMMARY_KEYS
            == FULL_FUTURE_WING_READBACK_CONTRACT_SUMMARY_KEYS
        )
        assert (
            REQUIRED_FUTURE_WING_READBACK_OPERATOR_NEXT_STEP
            == FULL_FUTURE_WING_READBACK_OPERATOR_NEXT_STEP
        )
        assert (
            REQUIRED_FUTURE_WING_READBACK_DENIAL_REASON
            == FULL_FUTURE_WING_READBACK_DENIAL_REASON
        )
        readback_summary_keys = tuple(
            item.split("=", 1)[0]
            for item in card["future_wing_readback_contract_summary"]
        )
        assert readback_summary_keys == (
            FULL_FUTURE_WING_READBACK_CONTRACT_SUMMARY_KEYS
        )
        assert (
            "operator_next_step="
            f"{FULL_FUTURE_WING_READBACK_OPERATOR_NEXT_STEP}"
        ) in card["future_wing_readback_contract_summary"]
        assert (
            "denial_reason="
            f"{FULL_FUTURE_WING_READBACK_DENIAL_REASON}"
        ) in card["future_wing_readback_contract_summary"]
        assert card["future_live_required_proofs"]
        assert sorted(card["future_live_requirement_details"]) == (
            _expected_future_live_requirement_details(card)
        )
        assert all(
            "live_apply_allowed=false" in item
            and "queue_import_allowed=false" in item
            and "requires_real_wing_proof_before_live=true" in item
            for item in card["future_live_requirement_details"]
        )
        assert card["blocked_live_actions"]
        assert sorted(card["safety_blocker_details"]) == (
            _expected_safety_blocker_details(card)
        )
        assert all(
            "live_apply_allowed=false" in item
            and "queue_import_allowed=false" in item
            and "requires_real_wing_proof_before_live=true" in item
            for item in card["safety_blocker_details"]
        )
        assert any(
            "source=required_safety_checks" in item
            for item in card["safety_blocker_details"]
        )
        assert any(
            "source=live_authority_boundary" in item
            for item in card["safety_blocker_details"]
        )
        assert sorted(card["headroom_feedback_true_peak_check_details"]) == (
            _expected_headroom_feedback_true_peak_check_details(card)
        )
        assert all(
            "live_apply_allowed=false" in item
            and "queue_import_allowed=false" in item
            and "requires_real_wing_proof_before_live=true" in item
            for item in card["headroom_feedback_true_peak_check_details"]
        )
        required_safety_check_text = ",".join(card["required_safety_checks"])
        check_detail_text = ",".join(
            card["headroom_feedback_true_peak_check_details"]
        )
        if (
            "headroom" in required_safety_check_text
            or "ceiling" in required_safety_check_text
            or "level_plane" in required_safety_check_text
        ):
            assert "check_type=headroom" in check_detail_text
        if "feedback" in required_safety_check_text:
            assert "check_type=feedback" in check_detail_text
        if "true_peak" in required_safety_check_text:
            assert "check_type=true_peak" in check_detail_text
        assert set(card["blocked_live_actions"]) <= set(
            ALLOWED_BLOCKED_LIVE_ACTIONS
        )
        assert (
            REQUIRED_LIVE_AUTHORITY_BOUNDARY_SUMMARY
            == FULL_LIVE_AUTHORITY_BOUNDARY_SUMMARY
        )
        assert tuple(card["live_authority_boundary_summary"]) == (
            FULL_LIVE_AUTHORITY_BOUNDARY_SUMMARY
        )
        assert "do_not_queue_or_apply_live" in card["operator_review_checklist"]
        assert sorted(card["operator_review_checklist_details"]) == sorted(
            _expected_operator_review_checklist_details(
                card["operator_review_checklist"]
            )
        )
        assert any(
            item.startswith(
                "do_not_queue_or_apply_live:source=live_authority_boundary"
            )
            for item in card["operator_review_checklist_details"]
        )
        assert (
            REQUIRED_OPERATOR_REVIEW_RECORD_FIELDS
            == FULL_OPERATOR_REVIEW_RECORD_FIELDS
        )
        assert tuple(card["operator_review_record_required_fields"]) == (
            FULL_OPERATOR_REVIEW_RECORD_FIELDS
        )
        assert sorted(card["operator_review_record_field_details"]) == sorted(
            FULL_OPERATOR_REVIEW_RECORD_FIELD_DETAILS
        )
        assert any(
            item.startswith("source_evidence_acknowledgement:source=source_evidence")
            for item in card["operator_review_record_field_details"]
        )
        assert sorted(card["operator_next_action_details"]) == (
            _expected_operator_next_action_details(card)
        )
        assert any(
            item.startswith(
                "do_not_queue_or_apply_live:source=live_authority_boundary"
            )
            for item in card["operator_next_action_details"]
        )
        assert all(
            "live_apply_allowed=false" in item
            and "queue_import_allowed=false" in item
            and "requires_real_wing_proof_before_live=true" in item
            for item in card["operator_next_action_details"]
        )
        assert REQUIRED_OFFLINE_REVIEW_DECISIONS == FULL_OFFLINE_REVIEW_DECISIONS
        assert tuple(card["operator_review_decisions"]) == (
            FULL_OFFLINE_REVIEW_DECISIONS
        )
        explanation_contract = card["operator_visible_explanation_contract"]
        assert explanation_contract["source_evidence_required"] is True
        assert explanation_contract["problem_statement_required"] is True
        assert explanation_contract["proposed_correction_required"] is True
        assert explanation_contract["confidence_required"] is True
        assert explanation_contract["bounded_delta_required"] is True
        assert explanation_contract["safety_blockers_required"] is True
        assert explanation_contract["safety_checks_required"] is True
        assert explanation_contract["operator_explanation_required"] is True
        assert explanation_contract["rollback_required"] is True
        assert explanation_contract["future_wing_readback_required"] is True
        assert explanation_contract["test_evidence_required"] is True
        assert card["read_only_or_dry_run_only"] is True
        assert card["operator_review_required"] is True
        assert card["future_wing_readback_required"] is True
        assert card["hardware_proof_required_before_live"] is True
        assert card["blocks_live_promotion"] is True
        assert card["blocks_queue_import"] is True
        assert card["live_authority_granted"] is False
        assert card["full_autonomous_ready"] is False
        assert card["readiness_closure_allowed"] is False
        assert card["live_apply_allowed"] is False
        assert card["queue_import_allowed"] is False
        assert card["can_apply_live"] is False
        assert card["can_import_to_queue"] is False
        assert card["can_authorize_live"] is False
        assert card["can_close_readiness"] is False
        assert card["queue_action"] == "not_queueable_until_promoted"
        assert card["contacts_wing"] is False
        assert card["contacts_backend"] is False
        assert card["mutates_mixer"] is False
        assert card["denial_reason"]


def test_correction_method_goal_operator_review_markdown_is_readonly_and_actionable():
    packet = build_correction_method_goal_operator_review_packet()
    markdown = render_correction_method_goal_operator_review_markdown()
    first_card = packet["method_review_cards"][0]

    assert markdown.startswith(
        "# Automixer Correction Method Operator Review Packet\n"
    )
    assert markdown.endswith("\n")
    assert (
        "automixer.correction_method_goal_operator_review_packet.v1"
        in markdown
    )
    assert "operator_review_packet_ready_proposal_only" in markdown
    assert "proposal_layer_contracts_ready_operator_review_required" in markdown
    assert "- read_only_or_dry_run_only: `true`" in markdown
    assert "- operator_review_required: `true`" in markdown
    assert "- future_wing_readback_required: `true`" in markdown
    assert "- hardware_proof_required_before_live: `true`" in markdown
    assert "- blocks_live_promotion: `true`" in markdown
    assert "- blocks_queue_import: `true`" in markdown
    assert "- live_authority_granted: `false`" in markdown
    assert "- full_autonomous_ready: `false`" in markdown
    assert "- readiness_closure_allowed: `false`" in markdown
    assert "- live_apply_allowed: `false`" in markdown
    assert "- queue_import_allowed: `false`" in markdown
    assert "- can_apply_live: `false`" in markdown
    assert "- can_import_to_queue: `false`" in markdown
    assert "- can_authorize_live: `false`" in markdown
    assert "- can_close_readiness: `false`" in markdown
    assert "- queue_action: `not_queueable_until_promoted`" in markdown
    assert "- contacts_wing: `false`" in markdown
    assert "- contacts_backend: `false`" in markdown
    assert "- mutates_mixer: `false`" in markdown
    assert "operator_review_packet_cannot_replace_real_wing_proof" in markdown
    assert f"- priority: `{first_card['priority']}`" in markdown
    assert f"- priority_label: `{first_card['priority_label']}`" in markdown
    assert (
        f"- priority_review_scope: `{first_card['priority_review_scope']}`"
        in markdown
    )
    expected_problem_review_line = (
        "- operator_problem_review_summary: "
        + ", ".join(
            f"`{item}`" for item in first_card["operator_problem_review_summary"]
        )
    )
    assert expected_problem_review_line in markdown
    expected_method_focus_line = (
        "- operator_method_review_focus: "
        + ", ".join(
            f"`{item}`" for item in first_card["operator_method_review_focus"]
        )
    )
    assert expected_method_focus_line in markdown
    assert "## First Priority Review" in markdown
    assert "- label: `Gain / Trim and Fader / Balance`" in markdown
    assert "- first_priority_method_count:" in markdown
    assert "- first_priority_method_ids:" in markdown
    assert "- first_priority_applicable_safety_checks:" in markdown
    assert "## Second Priority Review" in markdown
    assert "- label: `EQ and Compressor`" in markdown
    assert "- second_priority_method_count:" in markdown
    assert "- second_priority_method_ids:" in markdown
    assert "- second_priority_applicable_safety_checks:" in markdown
    assert "`manual_band`" in markdown
    assert "`gain_reduction`" in markdown
    assert "`eq_band_write`" in markdown
    assert "`compressor_write`" in markdown
    assert "## Third Priority Review" in markdown
    assert "- label: `Gate / Expander / De-esser`" in markdown
    assert "- third_priority_method_count:" in markdown
    assert "- third_priority_method_ids:" in markdown
    assert "- third_priority_applicable_safety_checks:" in markdown
    assert "`chatter_guard`" in markdown
    assert "`hysteresis`" in markdown
    assert "`tails_and_attack`" in markdown
    assert "`gate_write`" in markdown
    assert "`deesser_write`" in markdown
    assert "## Fourth Priority Review" in markdown
    assert "- label: `FX / Sends`" in markdown
    assert "- fourth_priority_method_count:" in markdown
    assert "- fourth_priority_method_ids:" in markdown
    assert "- fourth_priority_applicable_safety_checks:" in markdown
    assert "`return_level`" in markdown
    assert "`monitor_safety`" in markdown
    assert "`intelligibility`" in markdown
    assert "`fx_send_write`" in markdown
    assert "`fx_parameter_write`" in markdown
    assert "## Fifth Priority Review" in markdown
    assert "- label: `Pan / Stereo`" in markdown
    assert "- fifth_priority_method_count:" in markdown
    assert "- fifth_priority_method_ids:" in markdown
    assert "- fifth_priority_applicable_safety_checks:" in markdown
    assert "`mono_compatibility`" in markdown
    assert "`phase`" in markdown
    assert "`source_identity`" in markdown
    assert "`pan_write`" in markdown
    assert "`stereo_width_write`" in markdown
    assert "## Sixth Priority Review" in markdown
    assert "- label: `Routing / Groups / Scenes`" in markdown
    assert "- sixth_priority_method_count:" in markdown
    assert "- sixth_priority_method_ids:" in markdown
    assert "- sixth_priority_applicable_safety_checks:" in markdown
    assert "`read_only_map`" in markdown
    assert "`read_only_inventory`" in markdown
    assert "`destructive_scope`" in markdown
    assert "`routing_write`" in markdown
    assert "`scene_load`" in markdown
    assert "`mute_write`" in markdown
    assert "## Seventh Priority Review" in markdown
    assert "- label: `Master / Output`" in markdown
    assert "- seventh_priority_method_count:" in markdown
    assert "- seventh_priority_method_ids:" in markdown
    assert "- seventh_priority_applicable_safety_checks:" in markdown
    assert "`true_peak`" in markdown
    assert "`pa_feed_safety`" in markdown
    assert "`limiter_or_dynamics_proof`" in markdown
    assert "`reference_feed_exclusion`" in markdown
    assert "`master_bus_write`" in markdown
    assert "`limiter_write`" in markdown
    assert "`matrix_pa_feed_write`" in markdown
    assert "review_priority_group_offline_methods_and_collect_real_wing_readback_before_live_promotion" in markdown

    for heading in (
        "### 1. Gain / Trim and Fader / Balance",
        "### 2. EQ and Compressor",
        "### 3. Gate / Expander / De-esser",
        "### 4. FX / Sends",
        "### 5. Pan / Stereo",
        "### 6. Routing / Groups / Scenes",
        "### 7. Master / Output",
    ):
        assert heading in markdown

    for method_id in (
        "lufs_gain_staging",
        "auto_fader_v1",
        "auto_eq",
        "auto_compressor",
        "auto_gate_caig",
        "context_aware_gate_agent",
        "deesser_vocal_sibilance",
        "auto_fx",
        "fx_direct_apply",
        "fx_send_proposal",
        "auto_panner",
        "pan_spatial_proposal",
        "mute_management_proposal",
        "routing_management_proposal",
        "routing_snapshot_reset_writes",
        "scene_preset_management_proposal",
        "master_bus_optimization_proposal",
    ):
        assert f"`{method_id}`" in markdown

    for review_item in (
        "source_evidence_matches_current_offline_material",
        "measured_or_audible_problem_is_stated",
        "future_wing_readback_targets_are_known_before_live_promotion",
        "live_apply_and_queue_import_remain_blocked",
        "source_evidence",
        "operator_readable_explanation",
        "future_wing_readback_contract",
        "test_evidence",
        "live_authority_boundary",
    ):
        assert review_item in markdown

    for contract_line in (
        "- source_evidence:",
        "- source_evidence_details:",
        "- measured_or_audible_problem:",
        "- proposed_correction:",
        "- confidence:",
        "- bounded_delta_contract:",
        "- safety_blockers:",
        "- headroom_feedback_true_peak_checks:",
        "- operator_readable_explanation:",
        "- rollback_readback_contract:",
        "- future_wing_readback_contract:",
        "- test_evidence:",
        "- test_evidence_references:",
    ):
        assert contract_line in markdown

    assert "required_from_offline_proposal_payload" in markdown
    assert "confirm_problem_statement_matches_source_evidence" in markdown
    assert "confirm_proposed_correction_matches_the_problem" in markdown
    assert "review_confidence_and_confidence_blockers" in markdown
    assert "review_safety_blockers_before_acceptance" in markdown

    assert REQUIRED_FUTURE_PROOFS == FULL_FUTURE_LIVE_REQUIRED_PROOFS
    for future_live_proof in FULL_FUTURE_LIVE_REQUIRED_PROOFS:
        assert future_live_proof in markdown

    for forbidden in (
        "live_authority_granted: `true`",
        "live_apply_allowed: `true`",
        "queue_import_allowed: `true`",
        "contacts_wing: `true`",
        "contacts_backend: `true`",
        "mutates_mixer: `true`",
    ):
        assert forbidden not in markdown


def test_correction_method_goal_operator_review_export_writes_readonly_artifacts(tmp_path):
    run_id = "20260622T120000Z_correction_method_operator_review"
    manifest = export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T12:00:00Z",
    )

    run_dir = tmp_path / run_id
    packet_path = run_dir / "packet.json"
    markdown_path = run_dir / "operator-review.md"
    decision_template_path = run_dir / "operator-decision-template.json"
    goal_audit_path = run_dir / "goal-audit.json"
    goal_audit_markdown_path = run_dir / "goal-audit.md"
    manifest_path = run_dir / "artifact-manifest.json"

    assert packet_path.exists()
    assert markdown_path.exists()
    assert decision_template_path.exists()
    assert goal_audit_path.exists()
    assert goal_audit_markdown_path.exists()
    assert manifest_path.exists()

    disk_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    first_card = packet["method_review_cards"][0]
    markdown = markdown_path.read_text(encoding="utf-8")
    decision_template = json.loads(
        decision_template_path.read_text(encoding="utf-8")
    )
    goal_audit = json.loads(goal_audit_path.read_text(encoding="utf-8"))
    goal_audit_markdown = goal_audit_markdown_path.read_text(encoding="utf-8")

    assert manifest == disk_manifest
    assert manifest["schema"] == (
        "automixer.correction_method_operator_review_export_manifest.v1"
    )
    assert manifest["status"] == "operator_review_export_ready_proposal_only"
    assert manifest["packet_schema"] == (
        "automixer.correction_method_goal_operator_review_packet.v1"
    )
    assert manifest["packet_status"] == (
        "operator_review_packet_ready_proposal_only"
    )
    assert manifest["goal_audit_schema"] == (
        "automixer.correction_method_goal_audit.v1"
    )
    assert manifest["goal_audit_status"] == "passed"
    assert manifest["goal_audit_failed_check_count"] == 0
    assert manifest["goal_audit_summary"]["overall_offline_goal_percent"] == 100.0
    assert (
        manifest["goal_audit_summary"]["method_review_card_count"]
        == EXPECTED_METHOD_REVIEW_CARD_COUNT
    )
    assert (
        manifest["goal_audit_summary"]["expected_method_count"]
        == EXPECTED_METHOD_REVIEW_CARD_COUNT
    )
    assert manifest["goal_audit_summary"]["priority_section_count"] == 7
    assert manifest["goal_audit_summary"]["missing_required_objective_families"] == []
    assert manifest["goal_audit_summary"]["missing_expected_objective_families"] == []
    assert manifest["goal_audit_summary"]["no_live_authority"] is True
    assert manifest["goal_audit_summary"]["contacts_wing"] is False
    assert manifest["goal_audit_summary"]["live_authority_granted"] is False
    assert manifest["goal_audit_summary"]["readiness_closure_allowed"] is False
    assert manifest["goal_audit_summary"]["live_apply_allowed"] is False
    assert manifest["goal_audit_summary"]["queue_import_allowed"] is False
    assert manifest["operator_decision_template_schema"] == (
        "automixer.correction_method_operator_decision_template.v1"
    )
    assert manifest["operator_decision_template_status"] == (
        "operator_review_template_pending_proposal_only"
    )
    assert manifest["read_only_or_dry_run_only"] is True
    assert manifest["read_only"] is True
    assert manifest["contacts_wing"] is False
    assert manifest["contacts_backend"] is False
    assert manifest["queries_console"] is False
    assert manifest["mutates_mixer"] is False
    assert manifest["mutates_console_state"] is False
    assert manifest["live_write"] is False
    assert manifest["live_authority_granted"] is False
    assert manifest["readiness_closure_allowed"] is False
    assert manifest["live_apply_allowed"] is False
    assert manifest["queue_import_allowed"] is False
    assert manifest["hardware_proof_required_before_live"] is True
    assert manifest["operator_review_required"] is True
    assert manifest["future_wing_readback_required"] is True
    assert manifest["denial_reason"] == (
        "operator_review_packet_cannot_replace_real_wing_proof"
    )
    assert manifest["summary"]["priority_section_count"] == 7
    assert manifest["summary"]["live_authority_method_ids"] == []
    assert manifest["summary"]["live_apply_method_ids"] == []
    assert (
        manifest["summary"]["operator_review_contract_coverage"]
        == packet["summary"]["operator_review_contract_coverage"]
    )
    assert manifest["summary"]["queueable_method_ids"] == []
    decision_template_summary = manifest["operator_decision_template_summary"]
    assert (
        decision_template_summary["method_decision_count"]
        == EXPECTED_METHOD_REVIEW_CARD_COUNT
    )
    assert decision_template_summary["priority_decision_section_count"] == 7
    assert decision_template_summary["priority_decision_labels"] == [
        "Gain / Trim and Fader / Balance",
        "EQ and Compressor",
        "Gate / Expander / De-esser",
        "FX / Sends",
        "Pan / Stereo",
        "Routing / Groups / Scenes",
        "Master / Output",
    ]
    assert decision_template_summary["priority_method_decision_counts"] == {
        "1": 6,
        "2": 9,
        "3": 3,
        "4": 3,
        "5": 2,
        "6": 4,
        "7": 1,
    }
    assert decision_template_summary[
        "priority_operator_payload_required_field_counts"
    ]["1"] == 47
    assert decision_template_summary[
        "priority_operator_evidence_required_field_counts"
    ]["1"] == 38
    assert decision_template_summary[
        "priority_operator_problem_review_summary_counts"
    ]["1"] == len(REQUIRED_OPERATOR_PROBLEM_REVIEW_SUMMARY)
    assert decision_template_summary[
        "priority_operator_method_review_focus_counts"
    ]["1"] > len(REQUIRED_OPERATOR_METHOD_REVIEW_FOCUS_KEYS)
    assert decision_template_summary["priority_why_safe_summary_counts"]["1"] > (
        len(REQUIRED_OPERATOR_WHY_SAFE_SUMMARY_KEYS)
    )
    assert decision_template_summary["priority_why_blocked_summary_counts"]["1"] > (
        len(REQUIRED_OPERATOR_WHY_BLOCKED_SUMMARY_KEYS)
    )
    assert decision_template_summary[
        "priority_future_wing_readback_contract_summary_counts"
    ]["1"] == 23
    assert decision_template_summary["priority_test_evidence_counts"]["1"] == 4
    assert decision_template_summary[
        "priority_live_authority_boundary_summary_counts"
    ]["1"] == 7
    assert decision_template_summary[
        "priority_operator_review_record_required_field_counts"
    ]["1"] == len(FULL_OPERATOR_REVIEW_RECORD_FIELDS)
    assert decision_template_summary[
        "priority_operator_review_checklist_detail_counts"
    ]["1"] == 7
    assert decision_template_summary[
        "priority_operator_review_record_field_detail_counts"
    ]["1"] == len(FULL_OPERATOR_REVIEW_RECORD_FIELD_DETAILS)
    assert decision_template_summary[
        "priority_operator_acceptance_gate_detail_counts"
    ]["1"] == len(packet["priority_sections"][0]["operator_acceptance_gate_details"])
    assert decision_template_summary[
        "priority_goal_requirement_detail_counts"
    ]["1"] == len(packet["priority_sections"][0]["goal_requirement_details"])
    assert decision_template_summary[
        "priority_method_specific_review_detail_counts"
    ]["1"] == len(packet["priority_sections"][0]["method_specific_review_details"])
    assert decision_template_summary[
        "priority_operator_actionable_explanation_detail_counts"
    ]["1"] == len(
        packet["priority_sections"][0]["operator_actionable_explanation_details"]
    )
    assert decision_template_summary[
        "priority_operator_next_action_detail_counts"
    ]["1"] == len(packet["priority_sections"][0]["operator_next_action_details"])
    assert (
        "explicit_offline_decision_per_priority_section"
        in decision_template_summary["operator_completion_requires"]
    )
    assert decision_template_summary["read_only_or_dry_run_only"] is True
    assert decision_template_summary["read_only"] is True
    assert decision_template_summary["contacts_wing"] is False
    assert decision_template_summary["contacts_backend"] is False
    assert decision_template_summary["queries_console"] is False
    assert decision_template_summary["mutates_mixer"] is False
    assert decision_template_summary["mutates_console_state"] is False
    assert decision_template_summary["live_write"] is False
    assert decision_template_summary["live_authority_granted"] is False
    assert decision_template_summary["readiness_closure_allowed"] is False
    assert decision_template_summary["live_apply_allowed"] is False
    assert decision_template_summary["queue_import_allowed"] is False
    assert decision_template_summary["can_authorize_live"] is False
    assert decision_template_summary["hardware_proof_required_before_live"] is True
    assert decision_template_summary["operator_review_required"] is True
    assert decision_template_summary["future_wing_readback_required"] is True

    artifacts = {
        artifact["artifact_id"]: artifact
        for artifact in manifest["artifacts"]
    }
    assert set(artifacts) == {
        "packet_json",
        "operator_review_markdown",
        "operator_decision_template",
        "goal_audit_json",
        "goal_audit_markdown",
    }
    assert artifacts["packet_json"]["path"] == "packet.json"
    assert artifacts["operator_review_markdown"]["path"] == "operator-review.md"
    assert artifacts["operator_decision_template"]["path"] == (
        "operator-decision-template.json"
    )
    assert artifacts["goal_audit_json"]["path"] == "goal-audit.json"
    assert artifacts["goal_audit_markdown"]["path"] == "goal-audit.md"
    assert artifacts["packet_json"]["sha256"] == hashlib.sha256(
        packet_path.read_bytes()
    ).hexdigest()
    assert artifacts["operator_review_markdown"]["sha256"] == hashlib.sha256(
        markdown_path.read_bytes()
    ).hexdigest()
    assert artifacts["operator_decision_template"]["sha256"] == hashlib.sha256(
        decision_template_path.read_bytes()
    ).hexdigest()
    assert artifacts["goal_audit_json"]["sha256"] == hashlib.sha256(
        goal_audit_path.read_bytes()
    ).hexdigest()
    assert artifacts["goal_audit_markdown"]["sha256"] == hashlib.sha256(
        goal_audit_markdown_path.read_bytes()
    ).hexdigest()
    assert goal_audit["status"] == "passed"
    assert goal_audit["failed_checks"] == []
    assert goal_audit["overall_offline_goal_percent"] == 100.0
    assert goal_audit["method_review_card_count"] == len(
        packet["method_review_cards"]
    )
    assert goal_audit["priority_section_count"] == len(packet["priority_sections"])
    assert goal_audit["contacts_wing"] is False
    assert goal_audit["live_authority_granted"] is False
    assert goal_audit["readiness_closure_allowed"] is False
    assert "- status: `passed`" in goal_audit_markdown
    assert "- live_authority_granted: `false`" in goal_audit_markdown
    assert "- overall_offline_goal_percent: `100.0`" in goal_audit_markdown

    assert packet["status"] == "operator_review_packet_ready_proposal_only"
    assert packet["live_authority_granted"] is False
    assert packet["live_apply_allowed"] is False
    assert packet["queue_import_allowed"] is False
    assert packet["contacts_wing"] is False
    assert packet["contacts_backend"] is False
    assert packet["mutates_mixer"] is False
    for card in packet["method_review_cards"]:
        assert (
            tuple(card["future_live_required_proofs"])
            == FULL_FUTURE_LIVE_REQUIRED_PROOFS
        )
    packet_cards_by_method_id = {
        card["method_id"]: card for card in packet["method_review_cards"]
    }
    for row in decision_template["method_decisions"]:
        packet_card = packet_cards_by_method_id[row["method_id"]]
        assert row["priority"] == packet_card["priority"]
        assert row["priority_label"] == packet_card["priority_label"]
        assert row["priority_review_scope"] == packet_card["priority_review_scope"]
        assert row["operator_problem_review_summary"] == packet_card[
            "operator_problem_review_summary"
        ]
        assert row["operator_method_review_focus"] == packet_card[
            "operator_method_review_focus"
        ]
        assert row["why_safe_summary"] == packet_card["why_safe_summary"]
        assert row["why_blocked_summary"] == packet_card["why_blocked_summary"]
        assert row["goal_requirement_details"] == packet_card[
            "goal_requirement_details"
        ]
        assert row["method_specific_review_details"] == packet_card[
            "method_specific_review_details"
        ]
        assert row["operator_actionable_explanation_details"] == packet_card[
            "operator_actionable_explanation_details"
        ]
        assert row["evidence_problem_correction_chain_details"] == packet_card[
            "evidence_problem_correction_chain_details"
        ]
        assert row["required_confirmation_details"] == packet_card[
            "required_confirmation_details"
        ]
        assert row["operator_acceptance_gate_details"] == packet_card[
            "operator_acceptance_gate_details"
        ]
        assert row["operator_review_decision_option_details"] == packet_card[
            "operator_review_decision_option_details"
        ]
        assert row["operator_review_checklist_details"] == packet_card[
            "operator_review_checklist_details"
        ]
        assert row["operator_next_action_details"] == packet_card[
            "operator_next_action_details"
        ]
        assert row["operator_visible_explanation_contract"] == packet_card[
            "operator_visible_explanation_contract"
        ]
        assert (
            tuple(row["future_live_required_proofs"])
            == FULL_FUTURE_LIVE_REQUIRED_PROOFS
        )
        assert row["future_live_requirement_details"] == packet_card[
            "future_live_requirement_details"
        ]
        assert row["safety_blocker_details"] == packet_card[
            "safety_blocker_details"
        ]
        assert row["headroom_feedback_true_peak_check_details"] == packet_card[
            "headroom_feedback_true_peak_check_details"
        ]
    packet_priority_sections = {
        section["priority"]: section
        for section in packet["priority_sections"]
    }
    assert packet_priority_sections[1]["operator_payload_required_fields"]
    assert len(packet_priority_sections[1]["operator_payload_required_fields"]) == 47
    assert len(packet_priority_sections[1]["operator_evidence_required_fields"]) == 38
    assert sorted(packet_priority_sections[1]["operator_problem_review_summary"]) == (
        sorted(REQUIRED_OPERATOR_PROBLEM_REVIEW_SUMMARY)
    )
    assert (
        len(packet_priority_sections[1]["operator_method_review_focus"])
        == decision_template_summary["priority_operator_method_review_focus_counts"]["1"]
    )
    assert (
        len(packet_priority_sections[1]["operator_visible_explanation_contract"])
        == decision_template_summary[
            "priority_operator_visible_explanation_contract_counts"
        ]["1"]
    )
    assert any(
        item.startswith("method_id=lufs_gain_staging")
        for item in packet_priority_sections[1]["operator_method_review_focus"]
    )
    assert len(packet_priority_sections[1]["operator_review_checklist_details"]) == 7
    assert (
        len(packet_priority_sections[1]["required_confirmation_details"])
        == len(REQUIRED_METHOD_CARD_CONFIRMATIONS)
    )
    assert len(packet_priority_sections[1]["operator_review_decision_option_details"]) == (
        len(FULL_OFFLINE_REVIEW_DECISIONS)
    )
    assert any(
        item.startswith(
            "live_apply_and_queue_import_remain_blocked:"
            "source=live_authority_boundary"
        )
        for item in packet_priority_sections[1]["required_confirmation_details"]
    )
    assert any(
        item.startswith(
            "request_more_evidence:"
            "effect=keeps_proposal_blocked_until_required_evidence_complete"
        )
        for item in packet_priority_sections[1][
            "operator_review_decision_option_details"
        ]
    )
    assert any(
        item.startswith("verify_source_evidence_matches_current_offline_material:")
        for item in packet_priority_sections[1]["operator_review_checklist_details"]
    )
    assert (
        "operator_explanation_required=true"
        in packet_priority_sections[1]["operator_visible_explanation_contract"]
    )
    assert len(
        packet_priority_sections[1]["future_wing_readback_contract_summary"]
    ) == 23
    assert len(packet_priority_sections[1]["live_authority_boundary_summary"]) == 7
    assert len(
        packet_priority_sections[1]["operator_review_record_required_fields"]
    ) == len(FULL_OPERATOR_REVIEW_RECORD_FIELDS)
    assert len(
        packet_priority_sections[1]["operator_review_record_field_details"]
    ) == len(FULL_OPERATOR_REVIEW_RECORD_FIELD_DETAILS)
    assert packet_priority_sections[1]["operator_next_action_details"]
    assert any(
        item.startswith("review_operator_evidence_packet:source=operator_evidence_packet")
        for item in packet_priority_sections[1]["operator_next_action_details"]
    )
    assert packet_priority_sections[1]["future_live_requirement_details"]
    assert any(
        item.startswith(
            "real_wing_readonly_connection_artifact:"
            "source=future_live_promotion_requirements"
        )
        for item in packet_priority_sections[1]["future_live_requirement_details"]
    )
    assert packet_priority_sections[1]["safety_blocker_details"]
    assert packet_priority_sections[1]["goal_requirement_details"]
    assert packet_priority_sections[1]["method_specific_review_details"]
    assert packet_priority_sections[1]["operator_actionable_explanation_details"]
    assert any(
        item.startswith(
            "source_evidence:source=correction_method_goal_requirement"
        )
        for item in packet_priority_sections[1]["goal_requirement_details"]
    )
    assert any(
        "future_wing_readback_contract" in item
        and "coverage_status=covered" in item
        for item in packet_priority_sections[1]["goal_requirement_details"]
    )
    assert any(
        item.startswith(
            "method_problem_focus:"
            "source=correction_method_goal.method_specific_review"
        )
        for item in packet_priority_sections[1]["method_specific_review_details"]
    )
    assert any(
        "focus=safe_gain_down_or_trim" in item
        or "focus=channel_group_vocal_instrument_balance" in item
        for item in packet_priority_sections[1]["method_specific_review_details"]
    )
    assert any(
        item.startswith("what_is_wrong:source=operator_actionable_explanation")
        for item in packet_priority_sections[1][
            "operator_actionable_explanation_details"
        ]
    )
    assert any(
        "what_is_needed_before_live" in item
        and "requires_real_wing_proof_before_live=true" in item
        for item in packet_priority_sections[1][
            "operator_actionable_explanation_details"
        ]
    )
    assert packet_priority_sections[1]["evidence_problem_correction_chain_details"]
    assert any(
        item.startswith(
            "source_evidence:source=operator_review_contract.source_evidence"
        )
        for item in packet_priority_sections[1][
            "evidence_problem_correction_chain_details"
        ]
    )
    assert any(
        "chain_step=4_proposed_correction" in item
        for item in packet_priority_sections[1][
            "evidence_problem_correction_chain_details"
        ]
    )
    assert any(
        item.startswith("headroom:source=required_safety_checks")
        for item in packet_priority_sections[1]["safety_blocker_details"]
    )
    assert any(
        item.startswith("input_trim_write:source=live_authority_boundary")
        for item in packet_priority_sections[1]["safety_blocker_details"]
    )
    assert packet_priority_sections[1]["headroom_feedback_true_peak_check_details"]
    assert any(
        item.startswith(
            "headroom:"
            "source=headroom_feedback_true_peak_checks.required_checks"
        )
        for item in packet_priority_sections[1][
            "headroom_feedback_true_peak_check_details"
        ]
    )
    assert any(
        "check_type=true_peak" in item
        for item in packet_priority_sections[1][
            "headroom_feedback_true_peak_check_details"
        ]
    )
    for section in packet_priority_sections.values():
        assert section["method_count"] == len(section["methods"])
        assert section["review_status"] == "operator_review_required"
        assert section["required_evidence"]
        assert section["required_safety_checks"]
        assert set(section["required_safety_checks"]) <= set(
            ALLOWED_REQUIRED_SAFETY_CHECKS
        )
        assert len(section["required_safety_checks"]) == len(
            set(section["required_safety_checks"])
        )
        assert section["test_evidence_references"]
        for field in OPERATOR_REVIEW_CONTRACT_FIELDS:
            assert section[field]
        assert section["operator_payload_required_fields"]
        assert section["operator_evidence_required_fields"]
        assert sorted(section["operator_problem_review_summary"]) == (
            sorted(REQUIRED_OPERATOR_PROBLEM_REVIEW_SUMMARY)
        )
        assert section["operator_method_review_focus"]
        assert any(
            item.startswith("live_boundary=offline_only_no_queue_no_live_apply")
            for item in section["operator_method_review_focus"]
        )
        assert section["why_safe_summary"]
        assert {
            item.split("=", 1)[0] for item in section["why_safe_summary"]
        } == set(FULL_OPERATOR_WHY_SAFE_SUMMARY_KEYS)
        assert "read_only_or_dry_run_only=true" in section["why_safe_summary"]
        assert "live_apply_allowed=false" in section["why_safe_summary"]
        assert "queue_import_allowed=false" in section["why_safe_summary"]
        assert section["why_blocked_summary"]
        assert {
            item.split("=", 1)[0] for item in section["why_blocked_summary"]
        } == set(FULL_OPERATOR_WHY_BLOCKED_SUMMARY_KEYS)
        assert (
            "future_wing_readback_required=true"
            in section["why_blocked_summary"]
        )
        assert (
            "hardware_proof_required_before_live=true"
            in section["why_blocked_summary"]
        )
        assert section["operator_visible_explanation_contract"]
        assert {
            item.split("=", 1)[0]
            for item in section["operator_visible_explanation_contract"]
        } == set(REQUIRED_OPERATOR_VISIBLE_CONTRACT_FLAGS)
        assert section["bounded_delta_policies"]
        assert set(section["bounded_delta_policies"]) <= set(
            ALLOWED_BOUNDED_DELTA_POLICIES
        )
        assert len(section["bounded_delta_policies"]) == len(
            set(section["bounded_delta_policies"])
        )
        assert section["rollback_expectations"]
        assert set(section["rollback_expectations"]) <= set(
            ALLOWED_ROLLBACK_EXPECTATIONS
        )
        assert len(section["rollback_expectations"]) == len(
            set(section["rollback_expectations"])
        )
        assert section["future_wing_readback_targets"]
        assert set(section["future_wing_readback_targets"]) <= set(
            ALLOWED_FUTURE_WING_READBACK_TARGETS
        )
        assert len(section["future_wing_readback_targets"]) == len(
            set(section["future_wing_readback_targets"])
        )
        assert section["future_wing_readback_contract_summary"]
        section_readback_summary_keys = {
            item.split("=", 1)[0]
            for item in section["future_wing_readback_contract_summary"]
        }
        assert set(FULL_FUTURE_WING_READBACK_CONTRACT_SUMMARY_KEYS) <= (
            section_readback_summary_keys
        )
        assert (
            "operator_next_step="
            f"{FULL_FUTURE_WING_READBACK_OPERATOR_NEXT_STEP}"
        ) in section["future_wing_readback_contract_summary"]
        assert (
            "denial_reason="
            f"{FULL_FUTURE_WING_READBACK_DENIAL_REASON}"
        ) in section["future_wing_readback_contract_summary"]
        assert tuple(section["future_live_required_proofs"]) == tuple(
            sorted(FULL_FUTURE_LIVE_REQUIRED_PROOFS)
        )
        assert section["blocked_live_actions"]
        assert tuple(section["live_authority_boundary_summary"]) == (
            FULL_LIVE_AUTHORITY_BOUNDARY_SUMMARY
        )
        assert section["operator_review_record_required_fields"]
        assert section["missing_required_operator_surface_method_ids"] == []
        assert section["live_authority_method_ids"] == []
        assert section["live_apply_method_ids"] == []
        assert section["queueable_method_ids"] == []
        assert section["operator_review_required"] is True
        assert section["future_wing_readback_required"] is True
        assert section["hardware_proof_required_before_live"] is True
        assert section["contacts_wing"] is False
        assert section["contacts_backend"] is False
        assert section["mutates_mixer"] is False
        assert section["live_authority_granted"] is False
        assert section["readiness_closure_allowed"] is False
        assert section["live_apply_allowed"] is False
        assert section["queue_import_allowed"] is False
        for field in (
            "required_evidence",
            "required_safety_checks",
            "test_evidence_references",
            "operator_payload_required_fields",
            "operator_evidence_required_fields",
            "why_safe_summary",
            "why_blocked_summary",
            "future_wing_readback_targets",
            "future_wing_readback_contract_summary",
            "future_live_required_proofs",
            "blocked_live_actions",
            "live_authority_boundary_summary",
            "required_confirmation_details",
            "operator_acceptance_gate_details",
            "operator_review_decision_option_details",
            "operator_review_checklist_details",
            "operator_review_record_required_fields",
            "operator_review_record_field_details",
            "operator_next_action_details",
        ):
            expected_items = sorted(
                {
                    str(item)
                    for card in section["methods"]
                    for item in card[field]
                }
            )
            assert sorted(section[field]) == expected_items
        for field in OPERATOR_REVIEW_CONTRACT_FIELDS:
            expected_items = sorted(
                {
                    str(item)
                    for card in section["methods"]
                    for item in _operator_review_contract_items(card[field])
                }
            )
            assert sorted(section[field]) == expected_items
        expected_explanation_contract_items = sorted(
            {
                f"{key}={str(value).lower()}"
                for card in section["methods"]
                for key, value in card["operator_visible_explanation_contract"].items()
            }
        )
        assert sorted(section["operator_visible_explanation_contract"]) == (
            expected_explanation_contract_items
        )
        assert section["bounded_delta_policies"] == sorted(
            {
                str(card["bounded_delta_policy"])
                for card in section["methods"]
            }
        )
        assert section["rollback_expectations"] == sorted(
            {
                str(card["rollback_expectation"])
                for card in section["methods"]
            }
        )
    assert "# Automixer Correction Method Operator Review Packet" in markdown
    assert "## Operator Decision Workflow" in markdown
    assert "`explicit_offline_decision_per_priority_section`" in markdown
    assert "`explicit_offline_decision_per_method`" in markdown
    assert "`future_real_wing_readback_before_live_promotion`" in markdown
    assert "- workflow_scope: `offline_proposal_only`" in markdown
    assert (
        "- required_priority_operator_decision: "
        "`one_offline_decision_per_priority_section`"
    ) in markdown
    assert "- can_authorize_live: `false`" in markdown
    assert "- future_wing_readback_required: `true`" in markdown
    assert "- hardware_proof_required_before_live: `true`" in markdown
    assert (
        "### Decision Section 1: Gain / Trim and Fader / Balance" in markdown
    )
    assert "### Decision Section 2: EQ and Compressor" in markdown
    assert "### Decision Section 7: Master / Output" in markdown
    first_decision_start = markdown.index(
        "### Decision Section 1: Gain / Trim and Fader / Balance"
    )
    first_decision_end = markdown.index(
        "\n### Decision Section 2: EQ and Compressor",
        first_decision_start,
    )
    first_decision_block = markdown[first_decision_start:first_decision_end]
    assert "- required_evidence:" in first_decision_block
    assert "- required_safety_checks:" in first_decision_block
    assert "- source_evidence:" in first_decision_block
    assert "- source_evidence_details:" in first_decision_block
    assert "- measured_or_audible_problem:" in first_decision_block
    assert "- proposed_correction:" in first_decision_block
    assert "- goal_requirement_details:" in first_decision_block
    assert "- method_specific_review_details:" in first_decision_block
    assert "- operator_actionable_explanation_details:" in first_decision_block
    assert "- evidence_problem_correction_chain_details:" in first_decision_block
    assert "- confidence:" in first_decision_block
    assert "- bounded_delta_contract:" in first_decision_block
    assert "- safety_blockers:" in first_decision_block
    assert "- safety_blocker_details:" in first_decision_block
    assert "- headroom_feedback_true_peak_checks:" in first_decision_block
    assert "- headroom_feedback_true_peak_check_details:" in first_decision_block
    assert "- operator_readable_explanation:" in first_decision_block
    assert "- operator_payload_required_fields:" in first_decision_block
    assert "- operator_evidence_required_fields:" in first_decision_block
    assert "- bounded_delta_policies:" in first_decision_block
    assert "- rollback_expectations:" in first_decision_block
    assert "- rollback_readback_contract:" in first_decision_block
    assert "- future_wing_readback_contract_summary:" in first_decision_block
    assert "- future_wing_readback_contract:" in first_decision_block
    assert "- test_evidence:" in first_decision_block
    assert "- test_evidence_references:" in first_decision_block
    assert "- live_authority_boundary_summary:" in first_decision_block
    assert "- operator_review_record_required_fields:" in first_decision_block
    assert "- required_confirmation_details:" in first_decision_block
    assert "- operator_acceptance_gate_details:" in first_decision_block
    assert "- operator_review_decision_option_details:" in first_decision_block
    assert "- operator_next_action_details:" in first_decision_block
    assert "- future_live_requirement_details:" in first_decision_block
    assert "- missing_required_operator_surface_method_ids: `none`" in (
        first_decision_block
    )
    assert "- live_authority_method_ids: `none`" in first_decision_block
    assert "- live_apply_method_ids: `none`" in first_decision_block
    assert "- queueable_method_ids: `none`" in first_decision_block
    assert (
        "- operator_next_step: "
        "`review_priority_group_offline_methods_and_collect_real_wing_"
        "readback_before_live_promotion`"
    ) in first_decision_block
    assert "- contacts_wing: `false`" in first_decision_block
    assert "- contacts_backend: `false`" in first_decision_block
    assert "- mutates_mixer: `false`" in first_decision_block
    assert "`auto_eq`" in markdown
    assert "`master_bus_optimization_proposal`" in markdown
    assert "real_wing_readonly_connection_artifact" in markdown
    assert f"- operator_question: `{first_card['operator_question']}`" in markdown
    assert f"- denial_reason: `{first_card['denial_reason']}`" in markdown
    assert "- source_evidence:" in markdown
    assert "- source_evidence_details:" in markdown
    assert "- measured_or_audible_problem:" in markdown
    assert "- proposed_correction:" in markdown
    assert "- goal_requirement_details:" in markdown
    assert "- method_specific_review_details:" in markdown
    assert "- operator_actionable_explanation_details:" in markdown
    assert "- evidence_problem_correction_chain_details:" in markdown
    assert "- confidence:" in markdown
    assert "- bounded_delta_contract:" in markdown
    assert "- safety_blockers:" in markdown
    assert "- safety_blocker_details:" in markdown
    assert "- headroom_feedback_true_peak_checks:" in markdown
    assert "- headroom_feedback_true_peak_check_details:" in markdown
    assert "- operator_readable_explanation:" in markdown
    assert "- rollback_readback_contract:" in markdown
    assert "- test_evidence:" in markdown
    assert "- test_evidence_references:" in markdown
    assert "- required_operator_surfaces:" in markdown
    assert "- operator_payload_required_fields:" in markdown
    assert "- operator_evidence_required_fields:" in markdown
    assert "- must_confirm:" in markdown
    assert "- required_confirmation_details:" in markdown
    assert "- operator_acceptance_gate_details:" in markdown
    assert "- operator_review_decision_option_details:" in markdown
    assert "- operator_next_action_details:" in markdown
    assert "- future_live_requirement_details:" in markdown
    assert "- operator_review_checklist:" in markdown
    assert "- future_wing_readback_contract:" in markdown
    assert "- operator_review_record_required_fields:" in markdown
    assert "- operator_visible_explanation_contract:" in markdown
    assert "- future_wing_readback_contract_summary:" in markdown
    assert "- why_safe_summary:" in markdown
    assert "- why_blocked_summary:" in markdown
    assert "- live_authority_boundary_summary:" in markdown
    assert decision_template["schema"] == (
        "automixer.correction_method_operator_decision_template.v1"
    )
    assert decision_template["status"] == (
        "operator_review_template_pending_proposal_only"
    )
    assert (
        decision_template["method_decision_count"]
        == EXPECTED_METHOD_REVIEW_CARD_COUNT
    )
    assert decision_template["priority_decision_section_count"] == 7
    assert decision_template["read_only"] is True
    assert decision_template["contacts_wing"] is False
    assert decision_template["contacts_backend"] is False
    assert decision_template["live_authority_granted"] is False
    assert decision_template["readiness_closure_allowed"] is False
    assert decision_template["live_apply_allowed"] is False
    assert decision_template["queue_import_allowed"] is False
    assert decision_template["can_authorize_live"] is False
    assert tuple(decision_template["allowed_decisions"]) == (
        FULL_OFFLINE_REVIEW_DECISIONS
    )
    decision_rows = {
        row["method_id"]: row
        for row in decision_template["method_decisions"]
    }
    priority_decision_sections = {
        section["priority"]: section
        for section in decision_template["priority_decision_sections"]
    }
    assert [priority_decision_sections[index]["label"] for index in range(1, 8)] == [
        "Gain / Trim and Fader / Balance",
        "EQ and Compressor",
        "Gate / Expander / De-esser",
        "FX / Sends",
        "Pan / Stereo",
        "Routing / Groups / Scenes",
        "Master / Output",
    ]
    assert priority_decision_sections[1]["method_decision_count"] == 6
    assert priority_decision_sections[2]["method_decision_count"] == 9
    assert priority_decision_sections[7]["method_decision_count"] == 1
    for section in priority_decision_sections.values():
        assert section["review_scope"] == "offline_proposal_only"
        assert section["review_status"] == "operator_review_pending"
        assert section["operator_decision"] is None
        assert section["reviewer"] == ""
        assert section["reviewed_at_utc"] == ""
        assert section["rationale"] == ""
        assert section["evidence_packet_reference"] == ""
        assert section["source_material_reference"] == ""
        assert section["source_evidence_acknowledgement"] == ""
        assert section["test_evidence_reference"] == ""
        assert section["test_evidence_acknowledgement"] == ""
        assert section["verification_report_reference"] == ""
        assert section["verification_report_acknowledgement"] == ""
        assert section["measured_problem_acknowledgement"] == ""
        assert section["proposed_correction_acknowledgement"] == ""
        assert section["bounded_delta_acknowledgement"] == ""
        assert section["confidence_acknowledgement"] == ""
        assert section["operator_explanation_acknowledgement"] == ""
        assert section["future_wing_readback_acknowledgement"] == ""
        assert section["headroom_feedback_true_peak_acknowledgement"] == ""
        assert section["rollback_readback_acknowledgement"] == ""
        assert section["safety_blockers_acknowledgement"] == ""
        assert section["required_confirmations_acknowledgement"] == ""
        assert section["offline_acceptance_boundary_acknowledgement"] == ""
        assert section["operator_notes"] == ""
        assert section["expected_evidence_packet_reference"] == (
            f"packet.json#priority_sections/{section['priority']}"
        )
        assert section["expected_source_material_reference"] == (
            "packet.json#priority_sections/"
            f"{section['priority']}/source_evidence"
        )
        assert section["expected_test_evidence_references"] == (
            section["test_evidence_references"]
        )
        assert section["expected_verification_report_reference"] == (
            "verification-report.json"
        )
        assert section["required_priority_operator_decision"] == (
            "one_offline_decision_per_priority_section"
        )
        assert section["required_operator_decisions"] == (
            "one_offline_decision_per_method"
        )
        assert tuple(section["allowed_decisions"]) == FULL_OFFLINE_REVIEW_DECISIONS
        assert section["method_ids"]
        assert section["required_evidence"]
        assert section["required_safety_checks"]
        assert section["test_evidence_references"]
        assert section["bounded_delta_policies"]
        assert section["rollback_expectations"]
        assert section["operator_payload_required_fields"]
        assert section["operator_evidence_required_fields"]
        assert section["future_wing_readback_targets"]
        assert section["future_wing_readback_contract_summary"]
        assert section["future_live_required_proofs"]
        assert section["future_live_requirement_details"]
        assert section["safety_blocker_details"]
        assert section["headroom_feedback_true_peak_check_details"]
        assert section["blocked_live_actions"]
        assert section["why_safe_summary"]
        assert section["why_blocked_summary"]
        assert section["goal_requirement_details"]
        assert section["method_specific_review_details"]
        assert section["operator_actionable_explanation_details"]
        assert section["evidence_problem_correction_chain_details"]
        assert section["live_authority_boundary_summary"]
        assert section["required_confirmation_details"]
        assert section["operator_acceptance_gate_details"]
        assert section["operator_review_decision_option_details"]
        assert section["operator_review_checklist_details"]
        assert section["operator_review_record_required_fields"]
        assert section["operator_review_record_field_details"]
        assert section["operator_next_action_details"]
        assert section["operator_visible_explanation_contract"]
        section_rows = [
            decision_rows[method_id]
            for method_id in section["method_ids"]
        ]
        packet_section = packet_priority_sections[section["priority"]]
        assert section["required_evidence"] == packet_section["required_evidence"]
        assert section["required_safety_checks"] == packet_section[
            "required_safety_checks"
        ]
        assert section["test_evidence_references"] == packet_section[
            "test_evidence_references"
        ]
        for field in OPERATOR_REVIEW_CONTRACT_FIELDS:
            assert section[field] == packet_section[field]
        assert section["operator_visible_explanation_contract"] == (
            packet_section["operator_visible_explanation_contract"]
        )
        assert section["bounded_delta_policies"] == packet_section[
            "bounded_delta_policies"
        ]
        assert section["rollback_expectations"] == packet_section[
            "rollback_expectations"
        ]
        assert section["required_confirmation_details"] == packet_section[
            "required_confirmation_details"
        ]
        assert section["operator_acceptance_gate_details"] == packet_section[
            "operator_acceptance_gate_details"
        ]
        assert section["operator_review_decision_option_details"] == packet_section[
            "operator_review_decision_option_details"
        ]
        assert section["operator_review_checklist_details"] == packet_section[
            "operator_review_checklist_details"
        ]
        assert section["operator_review_record_field_details"] == packet_section[
            "operator_review_record_field_details"
        ]
        assert section["operator_next_action_details"] == packet_section[
            "operator_next_action_details"
        ]
        assert section["future_live_requirement_details"] == packet_section[
            "future_live_requirement_details"
        ]
        assert section["safety_blocker_details"] == packet_section[
            "safety_blocker_details"
        ]
        assert section["goal_requirement_details"] == (
            packet_section["goal_requirement_details"]
        )
        assert section["method_specific_review_details"] == (
            packet_section["method_specific_review_details"]
        )
        assert section["operator_actionable_explanation_details"] == (
            packet_section["operator_actionable_explanation_details"]
        )
        assert section["evidence_problem_correction_chain_details"] == (
            packet_section["evidence_problem_correction_chain_details"]
        )
        assert section["headroom_feedback_true_peak_check_details"] == (
            packet_section["headroom_feedback_true_peak_check_details"]
        )
        assert (
            decision_template_summary["priority_required_evidence_counts"][
                str(section["priority"])
            ]
            == len(section["required_evidence"])
        )
        assert (
            decision_template_summary["priority_required_safety_check_counts"][
                str(section["priority"])
            ]
            == len(section["required_safety_checks"])
        )
        assert (
            decision_template_summary["priority_test_evidence_reference_counts"][
                str(section["priority"])
            ]
            == len(section["test_evidence_references"])
        )
        contract_count_summary_keys = {
            "source_evidence": "priority_source_evidence_counts",
            "source_evidence_details": "priority_source_evidence_detail_counts",
            "measured_or_audible_problem": (
                "priority_measured_or_audible_problem_counts"
            ),
            "proposed_correction": "priority_proposed_correction_counts",
            "confidence": "priority_confidence_counts",
            "bounded_delta": "priority_bounded_delta_counts",
            "safety_blockers": "priority_safety_blocker_counts",
            "goal_requirement_details": (
                "priority_goal_requirement_detail_counts"
            ),
            "method_specific_review_details": (
                "priority_method_specific_review_detail_counts"
            ),
            "operator_actionable_explanation_details": (
                "priority_operator_actionable_explanation_detail_counts"
            ),
            "evidence_problem_correction_chain_details": (
                "priority_evidence_problem_correction_chain_detail_counts"
            ),
            "headroom_feedback_true_peak_checks": (
                "priority_headroom_feedback_true_peak_check_counts"
            ),
            "headroom_feedback_true_peak_check_details": (
                "priority_headroom_feedback_true_peak_check_detail_counts"
            ),
            "operator_readable_explanation": (
                "priority_operator_readable_explanation_counts"
            ),
            "rollback_readback_contract": (
                "priority_rollback_readback_contract_counts"
            ),
            "future_wing_readback_contract": (
                "priority_future_wing_readback_contract_counts"
            ),
            "test_evidence": "priority_test_evidence_counts",
        }
        for field, summary_key in contract_count_summary_keys.items():
            assert (
                decision_template_summary[summary_key][str(section["priority"])]
                == len(section[field])
            )
        assert (
            decision_template_summary[
                "priority_operator_visible_explanation_contract_counts"
            ][str(section["priority"])]
            == len(section["operator_visible_explanation_contract"])
        )
        assert (
            decision_template_summary["priority_why_safe_summary_counts"][
                str(section["priority"])
            ]
            == len(section["why_safe_summary"])
        )
        assert (
            decision_template_summary["priority_why_blocked_summary_counts"][
                str(section["priority"])
            ]
            == len(section["why_blocked_summary"])
        )
        assert (
            decision_template_summary["priority_bounded_delta_policy_counts"][
                str(section["priority"])
            ]
            == len(section["bounded_delta_policies"])
        )
        assert (
            decision_template_summary["priority_rollback_expectation_counts"][
                str(section["priority"])
            ]
            == len(section["rollback_expectations"])
        )
        assert (
            decision_template_summary[
                "priority_required_confirmation_detail_counts"
            ][str(section["priority"])]
            == len(section["required_confirmation_details"])
        )
        assert (
            decision_template_summary[
                "priority_operator_acceptance_gate_detail_counts"
            ][str(section["priority"])]
            == len(section["operator_acceptance_gate_details"])
        )
        assert (
            decision_template_summary[
                "priority_operator_review_decision_option_detail_counts"
            ][str(section["priority"])]
            == len(section["operator_review_decision_option_details"])
        )
        assert (
            decision_template_summary[
                "priority_operator_review_checklist_detail_counts"
            ][str(section["priority"])]
            == len(section["operator_review_checklist_details"])
        )
        assert (
            decision_template_summary[
                "priority_operator_review_record_field_detail_counts"
            ][str(section["priority"])]
            == len(section["operator_review_record_field_details"])
        )
        assert (
            decision_template_summary["priority_operator_next_action_detail_counts"][
                str(section["priority"])
            ]
            == len(section["operator_next_action_details"])
        )
        assert (
            decision_template_summary[
                "priority_future_live_requirement_detail_counts"
            ][str(section["priority"])]
            == len(section["future_live_requirement_details"])
        )
        assert (
            decision_template_summary[
                "priority_safety_blocker_detail_counts"
            ][str(section["priority"])]
            == len(section["safety_blocker_details"])
        )
        assert (
            decision_template_summary[
                "priority_goal_requirement_detail_counts"
            ][str(section["priority"])]
            == len(section["goal_requirement_details"])
        )
        assert (
            decision_template_summary[
                "priority_method_specific_review_detail_counts"
            ][str(section["priority"])]
            == len(section["method_specific_review_details"])
        )
        assert (
            decision_template_summary[
                "priority_operator_actionable_explanation_detail_counts"
            ][str(section["priority"])]
            == len(section["operator_actionable_explanation_details"])
        )
        assert (
            decision_template_summary[
                "priority_evidence_problem_correction_chain_detail_counts"
            ][str(section["priority"])]
            == len(section["evidence_problem_correction_chain_details"])
        )
        assert (
            decision_template_summary[
                "priority_headroom_feedback_true_peak_check_detail_counts"
            ][str(section["priority"])]
            == len(section["headroom_feedback_true_peak_check_details"])
        )
        for field in (
            "required_evidence",
            "required_safety_checks",
            "test_evidence_references",
            "operator_payload_required_fields",
            "operator_evidence_required_fields",
            "why_safe_summary",
            "why_blocked_summary",
            "safety_blocker_details",
            "future_wing_readback_contract_summary",
            "future_live_requirement_details",
            "live_authority_boundary_summary",
            "required_confirmation_details",
            "operator_review_decision_option_details",
            "operator_review_checklist_details",
            "operator_review_record_required_fields",
            "operator_review_record_field_details",
            "operator_next_action_details",
        ):
            expected_items = sorted(
                {
                    str(item)
                    for row in section_rows
                    for item in row[field]
                }
            )
            assert sorted(section[field]) == expected_items
        expected_explanation_contract_items = sorted(
            {
                f"{key}={str(value).lower()}"
                for row in section_rows
                for key, value in row[
                    "operator_visible_explanation_contract"
                ].items()
            }
        )
        assert sorted(section["operator_visible_explanation_contract"]) == (
            expected_explanation_contract_items
        )
        assert section["bounded_delta_policies"] == sorted(
            {
                str(row["bounded_delta_policy"])
                for row in section_rows
            }
        )
        assert section["rollback_expectations"] == sorted(
            {
                str(row["rollback_expectation"])
                for row in section_rows
            }
        )
        assert section["read_only"] is True
        assert section["can_authorize_live"] is False
        assert section["operator_review_required"] is True
        assert section["future_wing_readback_required"] is True
        assert section["hardware_proof_required_before_live"] is True
        assert section["live_authority_granted"] is False
        assert section["live_apply_allowed"] is False
        assert section["queue_import_allowed"] is False
        assert section["contacts_wing"] is False
        assert section["contacts_backend"] is False
    cards_by_method_id = {
        card["method_id"]: card
        for card in packet["method_review_cards"]
    }
    assert set(decision_rows) == {
        card["method_id"] for card in packet["method_review_cards"]
    }
    for row in decision_rows.values():
        card = cards_by_method_id[row["method_id"]]
        assert row["operator_decision"] is None
        assert row["reviewer"] == ""
        assert row["reviewed_at_utc"] == ""
        assert row["rationale"] == ""
        assert row["evidence_packet_reference"] == ""
        assert row["source_material_reference"] == ""
        assert row["source_evidence_acknowledgement"] == ""
        assert row["test_evidence_reference"] == ""
        assert row["test_evidence_acknowledgement"] == ""
        assert row["verification_report_reference"] == ""
        assert row["verification_report_acknowledgement"] == ""
        assert row["measured_problem_acknowledgement"] == ""
        assert row["proposed_correction_acknowledgement"] == ""
        assert row["bounded_delta_acknowledgement"] == ""
        assert row["confidence_acknowledgement"] == ""
        assert row["operator_explanation_acknowledgement"] == ""
        assert row["future_wing_readback_acknowledgement"] == ""
        assert row["headroom_feedback_true_peak_acknowledgement"] == ""
        assert row["rollback_readback_acknowledgement"] == ""
        assert row["safety_blockers_acknowledgement"] == ""
        assert row["required_confirmations_acknowledgement"] == ""
        assert row["offline_acceptance_boundary_acknowledgement"] == ""
        assert row["operator_notes"] == ""
        assert row["expected_evidence_packet_reference"] == (
            f"packet.json#method_review_cards/{row['method_id']}"
        )
        assert row["expected_source_material_reference"] == (
            f"packet.json#method_review_cards/{row['method_id']}/source_evidence"
        )
        assert row["expected_test_evidence_references"] == (
            row["test_evidence_references"]
        )
        assert row["expected_verification_report_reference"] == (
            "verification-report.json"
        )
        assert row["review_scope"] == "offline_proposal_only"
        assert row["review_status"] == "operator_review_pending"
        assert row["required_confirmations"]
        assert row["required_evidence"] == card["required_evidence"]
        assert row["required_safety_checks"] == card["required_safety_checks"]
        assert row["test_evidence_references"] == card["test_evidence_references"]
        for field in OPERATOR_REVIEW_CONTRACT_FIELDS:
            assert row[field] == card[field]
            assert row[field]["required"] is True
            assert row[field]["review_status"] == (
                "required_from_offline_proposal_payload"
            )
        assert row["operator_visible_explanation_contract"] == (
            card["operator_visible_explanation_contract"]
        )
        assert row["bounded_delta_unit"] == card["bounded_delta_unit"]
        assert row["bounded_delta_policy"] == card["bounded_delta_policy"]
        assert row["rollback_expectation"] == card["rollback_expectation"]
        assert row["operator_payload_required_fields"] == (
            card["operator_payload_required_fields"]
        )
        assert row["operator_evidence_required_fields"] == (
            card["operator_evidence_required_fields"]
        )
        assert row["future_wing_readback_targets"]
        assert row["future_wing_readback_contract_summary"] == (
            card["future_wing_readback_contract_summary"]
        )
        assert row["blocked_live_actions"]
        assert row["live_authority_boundary_summary"] == (
            card["live_authority_boundary_summary"]
        )
        assert row["why_safe_summary"] == card["why_safe_summary"]
        assert row["why_blocked_summary"] == card["why_blocked_summary"]
        assert row["goal_requirement_details"] == (
            card["goal_requirement_details"]
        )
        assert row["method_specific_review_details"] == (
            card["method_specific_review_details"]
        )
        assert row["operator_actionable_explanation_details"] == (
            card["operator_actionable_explanation_details"]
        )
        assert row["evidence_problem_correction_chain_details"] == (
            card["evidence_problem_correction_chain_details"]
        )
        assert row["required_confirmation_details"] == (
            card["required_confirmation_details"]
        )
        assert row["operator_review_decision_option_details"] == (
            card["operator_review_decision_option_details"]
        )
        assert row["operator_review_checklist_details"] == (
            card["operator_review_checklist_details"]
        )
        assert row["operator_review_record_required_fields"] == (
            card["operator_review_record_required_fields"]
        )
        assert row["operator_review_record_field_details"] == (
            card["operator_review_record_field_details"]
        )
        assert row["operator_next_action_details"] == (
            card["operator_next_action_details"]
        )
        assert row["future_live_requirement_details"] == (
            card["future_live_requirement_details"]
        )
        assert row["safety_blocker_details"] == card["safety_blocker_details"]
        assert row["headroom_feedback_true_peak_check_details"] == (
            card["headroom_feedback_true_peak_check_details"]
        )
        assert row["can_authorize_live"] is False
        assert row["live_authority_granted"] is False
        assert row["live_apply_allowed"] is False
        assert row["queue_import_allowed"] is False
        assert row["contacts_wing"] is False
        assert row["contacts_backend"] is False

    for forbidden in (
        "live_authority_granted: `true`",
        "live_apply_allowed: `true`",
        "queue_import_allowed: `true`",
        "contacts_wing: `true`",
        "contacts_backend: `true`",
        "mutates_mixer: `true`",
    ):
        assert forbidden not in markdown


def _write_filled_operator_decision_record(
    run_dir,
    *,
    claim_live_authority=False,
):
    template_path = run_dir / "operator-decision-template.json"
    record = json.loads(template_path.read_text(encoding="utf-8"))
    record["schema"] = "automixer.correction_method_operator_decision_record.v1"
    record["status"] = "operator_decisions_recorded_proposal_only"
    for row in record["method_decisions"]:
        row["operator_decision"] = "accept_as_offline_proposal"
        row["reviewer"] = "offline_test_operator"
        row["reviewed_at_utc"] = "2026-06-22T12:15:05Z"
        row["rationale"] = (
            "offline proposal accepted after reviewing evidence, safety "
            "blockers, rollback and future WING readback requirements"
        )
        row["evidence_packet_reference"] = row[
            "expected_evidence_packet_reference"
        ]
        row["source_material_reference"] = row[
            "expected_source_material_reference"
        ]
        row["source_evidence_acknowledgement"] = (
            REQUIRED_SOURCE_EVIDENCE_ACKNOWLEDGEMENT
        )
        row["test_evidence_reference"] = ";".join(
            row["expected_test_evidence_references"]
        )
        row["test_evidence_acknowledgement"] = (
            REQUIRED_TEST_EVIDENCE_ACKNOWLEDGEMENT
        )
        row["verification_report_reference"] = row[
            "expected_verification_report_reference"
        ]
        row["verification_report_acknowledgement"] = (
            REQUIRED_VERIFICATION_REPORT_ACKNOWLEDGEMENT
        )
        row["measured_problem_acknowledgement"] = (
            REQUIRED_MEASURED_PROBLEM_ACKNOWLEDGEMENT
        )
        row["proposed_correction_acknowledgement"] = (
            REQUIRED_PROPOSED_CORRECTION_ACKNOWLEDGEMENT
        )
        row["bounded_delta_acknowledgement"] = (
            REQUIRED_BOUNDED_DELTA_ACKNOWLEDGEMENT
        )
        row["confidence_acknowledgement"] = (
            REQUIRED_CONFIDENCE_ACKNOWLEDGEMENT
        )
        row["operator_explanation_acknowledgement"] = (
            REQUIRED_OPERATOR_EXPLANATION_ACKNOWLEDGEMENT
        )
        row["future_wing_readback_acknowledgement"] = (
            REQUIRED_FUTURE_WING_READBACK_ACKNOWLEDGEMENT
        )
        row["headroom_feedback_true_peak_acknowledgement"] = (
            REQUIRED_HEADROOM_FEEDBACK_TRUE_PEAK_ACKNOWLEDGEMENT
        )
        row["rollback_readback_acknowledgement"] = (
            REQUIRED_ROLLBACK_READBACK_ACKNOWLEDGEMENT
        )
        row["safety_blockers_acknowledgement"] = (
            REQUIRED_SAFETY_BLOCKERS_ACKNOWLEDGEMENT
        )
        row["required_confirmations_acknowledgement"] = (
            REQUIRED_OPERATOR_CONFIRMATION_ACKNOWLEDGEMENT
        )
        row["offline_acceptance_boundary_acknowledgement"] = (
            REQUIRED_OFFLINE_ACCEPTANCE_BOUNDARY_ACKNOWLEDGEMENT
        )
        row["operator_notes"] = "offline operator review only"
    for section in record["priority_decision_sections"]:
        section["operator_decision"] = "accept_as_offline_proposal"
        section["reviewer"] = "offline_test_operator"
        section["reviewed_at_utc"] = "2026-06-22T12:15:05Z"
        section["rationale"] = (
            "offline priority section accepted after reviewing all method "
            "evidence, safety blockers, rollback and future WING readback requirements"
        )
        section["evidence_packet_reference"] = section[
            "expected_evidence_packet_reference"
        ]
        section["source_material_reference"] = section[
            "expected_source_material_reference"
        ]
        section["source_evidence_acknowledgement"] = (
            REQUIRED_SOURCE_EVIDENCE_ACKNOWLEDGEMENT
        )
        section["test_evidence_reference"] = ";".join(
            section["expected_test_evidence_references"]
        )
        section["test_evidence_acknowledgement"] = (
            REQUIRED_TEST_EVIDENCE_ACKNOWLEDGEMENT
        )
        section["verification_report_reference"] = section[
            "expected_verification_report_reference"
        ]
        section["verification_report_acknowledgement"] = (
            REQUIRED_VERIFICATION_REPORT_ACKNOWLEDGEMENT
        )
        section["measured_problem_acknowledgement"] = (
            REQUIRED_MEASURED_PROBLEM_ACKNOWLEDGEMENT
        )
        section["proposed_correction_acknowledgement"] = (
            REQUIRED_PROPOSED_CORRECTION_ACKNOWLEDGEMENT
        )
        section["bounded_delta_acknowledgement"] = (
            REQUIRED_BOUNDED_DELTA_ACKNOWLEDGEMENT
        )
        section["confidence_acknowledgement"] = (
            REQUIRED_CONFIDENCE_ACKNOWLEDGEMENT
        )
        section["operator_explanation_acknowledgement"] = (
            REQUIRED_OPERATOR_EXPLANATION_ACKNOWLEDGEMENT
        )
        section["future_wing_readback_acknowledgement"] = (
            REQUIRED_FUTURE_WING_READBACK_ACKNOWLEDGEMENT
        )
        section["headroom_feedback_true_peak_acknowledgement"] = (
            REQUIRED_HEADROOM_FEEDBACK_TRUE_PEAK_ACKNOWLEDGEMENT
        )
        section["rollback_readback_acknowledgement"] = (
            REQUIRED_ROLLBACK_READBACK_ACKNOWLEDGEMENT
        )
        section["safety_blockers_acknowledgement"] = (
            REQUIRED_SAFETY_BLOCKERS_ACKNOWLEDGEMENT
        )
        section["required_confirmations_acknowledgement"] = (
            REQUIRED_OPERATOR_CONFIRMATION_ACKNOWLEDGEMENT
        )
        section["offline_acceptance_boundary_acknowledgement"] = (
            REQUIRED_OFFLINE_ACCEPTANCE_BOUNDARY_ACKNOWLEDGEMENT
        )
        section["operator_notes"] = "offline priority review only"
    if claim_live_authority:
        record["live_authority_granted"] = True
        record["can_authorize_live"] = True
        record["method_decisions"][0]["live_authority_granted"] = True
        record["method_decisions"][0]["can_authorize_live"] = True
        record["priority_decision_sections"][0]["live_authority_granted"] = True
        record["priority_decision_sections"][0]["can_authorize_live"] = True

    record_path = run_dir / "operator-decision-record.json"
    record_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record_path


def test_correction_method_goal_operator_review_export_verifier_accepts_bundle(tmp_path):
    run_id = "20260622T121500Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T12:15:00Z",
    )

    report = verify_correction_method_operator_review_export(tmp_path / run_id)

    assert report["schema"] == (
        "automixer.correction_method_operator_review_export_verification.v1"
    )
    assert report["status"] == "passed"
    assert report["failed_checks"] == []
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["contacts_backend"] is False
    assert report["queries_console"] is False
    assert report["mutates_console_state"] is False
    assert report["mutates_mixer"] is False
    assert report["live_write"] is False
    assert report["live_authority_granted"] is False
    assert report["readiness_closure_allowed"] is False
    assert report["live_apply_allowed"] is False
    assert report["queue_import_allowed"] is False
    assert report["method_review_card_count"] == EXPECTED_METHOD_REVIEW_CARD_COUNT
    assert report["priority_review_summary_count"] == 7
    assert report["priority_review_labels"] == [
        "Gain / Trim and Fader / Balance",
        "EQ and Compressor",
        "Gate / Expander / De-esser",
        "FX / Sends",
        "Pan / Stereo",
        "Routing / Groups / Scenes",
        "Master / Output",
    ]
    assert report["priority_decision_section_count"] == 7
    assert report["priority_decision_labels"] == report["priority_review_labels"]
    assert report["verified_artifacts"] == [
        "goal-audit.json",
        "goal-audit.md",
        "operator-decision-template.json",
        "operator-review.md",
        "packet.json",
    ]


def test_correction_method_goal_operator_review_export_verifier_rejects_tampered_goal_audit(tmp_path):
    run_id = "20260622T121499Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T12:14:59Z",
    )
    run_dir = tmp_path / run_id
    goal_audit_path = run_dir / "goal-audit.json"
    goal_audit = json.loads(goal_audit_path.read_text(encoding="utf-8"))
    goal_audit["live_authority_granted"] = True
    goal_audit_path.write_text(
        json.dumps(goal_audit, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_correction_method_operator_review_export(run_dir)
    failure_codes = {failure["code"] for failure in report["failed_checks"]}

    assert report["status"] == "failed"
    assert "goal_audit_json_sha256_mismatch" in failure_codes
    assert "goal_audit_live_authority_granted_not_false" in failure_codes
    assert "goal_audit_live_authority_granted_mismatch" in failure_codes


def test_correction_method_goal_operator_review_export_verifier_rejects_missing_priority_summary(tmp_path):
    run_id = "20260622T121501Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T12:15:01Z",
    )
    run_dir = tmp_path / run_id
    packet_path = run_dir / "packet.json"
    manifest_path = run_dir / "artifact-manifest.json"

    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet.pop("seventh_priority_review_summary")
    packet_path.write_text(
        json.dumps(packet, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packet_row = next(
        row for row in manifest["artifacts"] if row["artifact_id"] == "packet_json"
    )
    packet_row["sha256"] = hashlib.sha256(packet_path.read_bytes()).hexdigest()
    packet_row["bytes"] = packet_path.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_correction_method_operator_review_export(run_dir)
    failure_codes = {item["code"] for item in report["failed_checks"]}

    assert report["status"] == "failed"
    assert report["priority_review_summary_count"] == 6
    assert "packet_missing_seventh_priority_review_summary" in failure_codes
    assert "packet_json_sha256_mismatch" not in failure_codes


def test_correction_method_goal_operator_review_export_verifier_rejects_method_priority_drift(tmp_path):
    run_id = "20260622T121502Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T12:15:02Z",
    )
    run_dir = tmp_path / run_id
    packet_path = run_dir / "packet.json"
    manifest_path = run_dir / "artifact-manifest.json"

    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    method_id = packet["method_review_cards"][0]["method_id"]
    packet["method_review_cards"][0]["priority"] = 7
    packet["method_review_cards"][0]["priority_label"] = "Master / Output"
    packet["method_review_cards"][0]["priority_review_scope"] = (
        "supporting_context"
    )
    packet_path.write_text(
        json.dumps(packet, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packet_row = next(
        row for row in manifest["artifacts"] if row["artifact_id"] == "packet_json"
    )
    packet_row["sha256"] = hashlib.sha256(packet_path.read_bytes()).hexdigest()
    packet_row["bytes"] = packet_path.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_correction_method_operator_review_export(run_dir)
    failure_codes = {item["code"] for item in report["failed_checks"]}

    assert report["status"] == "failed"
    assert f"method_{method_id}_priority_mismatch" in failure_codes
    assert f"method_{method_id}_priority_label_mismatch" in failure_codes
    assert f"method_{method_id}_priority_review_scope_mismatch" in failure_codes
    assert "packet_json_sha256_mismatch" not in failure_codes


def test_correction_method_goal_operator_review_export_verifier_rejects_template_priority_drift(tmp_path):
    run_id = "20260622T121503Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T12:15:03Z",
    )
    run_dir = tmp_path / run_id
    template_path = run_dir / "operator-decision-template.json"
    manifest_path = run_dir / "artifact-manifest.json"

    template = json.loads(template_path.read_text(encoding="utf-8"))
    method_id = template["method_decisions"][0]["method_id"]
    template["method_decisions"][0]["priority"] = 7
    template["method_decisions"][0]["priority_label"] = "Master / Output"
    template["method_decisions"][0]["priority_review_scope"] = (
        "supporting_context"
    )
    template_path.write_text(
        json.dumps(template, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    template_row = next(
        row
        for row in manifest["artifacts"]
        if row["artifact_id"] == "operator_decision_template"
    )
    template_row["sha256"] = hashlib.sha256(template_path.read_bytes()).hexdigest()
    template_row["bytes"] = template_path.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_correction_method_operator_review_export(run_dir)
    failure_codes = {item["code"] for item in report["failed_checks"]}

    assert report["status"] == "failed"
    assert f"decision_template_{method_id}_priority_mismatch" in failure_codes
    assert f"decision_template_{method_id}_priority_label_mismatch" in failure_codes
    assert (
        f"decision_template_{method_id}_priority_review_scope_mismatch"
        in failure_codes
    )
    assert "operator_decision_template_sha256_mismatch" not in failure_codes


def test_correction_method_goal_operator_review_export_verifier_rejects_tampered_method_card_surfaces(tmp_path):
    run_id = "20260622T121510Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T12:15:10Z",
    )
    run_dir = tmp_path / run_id
    packet_path = run_dir / "packet.json"
    manifest_path = run_dir / "artifact-manifest.json"

    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    card = packet["method_review_cards"][0]
    method_id = card["method_id"]
    card["required_evidence"] = []
    _tamper_required_safety_checks(card)
    card["what_operator_must_confirm"] = [
        item
        for item in card["what_operator_must_confirm"]
        if item != "confidence_and_confidence_blockers_are_reviewed"
    ]
    card["what_operator_must_confirm"].extend(
        [
            "operator_can_apply_live_now",
            "live_apply_and_queue_import_remain_blocked",
        ]
    )
    card["operator_visible_explanation_contract"]["confidence_required"] = False
    card["proposed_correction"] = {}
    card["confidence"]["required"] = False
    card["bounded_delta_unit"] = "live_console_state"
    card["bounded_delta_policy"] = "unbounded_live_delta"
    card["rollback_expectation"] = "no_rollback_required"
    card["operator_review_decisions"] = [
        decision
        for decision in card["operator_review_decisions"]
        if decision != "defer_until_real_wing_proof"
    ]
    card["operator_review_decisions"].extend(
        ["authorize_live_now", "accept_as_offline_proposal"]
    )
    _tamper_why_safe_summary(card)
    _tamper_why_blocked_summary(card)
    card["future_live_required_proofs"] = [
        proof
        for proof in card["future_live_required_proofs"]
        if proof != "rollback_readback_proof"
    ]
    _tamper_future_wing_readback_targets(card)
    _tamper_future_wing_readback_contract_summary(card)
    card["headroom_feedback_true_peak_check_details"] = [
        "headroom:source=live_console;check_type=headroom;"
        "safety_check=headroom;operator_action=apply_live_now;"
        "status=live_authorized;denial_reason=none;"
        "required_before_operator_acceptance=false;"
        "required_before_live_promotion=false;"
        "required_before_queue_import=false;"
        "live_apply_allowed=true;queue_import_allowed=true;"
        "requires_real_wing_proof_before_live=false"
    ]
    card["blocked_live_actions"].extend(
        ["authorize_live_eq_now", card["blocked_live_actions"][0]]
    )
    _tamper_live_authority_boundary_summary(card)
    card["operator_review_record_required_fields"] = [
        field
        for field in card["operator_review_record_required_fields"]
        if field != "verification_report_reference"
    ]
    card["operator_review_record_required_fields"].extend(
        ["live_authorization_reference", "operator_notes"]
    )
    packet_path.write_text(
        json.dumps(packet, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packet_row = next(
        row for row in manifest["artifacts"] if row["artifact_id"] == "packet_json"
    )
    packet_row["sha256"] = hashlib.sha256(packet_path.read_bytes()).hexdigest()
    packet_row["bytes"] = packet_path.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_correction_method_operator_review_export(run_dir)
    failure_codes = {item["code"] for item in report["failed_checks"]}

    assert report["status"] == "failed"
    assert f"method_{method_id}_missing_required_evidence" in failure_codes
    assert f"method_{method_id}_unknown_required_safety_checks" in failure_codes
    assert f"method_{method_id}_duplicate_required_safety_checks" in failure_codes
    assert f"method_{method_id}_missing_operator_confirmations" in failure_codes
    assert f"method_{method_id}_unknown_operator_confirmations" in failure_codes
    assert f"method_{method_id}_duplicate_operator_confirmations" in failure_codes
    assert (
        "method_"
        f"{method_id}_operator_visible_explanation_contract_confidence_required_"
        "not_true"
    ) in failure_codes
    assert (
        f"method_{method_id}_proposed_correction_contract_required_not_true"
        in failure_codes
    )
    assert (
        f"method_{method_id}_proposed_correction_contract_source_fields_missing"
        in failure_codes
    )
    assert (
        f"method_{method_id}_confidence_contract_required_not_true"
        in failure_codes
    )
    assert f"method_{method_id}_unknown_bounded_delta_unit" in failure_codes
    assert f"method_{method_id}_unknown_bounded_delta_policy" in failure_codes
    assert f"method_{method_id}_unknown_rollback_expectation" in failure_codes
    assert (
        f"method_{method_id}_missing_operator_review_decisions"
        in failure_codes
    )
    assert (
        f"method_{method_id}_unknown_operator_review_decisions"
        in failure_codes
    )
    assert (
        f"method_{method_id}_duplicate_operator_review_decisions"
        in failure_codes
    )
    assert f"method_{method_id}_why_safe_summary_mismatch" in failure_codes
    assert (
        f"method_{method_id}_why_safe_summary_keys_mismatch" in failure_codes
    )
    assert f"method_{method_id}_why_blocked_summary_mismatch" in failure_codes
    assert (
        f"method_{method_id}_why_blocked_summary_keys_mismatch" in failure_codes
    )
    assert (
        f"method_{method_id}_missing_future_live_required_proofs" in failure_codes
    )
    assert f"method_{method_id}_unknown_future_wing_readback_targets" in failure_codes
    assert f"method_{method_id}_duplicate_future_wing_readback_targets" in failure_codes
    assert (
        f"method_{method_id}_missing_future_wing_readback_contract_summary"
        in failure_codes
    )
    assert (
        f"method_{method_id}_unknown_future_wing_readback_contract_summary"
        in failure_codes
    )
    assert (
        f"method_{method_id}_duplicate_future_wing_readback_contract_summary"
        in failure_codes
    )
    assert (
        f"method_{method_id}_invalid_future_wing_readback_operator_next_step"
        in failure_codes
    )
    assert (
        f"method_{method_id}_invalid_future_wing_readback_denial_reason"
        in failure_codes
    )
    assert (
        "method_"
        f"{method_id}_headroom_feedback_true_peak_check_details_mismatch"
        in failure_codes
    )
    assert f"method_{method_id}_unknown_blocked_live_actions" in failure_codes
    assert f"method_{method_id}_duplicate_blocked_live_actions" in failure_codes
    assert (
        f"method_{method_id}_missing_live_authority_boundary_summary"
        in failure_codes
    )
    assert (
        f"method_{method_id}_unknown_live_authority_boundary_summary"
        in failure_codes
    )
    assert (
        f"method_{method_id}_duplicate_live_authority_boundary_summary"
        in failure_codes
    )
    assert (
        f"method_{method_id}_missing_operator_review_record_required_fields"
        in failure_codes
    )
    assert (
        f"method_{method_id}_unknown_operator_review_record_required_fields"
        in failure_codes
    )
    assert (
        f"method_{method_id}_duplicate_operator_review_record_required_fields"
        in failure_codes
    )
    assert "packet_json_sha256_mismatch" not in failure_codes
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["live_authority_granted"] is False


def test_correction_method_goal_operator_review_export_verifier_rejects_invalid_test_evidence_reference(tmp_path):
    run_id = "20260622T215430Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T21:54:30Z",
    )
    run_dir = tmp_path / run_id
    packet_path = run_dir / "packet.json"
    manifest_path = run_dir / "artifact-manifest.json"

    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    card = packet["method_review_cards"][0]
    method_id = card["method_id"]
    card["test_evidence_references"] = ["tests/test_does_not_exist.py"]
    packet_path.write_text(
        json.dumps(packet, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packet_row = next(
        row for row in manifest["artifacts"] if row["artifact_id"] == "packet_json"
    )
    packet_row["sha256"] = hashlib.sha256(packet_path.read_bytes()).hexdigest()
    packet_row["bytes"] = packet_path.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_correction_method_operator_review_export(run_dir)
    failure_codes = {item["code"] for item in report["failed_checks"]}

    assert report["status"] == "failed"
    assert (
        f"method_{method_id}_test_evidence_reference_0_path_missing"
        in failure_codes
    )
    assert "packet_json_sha256_mismatch" not in failure_codes
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["live_authority_granted"] is False


def test_correction_method_goal_operator_review_export_verifier_rejects_invalid_test_evidence_node(tmp_path):
    run_id = "20260622T220230Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T22:02:30Z",
    )
    run_dir = tmp_path / run_id
    packet_path = run_dir / "packet.json"
    manifest_path = run_dir / "artifact-manifest.json"

    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    card = packet["method_review_cards"][0]
    method_id = card["method_id"]
    card["test_evidence_references"] = [
        "tests/test_processing_proposal_contracts.py::test_does_not_exist"
    ]
    packet_path.write_text(
        json.dumps(packet, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packet_row = next(
        row for row in manifest["artifacts"] if row["artifact_id"] == "packet_json"
    )
    packet_row["sha256"] = hashlib.sha256(packet_path.read_bytes()).hexdigest()
    packet_row["bytes"] = packet_path.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_correction_method_operator_review_export(run_dir)
    failure_codes = {item["code"] for item in report["failed_checks"]}

    assert report["status"] == "failed"
    assert (
        f"method_{method_id}_test_evidence_reference_0_node_missing"
        in failure_codes
    )
    assert "packet_json_sha256_mismatch" not in failure_codes
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["live_authority_granted"] is False


def test_correction_method_goal_operator_review_export_verifier_requires_concrete_test_evidence_node(tmp_path):
    run_id = "20260622T220840Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T22:08:40Z",
    )
    run_dir = tmp_path / run_id
    packet_path = run_dir / "packet.json"
    manifest_path = run_dir / "artifact-manifest.json"

    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    card = packet["method_review_cards"][0]
    method_id = card["method_id"]
    card["test_evidence_references"] = [
        "tests/test_processing_proposal_contracts.py"
    ]
    packet_path.write_text(
        json.dumps(packet, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packet_row = next(
        row for row in manifest["artifacts"] if row["artifact_id"] == "packet_json"
    )
    packet_row["sha256"] = hashlib.sha256(packet_path.read_bytes()).hexdigest()
    packet_row["bytes"] = packet_path.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_correction_method_operator_review_export(run_dir)
    failure_codes = {item["code"] for item in report["failed_checks"]}

    assert report["status"] == "failed"
    assert (
        f"method_{method_id}_test_evidence_references_missing_pytest_node"
        in failure_codes
    )
    assert "packet_json_sha256_mismatch" not in failure_codes
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["live_authority_granted"] is False


def test_correction_method_goal_operator_review_export_verifier_requires_test_evidence_reference_source_field(tmp_path):
    run_id = "20260622T221430Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T22:14:30Z",
    )
    run_dir = tmp_path / run_id
    packet_path = run_dir / "packet.json"
    manifest_path = run_dir / "artifact-manifest.json"

    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    card = packet["method_review_cards"][0]
    method_id = card["method_id"]
    card["test_evidence"]["source_fields"] = [
        source_field
        for source_field in card["test_evidence"]["source_fields"]
        if source_field != "test_evidence_references"
    ]
    packet_path.write_text(
        json.dumps(packet, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packet_row = next(
        row for row in manifest["artifacts"] if row["artifact_id"] == "packet_json"
    )
    packet_row["sha256"] = hashlib.sha256(packet_path.read_bytes()).hexdigest()
    packet_row["bytes"] = packet_path.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_correction_method_operator_review_export(run_dir)
    failure_codes = {item["code"] for item in report["failed_checks"]}

    assert report["status"] == "failed"
    assert (
        "method_"
        f"{method_id}_test_evidence_contract_source_field_"
        "test_evidence_references_missing"
    ) in failure_codes
    assert "packet_json_sha256_mismatch" not in failure_codes
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["live_authority_granted"] is False


def test_correction_method_goal_operator_review_export_verifier_rejects_unbound_contract_source_field(tmp_path):
    run_id = "20260622T222030Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T22:20:30Z",
    )
    run_dir = tmp_path / run_id
    packet_path = run_dir / "packet.json"
    manifest_path = run_dir / "artifact-manifest.json"

    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    card = packet["method_review_cards"][0]
    method_id = card["method_id"]
    card["confidence"]["source_fields"].append("missing_operator_payload_field")
    packet_path.write_text(
        json.dumps(packet, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packet_row = next(
        row for row in manifest["artifacts"] if row["artifact_id"] == "packet_json"
    )
    packet_row["sha256"] = hashlib.sha256(packet_path.read_bytes()).hexdigest()
    packet_row["bytes"] = packet_path.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_correction_method_operator_review_export(run_dir)
    failure_codes = {item["code"] for item in report["failed_checks"]}

    assert report["status"] == "failed"
    assert (
        f"method_{method_id}_confidence_contract_source_field_unbound"
        in failure_codes
    )
    assert "packet_json_sha256_mismatch" not in failure_codes
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["live_authority_granted"] is False


def test_correction_method_goal_operator_review_export_verifier_requires_full_future_live_proofs(tmp_path):
    run_id = "20260622T222830Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T22:28:30Z",
    )
    run_dir = tmp_path / run_id
    packet_path = run_dir / "packet.json"
    decision_template_path = run_dir / "operator-decision-template.json"
    manifest_path = run_dir / "artifact-manifest.json"
    removed_proof = "physical_console_readback_for_old_and_proposed_values"
    unknown_proof = "operator_claims_live_ready_without_wing"
    duplicate_proof = "rollback_readback_proof"

    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    card = packet["method_review_cards"][0]
    method_id = card["method_id"]
    card["future_live_required_proofs"] = [
        proof
        for proof in card["future_live_required_proofs"]
        if proof != removed_proof
    ]
    card["future_live_required_proofs"].extend(
        [unknown_proof, duplicate_proof]
    )
    for section in packet["priority_sections"]:
        for nested_card in section["methods"]:
            if nested_card["method_id"] == method_id:
                nested_card["future_live_required_proofs"] = list(
                    card["future_live_required_proofs"]
                )
    packet_path.write_text(
        json.dumps(packet, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    decision_template = json.loads(
        decision_template_path.read_text(encoding="utf-8")
    )
    decision_row = next(
        row
        for row in decision_template["method_decisions"]
        if row["method_id"] == method_id
    )
    decision_row["future_live_required_proofs"] = list(
        card["future_live_required_proofs"]
    )
    decision_template_path.write_text(
        json.dumps(decision_template, indent=2, ensure_ascii=True, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packet_row = next(
        row for row in manifest["artifacts"] if row["artifact_id"] == "packet_json"
    )
    packet_row["sha256"] = hashlib.sha256(packet_path.read_bytes()).hexdigest()
    packet_row["bytes"] = packet_path.stat().st_size
    template_row = next(
        row
        for row in manifest["artifacts"]
        if row["artifact_id"] == "operator_decision_template"
    )
    template_row["sha256"] = hashlib.sha256(
        decision_template_path.read_bytes()
    ).hexdigest()
    template_row["bytes"] = decision_template_path.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_correction_method_operator_review_export(run_dir)
    failure_codes = {item["code"] for item in report["failed_checks"]}

    assert report["status"] == "failed"
    assert (
        f"method_{method_id}_missing_future_live_required_proofs"
        in failure_codes
    )
    assert (
        f"method_{method_id}_unknown_future_live_required_proofs"
        in failure_codes
    )
    assert (
        f"method_{method_id}_duplicate_future_live_required_proofs"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_missing_future_live_required_proofs"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_unknown_future_live_required_proofs"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_duplicate_future_live_required_proofs"
        in failure_codes
    )
    assert "packet_json_sha256_mismatch" not in failure_codes
    assert "operator_decision_template_sha256_mismatch" not in failure_codes
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["live_authority_granted"] is False


def test_correction_method_goal_operator_review_export_verifier_rejects_drifted_priority_section_method_card(tmp_path):
    run_id = "20260622T121514Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T12:15:14Z",
    )
    run_dir = tmp_path / run_id
    packet_path = run_dir / "packet.json"
    manifest_path = run_dir / "artifact-manifest.json"

    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    nested_card = packet["priority_sections"][0]["methods"][0]
    method_id = nested_card["method_id"]
    nested_card["required_evidence"] = []
    nested_card["future_live_required_proofs"] = [
        proof
        for proof in nested_card["future_live_required_proofs"]
        if proof != "rollback_readback_proof"
    ]
    nested_card["blocked_live_actions"] = []
    nested_card["operator_visible_explanation_contract"][
        "confidence_required"
    ] = False
    nested_card["confidence"]["required"] = False
    packet_path.write_text(
        json.dumps(packet, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packet_row = next(
        row for row in manifest["artifacts"] if row["artifact_id"] == "packet_json"
    )
    packet_row["sha256"] = hashlib.sha256(packet_path.read_bytes()).hexdigest()
    packet_row["bytes"] = packet_path.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_correction_method_operator_review_export(run_dir)
    failure_codes = {item["code"] for item in report["failed_checks"]}

    assert report["status"] == "failed"
    assert (
        f"packet_priority_section_1_method_{method_id}_required_evidence_mismatch"
        in failure_codes
    )
    assert (
        "packet_priority_section_1_method_"
        f"{method_id}_future_live_required_proofs_mismatch"
    ) in failure_codes
    assert (
        f"packet_priority_section_1_method_{method_id}_blocked_live_actions_mismatch"
        in failure_codes
    )
    assert (
        "packet_priority_section_1_method_"
        f"{method_id}_operator_visible_explanation_contract_mismatch"
    ) in failure_codes
    assert (
        f"packet_priority_section_1_method_{method_id}_confidence_mismatch"
        in failure_codes
    )
    assert "packet_json_sha256_mismatch" not in failure_codes
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["live_authority_granted"] is False


def test_correction_method_goal_operator_review_export_verifier_rejects_drifted_priority_section_aggregates(tmp_path):
    run_id = "20260622T192950Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T19:29:50Z",
    )
    run_dir = tmp_path / run_id
    packet_path = run_dir / "packet.json"
    manifest_path = run_dir / "artifact-manifest.json"

    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    section = packet["priority_sections"][0]
    section["operator_payload_required_fields"] = []
    section["confidence"] = []
    _tamper_bounded_delta_policies(section)
    _tamper_future_wing_readback_targets(section)
    _tamper_rollback_expectations(section)
    section["future_wing_readback_contract_summary"] = []
    section["why_safe_summary"] = []
    section["why_blocked_summary"] = []
    section["live_authority_boundary_summary"] = []
    section["live_authority_method_ids"] = [section["methods"][0]["method_id"]]
    section["live_authority_granted"] = True
    section["operator_next_step"] = "apply_live_without_real_wing_readback"
    packet_path.write_text(
        json.dumps(packet, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packet_row = next(
        row for row in manifest["artifacts"] if row["artifact_id"] == "packet_json"
    )
    packet_row["sha256"] = hashlib.sha256(packet_path.read_bytes()).hexdigest()
    packet_row["bytes"] = packet_path.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_correction_method_operator_review_export(run_dir)
    failure_codes = {item["code"] for item in report["failed_checks"]}

    assert report["status"] == "failed"
    assert "packet_priority_section_1_confidence_mismatch" in failure_codes
    assert (
        "packet_priority_section_1_operator_payload_required_fields_mismatch"
        in failure_codes
    )
    assert "packet_priority_section_1_bounded_delta_policies_mismatch" in failure_codes
    assert "packet_priority_section_1_unknown_bounded_delta_policies" in failure_codes
    assert "packet_priority_section_1_duplicate_bounded_delta_policies" in failure_codes
    assert "packet_priority_section_1_rollback_expectations_mismatch" in failure_codes
    assert "packet_priority_section_1_unknown_rollback_expectations" in failure_codes
    assert "packet_priority_section_1_duplicate_rollback_expectations" in failure_codes
    assert (
        "packet_priority_section_1_future_wing_readback_targets_mismatch"
        in failure_codes
    )
    assert (
        "packet_priority_section_1_unknown_future_wing_readback_targets"
        in failure_codes
    )
    assert (
        "packet_priority_section_1_duplicate_future_wing_readback_targets"
        in failure_codes
    )
    assert (
        "packet_priority_section_1_future_wing_readback_contract_summary_mismatch"
        in failure_codes
    )
    assert "packet_priority_section_1_why_safe_summary_mismatch" in failure_codes
    assert (
        "packet_priority_section_1_why_blocked_summary_mismatch"
        in failure_codes
    )
    assert (
        "packet_priority_section_1_missing_future_wing_readback_contract_summary"
        in failure_codes
    )
    assert (
        "packet_priority_section_1_live_authority_boundary_summary_mismatch"
        in failure_codes
    )
    assert (
        "packet_priority_section_1_live_authority_method_ids_not_empty"
        in failure_codes
    )
    assert (
        "packet_priority_section_1_live_authority_granted_not_false"
        in failure_codes
    )
    assert "packet_priority_section_1_operator_next_step_mismatch" in failure_codes
    assert "packet_json_sha256_mismatch" not in failure_codes
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["live_authority_granted"] is False


def test_correction_method_goal_operator_review_export_verifier_rejects_drifted_supporting_method_card(tmp_path):
    run_id = "20260622T121519Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T12:15:19Z",
    )
    run_dir = tmp_path / run_id
    packet_path = run_dir / "packet.json"
    manifest_path = run_dir / "artifact-manifest.json"

    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    supporting_card = packet["supporting_method_cards"][0]
    method_id = supporting_card["method_id"]
    supporting_card["required_evidence"] = []
    supporting_card["future_live_required_proofs"] = [
        proof
        for proof in supporting_card["future_live_required_proofs"]
        if proof != "rollback_readback_proof"
    ]
    supporting_card["blocked_live_actions"] = []
    supporting_card["operator_visible_explanation_contract"][
        "confidence_required"
    ] = False
    packet_path.write_text(
        json.dumps(packet, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packet_row = next(
        row for row in manifest["artifacts"] if row["artifact_id"] == "packet_json"
    )
    packet_row["sha256"] = hashlib.sha256(packet_path.read_bytes()).hexdigest()
    packet_row["bytes"] = packet_path.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_correction_method_operator_review_export(run_dir)
    failure_codes = {item["code"] for item in report["failed_checks"]}

    assert report["status"] == "failed"
    assert (
        f"packet_supporting_method_{method_id}_required_evidence_mismatch"
        in failure_codes
    )
    assert (
        "packet_supporting_method_"
        f"{method_id}_future_live_required_proofs_mismatch"
    ) in failure_codes
    assert (
        f"packet_supporting_method_{method_id}_blocked_live_actions_mismatch"
        in failure_codes
    )
    assert (
        "packet_supporting_method_"
        f"{method_id}_operator_visible_explanation_contract_mismatch"
    ) in failure_codes
    assert "packet_json_sha256_mismatch" not in failure_codes
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["live_authority_granted"] is False


def test_correction_method_goal_operator_review_export_verifier_rejects_tampered_required_operator_surfaces(tmp_path):
    run_id = "20260622T121517Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T12:15:17Z",
    )
    run_dir = tmp_path / run_id
    packet_path = run_dir / "packet.json"
    manifest_path = run_dir / "artifact-manifest.json"

    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    removed_surface = "operator_readable_explanation"
    assert removed_surface in packet["required_operator_surfaces"]
    packet["required_operator_surfaces"] = [
        surface
        for surface in packet["required_operator_surfaces"]
        if surface != removed_surface
    ]
    card = packet["method_review_cards"][0]
    method_id = card["method_id"]
    card["required_operator_surfaces"] = [
        surface
        for surface in card["required_operator_surfaces"]
        if surface != removed_surface
    ]
    card["missing_required_operator_surfaces"] = [removed_surface]
    packet_path.write_text(
        json.dumps(packet, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packet_row = next(
        row for row in manifest["artifacts"] if row["artifact_id"] == "packet_json"
    )
    packet_row["sha256"] = hashlib.sha256(packet_path.read_bytes()).hexdigest()
    packet_row["bytes"] = packet_path.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_correction_method_operator_review_export(run_dir)
    failure_codes = {item["code"] for item in report["failed_checks"]}

    assert report["status"] == "failed"
    assert "packet_required_operator_surfaces_mismatch" in failure_codes
    assert (
        f"method_{method_id}_required_operator_surfaces_mismatch"
        in failure_codes
    )
    assert (
        f"method_{method_id}_missing_required_operator_surfaces_not_empty"
        in failure_codes
    )
    assert "packet_json_sha256_mismatch" not in failure_codes
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["live_authority_granted"] is False


def test_correction_method_goal_operator_review_export_verifier_rejects_tampered_decision_template_surfaces(tmp_path):
    run_id = "20260622T121511Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T12:15:11Z",
    )
    run_dir = tmp_path / run_id
    template_path = run_dir / "operator-decision-template.json"
    manifest_path = run_dir / "artifact-manifest.json"

    template = json.loads(template_path.read_text(encoding="utf-8"))
    row = template["method_decisions"][0]
    method_id = row["method_id"]
    row["allowed_decisions"] = [
        "accept_as_offline_proposal",
        "authorize_live_now",
        "accept_as_offline_proposal",
    ]
    row["required_confirmations"] = [
        "source_evidence_matches_current_offline_material",
        "operator_can_apply_live_now",
        "source_evidence_matches_current_offline_material",
    ]
    row["required_evidence"] = []
    _tamper_required_safety_checks(row)
    row["bounded_delta_unit"] = "live_console_state"
    row["bounded_delta_policy"] = "unbounded_live_delta"
    row["rollback_expectation"] = "no_rollback_required"
    row["confidence"]["required"] = False
    row["operator_readable_explanation"] = {}
    row["operator_visible_explanation_contract"]["confidence_required"] = False
    row["operator_payload_required_fields"] = []
    row["operator_evidence_required_fields"] = []
    row["operator_problem_review_summary"] = []
    row["operator_method_review_focus"] = []
    _tamper_why_safe_summary(row)
    _tamper_why_blocked_summary(row)
    row["goal_requirement_details"] = [
        "source_evidence:source=live_console;source_fields=none;"
        "operator_action=apply_live_now;coverage_status=covered;"
        "required=false;acceptance_scope=live_apply;"
        "required_before_operator_acceptance=false;"
        "required_before_live_promotion=false;"
        "required_before_queue_import=false;denial_reason=none;"
        "live_apply_allowed=true;queue_import_allowed=true;"
        "readiness_closure_allowed=true;"
        "requires_real_wing_proof_before_live=false"
    ]
    row["evidence_problem_correction_chain_details"] = [
        "source_evidence:source=live_console;chain_step=1_source_evidence;"
        "source_fields=none;operator_action=apply_live_now;"
        "review_status=live_authorized;required=false;denial_reason=none;"
        "required_before_operator_acceptance=false;"
        "required_before_live_promotion=false;"
        "required_before_queue_import=false;"
        "live_apply_allowed=true;queue_import_allowed=true;"
        "requires_real_wing_proof_before_live=false"
    ]
    row["future_live_required_proofs"] = [
        proof
        for proof in row["future_live_required_proofs"]
        if proof != "rollback_readback_proof"
    ]
    row["future_live_required_proofs"].extend(
        [
            "operator_claims_live_ready_without_wing",
            "emergency_stop_proof",
        ]
    )
    row["future_live_requirement_details"] = [
        "real_wing_readonly_connection_artifact:source=live_console;"
        "requirement_type=live_apply;proof=none;readback_target=none;"
        "operator_action=apply_live_now;status=live_authorized;"
        "required_before_operator_acceptance=false;"
        "required_before_live_promotion=false;"
        "required_before_queue_import=false;"
        "blocked_live_actions=none;live_apply_allowed=true;"
        "queue_import_allowed=true;requires_real_wing_proof_before_live=false"
    ]
    row["safety_blocker_details"] = [
        "headroom:source=live_console;blocker_type=live_apply;"
        "safety_check=none;blocked_live_action=none;"
        "missing_operator_surface=none;operator_action=apply_live_now;"
        "status=live_authorized;denial_reason=none;"
        "required_before_operator_acceptance=false;"
        "required_before_live_promotion=false;"
        "required_before_queue_import=false;"
        "live_apply_allowed=true;queue_import_allowed=true;"
        "requires_real_wing_proof_before_live=false"
    ]
    row["headroom_feedback_true_peak_check_details"] = [
        "headroom:source=live_console;check_type=headroom;"
        "safety_check=headroom;operator_action=apply_live_now;"
        "status=live_authorized;denial_reason=none;"
        "required_before_operator_acceptance=false;"
        "required_before_live_promotion=false;"
        "required_before_queue_import=false;"
        "live_apply_allowed=true;queue_import_allowed=true;"
        "requires_real_wing_proof_before_live=false"
    ]
    _tamper_future_wing_readback_targets(row)
    _tamper_future_wing_readback_contract_summary(row)
    row["blocked_live_actions"] = [
        "live_apply",
        "authorize_live_eq_now",
        "live_apply",
    ]
    _tamper_live_authority_boundary_summary(row)
    row["operator_review_record_required_fields"] = [
        "reviewer",
        "live_authorization_reference",
        "reviewer",
    ]
    row["operator_review_checklist_details"] = [
        "do_not_queue_or_apply_live:source=live_authority_boundary;"
        "action_type=live_apply;acknowledgement_fields=none;"
        "action=review_operator_checklist_item:do_not_queue_or_apply_live"
    ]
    row["required_confirmation_details"] = [
        "live_apply_and_queue_import_remain_blocked:source=live_console;"
        "action_type=write;acknowledgement_fields=none;"
        "action=confirm_operator_required_condition:"
        "live_apply_and_queue_import_remain_blocked"
    ]
    row["operator_acceptance_gate_details"] = [
        "live_apply_and_queue_import_remain_blocked:source=live_console;"
        "action_type=write;acknowledgement_fields=none;"
        "action=confirm_operator_required_condition:"
        "live_apply_and_queue_import_remain_blocked;"
        "acceptance_scope=live_apply;required=false;"
        "required_before_operator_acceptance=false;"
        "required_before_live_promotion=false;"
        "required_before_queue_import=false;"
        "live_apply_allowed=true;queue_import_allowed=true;"
        "readiness_closure_allowed=true;"
        "requires_real_wing_proof_before_live=false"
    ]
    row["operator_review_decision_option_details"] = [
        "accept_as_offline_proposal:effect=authorize_live_now;"
        "action_type=live_apply;operator_action=apply_live_now;"
        "blocks_offline_acceptance=false;live_apply_allowed=true;"
        "queue_import_allowed=true;requires_real_wing_proof_before_live=false"
    ]
    row["operator_next_action_details"] = [
        "do_not_queue_or_apply_live:source=live_console;"
        "action_type=live_apply;operator_action=apply_live_now;"
        "status=live_authorized;fields=none;readback_targets=none;"
        "proofs=none;blocked_live_actions=none;"
        "required_before_operator_acceptance=false;"
        "required_before_live_promotion=false;"
        "required_before_queue_import=false;"
        "live_apply_allowed=true;queue_import_allowed=true;"
        "requires_real_wing_proof_before_live=false"
    ]
    template_path.write_text(
        json.dumps(template, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    template_row = next(
        row
        for row in manifest["artifacts"]
        if row["artifact_id"] == "operator_decision_template"
    )
    template_row["sha256"] = hashlib.sha256(template_path.read_bytes()).hexdigest()
    template_row["bytes"] = template_path.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_correction_method_operator_review_export(run_dir)
    failure_codes = {item["code"] for item in report["failed_checks"]}

    assert report["status"] == "failed"
    assert (
        f"decision_template_{method_id}_missing_allowed_decisions"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_unknown_allowed_decisions"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_duplicate_allowed_decisions"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_required_confirmations_mismatch"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_required_evidence_mismatch"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_required_safety_checks_mismatch"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_bounded_delta_unit_mismatch"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_bounded_delta_policy_mismatch"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_unknown_bounded_delta_unit"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_unknown_bounded_delta_policy"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_rollback_expectation_mismatch"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_unknown_rollback_expectation"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_confidence_mismatch"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_operator_readable_explanation_mismatch"
        in failure_codes
    )
    assert (
        "decision_template_"
        f"{method_id}_operator_visible_explanation_contract_mismatch"
    ) in failure_codes
    assert (
        "decision_template_"
        f"{method_id}_operator_visible_explanation_contract_confidence_required_"
        "not_true"
    ) in failure_codes
    assert (
        f"decision_template_{method_id}_confidence_contract_required_not_true"
        in failure_codes
    )
    assert (
        "decision_template_"
        f"{method_id}_operator_readable_explanation_contract_required_not_true"
    ) in failure_codes
    assert (
        f"decision_template_{method_id}_operator_payload_required_fields_mismatch"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_operator_evidence_required_fields_mismatch"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_operator_problem_review_summary_mismatch"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_operator_method_review_focus_mismatch"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_why_safe_summary_mismatch"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_why_safe_summary_keys_mismatch"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_why_blocked_summary_mismatch"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_why_blocked_summary_keys_mismatch"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_goal_requirement_details_mismatch"
        in failure_codes
    )
    assert (
        "decision_template_"
        f"{method_id}_evidence_problem_correction_chain_details_mismatch"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_missing_required_confirmations"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_unknown_required_confirmations"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_duplicate_required_confirmations"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_missing_required_evidence"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_unknown_required_safety_checks"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_duplicate_required_safety_checks"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_future_live_required_proofs_mismatch"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_future_live_requirement_details_mismatch"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_safety_blocker_details_mismatch"
        in failure_codes
    )
    assert (
        "decision_template_"
        f"{method_id}_headroom_feedback_true_peak_check_details_mismatch"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_future_wing_readback_targets_mismatch"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_unknown_future_wing_readback_targets"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_duplicate_future_wing_readback_targets"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_future_wing_readback_contract_summary_mismatch"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_missing_future_wing_readback_contract_summary"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_unknown_future_wing_readback_contract_summary"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_duplicate_future_wing_readback_contract_summary"
        in failure_codes
    )
    assert (
        "decision_template_"
        f"{method_id}_invalid_future_wing_readback_operator_next_step"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_invalid_future_wing_readback_denial_reason"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_missing_future_live_required_proofs"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_unknown_future_live_required_proofs"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_duplicate_future_live_required_proofs"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_blocked_live_actions_mismatch"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_unknown_blocked_live_actions"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_duplicate_blocked_live_actions"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_live_authority_boundary_summary_mismatch"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_missing_live_authority_boundary_summary"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_unknown_live_authority_boundary_summary"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_duplicate_live_authority_boundary_summary"
        in failure_codes
    )
    assert (
        f"decision_template_{method_id}_operator_review_record_required_fields_mismatch"
        in failure_codes
    )
    assert (
        "decision_template_"
        f"{method_id}_missing_operator_review_record_required_fields"
        in failure_codes
    )
    assert (
        "decision_template_"
        f"{method_id}_unknown_operator_review_record_required_fields"
        in failure_codes
    )
    assert (
        "decision_template_"
        f"{method_id}_duplicate_operator_review_record_required_fields"
        in failure_codes
    )
    assert (
        "decision_template_"
        f"{method_id}_required_confirmation_details_mismatch"
        in failure_codes
    )
    assert (
        "decision_template_"
        f"{method_id}_operator_acceptance_gate_details_mismatch"
        in failure_codes
    )
    assert (
        "decision_template_"
        f"{method_id}_operator_review_decision_option_details_mismatch"
        in failure_codes
    )
    assert (
        "decision_template_"
        f"{method_id}_operator_review_checklist_details_mismatch"
        in failure_codes
    )
    assert (
        "decision_template_"
        f"{method_id}_operator_review_record_field_details_mismatch"
        in failure_codes
    )
    assert (
        "decision_template_"
        f"{method_id}_operator_next_action_details_mismatch"
        in failure_codes
    )
    assert "operator_decision_template_sha256_mismatch" not in failure_codes
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["live_authority_granted"] is False


def test_correction_method_goal_operator_review_export_verifier_rejects_tampered_decision_template_priority_surfaces(tmp_path):
    run_id = "20260622T121520Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T12:15:20Z",
    )
    run_dir = tmp_path / run_id
    template_path = run_dir / "operator-decision-template.json"
    manifest_path = run_dir / "artifact-manifest.json"

    template = json.loads(template_path.read_text(encoding="utf-8"))
    section = template["priority_decision_sections"][0]
    priority = section["priority"]
    section["allowed_decisions"] = [
        "accept_as_offline_proposal",
        "authorize_live_now",
        "accept_as_offline_proposal",
    ]
    section["required_evidence"] = []
    _tamper_required_safety_checks(section)
    section["confidence"] = []
    section["operator_readable_explanation"] = []
    _tamper_bounded_delta_policies(section)
    _tamper_rollback_expectations(section)
    section["operator_payload_required_fields"] = []
    section["operator_evidence_required_fields"] = []
    section["operator_problem_review_summary"] = []
    section["operator_method_review_focus"] = []
    section["why_safe_summary"] = []
    section["why_blocked_summary"] = []
    section["goal_requirement_details"] = [
        "source_evidence:source=live_console;source_fields=none;"
        "operator_action=apply_live_now;coverage_status=covered;"
        "required=false;acceptance_scope=live_apply;"
        "required_before_operator_acceptance=false;"
        "required_before_live_promotion=false;"
        "required_before_queue_import=false;denial_reason=none;"
        "live_apply_allowed=true;queue_import_allowed=true;"
        "readiness_closure_allowed=true;"
        "requires_real_wing_proof_before_live=false"
    ]
    section["evidence_problem_correction_chain_details"] = [
        "source_evidence:source=live_console;chain_step=1_source_evidence;"
        "source_fields=none;operator_action=apply_live_now;"
        "review_status=live_authorized;required=false;denial_reason=none;"
        "required_before_operator_acceptance=false;"
        "required_before_live_promotion=false;"
        "required_before_queue_import=false;"
        "live_apply_allowed=true;queue_import_allowed=true;"
        "requires_real_wing_proof_before_live=false"
    ]
    section["operator_visible_explanation_contract"] = []
    _tamper_future_wing_readback_targets(section)
    _tamper_future_wing_readback_contract_summary(section)
    section["future_live_required_proofs"] = [
        proof
        for proof in section["future_live_required_proofs"]
        if proof != "rollback_readback_proof"
    ]
    section["future_live_requirement_details"] = [
        "real_wing_readonly_connection_artifact:source=live_console;"
        "requirement_type=live_apply;proof=none;readback_target=none;"
        "operator_action=apply_live_now;status=live_authorized;"
        "required_before_operator_acceptance=false;"
        "required_before_live_promotion=false;"
        "required_before_queue_import=false;"
        "blocked_live_actions=none;live_apply_allowed=true;"
        "queue_import_allowed=true;requires_real_wing_proof_before_live=false"
    ]
    section["safety_blocker_details"] = [
        "headroom:source=live_console;blocker_type=live_apply;"
        "safety_check=none;blocked_live_action=none;"
        "missing_operator_surface=none;operator_action=apply_live_now;"
        "status=live_authorized;denial_reason=none;"
        "required_before_operator_acceptance=false;"
        "required_before_live_promotion=false;"
        "required_before_queue_import=false;"
        "live_apply_allowed=true;queue_import_allowed=true;"
        "requires_real_wing_proof_before_live=false"
    ]
    section["headroom_feedback_true_peak_check_details"] = [
        "headroom:source=live_console;check_type=headroom;"
        "safety_check=headroom;operator_action=apply_live_now;"
        "status=live_authorized;denial_reason=none;"
        "required_before_operator_acceptance=false;"
        "required_before_live_promotion=false;"
        "required_before_queue_import=false;"
        "live_apply_allowed=true;queue_import_allowed=true;"
        "requires_real_wing_proof_before_live=false"
    ]
    section["blocked_live_actions"] = [
        "live_apply",
        "authorize_live_eq_now",
        "live_apply",
    ]
    _tamper_live_authority_boundary_summary(section)
    section["operator_review_record_required_fields"] = [
        "reviewer",
        "live_authorization_reference",
        "reviewer",
    ]
    section["operator_review_checklist_details"] = [
        "do_not_queue_or_apply_live:source=live_authority_boundary;"
        "action_type=live_apply;acknowledgement_fields=none;"
        "action=review_operator_checklist_item:do_not_queue_or_apply_live"
    ]
    section["required_confirmation_details"] = [
        "live_apply_and_queue_import_remain_blocked:source=live_console;"
        "action_type=write;acknowledgement_fields=none;"
        "action=confirm_operator_required_condition:"
        "live_apply_and_queue_import_remain_blocked"
    ]
    section["operator_acceptance_gate_details"] = [
        "live_apply_and_queue_import_remain_blocked:source=live_console;"
        "action_type=write;acknowledgement_fields=none;"
        "action=confirm_operator_required_condition:"
        "live_apply_and_queue_import_remain_blocked;"
        "acceptance_scope=live_apply;required=false;"
        "required_before_operator_acceptance=false;"
        "required_before_live_promotion=false;"
        "required_before_queue_import=false;"
        "live_apply_allowed=true;queue_import_allowed=true;"
        "readiness_closure_allowed=true;"
        "requires_real_wing_proof_before_live=false"
    ]
    section["operator_review_decision_option_details"] = [
        "accept_as_offline_proposal:effect=authorize_live_now;"
        "action_type=live_apply;operator_action=apply_live_now;"
        "blocks_offline_acceptance=false;live_apply_allowed=true;"
        "queue_import_allowed=true;requires_real_wing_proof_before_live=false"
    ]
    section["operator_next_action_details"] = [
        "do_not_queue_or_apply_live:source=live_console;"
        "action_type=live_apply;operator_action=apply_live_now;"
        "status=live_authorized;fields=none;readback_targets=none;"
        "proofs=none;blocked_live_actions=none;"
        "required_before_operator_acceptance=false;"
        "required_before_live_promotion=false;"
        "required_before_queue_import=false;"
        "live_apply_allowed=true;queue_import_allowed=true;"
        "requires_real_wing_proof_before_live=false"
    ]
    template_path.write_text(
        json.dumps(template, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    template_row = next(
        row
        for row in manifest["artifacts"]
        if row["artifact_id"] == "operator_decision_template"
    )
    template_row["sha256"] = hashlib.sha256(template_path.read_bytes()).hexdigest()
    template_row["bytes"] = template_path.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_correction_method_operator_review_export(run_dir)
    failure_codes = {item["code"] for item in report["failed_checks"]}

    assert report["status"] == "failed"
    assert (
        f"decision_template_priority_{priority}_required_evidence_mismatch"
        in failure_codes
    )
    assert (
        f"decision_template_priority_{priority}_required_safety_checks_mismatch"
        in failure_codes
    )
    assert (
        f"decision_template_priority_{priority}_confidence_mismatch"
        in failure_codes
    )
    assert (
        "decision_template_priority_"
        f"{priority}_operator_readable_explanation_mismatch"
    ) in failure_codes
    assert (
        f"decision_template_priority_{priority}_bounded_delta_policies_mismatch"
        in failure_codes
    )
    assert (
        f"decision_template_priority_{priority}_rollback_expectations_mismatch"
        in failure_codes
    )
    assert (
        f"decision_template_priority_{priority}_missing_allowed_decisions"
        in failure_codes
    )
    assert (
        f"decision_template_priority_{priority}_unknown_allowed_decisions"
        in failure_codes
    )
    assert (
        f"decision_template_priority_{priority}_duplicate_allowed_decisions"
        in failure_codes
    )
    assert (
        f"decision_template_priority_{priority}_missing_required_evidence"
        in failure_codes
    )
    assert (
        f"decision_template_priority_{priority}_unknown_required_safety_checks"
        in failure_codes
    )
    assert (
        f"decision_template_priority_{priority}_duplicate_required_safety_checks"
        in failure_codes
    )
    assert (
        f"decision_template_priority_{priority}_missing_confidence"
        in failure_codes
    )
    assert (
        "decision_template_priority_"
        f"{priority}_missing_operator_readable_explanation"
    ) in failure_codes
    assert (
        f"decision_template_priority_{priority}_unknown_bounded_delta_policies"
        in failure_codes
    )
    assert (
        f"decision_template_priority_{priority}_duplicate_bounded_delta_policies"
        in failure_codes
    )
    assert (
        f"decision_template_priority_{priority}_unknown_rollback_expectations"
        in failure_codes
    )
    assert (
        f"decision_template_priority_{priority}_duplicate_rollback_expectations"
        in failure_codes
    )
    assert (
        "decision_template_priority_"
        f"{priority}_operator_payload_required_fields_mismatch"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_operator_evidence_required_fields_mismatch"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_operator_problem_review_summary_mismatch"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_operator_method_review_focus_mismatch"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_why_safe_summary_mismatch"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_missing_why_safe_summary_keys"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_why_blocked_summary_mismatch"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_goal_requirement_details_mismatch"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_evidence_problem_correction_chain_details_mismatch"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_missing_why_blocked_summary_keys"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_operator_visible_explanation_contract_mismatch"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_missing_operator_visible_explanation_contract"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_future_wing_readback_targets_mismatch"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_unknown_future_wing_readback_targets"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_duplicate_future_wing_readback_targets"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_future_live_required_proofs_mismatch"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_future_live_requirement_details_mismatch"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_safety_blocker_details_mismatch"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_headroom_feedback_true_peak_check_details_mismatch"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_future_wing_readback_contract_summary_mismatch"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_missing_future_wing_readback_contract_summary"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_unknown_future_wing_readback_contract_summary"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_duplicate_future_wing_readback_contract_summary"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_invalid_future_wing_readback_operator_next_step"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_invalid_future_wing_readback_denial_reason"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_missing_future_live_required_proofs"
    ) in failure_codes
    assert (
        f"decision_template_priority_{priority}_blocked_live_actions_mismatch"
        in failure_codes
    )
    assert (
        f"decision_template_priority_{priority}_unknown_blocked_live_actions"
        in failure_codes
    )
    assert (
        f"decision_template_priority_{priority}_duplicate_blocked_live_actions"
        in failure_codes
    )
    assert (
        f"decision_template_priority_{priority}_live_authority_boundary_summary_mismatch"
        in failure_codes
    )
    assert (
        "decision_template_priority_"
        f"{priority}_missing_live_authority_boundary_summary"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_unknown_live_authority_boundary_summary"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_duplicate_live_authority_boundary_summary"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_operator_review_record_required_fields_mismatch"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_missing_operator_review_record_required_fields"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_unknown_operator_review_record_required_fields"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_duplicate_operator_review_record_required_fields"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_required_confirmation_details_mismatch"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_operator_acceptance_gate_details_mismatch"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_operator_review_decision_option_details_mismatch"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_operator_review_checklist_details_mismatch"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_operator_review_record_field_details_mismatch"
    ) in failure_codes
    assert (
        "decision_template_priority_"
        f"{priority}_operator_next_action_details_mismatch"
    ) in failure_codes
    assert "operator_decision_template_sha256_mismatch" not in failure_codes
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["live_authority_granted"] is False


def test_correction_method_goal_operator_review_export_verifier_rejects_missing_markdown_decision_workflow(tmp_path):
    run_id = "20260622T121502Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T12:15:02Z",
    )
    run_dir = tmp_path / run_id
    markdown_path = run_dir / "operator-review.md"
    manifest_path = run_dir / "artifact-manifest.json"

    markdown_path.write_text(
        markdown_path.read_text(encoding="utf-8").replace(
            "## Operator Decision Workflow",
            "## Decision Workflow Removed",
            1,
        ),
        encoding="utf-8",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    markdown_row = next(
        row
        for row in manifest["artifacts"]
        if row["artifact_id"] == "operator_review_markdown"
    )
    markdown_row["sha256"] = hashlib.sha256(markdown_path.read_bytes()).hexdigest()
    markdown_row["bytes"] = markdown_path.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_correction_method_operator_review_export(run_dir)
    failure_codes = {item["code"] for item in report["failed_checks"]}

    assert report["status"] == "failed"
    assert "markdown_missing_operator_decision_workflow" in failure_codes
    assert "operator_review_markdown_sha256_mismatch" not in failure_codes
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["live_authority_granted"] is False


def test_correction_method_goal_operator_review_export_verifier_rejects_markdown_workflow_drift(tmp_path):
    run_id = "20260622T175831Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T17:58:31Z",
    )
    run_dir = tmp_path / run_id
    markdown_path = run_dir / "operator-review.md"
    manifest_path = run_dir / "artifact-manifest.json"

    markdown = markdown_path.read_text(encoding="utf-8")
    start = markdown.index("## Operator Decision Workflow")
    end = markdown.index("\n### Decision Section 1:", start)
    block = markdown[start:end]
    replacements = {
        (
            "- required_priority_operator_decision: "
            "`one_offline_decision_per_priority_section`"
        ): "- required_priority_operator_decision: `none`",
        (
            "- required_operator_decisions: "
            "`one_offline_decision_per_method`"
        ): "- required_operator_decisions: `none`",
        "- live_authority_granted: `false`": "- live_authority_granted: `none`",
        "- live_apply_allowed: `false`": "- live_apply_allowed: `none`",
        "- queue_import_allowed: `false`": "- queue_import_allowed: `none`",
        "- can_apply_live: `false`": "- can_apply_live: `none`",
        "- can_import_to_queue: `false`": "- can_import_to_queue: `none`",
        "- can_authorize_live: `false`": "- can_authorize_live: `none`",
        "- can_close_readiness: `false`": "- can_close_readiness: `none`",
        "- blocks_live_promotion: `true`": "- blocks_live_promotion: `none`",
        "- blocks_queue_import: `true`": "- blocks_queue_import: `none`",
        "- full_autonomous_ready: `false`": "- full_autonomous_ready: `none`",
        "- queue_action: `not_queueable_until_promoted`": "- queue_action: `none`",
    }
    for expected_line, forged_line in replacements.items():
        block = block.replace(expected_line, forged_line, 1)
    markdown_path.write_text(
        markdown[:start] + block + markdown[end:],
        encoding="utf-8",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    markdown_row = next(
        row
        for row in manifest["artifacts"]
        if row["artifact_id"] == "operator_review_markdown"
    )
    markdown_row["sha256"] = hashlib.sha256(markdown_path.read_bytes()).hexdigest()
    markdown_row["bytes"] = markdown_path.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_correction_method_operator_review_export(run_dir)
    failure_codes = {item["code"] for item in report["failed_checks"]}

    assert report["status"] == "failed"
    assert (
        "markdown_operator_decision_workflow_required_priority_operator_decision_mismatch"
        in failure_codes
    )
    assert (
        "markdown_operator_decision_workflow_required_operator_decisions_mismatch"
        in failure_codes
    )
    assert "markdown_operator_decision_workflow_live_authority_granted_mismatch" in failure_codes
    assert "markdown_operator_decision_workflow_live_apply_allowed_mismatch" in failure_codes
    assert "markdown_operator_decision_workflow_queue_import_allowed_mismatch" in failure_codes
    assert "markdown_operator_decision_workflow_can_apply_live_mismatch" in failure_codes
    assert "markdown_operator_decision_workflow_can_import_to_queue_mismatch" in failure_codes
    assert "markdown_operator_decision_workflow_can_authorize_live_mismatch" in failure_codes
    assert "markdown_operator_decision_workflow_can_close_readiness_mismatch" in failure_codes
    assert "markdown_operator_decision_workflow_blocks_live_promotion_mismatch" in failure_codes
    assert "markdown_operator_decision_workflow_blocks_queue_import_mismatch" in failure_codes
    assert "markdown_operator_decision_workflow_full_autonomous_ready_mismatch" in failure_codes
    assert "markdown_operator_decision_workflow_queue_action_mismatch" in failure_codes
    assert "operator_review_markdown_sha256_mismatch" not in failure_codes
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["live_authority_granted"] is False


def test_correction_method_goal_operator_review_export_verifier_rejects_markdown_boundary_drift(tmp_path):
    run_id = "20260622T121505Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T12:15:05Z",
    )
    run_dir = tmp_path / run_id
    markdown_path = run_dir / "operator-review.md"
    manifest_path = run_dir / "artifact-manifest.json"

    markdown = markdown_path.read_text(encoding="utf-8")
    markdown = markdown.replace(
        "- denial_reason: `operator_review_packet_cannot_replace_real_wing_proof`",
        "- denial_reason: `tampered_denial_reason`",
        1,
    )
    markdown = markdown.replace(
        "review_offline_method_cards_and_collect_real_wing_readback_before_live_promotion",
        "tampered_operator_next_step",
        1,
    )
    markdown_path.write_text(markdown, encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    markdown_row = next(
        row
        for row in manifest["artifacts"]
        if row["artifact_id"] == "operator_review_markdown"
    )
    markdown_row["sha256"] = hashlib.sha256(markdown_path.read_bytes()).hexdigest()
    markdown_row["bytes"] = markdown_path.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_correction_method_operator_review_export(run_dir)
    failure_codes = {item["code"] for item in report["failed_checks"]}

    assert report["status"] == "failed"
    assert "markdown_denial_reason_mismatch" in failure_codes
    assert "markdown_operator_next_step_mismatch" in failure_codes
    assert "operator_review_markdown_sha256_mismatch" not in failure_codes
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["live_authority_granted"] is False


def test_correction_method_goal_operator_review_export_verifier_rejects_markdown_summary_drift(tmp_path):
    run_id = "20260622T121506Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T12:15:06Z",
    )
    run_dir = tmp_path / run_id
    packet_path = run_dir / "packet.json"
    markdown_path = run_dir / "operator-review.md"
    manifest_path = run_dir / "artifact-manifest.json"

    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    summary = packet["summary"]
    markdown = markdown_path.read_text(encoding="utf-8")
    markdown = markdown.replace(
        f"- method_review_card_count: `{summary['method_review_card_count']}`",
        "- method_review_card_count: `1`",
        1,
    )
    markdown = markdown.replace(
        f"- first_priority_method_ids: `{summary['first_priority_method_ids']}`",
        "- first_priority_method_ids: `['forged_method']`",
        1,
    )
    markdown = markdown.replace(
        f"- live_authority_method_ids: `{summary['live_authority_method_ids']}`",
        "- live_authority_method_ids: `['forged_live_method']`",
        1,
    )
    markdown_path.write_text(markdown, encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    markdown_row = next(
        row
        for row in manifest["artifacts"]
        if row["artifact_id"] == "operator_review_markdown"
    )
    markdown_row["sha256"] = hashlib.sha256(markdown_path.read_bytes()).hexdigest()
    markdown_row["bytes"] = markdown_path.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_correction_method_operator_review_export(run_dir)
    failure_codes = {item["code"] for item in report["failed_checks"]}

    assert report["status"] == "failed"
    assert "markdown_summary_method_review_card_count_mismatch" in failure_codes
    assert "markdown_summary_first_priority_method_ids_mismatch" in failure_codes
    assert "markdown_summary_live_authority_method_ids_mismatch" in failure_codes
    assert "operator_review_markdown_sha256_mismatch" not in failure_codes
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["live_authority_granted"] is False


def test_correction_method_goal_operator_review_export_verifier_rejects_markdown_priority_review_drift(tmp_path):
    run_id = "20260622T121507Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T12:15:07Z",
    )
    run_dir = tmp_path / run_id
    packet_path = run_dir / "packet.json"
    markdown_path = run_dir / "operator-review.md"
    manifest_path = run_dir / "artifact-manifest.json"

    def _md_items(values):
        return ", ".join(f"`{value}`" for value in values)

    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    first_priority = packet["first_priority_review_summary"]
    markdown = markdown_path.read_text(encoding="utf-8")
    start = markdown.index("## First Priority Review")
    end = markdown.index("\n## Second Priority Review", start)
    block = markdown[start:end]
    block = block.replace(
        f"- label: `{first_priority['label']}`",
        "- label: `Forged Priority Label`",
        1,
    )
    block = block.replace(
        f"- method_ids: {_md_items(first_priority['method_ids'])}",
        "- method_ids: `forged_method`",
        1,
    )
    block = block.replace(
        f"- operator_next_step: `{first_priority['operator_next_step']}`",
        "- operator_next_step: `forged_priority_next_step`",
        1,
    )
    block = block.replace(
        "- blocks_live_promotion: `true`",
        "- blocks_live_promotion: `false`",
        1,
    )
    block = block.replace(
        "- can_apply_live: `false`",
        "- can_apply_live: `true`",
        1,
    )
    block = block.replace(
        "- queue_action: `not_queueable_until_promoted`",
        "- queue_action: `queue_now`",
        1,
    )
    markdown_path.write_text(
        markdown[:start] + block + markdown[end:],
        encoding="utf-8",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    markdown_row = next(
        row
        for row in manifest["artifacts"]
        if row["artifact_id"] == "operator_review_markdown"
    )
    markdown_row["sha256"] = hashlib.sha256(markdown_path.read_bytes()).hexdigest()
    markdown_row["bytes"] = markdown_path.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_correction_method_operator_review_export(run_dir)
    failure_codes = {item["code"] for item in report["failed_checks"]}

    assert report["status"] == "failed"
    assert "markdown_first_priority_label_mismatch" in failure_codes
    assert "markdown_first_priority_method_ids_mismatch" in failure_codes
    assert "markdown_first_priority_operator_next_step_mismatch" in failure_codes
    assert "markdown_first_priority_blocks_live_promotion_mismatch" in failure_codes
    assert "markdown_first_priority_can_apply_live_mismatch" in failure_codes
    assert "markdown_first_priority_queue_action_mismatch" in failure_codes
    assert "operator_review_markdown_sha256_mismatch" not in failure_codes
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["live_authority_granted"] is False


def test_correction_method_goal_operator_review_export_verifier_rejects_markdown_decision_section_drift(tmp_path):
    run_id = "20260622T174815Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T17:48:15Z",
    )
    run_dir = tmp_path / run_id
    markdown_path = run_dir / "operator-review.md"
    manifest_path = run_dir / "artifact-manifest.json"

    markdown = markdown_path.read_text(encoding="utf-8")
    start = markdown.index("### Decision Section 1: Gain / Trim and Fader / Balance")
    end = markdown.index("\n### Decision Section 2: EQ and Compressor", start)
    block = markdown[start:end]
    tampered_lines = []
    for line in block.splitlines():
        if line.startswith("- method_ids: "):
            tampered_lines.append("- method_ids: `forged_decision_method`")
        elif line.startswith("- allowed_decisions: "):
            tampered_lines.append("- allowed_decisions: `accept_as_live`")
        elif line.startswith("- required_evidence: "):
            tampered_lines.append("- required_evidence: `none`")
        elif line.startswith("- required_safety_checks: "):
            tampered_lines.append("- required_safety_checks: `none`")
        elif line.startswith("- expected_evidence_packet_reference: "):
            tampered_lines.append(
                "- expected_evidence_packet_reference: "
                "`packet.json#priority_sections/999`"
            )
        elif line.startswith("- expected_source_material_reference: "):
            tampered_lines.append(
                "- expected_source_material_reference: "
                "`packet.json#priority_sections/999/source_evidence`"
            )
        elif line.startswith("- expected_test_evidence_references: "):
            tampered_lines.append(
                "- expected_test_evidence_references: `tests/test_stale.py`"
            )
        elif line.startswith("- expected_verification_report_reference: "):
            tampered_lines.append(
                "- expected_verification_report_reference: "
                "`stale-verification-report.json`"
            )
        elif line.startswith("- goal_requirement_details: "):
            tampered_lines.append(
                "- goal_requirement_details: "
                "`source_evidence:source=live_console;live_apply_allowed=true`"
            )
        elif line.startswith("- evidence_problem_correction_chain_details: "):
            tampered_lines.append(
                "- evidence_problem_correction_chain_details: "
                "`source_evidence:source=live_console;live_apply_allowed=true`"
            )
        elif line.startswith("- confidence: "):
            tampered_lines.append("- confidence: `none`")
        elif line.startswith("- operator_readable_explanation: "):
            tampered_lines.append("- operator_readable_explanation: `none`")
        elif line.startswith("- operator_visible_explanation_contract: "):
            tampered_lines.append("- operator_visible_explanation_contract: `none`")
        elif line.startswith("- operator_payload_required_fields: "):
            tampered_lines.append("- operator_payload_required_fields: `none`")
        elif line.startswith("- operator_evidence_required_fields: "):
            tampered_lines.append("- operator_evidence_required_fields: `none`")
        elif line.startswith("- operator_problem_review_summary: "):
            tampered_lines.append("- operator_problem_review_summary: `none`")
        elif line.startswith("- operator_method_review_focus: "):
            tampered_lines.append("- operator_method_review_focus: `none`")
        elif line.startswith("- why_safe_summary: "):
            tampered_lines.append("- why_safe_summary: `live_apply_allowed=true`")
        elif line.startswith("- why_blocked_summary: "):
            tampered_lines.append(
                "- why_blocked_summary: `future_wing_readback_required=false`"
            )
        elif line.startswith("- bounded_delta_policies: "):
            tampered_lines.append("- bounded_delta_policies: `unbounded_live_delta`")
        elif line.startswith("- rollback_expectations: "):
            tampered_lines.append("- rollback_expectations: `no_rollback_required`")
        elif line.startswith("- safety_blocker_details: "):
            tampered_lines.append(
                "- safety_blocker_details: "
                "`headroom:source=live_console;live_apply_allowed=true`"
            )
        elif line.startswith("- headroom_feedback_true_peak_check_details: "):
            tampered_lines.append(
                "- headroom_feedback_true_peak_check_details: "
                "`headroom:source=live_console;live_apply_allowed=true`"
            )
        elif line.startswith("- future_wing_readback_contract_summary: "):
            tampered_lines.append(
                "- future_wing_readback_contract_summary: "
                "`target_kind=unverified_live_write`"
            )
        elif line.startswith("- future_live_requirement_details: "):
            tampered_lines.append(
                "- future_live_requirement_details: "
                "`real_wing_readonly_connection_artifact:source=live_console;"
                "live_apply_allowed=true`"
            )
        elif line.startswith("- blocked_live_actions: "):
            tampered_lines.append("- blocked_live_actions: `none`")
        elif line.startswith("- live_authority_boundary_summary: "):
            tampered_lines.append(
                "- live_authority_boundary_summary: "
                "`live_authority_granted=true`"
            )
        elif line.startswith("- operator_review_record_required_fields: "):
            tampered_lines.append(
                "- operator_review_record_required_fields: `none`"
            )
        elif line.startswith("- required_confirmation_details: "):
            tampered_lines.append(
                "- required_confirmation_details: "
                "`live_apply_and_queue_import_remain_blocked:"
                "source=live_console;action_type=write`"
            )
        elif line.startswith("- operator_acceptance_gate_details: "):
            tampered_lines.append(
                "- operator_acceptance_gate_details: "
                "`live_apply_and_queue_import_remain_blocked:"
                "source=live_console;live_apply_allowed=true`"
            )
        elif line.startswith("- operator_review_decision_option_details: "):
            tampered_lines.append(
                "- operator_review_decision_option_details: "
                "`accept_as_offline_proposal:effect=authorize_live_now;"
                "live_apply_allowed=true`"
            )
        elif line.startswith("- operator_review_checklist_details: "):
            tampered_lines.append(
                "- operator_review_checklist_details: "
                "`do_not_queue_or_apply_live:source=live_console;action_type=write`"
            )
        elif line.startswith("- operator_review_record_field_details: "):
            tampered_lines.append(
                "- operator_review_record_field_details: "
                "`reviewer:source=live_console;type=write`"
            )
        elif line.startswith("- operator_next_action_details: "):
            tampered_lines.append(
                "- operator_next_action_details: "
                "`do_not_queue_or_apply_live:source=live_console;"
                "live_apply_allowed=true`"
            )
        elif line.startswith("- missing_required_operator_surface_method_ids: "):
            tampered_lines.append(
                "- missing_required_operator_surface_method_ids: `lufs_gain_staging`"
            )
        elif line.startswith("- live_authority_method_ids: "):
            tampered_lines.append("- live_authority_method_ids: `lufs_gain_staging`")
        elif line.startswith("- live_apply_method_ids: "):
            tampered_lines.append("- live_apply_method_ids: `lufs_gain_staging`")
        elif line.startswith("- queueable_method_ids: "):
            tampered_lines.append("- queueable_method_ids: `lufs_gain_staging`")
        elif line.startswith("- operator_next_step: "):
            tampered_lines.append("- operator_next_step: `apply_live_now`")
        elif line.startswith("- blocks_live_promotion: "):
            tampered_lines.append("- blocks_live_promotion: `false`")
        elif line.startswith("- blocks_queue_import: "):
            tampered_lines.append("- blocks_queue_import: `false`")
        elif line.startswith("- full_autonomous_ready: "):
            tampered_lines.append("- full_autonomous_ready: `true`")
        elif line.startswith("- can_apply_live: "):
            tampered_lines.append("- can_apply_live: `true`")
        elif line.startswith("- can_import_to_queue: "):
            tampered_lines.append("- can_import_to_queue: `true`")
        elif line.startswith("- can_close_readiness: "):
            tampered_lines.append("- can_close_readiness: `true`")
        elif line.startswith("- queue_action: "):
            tampered_lines.append("- queue_action: `queue_now`")
        elif line.startswith("- contacts_wing: "):
            tampered_lines.append("- contacts_wing: `true`")
        elif line.startswith("- contacts_backend: "):
            tampered_lines.append("- contacts_backend: `true`")
        elif line.startswith("- mutates_mixer: "):
            tampered_lines.append("- mutates_mixer: `true`")
        else:
            tampered_lines.append(line)
    block = "\n".join(tampered_lines) + "\n"
    markdown_path.write_text(
        markdown[:start] + block + markdown[end:],
        encoding="utf-8",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    markdown_row = next(
        row
        for row in manifest["artifacts"]
        if row["artifact_id"] == "operator_review_markdown"
    )
    markdown_row["sha256"] = hashlib.sha256(markdown_path.read_bytes()).hexdigest()
    markdown_row["bytes"] = markdown_path.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_correction_method_operator_review_export(run_dir)
    failure_codes = {item["code"] for item in report["failed_checks"]}

    assert report["status"] == "failed"
    assert "markdown_decision_section_1_method_ids_mismatch" in failure_codes
    assert "markdown_decision_section_1_allowed_decisions_mismatch" in failure_codes
    assert "markdown_decision_section_1_required_evidence_mismatch" in failure_codes
    assert (
        "markdown_decision_section_1_required_safety_checks_mismatch"
        in failure_codes
    )
    assert (
        "markdown_decision_section_1_expected_evidence_packet_reference_mismatch"
        in failure_codes
    )
    assert (
        "markdown_decision_section_1_expected_source_material_reference_mismatch"
        in failure_codes
    )
    assert (
        "markdown_decision_section_1_expected_test_evidence_references_mismatch"
        in failure_codes
    )
    assert (
        "markdown_decision_section_1_expected_verification_report_reference_mismatch"
        in failure_codes
    )
    assert "markdown_decision_section_1_confidence_mismatch" in failure_codes
    assert (
        "markdown_decision_section_1_operator_readable_explanation_mismatch"
        in failure_codes
    )
    assert (
        "markdown_decision_section_1_operator_visible_explanation_contract_mismatch"
        in failure_codes
    )
    assert (
        "markdown_decision_section_1_operator_payload_required_fields_mismatch"
        in failure_codes
    )
    assert (
        "markdown_decision_section_1_operator_evidence_required_fields_mismatch"
        in failure_codes
    )
    assert (
        "markdown_decision_section_1_operator_problem_review_summary_mismatch"
        in failure_codes
    )
    assert (
        "markdown_decision_section_1_operator_method_review_focus_mismatch"
        in failure_codes
    )
    assert (
        "markdown_decision_section_1_why_safe_summary_mismatch"
        in failure_codes
    )
    assert (
        "markdown_decision_section_1_why_blocked_summary_mismatch"
        in failure_codes
    )
    assert (
        "markdown_decision_section_1_bounded_delta_policies_mismatch"
        in failure_codes
    )
    assert (
        "markdown_decision_section_1_rollback_expectations_mismatch"
        in failure_codes
    )
    assert (
        "markdown_decision_section_1_future_wing_readback_contract_summary_mismatch"
        in failure_codes
    )
    assert (
        "markdown_decision_section_1_future_live_requirement_details_mismatch"
        in failure_codes
    )
    assert (
        "markdown_decision_section_1_goal_requirement_details_mismatch"
        in failure_codes
    )
    assert (
        "markdown_decision_section_1_"
        "evidence_problem_correction_chain_details_mismatch"
        in failure_codes
    )
    assert "markdown_decision_section_1_safety_blocker_details_mismatch" in (
        failure_codes
    )
    assert (
        "markdown_decision_section_1_"
        "headroom_feedback_true_peak_check_details_mismatch"
        in failure_codes
    )
    assert "markdown_decision_section_1_blocked_live_actions_mismatch" in failure_codes
    assert (
        "markdown_decision_section_1_live_authority_boundary_summary_mismatch"
        in failure_codes
    )
    assert (
        "markdown_decision_section_1_operator_review_record_required_fields_mismatch"
        in failure_codes
    )
    assert (
        "markdown_decision_section_1_required_confirmation_details_mismatch"
        in failure_codes
    )
    assert (
        "markdown_decision_section_1_operator_acceptance_gate_details_mismatch"
        in failure_codes
    )
    assert (
        "markdown_decision_section_1_"
        "operator_review_decision_option_details_mismatch"
        in failure_codes
    )
    assert (
        "markdown_decision_section_1_operator_review_checklist_details_mismatch"
        in failure_codes
    )
    assert (
        "markdown_decision_section_1_operator_review_record_field_details_mismatch"
        in failure_codes
    )
    assert (
        "markdown_decision_section_1_operator_next_action_details_mismatch"
        in failure_codes
    )
    assert (
        "markdown_decision_section_1_"
        "missing_required_operator_surface_method_ids_mismatch"
    ) in failure_codes
    assert "markdown_decision_section_1_live_authority_method_ids_mismatch" in (
        failure_codes
    )
    assert "markdown_decision_section_1_live_apply_method_ids_mismatch" in (
        failure_codes
    )
    assert "markdown_decision_section_1_queueable_method_ids_mismatch" in (
        failure_codes
    )
    assert "markdown_decision_section_1_operator_next_step_mismatch" in failure_codes
    assert "markdown_decision_section_1_blocks_live_promotion_mismatch" in failure_codes
    assert "markdown_decision_section_1_blocks_queue_import_mismatch" in failure_codes
    assert "markdown_decision_section_1_full_autonomous_ready_mismatch" in failure_codes
    assert "markdown_decision_section_1_can_apply_live_mismatch" in failure_codes
    assert "markdown_decision_section_1_can_import_to_queue_mismatch" in failure_codes
    assert "markdown_decision_section_1_can_close_readiness_mismatch" in failure_codes
    assert "markdown_decision_section_1_queue_action_mismatch" in failure_codes
    assert "markdown_decision_section_1_contacts_wing_mismatch" in failure_codes
    assert "markdown_decision_section_1_contacts_backend_mismatch" in failure_codes
    assert "markdown_decision_section_1_mutates_mixer_mismatch" in failure_codes
    assert "operator_review_markdown_sha256_mismatch" not in failure_codes
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["live_authority_granted"] is False


def test_correction_method_goal_operator_review_export_verifier_rejects_tampered_method_card_markdown_surfaces(tmp_path):
    run_id = "20260622T121516Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T12:15:16Z",
    )
    run_dir = tmp_path / run_id
    packet_path = run_dir / "packet.json"
    markdown_path = run_dir / "operator-review.md"
    manifest_path = run_dir / "artifact-manifest.json"

    def _md_code(value):
        text = "none" if value in (None, "") else str(value)
        escaped = text.replace("`", "\\`")
        return f"`{escaped}`"

    def _md_items(values):
        if not values:
            return "`none`"
        return ", ".join(_md_code(value) for value in values)

    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    card = packet["method_review_cards"][0]
    method_id = card["method_id"]
    heading = f"#### {card['method_label']} ({_md_code(method_id)})"

    markdown = markdown_path.read_text(encoding="utf-8")
    start = markdown.index(heading)
    end = markdown.find("\n#### ", start + len(heading))
    if end == -1:
        end = len(markdown)
    card_block = markdown[start:end]
    removed_lines = {
        "required_evidence": f"- required_evidence: {_md_items(card['required_evidence'])}",
        "operator_method_review_focus": (
            "- operator_method_review_focus: "
            f"{_md_items(card['operator_method_review_focus'])}"
        ),
        "why_safe_summary": (
            "- why_safe_summary: "
            f"{_md_items(card['why_safe_summary'])}"
        ),
        "why_blocked_summary": (
            "- why_blocked_summary: "
            f"{_md_items(card['why_blocked_summary'])}"
        ),
        "safety_blocker_details": (
            "- safety_blocker_details: "
            f"{_md_items(card['safety_blocker_details'])}"
        ),
        "goal_requirement_details": (
            "- goal_requirement_details: "
            f"{_md_items(card['goal_requirement_details'])}"
        ),
        "evidence_problem_correction_chain_details": (
            "- evidence_problem_correction_chain_details: "
            f"{_md_items(card['evidence_problem_correction_chain_details'])}"
        ),
        "headroom_feedback_true_peak_check_details": (
            "- headroom_feedback_true_peak_check_details: "
            f"{_md_items(card['headroom_feedback_true_peak_check_details'])}"
        ),
        "must_confirm": f"- must_confirm: {_md_items(card['what_operator_must_confirm'])}",
        "required_confirmation_details": (
            "- required_confirmation_details: "
            f"{_md_items(card['required_confirmation_details'])}"
        ),
        "operator_acceptance_gate_details": (
            "- operator_acceptance_gate_details: "
            f"{_md_items(card['operator_acceptance_gate_details'])}"
        ),
        "operator_review_decision_option_details": (
            "- operator_review_decision_option_details: "
            f"{_md_items(card['operator_review_decision_option_details'])}"
        ),
        "operator_next_action_details": (
            "- operator_next_action_details: "
            f"{_md_items(card['operator_next_action_details'])}"
        ),
        "rollback_expectation": (
            f"- rollback_expectation: {_md_code(card['rollback_expectation'])}"
        ),
        "future_live_required_proofs": (
            "- future_live_required_proofs: "
            f"{_md_items(card['future_live_required_proofs'])}"
        ),
        "future_live_requirement_details": (
            "- future_live_requirement_details: "
            f"{_md_items(card['future_live_requirement_details'])}"
        ),
        "blocks_live_promotion": "- blocks_live_promotion: `true`",
        "blocks_queue_import": "- blocks_queue_import: `true`",
        "full_autonomous_ready": "- full_autonomous_ready: `false`",
        "can_apply_live": "- can_apply_live: `false`",
        "can_import_to_queue": "- can_import_to_queue: `false`",
        "can_close_readiness": "- can_close_readiness: `false`",
        "queue_action": "- queue_action: `not_queueable_until_promoted`",
    }
    for line in removed_lines.values():
        assert line in card_block
        card_block = card_block.replace(line + "\n", "", 1)
    markdown_path.write_text(
        markdown[:start] + card_block + markdown[end:],
        encoding="utf-8",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    markdown_row = next(
        row
        for row in manifest["artifacts"]
        if row["artifact_id"] == "operator_review_markdown"
    )
    markdown_row["sha256"] = hashlib.sha256(markdown_path.read_bytes()).hexdigest()
    markdown_row["bytes"] = markdown_path.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_correction_method_operator_review_export(run_dir)
    failure_codes = {item["code"] for item in report["failed_checks"]}

    assert report["status"] == "failed"
    for field in removed_lines:
        assert f"markdown_method_{method_id}_{field}_missing" in failure_codes
    assert "operator_review_markdown_sha256_mismatch" not in failure_codes
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["live_authority_granted"] is False


def test_correction_method_goal_operator_review_export_verifier_rejects_tampered_method_card_operator_explanation_markdown_surfaces(tmp_path):
    run_id = "20260622T180813Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T18:08:13Z",
    )
    run_dir = tmp_path / run_id
    packet_path = run_dir / "packet.json"
    markdown_path = run_dir / "operator-review.md"
    manifest_path = run_dir / "artifact-manifest.json"

    def _md_code(value):
        text = "none" if value in (None, "") else str(value)
        escaped = text.replace("`", "\\`")
        return f"`{escaped}`"

    def _md_items(values):
        if not values:
            return "`none`"
        return ", ".join(_md_code(value) for value in values)

    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    card = packet["method_review_cards"][0]
    method_id = card["method_id"]
    heading = f"#### {card['method_label']} ({_md_code(method_id)})"
    explanation_contract_items = [
        f"{key}={str(value).lower()}"
        for key, value in sorted(
            card["operator_visible_explanation_contract"].items()
        )
    ]

    markdown = markdown_path.read_text(encoding="utf-8")
    start = markdown.index(heading)
    end = markdown.find("\n#### ", start + len(heading))
    if end == -1:
        end = len(markdown)
    card_block = markdown[start:end]
    removed_lines = {
        "operator_question": f"- operator_question: {_md_code(card['operator_question'])}",
        "denial_reason": f"- denial_reason: {_md_code(card['denial_reason'])}",
        "source_evidence": (
            "- source_evidence: "
            f"{_md_items(_operator_review_contract_items(card['source_evidence']))}"
        ),
        "source_evidence_details": (
            "- source_evidence_details: "
            f"{_md_items(_operator_review_contract_items(card['source_evidence_details']))}"
        ),
        "bounded_delta": (
            "- bounded_delta_contract: "
            f"{_md_items(_operator_review_contract_items(card['bounded_delta']))}"
        ),
        "operator_problem_review_summary": (
            "- operator_problem_review_summary: "
            f"{_md_items(card['operator_problem_review_summary'])}"
        ),
        "operator_method_review_focus": (
            "- operator_method_review_focus: "
            f"{_md_items(card['operator_method_review_focus'])}"
        ),
        "why_safe_summary": (
            "- why_safe_summary: "
            f"{_md_items(card['why_safe_summary'])}"
        ),
        "why_blocked_summary": (
            "- why_blocked_summary: "
            f"{_md_items(card['why_blocked_summary'])}"
        ),
        "expected_evidence_packet_reference": (
            "- expected_evidence_packet_reference: "
            f"{_md_code(f'packet.json#method_review_cards/{method_id}')}"
        ),
        "expected_source_material_reference": (
            "- expected_source_material_reference: "
            f"{_md_code(f'packet.json#method_review_cards/{method_id}/source_evidence')}"
        ),
        "expected_test_evidence_references": (
            "- expected_test_evidence_references: "
            f"{_md_items(card['test_evidence_references'])}"
        ),
        "expected_verification_report_reference": (
            "- expected_verification_report_reference: "
            f"{_md_code('verification-report.json')}"
        ),
        "required_operator_surfaces": (
            "- required_operator_surfaces: "
            f"{_md_items(card['required_operator_surfaces'])}"
        ),
        "operator_payload_required_fields": (
            "- operator_payload_required_fields: "
            f"{_md_items(card['operator_payload_required_fields'])}"
        ),
        "operator_evidence_required_fields": (
            "- operator_evidence_required_fields: "
            f"{_md_items(card['operator_evidence_required_fields'])}"
        ),
        "operator_review_checklist": (
            "- operator_review_checklist: "
            f"{_md_items(card['operator_review_checklist'])}"
        ),
        "required_confirmation_details": (
            "- required_confirmation_details: "
            f"{_md_items(card['required_confirmation_details'])}"
        ),
        "operator_acceptance_gate_details": (
            "- operator_acceptance_gate_details: "
            f"{_md_items(card['operator_acceptance_gate_details'])}"
        ),
        "operator_review_decision_option_details": (
            "- operator_review_decision_option_details: "
            f"{_md_items(card['operator_review_decision_option_details'])}"
        ),
        "operator_review_checklist_details": (
            "- operator_review_checklist_details: "
            f"{_md_items(card['operator_review_checklist_details'])}"
        ),
        "operator_review_record_required_fields": (
            "- operator_review_record_required_fields: "
            f"{_md_items(card['operator_review_record_required_fields'])}"
        ),
        "operator_review_record_field_details": (
            "- operator_review_record_field_details: "
            f"{_md_items(card['operator_review_record_field_details'])}"
        ),
        "operator_next_action_details": (
            "- operator_next_action_details: "
            f"{_md_items(card['operator_next_action_details'])}"
        ),
        "operator_visible_explanation_contract": (
            "- operator_visible_explanation_contract: "
            f"{_md_items(explanation_contract_items)}"
        ),
        "future_wing_readback_contract_summary": (
            "- future_wing_readback_contract_summary: "
            f"{_md_items(card['future_wing_readback_contract_summary'])}"
        ),
        "rollback_readback_contract": (
            "- rollback_readback_contract: "
            f"{_md_items(_operator_review_contract_items(card['rollback_readback_contract']))}"
        ),
        "future_wing_readback_contract": (
            "- future_wing_readback_contract: "
            f"{_md_items(_operator_review_contract_items(card['future_wing_readback_contract']))}"
        ),
        "live_authority_boundary_summary": (
            "- live_authority_boundary_summary: "
            f"{_md_items(card['live_authority_boundary_summary'])}"
        ),
    }
    for line in removed_lines.values():
        assert line in card_block
        card_block = card_block.replace(line + "\n", "", 1)
    markdown_path.write_text(
        markdown[:start] + card_block + markdown[end:],
        encoding="utf-8",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    markdown_row = next(
        row
        for row in manifest["artifacts"]
        if row["artifact_id"] == "operator_review_markdown"
    )
    markdown_row["sha256"] = hashlib.sha256(markdown_path.read_bytes()).hexdigest()
    markdown_row["bytes"] = markdown_path.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_correction_method_operator_review_export(run_dir)
    failure_codes = {item["code"] for item in report["failed_checks"]}

    assert report["status"] == "failed"
    for field in removed_lines:
        assert f"markdown_method_{method_id}_{field}_missing" in failure_codes
    assert "operator_review_markdown_sha256_mismatch" not in failure_codes
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["live_authority_granted"] is False


def test_correction_method_goal_operator_review_export_verifier_rejects_tampered_required_operator_surface_markdown(tmp_path):
    run_id = "20260622T121518Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T12:15:18Z",
    )
    run_dir = tmp_path / run_id
    markdown_path = run_dir / "operator-review.md"
    manifest_path = run_dir / "artifact-manifest.json"

    removed_line = "- `operator_readable_explanation`\n"
    markdown = markdown_path.read_text(encoding="utf-8")
    assert removed_line in markdown
    markdown_path.write_text(
        markdown.replace(removed_line, "", 1),
        encoding="utf-8",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    markdown_row = next(
        row
        for row in manifest["artifacts"]
        if row["artifact_id"] == "operator_review_markdown"
    )
    markdown_row["sha256"] = hashlib.sha256(markdown_path.read_bytes()).hexdigest()
    markdown_row["bytes"] = markdown_path.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_correction_method_operator_review_export(run_dir)
    failure_codes = {item["code"] for item in report["failed_checks"]}

    assert report["status"] == "failed"
    assert "markdown_missing_required_operator_surface" in failure_codes
    assert "operator_review_markdown_sha256_mismatch" not in failure_codes
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["live_authority_granted"] is False


def test_correction_method_goal_operator_review_export_verifier_rejects_manifest_decision_summary_authority_claim(tmp_path):
    run_id = "20260622T121503Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T12:15:03Z",
    )
    run_dir = tmp_path / run_id
    manifest_path = run_dir / "artifact-manifest.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = manifest["operator_decision_template_summary"]
    summary["live_authority_granted"] = True
    summary["can_authorize_live"] = True
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_correction_method_operator_review_export(run_dir)
    failure_codes = {item["code"] for item in report["failed_checks"]}

    assert report["status"] == "failed"
    assert (
        "manifest_decision_template_summary_live_authority_granted_not_false"
        in failure_codes
    )
    assert (
        "manifest_decision_template_summary_can_authorize_live_not_false"
        in failure_codes
    )
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["contacts_backend"] is False
    assert report["live_authority_granted"] is False


def test_correction_method_goal_operator_review_export_verifier_rejects_manifest_decision_summary_contract_count_drift(tmp_path):
    run_id = "20260622T121504Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T12:15:04Z",
    )
    run_dir = tmp_path / run_id
    manifest_path = run_dir / "artifact-manifest.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = manifest["operator_decision_template_summary"]
    summary["priority_required_evidence_counts"]["1"] = 1
    summary["priority_source_evidence_counts"]["1"] = 1
    summary["priority_source_evidence_detail_counts"]["1"] = 1
    summary["priority_confidence_counts"]["1"] = 1
    summary["priority_bounded_delta_counts"]["1"] = 1
    summary["priority_future_wing_readback_contract_counts"]["1"] = 1
    summary["priority_test_evidence_counts"]["1"] = 1
    summary["priority_test_evidence_reference_counts"]["1"] = 1
    summary["priority_rollback_readback_contract_counts"]["1"] = 1
    summary["priority_rollback_expectation_counts"]["1"] = 1
    summary["priority_operator_payload_required_field_counts"]["1"] = 1
    summary["priority_operator_method_review_focus_counts"]["1"] = 1
    summary["priority_operator_visible_explanation_contract_counts"]["1"] = 1
    summary["priority_future_wing_readback_contract_summary_counts"]["1"] = 1
    summary["priority_future_live_requirement_detail_counts"]["1"] = 1
    summary["priority_goal_requirement_detail_counts"]["1"] = 1
    summary["priority_evidence_problem_correction_chain_detail_counts"]["1"] = 1
    summary["priority_safety_blocker_detail_counts"]["1"] = 1
    summary["priority_headroom_feedback_true_peak_check_detail_counts"]["1"] = 1
    summary["priority_required_confirmation_detail_counts"]["1"] = 1
    summary["priority_operator_acceptance_gate_detail_counts"]["1"] = 1
    summary["priority_operator_review_decision_option_detail_counts"]["1"] = 1
    summary["priority_operator_review_checklist_detail_counts"]["1"] = 1
    summary["priority_operator_review_record_field_detail_counts"]["1"] = 1
    summary["priority_operator_next_action_detail_counts"]["1"] = 1
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_correction_method_operator_review_export(run_dir)
    failure_codes = {item["code"] for item in report["failed_checks"]}

    assert report["status"] == "failed"
    assert (
        "manifest_decision_template_summary_"
        "priority_required_evidence_counts_mismatch"
    ) in failure_codes
    assert (
        "manifest_decision_template_summary_"
        "priority_source_evidence_counts_mismatch"
    ) in failure_codes
    assert (
        "manifest_decision_template_summary_"
        "priority_source_evidence_detail_counts_mismatch"
    ) in failure_codes
    assert (
        "manifest_decision_template_summary_"
        "priority_confidence_counts_mismatch"
    ) in failure_codes
    assert (
        "manifest_decision_template_summary_"
        "priority_bounded_delta_counts_mismatch"
    ) in failure_codes
    assert (
        "manifest_decision_template_summary_"
        "priority_future_wing_readback_contract_counts_mismatch"
    ) in failure_codes
    assert (
        "manifest_decision_template_summary_"
        "priority_test_evidence_counts_mismatch"
    ) in failure_codes
    assert (
        "manifest_decision_template_summary_"
        "priority_test_evidence_reference_counts_mismatch"
    ) in failure_codes
    assert (
        "manifest_decision_template_summary_"
        "priority_rollback_readback_contract_counts_mismatch"
    ) in failure_codes
    assert (
        "manifest_decision_template_summary_"
        "priority_rollback_expectation_counts_mismatch"
    ) in failure_codes
    assert (
        "manifest_decision_template_summary_"
        "priority_operator_payload_required_field_counts_mismatch"
    ) in failure_codes
    assert (
        "manifest_decision_template_summary_"
        "priority_operator_method_review_focus_counts_mismatch"
    ) in failure_codes
    assert (
        "manifest_decision_template_summary_"
        "priority_operator_visible_explanation_contract_counts_mismatch"
    ) in failure_codes
    assert (
        "manifest_decision_template_summary_"
        "priority_future_wing_readback_contract_summary_counts_mismatch"
    ) in failure_codes
    assert (
        "manifest_decision_template_summary_"
        "priority_future_live_requirement_detail_counts_mismatch"
    ) in failure_codes
    assert (
        "manifest_decision_template_summary_"
        "priority_goal_requirement_detail_counts_mismatch"
    ) in failure_codes
    assert (
        "manifest_decision_template_summary_"
        "priority_evidence_problem_correction_chain_detail_counts_mismatch"
    ) in failure_codes
    assert (
        "manifest_decision_template_summary_"
        "priority_safety_blocker_detail_counts_mismatch"
    ) in failure_codes
    assert (
        "manifest_decision_template_summary_"
        "priority_headroom_feedback_true_peak_check_detail_counts_mismatch"
    ) in failure_codes
    assert (
        "manifest_decision_template_summary_"
        "priority_required_confirmation_detail_counts_mismatch"
    ) in failure_codes
    assert (
        "manifest_decision_template_summary_"
        "priority_operator_acceptance_gate_detail_counts_mismatch"
    ) in failure_codes
    assert (
        "manifest_decision_template_summary_"
        "priority_operator_review_decision_option_detail_counts_mismatch"
    ) in failure_codes
    assert (
        "manifest_decision_template_summary_"
        "priority_operator_review_checklist_detail_counts_mismatch"
    ) in failure_codes
    assert (
        "manifest_decision_template_summary_"
        "priority_operator_review_record_field_detail_counts_mismatch"
    ) in failure_codes
    assert (
        "manifest_decision_template_summary_"
        "priority_operator_next_action_detail_counts_mismatch"
    ) in failure_codes
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["live_authority_granted"] is False


def test_correction_method_goal_operator_review_export_verifier_rejects_manifest_summary_count_drift(tmp_path):
    run_id = "20260622T121521Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T12:15:21Z",
    )
    run_dir = tmp_path / run_id
    manifest_path = run_dir / "artifact-manifest.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = manifest["summary"]
    summary["required_objective_family_count"] = 1
    summary["covered_objective_family_count"] = 1
    summary["method_review_card_count"] = 1
    summary["supporting_method_review_card_count"] = 0
    summary["operator_review_contract_coverage"]["confidence"]["covered"] = False
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_correction_method_operator_review_export(run_dir)
    failure_codes = {item["code"] for item in report["failed_checks"]}

    assert report["status"] == "failed"
    assert "manifest_summary_required_objective_family_count_mismatch" in failure_codes
    assert "manifest_summary_covered_objective_family_count_mismatch" in failure_codes
    assert "manifest_summary_method_review_card_count_mismatch" in failure_codes
    assert (
        "manifest_summary_supporting_method_review_card_count_mismatch"
        in failure_codes
    )
    assert "manifest_summary_operator_review_contract_coverage_mismatch" in failure_codes
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["live_authority_granted"] is False


def test_correction_method_goal_operator_review_export_verifier_rejects_manifest_priority_summary_drift(tmp_path):
    run_id = "20260622T121523Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T12:15:23Z",
    )
    run_dir = tmp_path / run_id
    manifest_path = run_dir / "artifact-manifest.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = manifest["summary"]
    summary["first_priority_method_count"] = 1
    summary["first_priority_method_ids"] = ["forged_method_id"]
    summary["first_priority_applicable_safety_checks"] = []
    summary["first_priority_future_wing_readback_target_count"] = 0
    summary["first_priority_blocked_live_action_count"] = 0
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_correction_method_operator_review_export(run_dir)
    failure_codes = {item["code"] for item in report["failed_checks"]}

    assert report["status"] == "failed"
    assert "manifest_summary_first_priority_method_count_mismatch" in failure_codes
    assert "manifest_summary_first_priority_method_ids_mismatch" in failure_codes
    assert (
        "manifest_summary_first_priority_applicable_safety_checks_mismatch"
        in failure_codes
    )
    assert (
        "manifest_summary_first_priority_future_wing_readback_target_count_mismatch"
        in failure_codes
    )
    assert (
        "manifest_summary_first_priority_blocked_live_action_count_mismatch"
        in failure_codes
    )
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["live_authority_granted"] is False


def test_correction_method_goal_operator_review_export_verifier_rejects_manifest_boundary_drift(tmp_path):
    run_id = "20260622T121524Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T12:15:24Z",
    )
    run_dir = tmp_path / run_id
    manifest_path = run_dir / "artifact-manifest.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["denial_reason"] = "tampered_denial_reason"
    manifest["operator_next_step"] = "tampered_operator_next_step"
    manifest["summary"]["blocked_by"] = ["tampered_blocker"]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_correction_method_operator_review_export(run_dir)
    failure_codes = {item["code"] for item in report["failed_checks"]}

    assert report["status"] == "failed"
    assert "manifest_denial_reason_mismatch" in failure_codes
    assert "manifest_operator_next_step_mismatch" in failure_codes
    assert "manifest_summary_blocked_by_mismatch" in failure_codes
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["live_authority_granted"] is False


def test_correction_method_goal_operator_review_export_verifier_rejects_packet_summary_count_drift(tmp_path):
    run_id = "20260622T121522Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T12:15:22Z",
    )
    run_dir = tmp_path / run_id
    packet_path = run_dir / "packet.json"
    manifest_path = run_dir / "artifact-manifest.json"

    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    summary = packet["summary"]
    summary["required_objective_family_count"] = 1
    summary["covered_objective_family_count"] = 1
    summary["method_review_card_count"] = 1
    summary["operator_review_contract_coverage"]["confidence"]["covered"] = False
    packet_path.write_text(
        json.dumps(packet, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packet_row = next(
        row for row in manifest["artifacts"] if row["artifact_id"] == "packet_json"
    )
    packet_row["sha256"] = hashlib.sha256(packet_path.read_bytes()).hexdigest()
    packet_row["bytes"] = packet_path.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_correction_method_operator_review_export(run_dir)
    failure_codes = {item["code"] for item in report["failed_checks"]}

    assert report["status"] == "failed"
    assert "packet_summary_required_objective_family_count_mismatch" in failure_codes
    assert "packet_summary_covered_objective_family_count_mismatch" in failure_codes
    assert "packet_summary_method_review_card_count_mismatch" in failure_codes
    assert (
        "packet_summary_confidence_contract_coverage_covered_mismatch"
        in failure_codes
    )
    assert "packet_json_sha256_mismatch" not in failure_codes
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["live_authority_granted"] is False


def test_correction_method_goal_operator_review_export_verifier_persists_report(tmp_path):
    run_id = "20260622T121504Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T12:15:04Z",
    )
    report = verify_correction_method_operator_review_export(tmp_path / run_id)
    report_path = tmp_path / run_id / "verification-report.json"

    written_path = write_correction_method_operator_review_verification_report(
        report,
        report_path,
    )
    disk_report = json.loads(report_path.read_text(encoding="utf-8"))

    assert written_path == report_path
    assert disk_report == report
    assert disk_report["status"] == "passed"
    assert disk_report["failed_checks"] == []
    assert disk_report["read_only"] is True
    assert disk_report["contacts_wing"] is False
    assert disk_report["contacts_backend"] is False
    assert disk_report["live_authority_granted"] is False
    assert disk_report["readiness_closure_allowed"] is False
    assert (
        disk_report["method_review_card_count"]
        == EXPECTED_METHOD_REVIEW_CARD_COUNT
    )
    assert disk_report["priority_review_summary_count"] == 7
    assert disk_report["priority_decision_section_count"] == 7


def test_correction_method_goal_operator_decision_record_validator_accepts_offline_decisions(tmp_path):
    run_id = "20260622T121505Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T12:15:05Z",
    )
    run_dir = tmp_path / run_id
    record_path = _write_filled_operator_decision_record(run_dir)

    report = validate_correction_method_operator_decision_record(
        run_dir,
        record_path,
    )
    report_path = run_dir / "decision-validation-report.json"
    written_path = write_correction_method_operator_decision_validation_report(
        report,
        report_path,
    )
    disk_report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["schema"] == (
        "automixer.correction_method_operator_decision_record_validation.v1"
    )
    assert report["status"] == "passed"
    assert report["failed_checks"] == []
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["contacts_backend"] is False
    assert report["queries_console"] is False
    assert report["mutates_console_state"] is False
    assert report["mutates_mixer"] is False
    assert report["live_write"] is False
    assert report["live_authority_granted"] is False
    assert report["readiness_closure_allowed"] is False
    assert report["live_apply_allowed"] is False
    assert report["queue_import_allowed"] is False
    assert report["can_authorize_live"] is False
    assert report["operator_review_required"] is True
    assert report["future_wing_readback_required"] is True
    assert report["hardware_proof_required_before_live"] is True
    assert report["source_authority_denied"] is True
    assert report["denial_reason"] == (
        "operator_decision_record_cannot_authorize_live_or_close_readiness"
    )
    assert report["operator_next_step"] == (
        "use_offline_decisions_for_engineer_review_and_collect_real_wing_"
        "readback_before_live_promotion"
    )
    assert report["method_decision_count"] == EXPECTED_METHOD_REVIEW_CARD_COUNT
    assert (
        report["accepted_offline_proposal_count"]
        == EXPECTED_METHOD_REVIEW_CARD_COUNT
    )
    assert report["decision_counts"] == {
        "accept_as_offline_proposal": EXPECTED_METHOD_REVIEW_CARD_COUNT,
        "defer_until_real_wing_proof": 0,
        "missing_operator_decision": 0,
        "reject_proposal": 0,
        "request_more_evidence": 0,
    }
    assert (
        len(report["method_ids_by_decision"]["accept_as_offline_proposal"])
        == EXPECTED_METHOD_REVIEW_CARD_COUNT
    )
    assert report["method_ids_by_decision"]["missing_operator_decision"] == []
    assert report["priority_decision_counts"] == {
        "accept_as_offline_proposal": 7,
        "defer_until_real_wing_proof": 0,
        "missing_operator_decision": 0,
        "reject_proposal": 0,
        "request_more_evidence": 0,
    }
    assert report["priority_ids_by_decision"]["accept_as_offline_proposal"] == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    ]
    assert report["priority_ids_by_decision"]["missing_operator_decision"] == []
    assert report["future_wing_readback_target_count"] > 0
    assert report["future_live_required_proof_count"] > 0
    assert report["blocked_live_action_count"] > 0
    assert report["priority_decision_section_count"] == 7
    assert len(report["validated_priority_decision_sections"]) == 7
    priority_section_method_count = sum(
        section["method_decision_count"]
        for section in report["validated_priority_decision_sections"]
    )
    assert sum(
        section["accepted_offline_proposal_count"]
        for section in report["validated_priority_decision_sections"]
    ) == priority_section_method_count
    assert [section["label"] for section in report["validated_priority_decision_sections"]] == [
        "Gain / Trim and Fader / Balance",
        "EQ and Compressor",
        "Gate / Expander / De-esser",
        "FX / Sends",
        "Pan / Stereo",
        "Routing / Groups / Scenes",
        "Master / Output",
    ]
    for section in report["validated_priority_decision_sections"]:
        assert section["review_scope"] == "offline_proposal_only"
        assert section["operator_decision"] == "accept_as_offline_proposal"
        assert section["method_decision_count"] >= 1
        assert section["accepted_offline_proposal_count"] == section[
            "method_decision_count"
        ]
        assert section["decision_counts"]["accept_as_offline_proposal"] == section[
            "method_decision_count"
        ]
        assert len(
            section["method_ids_by_decision"]["accept_as_offline_proposal"]
        ) == section["method_decision_count"]
        assert section["future_wing_readback_target_count"] > 0
        assert section["future_live_required_proof_count"] > 0
        assert section["blocked_live_action_count"] > 0
        assert section["operator_payload_required_field_count"] > 0
        assert section["operator_evidence_required_field_count"] > 0
        assert section["required_evidence_count"] > 0
        assert section["required_safety_check_count"] > 0
        assert section["bounded_delta_policy_count"] > 0
        assert section["rollback_expectation_count"] > 0
        assert set(section["operator_review_contract_counts"]) == set(
            OPERATOR_REVIEW_CONTRACT_FIELDS
        )
        assert all(
            count > 0
            for count in section["operator_review_contract_counts"].values()
        )
        assert section["operator_visible_explanation_contract_count"] > 0
        assert section["future_wing_readback_contract_summary_count"] > 0
        assert section["live_authority_boundary_summary_count"] > 0
        assert section["why_safe_summary_count"] > 0
        assert section["why_blocked_summary_count"] > 0
        assert section["required_confirmation_detail_count"] > 0
        assert section["operator_review_decision_option_detail_count"] > 0
        assert section["operator_review_checklist_detail_count"] > 0
        assert section["operator_review_record_required_field_count"] > 0
        assert section["operator_review_record_completed_field_count"] == len(
            FULL_OPERATOR_REVIEW_RECORD_FIELDS
        )
        assert section["read_only"] is True
        assert section["live_authority_granted"] is False
        assert section["readiness_closure_allowed"] is False
        assert section["live_apply_allowed"] is False
        assert section["queue_import_allowed"] is False
        assert section["can_authorize_live"] is False
    assert (
        len(report["validated_method_decisions"])
        == EXPECTED_METHOD_REVIEW_CARD_COUNT
    )
    for row in report["validated_method_decisions"]:
        assert row["operator_decision"] == "accept_as_offline_proposal"
        assert row["review_scope"] == "offline_proposal_only"
        assert row["read_only"] is True
        assert row["live_authority_granted"] is False
        assert row["readiness_closure_allowed"] is False
        assert row["live_apply_allowed"] is False
        assert row["queue_import_allowed"] is False
        assert row["can_authorize_live"] is False
        assert row["future_wing_readback_required"] is True
        assert row["future_wing_readback_target_count"] > 0
        assert row["operator_payload_required_field_count"] > 0
        assert row["operator_evidence_required_field_count"] > 0
        assert row["required_evidence_count"] > 0
        assert row["required_safety_check_count"] > 0
        assert set(row["operator_review_contract_counts"]) == set(
            OPERATOR_REVIEW_CONTRACT_FIELDS
        )
        assert all(
            count > 0
            for count in row["operator_review_contract_counts"].values()
        )
        assert row["operator_visible_explanation_contract_count"] > 0
        assert row["future_wing_readback_contract_summary_count"] > 0
        assert row["future_live_required_proof_count"] > 0
        assert row["blocked_live_action_count"] > 0
        assert row["live_authority_boundary_summary_count"] > 0
        assert row["why_safe_summary_count"] > 0
        assert row["why_blocked_summary_count"] > 0
        assert row["required_confirmation_detail_count"] > 0
        assert row["operator_review_decision_option_detail_count"] > 0
        assert row["operator_review_checklist_detail_count"] > 0
        assert row["operator_review_record_required_field_count"] > 0
        assert row["operator_review_record_completed_field_count"] == len(
            FULL_OPERATOR_REVIEW_RECORD_FIELDS
        )
    assert written_path == report_path
    assert disk_report == report


def test_correction_method_goal_operator_decision_record_validator_rejects_live_authority_claim(tmp_path):
    run_id = "20260622T121506Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T12:15:06Z",
    )
    run_dir = tmp_path / run_id
    record_path = _write_filled_operator_decision_record(
        run_dir,
        claim_live_authority=True,
    )

    report = validate_correction_method_operator_decision_record(
        run_dir,
        record_path,
    )
    failure_codes = {failure["code"] for failure in report["failed_checks"]}

    assert report["status"] == "failed"
    assert "decision_record_live_authority_granted_not_false" in failure_codes
    assert "decision_record_can_authorize_live_not_false" in failure_codes
    assert any(
        code.startswith("decision_record_")
        and code.endswith("_live_authority_granted_not_false")
        and code != "decision_record_live_authority_granted_not_false"
        for code in failure_codes
    )
    assert any(
        code.startswith("decision_record_")
        and code.endswith("_can_authorize_live_not_false")
        and code != "decision_record_can_authorize_live_not_false"
        for code in failure_codes
    )
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["contacts_backend"] is False
    assert report["live_authority_granted"] is False
    assert report["readiness_closure_allowed"] is False
    assert report["live_apply_allowed"] is False
    assert report["queue_import_allowed"] is False
    assert report["can_authorize_live"] is False
    assert report["operator_review_required"] is True
    assert report["future_wing_readback_required"] is True
    assert report["hardware_proof_required_before_live"] is True
    assert report["source_authority_denied"] is True
    assert report["denial_reason"] == (
        "operator_decision_record_cannot_authorize_live_or_close_readiness"
    )
    assert (
        report["accepted_offline_proposal_count"]
        == EXPECTED_METHOD_REVIEW_CARD_COUNT
    )


def test_correction_method_goal_operator_decision_record_validator_rejects_tampered_evidence_references(tmp_path):
    run_id = "20260622T121507Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T12:15:07Z",
    )
    run_dir = tmp_path / run_id
    record_path = _write_filled_operator_decision_record(run_dir)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    row = record["method_decisions"][0]
    method_id = row["method_id"]
    row["evidence_packet_reference"] = "packet.json#method_review_cards/stale"
    row["source_material_reference"] = (
        "packet.json#method_review_cards/stale/source_evidence"
    )
    row["test_evidence_reference"] = (
        "tests/test_processing_proposal_contracts.py::test_stale_reference"
    )
    row["verification_report_reference"] = "stale-verification-report.json"
    row["source_evidence_acknowledgement"] = (
        "operator_skipped_source_evidence_review"
    )
    row["test_evidence_acknowledgement"] = (
        "operator_skipped_test_evidence_review"
    )
    row["verification_report_acknowledgement"] = (
        "operator_skipped_verification_report_review"
    )
    row["measured_problem_acknowledgement"] = (
        "operator_skipped_measured_problem_review"
    )
    row["proposed_correction_acknowledgement"] = (
        "operator_skipped_proposed_correction_review"
    )
    row["bounded_delta_acknowledgement"] = (
        "operator_skipped_bounded_delta_review"
    )
    row["confidence_acknowledgement"] = "operator_skipped_confidence_review"
    row["operator_explanation_acknowledgement"] = (
        "operator_skipped_operator_explanation_review"
    )
    row["future_wing_readback_acknowledgement"] = (
        "operator_skipped_future_wing_readback_contract_review"
    )
    row["headroom_feedback_true_peak_acknowledgement"] = (
        "operator_skipped_headroom_feedback_true_peak_review"
    )
    row["rollback_readback_acknowledgement"] = (
        "operator_skipped_rollback_readback_review"
    )
    row["safety_blockers_acknowledgement"] = (
        "operator_skipped_safety_blockers"
    )
    row["required_confirmations_acknowledgement"] = (
        "operator_skipped_required_confirmations"
    )
    row["offline_acceptance_boundary_acknowledgement"] = (
        "offline_acceptance_authorizes_live_apply_now"
    )
    section = record["priority_decision_sections"][0]
    priority = section["priority"]
    section["evidence_packet_reference"] = "packet.json#priority_sections/999"
    section["source_material_reference"] = (
        "packet.json#priority_sections/999/source_evidence"
    )
    section["test_evidence_reference"] = (
        "tests/test_processing_proposal_contracts.py::test_stale_reference"
    )
    section["verification_report_reference"] = "stale-verification-report.json"
    section["source_evidence_acknowledgement"] = (
        "operator_skipped_source_evidence_review"
    )
    section["test_evidence_acknowledgement"] = (
        "operator_skipped_test_evidence_review"
    )
    section["verification_report_acknowledgement"] = (
        "operator_skipped_verification_report_review"
    )
    section["measured_problem_acknowledgement"] = (
        "operator_skipped_measured_problem_review"
    )
    section["proposed_correction_acknowledgement"] = (
        "operator_skipped_proposed_correction_review"
    )
    section["bounded_delta_acknowledgement"] = (
        "operator_skipped_bounded_delta_review"
    )
    section["confidence_acknowledgement"] = "operator_skipped_confidence_review"
    section["operator_explanation_acknowledgement"] = (
        "operator_skipped_operator_explanation_review"
    )
    section["future_wing_readback_acknowledgement"] = (
        "operator_skipped_future_wing_readback_contract_review"
    )
    section["headroom_feedback_true_peak_acknowledgement"] = (
        "operator_skipped_headroom_feedback_true_peak_review"
    )
    section["rollback_readback_acknowledgement"] = (
        "operator_skipped_rollback_readback_review"
    )
    section["safety_blockers_acknowledgement"] = (
        "operator_skipped_safety_blockers"
    )
    section["required_confirmations_acknowledgement"] = (
        "operator_skipped_required_confirmations"
    )
    section["offline_acceptance_boundary_acknowledgement"] = (
        "offline_acceptance_authorizes_live_apply_now"
    )
    record_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = validate_correction_method_operator_decision_record(
        run_dir,
        record_path,
    )
    failure_codes = {failure["code"] for failure in report["failed_checks"]}

    assert report["status"] == "failed"
    assert (
        f"decision_record_{method_id}_evidence_packet_reference_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_source_material_reference_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_test_evidence_reference_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_verification_report_reference_mismatch"
        in failure_codes
    )
    assert (
        "decision_record_"
        f"{method_id}_decision_record_bounded_delta_acknowledgement_invalid"
        in failure_codes
    )
    assert (
        "decision_record_"
        f"{method_id}_decision_record_source_evidence_acknowledgement_invalid"
        in failure_codes
    )
    assert (
        "decision_record_"
        f"{method_id}_decision_record_test_evidence_acknowledgement_invalid"
        in failure_codes
    )
    assert (
        "decision_record_"
        f"{method_id}_decision_record_verification_report_acknowledgement_invalid"
        in failure_codes
    )
    assert (
        "decision_record_"
        f"{method_id}_decision_record_measured_problem_acknowledgement_invalid"
        in failure_codes
    )
    assert (
        "decision_record_"
        f"{method_id}_decision_record_proposed_correction_acknowledgement_invalid"
        in failure_codes
    )
    assert (
        "decision_record_"
        f"{method_id}_decision_record_confidence_acknowledgement_invalid"
        in failure_codes
    )
    assert (
        "decision_record_"
        f"{method_id}_decision_record_operator_explanation_acknowledgement_invalid"
        in failure_codes
    )
    assert (
        "decision_record_"
        f"{method_id}_decision_record_future_wing_readback_acknowledgement_invalid"
        in failure_codes
    )
    assert (
        "decision_record_"
        f"{method_id}_decision_record_headroom_feedback_true_peak_acknowledgement_invalid"
        in failure_codes
    )
    assert (
        "decision_record_"
        f"{method_id}_decision_record_rollback_readback_acknowledgement_invalid"
        in failure_codes
    )
    assert (
        "decision_record_"
        f"{method_id}_decision_record_safety_blockers_acknowledgement_invalid"
        in failure_codes
    )
    assert (
        "decision_record_"
        f"{method_id}_decision_record_required_confirmations_acknowledgement_invalid"
        in failure_codes
    )
    assert (
        "decision_record_"
        f"{method_id}_decision_record_offline_acceptance_boundary_acknowledgement_invalid"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_evidence_packet_reference_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_source_material_reference_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_test_evidence_reference_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_verification_report_reference_mismatch"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_decision_record_bounded_delta_acknowledgement_invalid"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_decision_record_source_evidence_acknowledgement_invalid"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_decision_record_test_evidence_acknowledgement_invalid"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_decision_record_verification_report_acknowledgement_invalid"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_decision_record_measured_problem_acknowledgement_invalid"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_decision_record_proposed_correction_acknowledgement_invalid"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_decision_record_confidence_acknowledgement_invalid"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_decision_record_operator_explanation_acknowledgement_invalid"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_decision_record_future_wing_readback_acknowledgement_invalid"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_decision_record_headroom_feedback_true_peak_acknowledgement_invalid"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_decision_record_rollback_readback_acknowledgement_invalid"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_decision_record_safety_blockers_acknowledgement_invalid"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_decision_record_required_confirmations_acknowledgement_invalid"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_decision_record_offline_acceptance_boundary_acknowledgement_invalid"
        in failure_codes
    )
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["live_authority_granted"] is False


def test_correction_method_goal_operator_decision_record_validator_rejects_tampered_record_surfaces(tmp_path):
    run_id = "20260622T121512Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T12:15:12Z",
    )
    run_dir = tmp_path / run_id
    record_path = _write_filled_operator_decision_record(run_dir)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    row = record["method_decisions"][0]
    method_id = row["method_id"]
    row["allowed_decisions"] = [
        "accept_as_offline_proposal",
        "authorize_live_now",
        "accept_as_offline_proposal",
    ]
    row["required_confirmations"] = [
        "source_evidence_matches_current_offline_material",
        "operator_can_apply_live_now",
        "source_evidence_matches_current_offline_material",
    ]
    row["required_evidence"] = []
    _tamper_required_safety_checks(row)
    row["bounded_delta_unit"] = "live_console_state"
    row["bounded_delta_policy"] = "unbounded_live_delta"
    row["rollback_expectation"] = "no_rollback_required"
    row["confidence"]["required"] = False
    row["operator_readable_explanation"] = {}
    row["operator_visible_explanation_contract"]["confidence_required"] = False
    row["operator_payload_required_fields"] = []
    row["operator_evidence_required_fields"] = []
    row["operator_problem_review_summary"] = []
    row["operator_method_review_focus"] = []
    _tamper_why_safe_summary(row)
    _tamper_why_blocked_summary(row)
    row["goal_requirement_details"] = [
        "source_evidence:source=live_console;source_fields=none;"
        "operator_action=apply_live_now;coverage_status=covered;"
        "required=false;acceptance_scope=live_apply;"
        "required_before_operator_acceptance=false;"
        "required_before_live_promotion=false;"
        "required_before_queue_import=false;denial_reason=none;"
        "live_apply_allowed=true;queue_import_allowed=true;"
        "readiness_closure_allowed=true;"
        "requires_real_wing_proof_before_live=false"
    ]
    row["future_live_required_proofs"] = [
        proof
        for proof in row["future_live_required_proofs"]
        if proof != "rollback_readback_proof"
    ]
    row["future_live_required_proofs"].extend(
        [
            "operator_claims_live_ready_without_wing",
            "emergency_stop_proof",
        ]
    )
    row["future_live_requirement_details"] = [
        "real_wing_readonly_connection_artifact:source=live_console;"
        "requirement_type=live_apply;proof=none;readback_target=none;"
        "operator_action=apply_live_now;status=live_authorized;"
        "required_before_operator_acceptance=false;"
        "required_before_live_promotion=false;"
        "required_before_queue_import=false;"
        "blocked_live_actions=none;live_apply_allowed=true;"
        "queue_import_allowed=true;requires_real_wing_proof_before_live=false"
    ]
    row["evidence_problem_correction_chain_details"] = [
        "source_evidence:source=live_console;chain_step=1_source_evidence;"
        "source_fields=none;operator_action=apply_live_now;"
        "review_status=live_authorized;required=false;denial_reason=none;"
        "required_before_operator_acceptance=false;"
        "required_before_live_promotion=false;"
        "required_before_queue_import=false;"
        "live_apply_allowed=true;queue_import_allowed=true;"
        "requires_real_wing_proof_before_live=false"
    ]
    row["safety_blocker_details"] = [
        "headroom:source=live_console;blocker_type=live_apply;"
        "safety_check=none;blocked_live_action=none;"
        "missing_operator_surface=none;operator_action=apply_live_now;"
        "status=live_authorized;denial_reason=none;"
        "required_before_operator_acceptance=false;"
        "required_before_live_promotion=false;"
        "required_before_queue_import=false;"
        "live_apply_allowed=true;queue_import_allowed=true;"
        "requires_real_wing_proof_before_live=false"
    ]
    row["headroom_feedback_true_peak_check_details"] = [
        "headroom:source=live_console;check_type=headroom;"
        "safety_check=headroom;operator_action=apply_live_now;"
        "status=live_authorized;denial_reason=none;"
        "required_before_operator_acceptance=false;"
        "required_before_live_promotion=false;"
        "required_before_queue_import=false;"
        "live_apply_allowed=true;queue_import_allowed=true;"
        "requires_real_wing_proof_before_live=false"
    ]
    _tamper_future_wing_readback_targets(row)
    _tamper_future_wing_readback_contract_summary(row)
    row["blocked_live_actions"] = [
        "live_apply",
        "authorize_live_eq_now",
        "live_apply",
    ]
    _tamper_live_authority_boundary_summary(row)
    row["operator_review_record_required_fields"] = [
        "reviewer",
        "live_authorization_reference",
        "reviewer",
    ]
    row["required_confirmation_details"] = [
        "live_apply_and_queue_import_remain_blocked:source=live_console;"
        "action_type=write;acknowledgement_fields=none;"
        "action=confirm_operator_required_condition:"
        "live_apply_and_queue_import_remain_blocked"
    ]
    row["operator_acceptance_gate_details"] = [
        "live_apply_and_queue_import_remain_blocked:source=live_console;"
        "action_type=write;acknowledgement_fields=none;"
        "action=confirm_operator_required_condition:"
        "live_apply_and_queue_import_remain_blocked;"
        "acceptance_scope=live_apply;required=false;"
        "required_before_operator_acceptance=false;"
        "required_before_live_promotion=false;"
        "required_before_queue_import=false;"
        "live_apply_allowed=true;queue_import_allowed=true;"
        "readiness_closure_allowed=true;"
        "requires_real_wing_proof_before_live=false"
    ]
    row["operator_review_decision_option_details"] = [
        "accept_as_offline_proposal:effect=authorize_live_now;"
        "action_type=live_apply;operator_action=apply_live_now;"
        "blocks_offline_acceptance=false;live_apply_allowed=true;"
        "queue_import_allowed=true;requires_real_wing_proof_before_live=false"
    ]
    row["operator_next_action_details"] = [
        "do_not_queue_or_apply_live:source=live_console;"
        "action_type=live_apply;operator_action=apply_live_now;"
        "status=live_authorized;fields=none;readback_targets=none;"
        "proofs=none;blocked_live_actions=none;"
        "required_before_operator_acceptance=false;"
        "required_before_live_promotion=false;"
        "required_before_queue_import=false;"
        "live_apply_allowed=true;queue_import_allowed=true;"
        "requires_real_wing_proof_before_live=false"
    ]
    row["rationale"] = ""
    row["source_material_reference"] = ""
    row["source_evidence_acknowledgement"] = ""
    row["test_evidence_reference"] = ""
    row["test_evidence_acknowledgement"] = ""
    row["verification_report_reference"] = ""
    row["verification_report_acknowledgement"] = ""
    row["measured_problem_acknowledgement"] = ""
    row["proposed_correction_acknowledgement"] = ""
    row["bounded_delta_acknowledgement"] = ""
    row["confidence_acknowledgement"] = ""
    row["operator_explanation_acknowledgement"] = ""
    row["future_wing_readback_acknowledgement"] = ""
    row["headroom_feedback_true_peak_acknowledgement"] = ""
    row["rollback_readback_acknowledgement"] = ""
    row["safety_blockers_acknowledgement"] = ""
    row["required_confirmations_acknowledgement"] = ""
    row["offline_acceptance_boundary_acknowledgement"] = ""
    record_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = validate_correction_method_operator_decision_record(
        run_dir,
        record_path,
    )
    failure_codes = {failure["code"] for failure in report["failed_checks"]}

    assert report["status"] == "failed"
    assert (
        f"decision_record_{method_id}_allowed_decisions_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_missing_allowed_decisions"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_unknown_allowed_decisions"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_duplicate_allowed_decisions"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_required_confirmations_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_required_evidence_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_required_safety_checks_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_bounded_delta_unit_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_bounded_delta_policy_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_unknown_bounded_delta_unit"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_unknown_bounded_delta_policy"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_rollback_expectation_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_unknown_rollback_expectation"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_confidence_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_operator_readable_explanation_mismatch"
        in failure_codes
    )
    assert (
        "decision_record_"
        f"{method_id}_operator_visible_explanation_contract_mismatch"
    ) in failure_codes
    assert (
        "decision_record_"
        f"{method_id}_operator_visible_explanation_contract_confidence_required_"
        "not_true"
    ) in failure_codes
    assert (
        f"decision_record_{method_id}_confidence_contract_required_not_true"
        in failure_codes
    )
    assert (
        "decision_record_"
        f"{method_id}_operator_readable_explanation_contract_required_not_true"
    ) in failure_codes
    assert (
        f"decision_record_{method_id}_decision_record_rationale_missing"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_decision_record_source_material_reference_missing"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_decision_record_test_evidence_reference_missing"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_decision_record_verification_report_reference_missing"
        in failure_codes
    )
    assert (
        "decision_record_"
        f"{method_id}_decision_record_bounded_delta_acknowledgement_missing"
        in failure_codes
    )
    assert (
        "decision_record_"
        f"{method_id}_decision_record_source_evidence_acknowledgement_missing"
        in failure_codes
    )
    assert (
        "decision_record_"
        f"{method_id}_decision_record_test_evidence_acknowledgement_missing"
        in failure_codes
    )
    assert (
        "decision_record_"
        f"{method_id}_decision_record_verification_report_acknowledgement_missing"
        in failure_codes
    )
    assert (
        "decision_record_"
        f"{method_id}_decision_record_measured_problem_acknowledgement_missing"
        in failure_codes
    )
    assert (
        "decision_record_"
        f"{method_id}_decision_record_proposed_correction_acknowledgement_missing"
        in failure_codes
    )
    assert (
        "decision_record_"
        f"{method_id}_decision_record_confidence_acknowledgement_missing"
        in failure_codes
    )
    assert (
        "decision_record_"
        f"{method_id}_decision_record_operator_explanation_acknowledgement_missing"
        in failure_codes
    )
    assert (
        "decision_record_"
        f"{method_id}_decision_record_future_wing_readback_acknowledgement_missing"
        in failure_codes
    )
    assert (
        "decision_record_"
        f"{method_id}_decision_record_headroom_feedback_true_peak_acknowledgement_missing"
        in failure_codes
    )
    assert (
        "decision_record_"
        f"{method_id}_decision_record_rollback_readback_acknowledgement_missing"
        in failure_codes
    )
    assert (
        "decision_record_"
        f"{method_id}_decision_record_safety_blockers_acknowledgement_missing"
        in failure_codes
    )
    assert (
        "decision_record_"
        f"{method_id}_decision_record_required_confirmations_acknowledgement_missing"
        in failure_codes
    )
    assert (
        "decision_record_"
        f"{method_id}_decision_record_offline_acceptance_boundary_acknowledgement_missing"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_operator_payload_required_fields_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_operator_evidence_required_fields_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_operator_problem_review_summary_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_operator_method_review_focus_mismatch"
        in failure_codes
    )
    assert f"decision_record_{method_id}_why_safe_summary_mismatch" in failure_codes
    assert (
        f"decision_record_{method_id}_why_safe_summary_keys_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_why_blocked_summary_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_why_blocked_summary_keys_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_missing_required_confirmations"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_unknown_required_confirmations"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_duplicate_required_confirmations"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_missing_required_evidence"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_unknown_required_safety_checks"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_duplicate_required_safety_checks"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_future_live_required_proofs_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_future_live_requirement_details_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_goal_requirement_details_mismatch"
        in failure_codes
    )
    assert (
        "decision_record_"
        f"{method_id}_evidence_problem_correction_chain_details_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_safety_blocker_details_mismatch"
        in failure_codes
    )
    assert (
        "decision_record_"
        f"{method_id}_headroom_feedback_true_peak_check_details_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_future_wing_readback_targets_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_unknown_future_wing_readback_targets"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_duplicate_future_wing_readback_targets"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_future_wing_readback_contract_summary_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_missing_future_wing_readback_contract_summary"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_unknown_future_wing_readback_contract_summary"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_duplicate_future_wing_readback_contract_summary"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_invalid_future_wing_readback_operator_next_step"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_invalid_future_wing_readback_denial_reason"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_missing_future_live_required_proofs"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_unknown_future_live_required_proofs"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_duplicate_future_live_required_proofs"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_blocked_live_actions_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_unknown_blocked_live_actions"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_duplicate_blocked_live_actions"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_live_authority_boundary_summary_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_missing_live_authority_boundary_summary"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_unknown_live_authority_boundary_summary"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_duplicate_live_authority_boundary_summary"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_operator_review_record_required_fields_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_required_confirmation_details_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_operator_acceptance_gate_details_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_operator_review_decision_option_details_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_{method_id}_operator_next_action_details_mismatch"
        in failure_codes
    )
    assert (
        "decision_record_"
        f"{method_id}_missing_operator_review_record_required_fields"
        in failure_codes
    )
    assert (
        "decision_record_"
        f"{method_id}_unknown_operator_review_record_required_fields"
        in failure_codes
    )
    assert (
        "decision_record_"
        f"{method_id}_duplicate_operator_review_record_required_fields"
        in failure_codes
    )
    assert (
        report["accepted_offline_proposal_count"]
        == EXPECTED_METHOD_REVIEW_CARD_COUNT
    )
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["live_authority_granted"] is False


def test_correction_method_goal_operator_decision_record_validator_rejects_tampered_priority_surfaces(tmp_path):
    run_id = "20260622T121513Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T12:15:13Z",
    )
    run_dir = tmp_path / run_id
    record_path = _write_filled_operator_decision_record(run_dir)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    section = record["priority_decision_sections"][0]
    priority = section["priority"]
    section["allowed_decisions"] = [
        "accept_as_offline_proposal",
        "authorize_live_now",
        "accept_as_offline_proposal",
    ]
    section["required_evidence"] = []
    _tamper_required_safety_checks(section)
    section["confidence"] = []
    section["operator_readable_explanation"] = []
    _tamper_bounded_delta_policies(section)
    _tamper_rollback_expectations(section)
    section["operator_payload_required_fields"] = []
    section["operator_evidence_required_fields"] = []
    section["operator_problem_review_summary"] = []
    section["operator_method_review_focus"] = []
    section["why_safe_summary"] = []
    section["why_blocked_summary"] = []
    section["goal_requirement_details"] = [
        "source_evidence:source=live_console;source_fields=none;"
        "operator_action=apply_live_now;coverage_status=covered;"
        "required=false;acceptance_scope=live_apply;"
        "required_before_operator_acceptance=false;"
        "required_before_live_promotion=false;"
        "required_before_queue_import=false;denial_reason=none;"
        "live_apply_allowed=true;queue_import_allowed=true;"
        "readiness_closure_allowed=true;"
        "requires_real_wing_proof_before_live=false"
    ]
    section["operator_visible_explanation_contract"] = []
    _tamper_future_wing_readback_targets(section)
    _tamper_future_wing_readback_contract_summary(section)
    section["future_live_required_proofs"] = [
        proof
        for proof in section["future_live_required_proofs"]
        if proof != "rollback_readback_proof"
    ]
    section["future_live_requirement_details"] = [
        "real_wing_readonly_connection_artifact:source=live_console;"
        "requirement_type=live_apply;proof=none;readback_target=none;"
        "operator_action=apply_live_now;status=live_authorized;"
        "required_before_operator_acceptance=false;"
        "required_before_live_promotion=false;"
        "required_before_queue_import=false;"
        "blocked_live_actions=none;live_apply_allowed=true;"
        "queue_import_allowed=true;requires_real_wing_proof_before_live=false"
    ]
    section["evidence_problem_correction_chain_details"] = [
        "source_evidence:source=live_console;chain_step=1_source_evidence;"
        "source_fields=none;operator_action=apply_live_now;"
        "review_status=live_authorized;required=false;denial_reason=none;"
        "required_before_operator_acceptance=false;"
        "required_before_live_promotion=false;"
        "required_before_queue_import=false;"
        "live_apply_allowed=true;queue_import_allowed=true;"
        "requires_real_wing_proof_before_live=false"
    ]
    section["safety_blocker_details"] = [
        "headroom:source=live_console;blocker_type=live_apply;"
        "safety_check=none;blocked_live_action=none;"
        "missing_operator_surface=none;operator_action=apply_live_now;"
        "status=live_authorized;denial_reason=none;"
        "required_before_operator_acceptance=false;"
        "required_before_live_promotion=false;"
        "required_before_queue_import=false;"
        "live_apply_allowed=true;queue_import_allowed=true;"
        "requires_real_wing_proof_before_live=false"
    ]
    section["headroom_feedback_true_peak_check_details"] = [
        "headroom:source=live_console;check_type=headroom;"
        "safety_check=headroom;operator_action=apply_live_now;"
        "status=live_authorized;denial_reason=none;"
        "required_before_operator_acceptance=false;"
        "required_before_live_promotion=false;"
        "required_before_queue_import=false;"
        "live_apply_allowed=true;queue_import_allowed=true;"
        "requires_real_wing_proof_before_live=false"
    ]
    section["blocked_live_actions"] = [
        "live_apply",
        "authorize_live_eq_now",
        "live_apply",
    ]
    _tamper_live_authority_boundary_summary(section)
    section["operator_review_record_required_fields"] = [
        "reviewer",
        "live_authorization_reference",
        "reviewer",
    ]
    section["required_confirmation_details"] = [
        "live_apply_and_queue_import_remain_blocked:source=live_console;"
        "action_type=write;acknowledgement_fields=none;"
        "action=confirm_operator_required_condition:"
        "live_apply_and_queue_import_remain_blocked"
    ]
    section["operator_acceptance_gate_details"] = [
        "live_apply_and_queue_import_remain_blocked:source=live_console;"
        "action_type=write;acknowledgement_fields=none;"
        "action=confirm_operator_required_condition:"
        "live_apply_and_queue_import_remain_blocked;"
        "acceptance_scope=live_apply;required=false;"
        "required_before_operator_acceptance=false;"
        "required_before_live_promotion=false;"
        "required_before_queue_import=false;"
        "live_apply_allowed=true;queue_import_allowed=true;"
        "readiness_closure_allowed=true;"
        "requires_real_wing_proof_before_live=false"
    ]
    section["operator_review_decision_option_details"] = [
        "accept_as_offline_proposal:effect=authorize_live_now;"
        "action_type=live_apply;operator_action=apply_live_now;"
        "blocks_offline_acceptance=false;live_apply_allowed=true;"
        "queue_import_allowed=true;requires_real_wing_proof_before_live=false"
    ]
    section["operator_next_action_details"] = [
        "do_not_queue_or_apply_live:source=live_console;"
        "action_type=live_apply;operator_action=apply_live_now;"
        "status=live_authorized;fields=none;readback_targets=none;"
        "proofs=none;blocked_live_actions=none;"
        "required_before_operator_acceptance=false;"
        "required_before_live_promotion=false;"
        "required_before_queue_import=false;"
        "live_apply_allowed=true;queue_import_allowed=true;"
        "requires_real_wing_proof_before_live=false"
    ]
    section["rationale"] = ""
    section["source_material_reference"] = ""
    section["source_evidence_acknowledgement"] = ""
    section["test_evidence_reference"] = ""
    section["test_evidence_acknowledgement"] = ""
    section["verification_report_reference"] = ""
    section["verification_report_acknowledgement"] = ""
    section["measured_problem_acknowledgement"] = ""
    section["proposed_correction_acknowledgement"] = ""
    section["bounded_delta_acknowledgement"] = ""
    section["confidence_acknowledgement"] = ""
    section["operator_explanation_acknowledgement"] = ""
    section["future_wing_readback_acknowledgement"] = ""
    section["headroom_feedback_true_peak_acknowledgement"] = ""
    section["rollback_readback_acknowledgement"] = ""
    section["safety_blockers_acknowledgement"] = ""
    section["required_confirmations_acknowledgement"] = ""
    section["offline_acceptance_boundary_acknowledgement"] = ""
    record_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = validate_correction_method_operator_decision_record(
        run_dir,
        record_path,
    )
    failure_codes = {failure["code"] for failure in report["failed_checks"]}

    assert report["status"] == "failed"
    assert (
        f"decision_record_priority_{priority}_allowed_decisions_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_missing_allowed_decisions"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_unknown_allowed_decisions"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_duplicate_allowed_decisions"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_required_evidence_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_required_safety_checks_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_confidence_mismatch"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_operator_readable_explanation_mismatch"
    ) in failure_codes
    assert (
        f"decision_record_priority_{priority}_bounded_delta_policies_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_rollback_expectations_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_missing_required_evidence"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_unknown_required_safety_checks"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_duplicate_required_safety_checks"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_missing_confidence"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_missing_operator_readable_explanation"
    ) in failure_codes
    assert (
        f"decision_record_priority_{priority}_unknown_bounded_delta_policies"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_duplicate_bounded_delta_policies"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_unknown_rollback_expectations"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_duplicate_rollback_expectations"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_operator_payload_required_fields_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_operator_evidence_required_fields_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_operator_problem_review_summary_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_operator_method_review_focus_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_why_safe_summary_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_missing_why_safe_summary_keys"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_why_blocked_summary_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_missing_why_blocked_summary_keys"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_decision_record_rationale_missing"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_decision_record_source_material_reference_missing"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_decision_record_test_evidence_reference_missing"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_decision_record_verification_report_reference_missing"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_decision_record_bounded_delta_acknowledgement_missing"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_decision_record_source_evidence_acknowledgement_missing"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_decision_record_test_evidence_acknowledgement_missing"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_decision_record_verification_report_acknowledgement_missing"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_decision_record_measured_problem_acknowledgement_missing"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_decision_record_proposed_correction_acknowledgement_missing"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_decision_record_confidence_acknowledgement_missing"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_decision_record_operator_explanation_acknowledgement_missing"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_decision_record_future_wing_readback_acknowledgement_missing"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_decision_record_headroom_feedback_true_peak_acknowledgement_missing"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_decision_record_rollback_readback_acknowledgement_missing"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_decision_record_safety_blockers_acknowledgement_missing"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_decision_record_required_confirmations_acknowledgement_missing"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_decision_record_offline_acceptance_boundary_acknowledgement_missing"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_operator_visible_explanation_contract_mismatch"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_missing_operator_visible_explanation_contract"
    ) in failure_codes
    assert (
        f"decision_record_priority_{priority}_future_wing_readback_targets_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_unknown_future_wing_readback_targets"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_duplicate_future_wing_readback_targets"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_future_live_required_proofs_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_future_live_requirement_details_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_goal_requirement_details_mismatch"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_evidence_problem_correction_chain_details_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_safety_blocker_details_mismatch"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_headroom_feedback_true_peak_check_details_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_future_wing_readback_contract_summary_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_missing_future_wing_readback_contract_summary"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_unknown_future_wing_readback_contract_summary"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_duplicate_future_wing_readback_contract_summary"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_invalid_future_wing_readback_operator_next_step"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_invalid_future_wing_readback_denial_reason"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_missing_future_live_required_proofs"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_blocked_live_actions_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_unknown_blocked_live_actions"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_duplicate_blocked_live_actions"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_live_authority_boundary_summary_mismatch"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_missing_live_authority_boundary_summary"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_unknown_live_authority_boundary_summary"
        in failure_codes
    )
    assert (
        f"decision_record_priority_{priority}_duplicate_live_authority_boundary_summary"
        in failure_codes
    )
    assert (
        "decision_record_priority_"
        f"{priority}_operator_review_record_required_fields_mismatch"
    ) in failure_codes
    assert (
        "decision_record_priority_"
        f"{priority}_required_confirmation_details_mismatch"
    ) in failure_codes
    assert (
        "decision_record_priority_"
        f"{priority}_operator_acceptance_gate_details_mismatch"
    ) in failure_codes
    assert (
        "decision_record_priority_"
        f"{priority}_operator_review_decision_option_details_mismatch"
    ) in failure_codes
    assert (
        "decision_record_priority_"
        f"{priority}_operator_next_action_details_mismatch"
    ) in failure_codes
    assert (
        "decision_record_priority_"
        f"{priority}_missing_operator_review_record_required_fields"
    ) in failure_codes
    assert (
        "decision_record_priority_"
        f"{priority}_unknown_operator_review_record_required_fields"
    ) in failure_codes
    assert (
        "decision_record_priority_"
        f"{priority}_duplicate_operator_review_record_required_fields"
    ) in failure_codes
    assert report["priority_decision_counts"]["accept_as_offline_proposal"] == 7
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["live_authority_granted"] is False


def test_correction_method_goal_operator_decision_record_validator_rejects_missing_priority_decision(tmp_path):
    run_id = "20260622T121507Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T12:15:07Z",
    )
    run_dir = tmp_path / run_id
    record_path = _write_filled_operator_decision_record(run_dir)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["priority_decision_sections"][0]["operator_decision"] = None
    record_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = validate_correction_method_operator_decision_record(
        run_dir,
        record_path,
    )
    failure_codes = {failure["code"] for failure in report["failed_checks"]}

    assert report["status"] == "failed"
    assert (
        "decision_record_priority_1_operator_decision_missing" in failure_codes
    )
    assert report["priority_decision_counts"]["missing_operator_decision"] == 1
    assert report["priority_ids_by_decision"]["missing_operator_decision"] == [1]
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["live_authority_granted"] is False


def test_correction_method_goal_operator_decision_record_validator_rejects_priority_accept_with_rejected_method(tmp_path):
    run_id = "20260622T121508Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T12:15:08Z",
    )
    run_dir = tmp_path / run_id
    record_path = _write_filled_operator_decision_record(run_dir)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    method_id = record["priority_decision_sections"][0]["method_ids"][0]
    for row in record["method_decisions"]:
        if row["method_id"] == method_id:
            row["operator_decision"] = "reject_proposal"
            row["operator_notes"] = "offline method rejection conflicts with group"
            break
    else:
        raise AssertionError(f"missing method decision for {method_id!r}")
    record_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = validate_correction_method_operator_decision_record(
        run_dir,
        record_path,
    )
    failure_codes = {failure["code"] for failure in report["failed_checks"]}
    priority_one = next(
        section
        for section in report["validated_priority_decision_sections"]
        if section["priority"] == 1
    )

    assert report["status"] == "failed"
    assert (
        "decision_record_priority_1_accept_conflicts_with_method_decisions"
        in failure_codes
    )
    assert report["priority_decision_counts"]["accept_as_offline_proposal"] == 7
    assert report["decision_counts"]["reject_proposal"] == 1
    assert priority_one["operator_decision"] == "accept_as_offline_proposal"
    assert priority_one["accepted_offline_proposal_count"] == (
        priority_one["method_decision_count"] - 1
    )
    assert priority_one["decision_counts"]["reject_proposal"] == 1
    assert priority_one["method_ids_by_decision"]["reject_proposal"] == [method_id]
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["live_authority_granted"] is False


def test_correction_method_goal_operator_decision_record_validator_rejects_priority_reject_with_all_methods_accepted(tmp_path):
    run_id = "20260622T121509Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T12:15:09Z",
    )
    run_dir = tmp_path / run_id
    record_path = _write_filled_operator_decision_record(run_dir)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["priority_decision_sections"][0]["operator_decision"] = "reject_proposal"
    record["priority_decision_sections"][0]["operator_notes"] = (
        "offline priority rejection conflicts with accepted methods"
    )
    record_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = validate_correction_method_operator_decision_record(
        run_dir,
        record_path,
    )
    failure_codes = {failure["code"] for failure in report["failed_checks"]}
    priority_one = next(
        section
        for section in report["validated_priority_decision_sections"]
        if section["priority"] == 1
    )

    assert report["status"] == "failed"
    assert (
        "decision_record_priority_1_non_accept_conflicts_with_method_decisions"
        in failure_codes
    )
    assert report["priority_decision_counts"]["reject_proposal"] == 1
    assert (
        report["decision_counts"]["accept_as_offline_proposal"]
        == EXPECTED_METHOD_REVIEW_CARD_COUNT
    )
    assert priority_one["operator_decision"] == "reject_proposal"
    assert priority_one["accepted_offline_proposal_count"] == priority_one[
        "method_decision_count"
    ]
    assert priority_one["decision_counts"]["accept_as_offline_proposal"] == (
        priority_one["method_decision_count"]
    )
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["live_authority_granted"] is False


def test_correction_method_goal_operator_review_export_verifier_rejects_tampered_markdown(tmp_path):
    run_id = "20260622T121501Z_correction_method_operator_review"
    export_correction_method_operator_review_packet(
        artifact_root=tmp_path,
        run_id=run_id,
        generated_at_utc="2026-06-22T12:15:01Z",
    )
    markdown_path = tmp_path / run_id / "operator-review.md"
    markdown_path.write_text(
        markdown_path.read_text(encoding="utf-8")
        + "\n- live_authority_granted: `true`\n",
        encoding="utf-8",
    )

    report = verify_correction_method_operator_review_export(tmp_path / run_id)
    failure_codes = {failure["code"] for failure in report["failed_checks"]}

    assert report["status"] == "failed"
    assert "operator_review_markdown_sha256_mismatch" in failure_codes
    assert "operator_review_markdown_byte_count_mismatch" in failure_codes
    assert "markdown_forbidden_true_flag" in failure_codes
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["contacts_backend"] is False
    assert report["live_authority_granted"] is False


def test_registry_correction_proposal_methods_are_covered_by_contract_suite():
    contracts = _contracts_by_method()
    covered_methods = {
        method_id
        for method_ids in OBJECTIVE_FAMILY_METHODS.values()
        for method_id in method_ids
    }

    missing = sorted(
        method.id
        for method in iter_processing_methods()
        if method.status == "proposal_only"
        and method.can_propose
        and method.domain in CORRECTION_PROPOSAL_DOMAINS
        and (method.id not in contracts or method.id not in covered_methods)
    )

    assert missing == []


def test_method_family_safety_checks_match_correction_domains():
    contracts = _contracts_by_method()

    expected_checks = {
        "lufs_gain_staging": {"headroom", "feedback", "true_peak", "bleed"},
        "auto_fader_v1": {"headroom", "feedback", "true_peak", "fader_ceiling", "ayaic_level_plane"},
        "live_shared_mix": {"headroom", "feedback", "true_peak", "fader_ceiling", "ayaic_level_plane"},
        "auto_soundcheck_engine": {"headroom", "feedback", "true_peak", "dry_run"},
        "bus_group_balance_proposal": {"headroom", "feedback", "true_peak", "main_bus_headroom", "ayaic_level_plane"},
        "auto_eq": {"headroom", "feedback", "true_peak", "manual_band", "phase"},
        "corrective_eq_proposal": {"headroom", "feedback", "true_peak", "manual_band", "phase"},
        "cross_adaptive_eq": {"headroom", "feedback", "true_peak", "manual_band", "phase"},
        "frequency_separation_proposal": {"headroom", "feedback", "true_peak", "sum_bus", "source_identity"},
        "auto_compressor": {"headroom", "feedback", "true_peak", "gain_reduction", "makeup_gain"},
        "auto_compressor_cf": {"headroom", "feedback", "true_peak", "gain_reduction", "makeup_gain"},
        "compressor_dynamics_proposal": {"headroom", "feedback", "true_peak", "gain_reduction", "makeup_gain"},
        "auto_gate_caig": {"source_identity", "chatter_guard", "hysteresis", "tails_and_attack"},
        "context_aware_gate_agent": {"source_identity", "chatter_guard", "hysteresis", "tails_and_attack"},
        "deesser_vocal_sibilance": {"source_identity", "arrangement_role", "true_peak", "speech_intelligibility", "tails_and_attack"},
        "auto_fx": {"headroom", "feedback", "return_level", "monitor_safety", "intelligibility"},
        "fx_send_proposal": {"headroom", "feedback", "return_level", "monitor_safety", "intelligibility"},
        "auto_panner": {"mono_compatibility", "phase", "source_identity"},
        "pan_spatial_proposal": {"mono_compatibility", "phase", "source_identity"},
        "phase_alignment": {"phase_correlation", "polarity_readback", "delay_readback", "sum_bus"},
        "phase_aware_eq_guard": {"phase_correlation", "sum_bus", "bleed", "eq_hold_policy"},
        "routing_management_proposal": {"read_only_map", "destructive_scope", "active_performance_guard"},
        "scene_preset_management_proposal": {"read_only_inventory", "destructive_scope", "active_signal_guard"},
        "mute_management_proposal": {"source_identity", "active_signal_guard", "feedback", "destructive_scope"},
        "master_bus_optimization_proposal": {
            "headroom",
            "true_peak",
            "pa_feed_safety",
            "reference_feed_exclusion",
            "limiter_or_dynamics_proof",
        },
        "routing_snapshot_reset_writes": {
            "read_only_map",
            "destructive_scope",
            "active_performance_guard",
        },
        "fx_direct_apply": {
            "headroom",
            "feedback",
            "return_level",
            "insert_bypass_state",
            "parameter_bounds",
        },
        "source_identity_replay_calibration": {
            "operator_outcomes",
            "replay_match",
            "safety_regressions",
            "operator_acceptance",
            "context_freshness",
        },
        "source_identity_confidence_proposal": {
            "classifier_confidence",
            "operator_acceptance",
            "context_freshness",
            "downstream_rejection",
            "replay_outcome",
        },
        "mix_variant_profile_proposal": {
            "headroom",
            "feedback",
            "true_peak",
            "arrangement_context",
            "source_identity",
            "style_clone",
        },
        "performance_adaptation_proposal": {
            "headroom",
            "feedback",
            "true_peak",
            "cooldown",
            "manual_override",
            "source_identity",
        },
        "arrangement_role_proposal": {
            "source_identity",
            "arrangement_context",
            "inter_channel_context",
            "role_persistence",
            "masking",
        },
        "decision_evidence_packet_proposal": {
            "readback",
            "rollback",
            "headroom",
            "feedback",
            "source_identity",
            "context_freshness",
        },
        "long_term_memory_learning_proposal": {
            "offline_replay",
            "operator_outcomes",
            "safety_regressions",
            "manual_overrides",
            "source_identity",
            "context_freshness",
        },
    }

    for method_id, required_checks in expected_checks.items():
        operator_contract = contracts[method_id]["proposal_contract"][
            "operator_correction_contract"
        ]
        actual_checks = set(operator_contract["applicable_safety_checks"])
        assert required_checks <= actual_checks


def test_gate_and_deesser_contracts_require_tail_attack_preservation_evidence():
    contracts = _contracts_by_method()
    expected_evidence_by_method = {
        "auto_gate_caig": {
            "chatter_guard",
            "hysteresis",
            "source_identity_downstream_trust",
            "quiet_words_tail_attack_preservation",
        },
        "context_aware_gate_agent": {
            "chatter_guard",
            "hysteresis",
            "source_identity_downstream_trust",
            "quiet_words_tail_attack_preservation",
        },
        "deesser_vocal_sibilance": {
            "speech_intelligibility",
            "source_identity_downstream_trust",
            "quiet_words_tail_attack_preservation",
        },
    }

    for method_id, expected_evidence in expected_evidence_by_method.items():
        proposal_contract = contracts[method_id]["proposal_contract"]
        operator_contract = proposal_contract["operator_correction_contract"]

        assert expected_evidence <= set(proposal_contract["safety_evidence_required"])
        assert expected_evidence <= set(operator_contract["required_evidence_fields"])

    gate_readback_targets = set(
        contracts["auto_gate_caig"]["proposal_contract"][
            "operator_correction_contract"
        ]["future_wing_readback_contract"]["readback_targets"]
    )
    assert {"attack", "hold", "release", "hysteresis"} <= gate_readback_targets
    assert "tails_and_attack" in contracts["deesser_vocal_sibilance"][
        "proposal_contract"
    ]["operator_correction_contract"]["applicable_safety_checks"]


def test_auto_fx_contract_requires_monitor_and_intelligibility_evidence():
    contract = _contracts_by_method()["auto_fx"]["proposal_contract"]

    assert {
        "return_level_safety",
        "monitor_safety",
        "intelligibility_risk",
        "current_fx_send_return_cache_freshness",
        "current_fx_send_return_cache_no_live_authority",
        "current_fx_parameter_schema_hints",
        "fx_parameter_schema_no_live_authority",
        "source_identity_downstream_trust",
        "old_value_source",
        "rollback_fx_readback",
    } <= set(contract["safety_evidence_required"])


def test_auto_panner_contract_requires_mono_phase_and_readback_evidence():
    contract = _contracts_by_method()["auto_panner"]["proposal_contract"]

    assert {
        "bounded_pan_delta",
        "current_pan_cache_freshness",
        "current_pan_cache_no_live_authority",
        "mono_phase_safety",
        "mono_phase_check_no_live_authority",
        "source_identity_downstream_trust",
        "rollback_pan_readback",
    } <= set(contract["safety_evidence_required"])


def test_mute_management_contract_requires_manual_authority_and_rollback_readback():
    contract = _contracts_by_method()["mute_management_proposal"]["proposal_contract"]

    assert {
        "current_mute_state",
        "current_mute_cache_freshness",
        "current_mute_cache_no_live_authority",
        "source_identity_downstream_trust",
        "rollback_mute_readback",
        "manual_mute_authority_preserved",
    } <= set(contract["safety_evidence_required"])


def test_auto_eq_contract_requires_readback_context_and_rollback_evidence():
    contract = _contracts_by_method()["auto_eq"]["proposal_contract"]

    assert {
        "frequency_match",
        "cut_or_bounded_boost",
        "feedback_notch_preserved",
        "manual_band_preserved",
        "eq_band_readback",
        "current_eq_context_cache_freshness",
        "current_eq_context_cache_no_live_authority",
        "rollback_eq_band_readback",
    } <= set(contract["safety_evidence_required"])


def test_auto_compressor_contract_requires_current_context_and_rollback_evidence():
    contract = _contracts_by_method()["auto_compressor"]["proposal_contract"]

    assert {
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
    } <= set(contract["safety_evidence_required"])


def test_auto_gate_contract_requires_current_context_and_rollback_evidence():
    contract = _contracts_by_method()["auto_gate_caig"]["proposal_contract"]

    assert {
        "noise_floor",
        "hysteresis",
        "source_role",
        "bleed_risk",
        "current_gate_bleed_context_cache_freshness",
        "current_gate_bleed_context_cache_no_live_authority",
        "source_identity_downstream_trust",
        "old_value_source",
        "rollback_gate_readback",
    } <= set(contract["safety_evidence_required"])


def test_deesser_contract_requires_bounded_speech_readback_and_rollback_evidence():
    contract = _contracts_by_method()["deesser_vocal_sibilance"]["proposal_contract"]

    assert {
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
        "dynamic_eq_or_deesser_readback",
        "old_value_source",
        "rollback_dynamics_or_dynamic_eq_readback",
        "bypass_state",
    } <= set(contract["safety_evidence_required"])


def test_routing_snapshot_reset_writes_contract_requires_read_only_route_evidence():
    contract = _contracts_by_method()["routing_snapshot_reset_writes"]["proposal_contract"]

    assert {
        "current_route_readback",
        "current_route_cache_freshness",
        "current_route_cache_no_live_authority",
        "proposed_source_readback",
        "active_performance_guard",
        "destructive_scope_check",
        "rollback_route",
        "rollback_snapshot",
        "operator_confirmation",
    } <= set(contract["safety_evidence_required"])


def test_quarantined_write_surface_contracts_remain_blocked_and_non_queueable():
    contracts = _contracts_by_method()

    expected_blockers = {
        "routing_snapshot_reset_writes": "destructive_or_broad_mixer_state_surface",
        "fx_direct_apply": "direct_fx_or_insert_write_surface",
    }

    for method_id, specific_blocker in expected_blockers.items():
        contract = contracts[method_id]
        proposal_contract = contract["proposal_contract"]
        operator_contract = proposal_contract["operator_correction_contract"]
        readback_contract = operator_contract["future_wing_readback_contract"]

        assert contract["risk"] == "blocked"
        assert contract["can_import_to_queue"] is False
        assert contract["can_apply_live"] is False
        assert contract["queue_import_allowed"] is False
        assert contract["live_apply_allowed"] is False
        assert contract["read_only_or_dry_run_only"] is True
        assert contract["live_authority_granted"] is False
        assert contract["readiness_closure_allowed"] is False
        assert contract["mutates_mixer"] is False
        assert specific_blocker in contract["blockers"]
        assert "not_in_supervised_fader_gain_executor" in contract["blockers"]
        assert "hardware_readback_and_rollback_not_proven" in contract["blockers"]
        assert proposal_contract["live_apply_allowed"] is False
        assert proposal_contract["queue_import_allowed"] is False
        assert operator_contract["read_only_or_dry_run_only"] is True
        assert operator_contract["live_apply_allowed"] is False
        assert operator_contract["queue_import_allowed"] is False
        assert readback_contract["required_before_live_promotion"] is True
        assert readback_contract["blocked_live_actions"]


def test_master_bus_contract_requires_rollback_readback_evidence():
    contract = _contracts_by_method()["master_bus_optimization_proposal"][
        "proposal_contract"
    ]

    assert {
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
    } <= set(contract["safety_evidence_required"])


def test_bus_group_balance_contract_requires_bounded_delta_and_rollback_evidence():
    contract = _contracts_by_method()["bus_group_balance_proposal"][
        "proposal_contract"
    ]

    assert {
        "ayaic_target_plane",
        "summed_child_lufs",
        "bounded_bus_group_delta",
        "assignment_readback",
        "current_bus_group_dca_matrix_cache_no_live_authority",
        "rollback_bus_group_dca_matrix_readback",
    } <= set(contract["safety_evidence_required"])


def test_future_wing_readback_contracts_do_not_grant_live_authority():
    contracts = _contracts_by_method()
    method_ids = sorted(
        {
            method_id
            for family_method_ids in OBJECTIVE_FAMILY_METHODS.values()
            for method_id in family_method_ids
        }
        | set(QUARANTINED_WRITE_SURFACE_METHODS)
    )

    for method_id in method_ids:
        contract = contracts[method_id]
        proposal_contract = contract["proposal_contract"]
        operator_contract = proposal_contract["operator_correction_contract"]
        readback_contract = operator_contract["future_wing_readback_contract"]

        assert "not_in_supervised_fader_gain_executor" in contract["blockers"]
        assert "hardware_readback_and_rollback_not_proven" in contract["blockers"]
        assert proposal_contract["allowed_live_write_kinds"] == ["fader", "gain"]
        assert operator_contract["rollback_required"] is True
        assert readback_contract["required_before_live_promotion"] is True
        assert readback_contract["blocked_live_actions"]
        assert readback_contract["read_only_or_dry_run_only"] is True
        assert readback_contract["live_apply_allowed"] is False
        assert readback_contract["queue_import_allowed"] is False
        assert readback_contract["live_authority_granted"] is False
        assert readback_contract["full_autonomous_ready"] is False
        assert readback_contract["readiness_closure_allowed"] is False
        assert readback_contract["can_authorize_live"] is False
        assert readback_contract["can_close_readiness"] is False
        assert readback_contract["mutates_mixer"] is False
        assert readback_contract["operator_review_required"] is True
        assert readback_contract["operator_approval_required"] is True
        assert readback_contract["hardware_proof_required"] is True
        assert readback_contract["blocks_live_promotion"] is True
        assert readback_contract["blocks_queue_import"] is True
        assert readback_contract["future_wing_readback_required"] is True
        assert (
            readback_contract["denial_reason"]
            == "real_wing_hardware_readback_rollback_and_operator_review_required"
        )
        assert (
            readback_contract["operator_next_step"]
            == "capture_real_wing_readback_and_rollback_proof_before_live_promotion"
        )
        assert readback_contract["future_live_promotion_requires"] == operator_contract[
            "future_live_promotion_requirements"
        ]["required_proofs"]
        assert "live_apply_denied_until_real_wing_proof" in readback_contract[
            "why_blocked"
        ]
        assert "future_wing_readback_required" in readback_contract["why_blocked"]


def test_complete_contract_payload_surfaces_all_required_evidence_checks():
    contracts = _contracts_by_method()

    for method_id, contract in contracts.items():
        proposal_contract = contract["proposal_contract"]
        required_evidence = proposal_contract["safety_evidence_required"]
        raw_evidence = {
            field: False
            if field in {"live_write", "queue_mutation"}
            else {"field": field, "synthetic_complete_contract_evidence": True}
            for field in required_evidence
        }

        payload = build_operator_correction_payload(
            proposal={
                "source": "synthetic_contract_visibility_audit",
                "target": f"target:{method_id}",
                "reason": f"Synthetic complete evidence for {method_id}.",
                "confidence": 0.75,
                "raw": raw_evidence,
            },
            method_id=method_id,
            target=f"target:{method_id}",
        )

        completeness = payload["source_evidence"]["evidence_completeness"]
        assert completeness["status"] == "complete"
        assert completeness["missing"] == []
        _assert_required_evidence_status(
            payload["source_evidence"]["required_evidence_status"],
            completeness=completeness,
            evidence_gate=payload["evidence_gate"],
        )
        assert {
            row["status"]
            for row in payload["source_evidence"]["required_evidence_status"]["rows"]
        } == {"provided_operator_review_required"}
        evidence_checks = payload["headroom_feedback_true_peak_checks"][
            "evidence_checks"
        ]
        for field in required_evidence:
            assert evidence_checks[field] == raw_evidence[field]
        _assert_raw_input_evidence_boundary(
            payload["headroom_feedback_true_peak_checks"]["evidence_checks_boundary"],
            surface="headroom_feedback_true_peak_checks.evidence_checks",
            evidence_fields=required_evidence,
        )


def test_operator_readable_explanation_wraps_proposal_reason_with_safety_context():
    method_id = "lufs_gain_staging"
    contract = _contracts_by_method()[method_id]
    proposal_contract = contract["proposal_contract"]
    operator_contract = proposal_contract["operator_correction_contract"]
    required_evidence = proposal_contract["safety_evidence_required"]
    raw_evidence = {
        field: f"synthetic_operator_explanation_evidence:{field}"
        for field in required_evidence
    }

    payload = build_operator_correction_payload(
        proposal={
            "source": "synthetic_operator_explanation_audit",
            "target": "lead_vocal",
            "reason": "Turn it down a little.",
            "confidence": 0.75,
            "raw": raw_evidence,
        },
        method_id=method_id,
        target="lead_vocal",
    )

    explanation = payload["operator_readable_explanation"]
    assert "Turn it down a little." in explanation
    assert "little.." not in explanation
    assert method_id in explanation
    assert operator_contract["objective_family"] in explanation
    assert (
        operator_contract["future_wing_readback_contract"]["target_kind"]
        in explanation
    )
    assert "Required evidence is complete for" in explanation
    assert "Evidence is available" not in explanation
    assert "offline/dry-run proposal-only review" in explanation
    assert "Live apply and queue import remain blocked" in explanation
    assert "real WING readback, rollback proof, and operator approval" in explanation


def test_contract_only_payload_fallbacks_are_method_specific_and_non_queueable():
    contracts = _contracts_by_method()

    for method_id, contract in contracts.items():
        proposal_contract = contract["proposal_contract"]
        operator_contract = proposal_contract["operator_correction_contract"]
        required_evidence = proposal_contract["safety_evidence_required"]
        raw_evidence = {
            field: False
            if field in {"live_write", "queue_mutation"}
            else f"synthetic_complete_contract_evidence:{field}"
            for field in required_evidence
        }

        payload = build_operator_correction_payload(
            proposal={
                "source": "synthetic_contract_fallback_audit",
                "target": f"target:{method_id}",
                "confidence": 0.75,
                "raw": raw_evidence,
            },
            method_id=method_id,
            target=f"target:{method_id}",
        )

        assert payload["method_id"] == method_id
        assert payload["source"] == "synthetic_contract_fallback_audit"
        assert payload["target"] == f"target:{method_id}"
        assert payload["value_type"] == proposal_contract["value_type"]
        assert payload["target_kind"] == proposal_contract["target_kind"]
        assert payload["old_value_source"] == proposal_contract["old_value_source"]
        assert payload["method_id"] == payload["source_evidence"]["method_id"]
        assert payload["source"] == payload["source_evidence"]["source"]
        assert payload["target"] == payload["source_evidence"]["target"]
        assert payload["target_kind"] == payload["future_wing_readback_contract"][
            "target_kind"
        ]
        assert payload["old_value_source"] == payload["future_wing_readback_contract"][
            "old_value_source"
        ]
        for packet_key in ("future_wing_readback_contract", "rollback_readback_contract"):
            packet = payload[packet_key]
            for field in OPERATOR_IDENTITY_PROFILE_FIELDS:
                assert packet[field] == payload[field]
        future_requirements = payload["future_live_promotion_requirements"]
        for field in OPERATOR_IDENTITY_PROFILE_FIELDS:
            assert future_requirements[field] == payload[field]
        for row in payload["future_live_requirement_details"]:
            for field in OPERATOR_IDENTITY_PROFILE_FIELDS:
                assert row[field] == payload["future_wing_readback_contract"][field]
        for row in payload["rollback_readback_details"]:
            for field in OPERATOR_IDENTITY_PROFILE_FIELDS:
                assert row[field] == payload["rollback_readback_contract"][field]
        for packet_key in (
            "live_authority_boundary",
            "operator_proposal_review_gate",
            "operator_safety_review_matrix",
        ):
            packet = payload[packet_key]
            for field in OPERATOR_IDENTITY_PROFILE_FIELDS:
                assert packet[field] == payload[field]
        for row in payload["operator_safety_review_matrix"]["checks"]:
            _assert_correction_context_profile(row, payload)
            for field in OPERATOR_IDENTITY_PROFILE_FIELDS:
                assert row[field] == payload[field]
        for packet_key in (
            "operator_review_record_template",
        ):
            packet = payload[packet_key]
            for field in OPERATOR_IDENTITY_PROFILE_FIELDS:
                assert packet[field] == payload[field]
        for packet_key in (
            "operator_review_decision_options",
            "operator_review_decision_option_details",
            "source_evidence_details",
        ):
            for row in payload[packet_key]:
                for field in OPERATOR_IDENTITY_PROFILE_FIELDS:
                    assert row[field] == payload[field]
        for packet_key in (
            "operator_review_decision_options",
            "operator_next_action_details",
            "operator_review_decision_option_details",
            "headroom_feedback_true_peak_check_details",
            "source_evidence_details",
            "why_safe_details",
            "why_blocked_details",
            "future_live_requirement_details",
            "operator_review_checklist_details",
            "operator_review_record_field_details",
            "bounded_delta_details",
            "confidence_review_details",
            "safety_blocker_details",
            "rollback_readback_details",
        ):
            for row in payload[packet_key]:
                _assert_correction_context_profile(row, payload)
        assert payload["denial_reason"] == payload["source_evidence"][
            "denial_reason"
        ]
        assert payload["operator_next_step"] == payload["source_evidence"][
            "operator_next_step"
        ]
        assert payload["denial_reason"] == payload[
            "future_live_promotion_requirements"
        ]["blocked_until"]
        assert payload["operator_next_step"] == payload["evidence_gate"][
            "operator_action"
        ]
        for packet_key in ("operator_explanation_packet", "operator_review_summary"):
            packet = payload[packet_key]
            assert packet["method_id"] == payload["method_id"]
            assert packet["source"] == payload["source"]
            assert packet["target"] == payload["target"]
            assert packet["value_type"] == payload["value_type"]
            assert packet["target_kind"] == payload["target_kind"]
            assert packet["old_value_source"] == payload["old_value_source"]
            assert packet["objective_family"] == payload["objective_family"]
            assert packet["bounded_delta_unit"] == payload["bounded_delta_unit"]
            assert packet["bounded_delta_policy"] == payload["bounded_delta_policy"]
        evidence_packet = payload["operator_evidence_packet"]
        assert evidence_packet["method_id"] == payload["method_id"]
        assert evidence_packet["source"] == payload["source"]
        assert evidence_packet["target"] == payload["target"]
        assert evidence_packet["value_type"] == payload["value_type"]
        assert evidence_packet["target_kind"] == payload["target_kind"]
        assert evidence_packet["old_value_source"] == payload["old_value_source"]
        assert evidence_packet["objective_family"] == payload["objective_family"]
        assert evidence_packet["bounded_delta_unit"] == payload["bounded_delta_unit"]
        assert evidence_packet["bounded_delta_policy"] == payload[
            "bounded_delta_policy"
        ]
        for packet_key in ("measured_or_audible_problem", "proposed_correction"):
            packet = payload[packet_key]
            assert packet["source"] == payload["source"]
            assert packet["target"] == payload["target"]
            if packet_key == "proposed_correction":
                assert packet["contract_value_type"] == payload["value_type"]
                assert packet["contract_target_kind"] == payload["target_kind"]
                assert packet["contract_old_value_source"] == payload[
                    "old_value_source"
                ]
            else:
                assert packet["value_type"] == payload["value_type"]
                assert packet["target_kind"] == payload["target_kind"]
                assert packet["old_value_source"] == payload["old_value_source"]
        measured_problem = payload["measured_or_audible_problem"]
        assert measured_problem["method_id"] == payload["method_id"]
        assert measured_problem["objective_family"] == payload["objective_family"]
        assert measured_problem["bounded_delta_unit"] == payload["bounded_delta_unit"]
        assert measured_problem["bounded_delta_policy"] == payload[
            "bounded_delta_policy"
        ]

        problem = payload["measured_or_audible_problem"]["summary"]
        assert problem != "Analyzer produced a proposal-only recommendation."
        assert method_id in problem
        assert operator_contract["objective_family"] in problem
        assert "produced proposal-only evidence" in problem

        correction = payload["proposed_correction"]
        assert correction["status"] == "proposal_contract_evidence_review_only"
        assert correction["method_id"] == method_id
        assert correction["objective_family"] == operator_contract["objective_family"]
        assert correction["target_kind"] == proposal_contract["target_kind"]
        assert correction["bounded_delta_unit"] == operator_contract[
            "bounded_delta_unit"
        ]
        assert correction["bounded_delta_policy"] == operator_contract[
            "bounded_delta_policy"
        ]
        assert correction["required_evidence_fields"] == required_evidence
        assert correction["missing_required_evidence"] == []
        assert correction["concrete_proposed_action_required"] is True
        assert correction["read_only_or_dry_run_only"] is True
        assert correction["live_apply_allowed"] is False
        assert correction["queue_import_allowed"] is False
        assert correction["can_apply_live"] is False
        assert correction["can_import_to_queue"] is False
        assert correction["live_authority_granted"] is False
        assert correction["full_autonomous_ready"] is False
        assert correction["readiness_closure_allowed"] is False
        assert correction["can_authorize_live"] is False
        assert correction["can_close_readiness"] is False
        assert correction["mutates_mixer"] is False

        effective_blockers = payload["effective_safety_blockers"]
        assert effective_blockers
        assert "live_authority_denied" in effective_blockers
        assert "real_wing_proof_required" in effective_blockers
        assert payload["operator_evidence_packet"][
            "effective_safety_blockers"
        ] == effective_blockers
        assert payload["operator_explanation_packet"][
            "effective_safety_blockers"
        ] == effective_blockers
        assert payload["operator_review_summary"][
            "effective_safety_blockers"
        ] == effective_blockers
        assert set(effective_blockers) <= set(
            payload["operator_review_summary"]["why_blocked"]
        )

    explanation = payload["operator_readable_explanation"]
    assert explanation != "Proposal-only correction requires operator review."
    assert method_id in explanation
    assert operator_contract["objective_family"] in explanation
    assert "Live apply and queue import remain blocked" in explanation


def test_safety_matrix_no_direct_mapping_rows_expose_review_context():
    contract = _contracts_by_method()["routing_management_proposal"]
    proposal_contract = contract["proposal_contract"]
    required_evidence = proposal_contract["safety_evidence_required"]
    raw_evidence = {
        field: f"synthetic_complete_contract_evidence:{field}"
        for field in required_evidence
    }

    payload = build_operator_correction_payload(
        proposal={
            "source": "synthetic_mapping_review_audit",
            "target": "route:lead_vocal",
            "reason": "Synthetic route-map evidence for mapping review.",
            "confidence": 0.75,
            "raw": raw_evidence,
        },
        method_id="routing_management_proposal",
        target="route:lead_vocal",
    )

    matrix = payload["operator_safety_review_matrix"]
    no_direct_rows = [
        row for row in matrix["checks"] if row["mapping_review_required"]
    ]
    assert no_direct_rows
    assert matrix["summary"]["checks_requiring_mapping_review"] == len(
        no_direct_rows
    )
    for row in no_direct_rows:
        assert row["status"] == "operator_review_required_no_direct_evidence_mapping"
        assert row["evidence_mapping_status"] == (
            "no_direct_required_evidence_field_match"
        )
        assert row["required_evidence_fields"] == []
        assert row["provided_evidence_fields"] == []
        assert row["missing_required_evidence"] == []
        assert row["available_contract_evidence_fields"] == sorted(required_evidence)
        assert row["all_contract_evidence_fields"] == required_evidence
        assert "full evidence packet" in row["review_note"]
        assert row["read_only_or_dry_run_only"] is True
        assert row["live_apply_allowed"] is False
        assert row["queue_import_allowed"] is False
        assert row["blocks_live_promotion"] is True
        assert row["blocks_queue_import"] is True


def test_operator_correction_payload_reports_evidence_completeness():
    complete = build_operator_correction_payload(
        proposal={
            "source": "offline_gain_analyzer",
            "target": 1,
            "reason": "Input overload leaves too little headroom.",
            "confidence": 0.92,
            "raw": {
                "input_level_evidence": {"peak_dbfs": -0.4},
                "overload_or_weak_signal_problem": "overload",
                "headroom_margin": 0.4,
                "true_peak_headroom": 0.2,
                "bleed_risk": "low",
                "bounded_gain_delta": -2.0,
                "gain_down_or_operator_review_gain_up": "gain_down",
                "rollback_input_trim_readback": 0.0,
            },
            "evidence": {
                "live_authority_granted": True,
                "can_apply_live": True,
                "queue_action": "queue_now",
                "readback_required": False,
                "nested_authority_claim": {
                    "apply_now": True,
                    "dispatch_supervised_apply": True,
                },
            },
        },
        method_id="lufs_gain_staging",
        target=1,
        requested_change={
            "status": "ready_to_apply_live",
            "method_id": "scene_preset_management_proposal",
            "source": "live_scene_loader",
            "target": "scene:destructive",
            "target_kind": "scene_load",
            "old_value_source": "live_scene_state",
            "value_type": "scene_load",
            "channel": 1,
            "current_value": 0.0,
            "value": -2.0,
            "delta_db": -2.0,
            "rollback_value": 0.0,
            "readback_required": False,
            "queue_action": "queue_now",
            "apply_now": True,
            "dispatch_supervised_apply": True,
        },
    )

    completeness = complete["source_evidence"]["evidence_completeness"]
    assert completeness["status"] == "complete"
    assert completeness["missing"] == []
    assert completeness["complete"] is True
    source_evidence = complete["source_evidence"]
    assert source_evidence["evidence"]["live_authority_granted"] is False
    assert source_evidence["evidence"]["can_apply_live"] is False
    assert source_evidence["evidence"]["queue_action"] == (
        "not_queueable_until_promoted"
    )
    assert source_evidence["evidence"]["readback_required"] is True
    assert source_evidence["evidence"]["nested_authority_claim"]["apply_now"] is False
    assert source_evidence["evidence"]["nested_authority_claim"][
        "dispatch_supervised_apply"
    ] is False
    claims_by_path = {
        claim["path"]: claim for claim in source_evidence["raw_input_authority_claims"]
    }
    assert claims_by_path["live_authority_granted"]["requested_value"] is True
    assert claims_by_path["can_apply_live"]["requested_value"] is True
    assert claims_by_path["queue_action"]["requested_value"] == "queue_now"
    assert claims_by_path["readback_required"]["requested_value"] is False
    assert claims_by_path["nested_authority_claim.apply_now"][
        "requested_value"
    ] is True
    assert claims_by_path["nested_authority_claim.dispatch_supervised_apply"][
        "requested_value"
    ] is True
    assert complete["headroom_feedback_true_peak_checks"][
        "evidence_completeness"
    ] == completeness
    complete_checks = complete["headroom_feedback_true_peak_checks"][
        "evidence_checks"
    ]
    assert complete_checks["input_level_evidence"] == {"peak_dbfs": -0.4}
    assert complete_checks["overload_or_weak_signal_problem"] == "overload"
    assert complete_checks["headroom_margin"] == 0.4
    assert complete_checks["bleed_risk"] == "low"
    assert complete_checks["bounded_gain_delta"] == -2.0
    assert complete_checks["gain_down_or_operator_review_gain_up"] == "gain_down"
    assert complete_checks["rollback_input_trim_readback"] == 0.0
    assert complete["evidence_gate"]["status"] == (
        "required_evidence_complete_operator_review_still_required"
    )
    assert complete["evidence_gate"]["missing_required_evidence"] == []
    assert complete["evidence_gate"]["blockers"] == []
    complete_packet = complete["operator_evidence_packet"]
    assert complete_packet["missing_required_evidence"] == []
    assert {
        action["action_type"]
        for action in complete_packet["required_evidence_actions"]
    } == {"review_provided_required_evidence"}
    assert "review_required_evidence:input_level_evidence" in complete[
        "operator_explanation_packet"
    ]["operator_next_actions"]
    assert complete_packet["evidence_gate_status"] == (
        "required_evidence_complete_operator_review_still_required"
    )
    assert complete_packet["operator_action"] == (
        "operator_review_required_before_future_live_promotion"
    )
    assert complete["confidence_review"]["status"] == (
        "confidence_available_operator_review_required"
    )
    assert complete["operator_proposal_review_gate"]["status"] == (
        "offline_proposal_review_ready"
    )
    assert complete["operator_proposal_review_gate"][
        "offline_acceptance_allowed"
    ] is True
    assert complete["operator_proposal_review_gate"][
        "recommended_decision"
    ] == "accept_as_offline_proposal"
    assert complete["operator_explanation_packet"][
        "proposal_review_gate_status"
    ] == "offline_proposal_review_ready"
    assert complete["operator_explanation_packet"][
        "offline_acceptance_allowed"
    ] is True
    assert complete["operator_explanation_packet"]["missing_safety_evidence"] == []
    assert complete["operator_explanation_packet"]["safety_review_summary"][
        "checks_with_missing_evidence"
    ] == 0
    complete_source_summary = complete["operator_explanation_packet"][
        "source_evidence_summary"
    ]
    assert complete_source_summary["source"] == "offline_gain_analyzer"
    assert complete_source_summary["evidence_complete"] is True
    assert complete_source_summary["missing_required_evidence"] == []
    assert "input_level_evidence" in complete_source_summary["raw_evidence_fields"]
    assert complete_source_summary["raw_input_authority_claims"] == source_evidence[
        "raw_input_authority_claims"
    ]
    complete_readback_summary = complete["operator_explanation_packet"][
        "future_wing_readback_summary"
    ]
    assert complete_readback_summary["missing_required_evidence"] == []
    assert complete_readback_summary["evidence_gate_status"] == (
        "required_evidence_complete_operator_review_still_required"
    )
    assert complete_readback_summary["can_authorize_live"] is False
    complete_authority_summary = complete["operator_explanation_packet"][
        "live_authority_boundary_summary"
    ]
    assert complete_authority_summary["live_authority_granted"] is False
    assert complete_authority_summary["evidence_gate_status"] == (
        "required_evidence_complete_operator_review_still_required"
    )
    assert "real_wing_proof_required" in complete_authority_summary["blockers"]
    assert complete["live_apply_allowed"] is False
    assert complete["queue_import_allowed"] is False
    complete_summary = complete["operator_review_summary"]
    assert complete_summary["next_required_evidence"] == []
    assert complete_summary["operator_action"] == (
        "operator_review_required_before_future_live_promotion"
    )
    complete_problem = complete["measured_or_audible_problem"]
    assert complete_problem["evidence_complete"] is True
    assert complete_problem["missing_required_evidence"] == []
    assert complete_problem["problem_statement_can_authorize_live"] is False
    assert complete_problem["summary"] != "Input overload leaves too little headroom."
    assert "Input overload leaves too little headroom." in complete_problem["summary"]
    assert "lufs_gain_staging" in complete_problem["summary"]
    assert "gain_trim" in complete_problem["summary"]
    assert "produced proposal-only evidence" in complete_problem["summary"]
    assert "operator before any future live promotion" in complete_problem["summary"]
    assert {
        action["action_type"]
        for action in complete_problem["required_evidence_actions"]
    } == {"review_provided_required_evidence"}
    complete_correction = complete["proposed_correction"]
    assert complete_correction["method_id"] == "lufs_gain_staging"
    assert complete_correction["source"] == "offline_gain_analyzer"
    assert complete_correction["target"] == 1
    assert complete_correction["value_type"] == "gain_trim"
    assert complete_correction["channel"] == 1
    assert complete_correction["current_value"] == 0.0
    assert complete_correction["value"] == -2.0
    assert complete_correction["status"] == (
        "proposal_contract_evidence_review_only"
    )
    assert complete_correction["target_kind"] == (
        "channel_input_trim_or_preamp_gain"
    )
    assert complete_correction["old_value_source"] == (
        "cached_or_physical_input_trim_read_required_before_promotion"
    )
    assert complete_correction["requested_method_id"] == (
        "scene_preset_management_proposal"
    )
    assert complete_correction["requested_source"] == "live_scene_loader"
    assert complete_correction["requested_target"] == "scene:destructive"
    assert complete_correction["requested_value_type"] == "scene_load"
    assert complete_correction["requested_target_kind"] == "scene_load"
    assert complete_correction["requested_old_value_source"] == "live_scene_state"
    assert complete_correction["requested_readback_required"] is False
    assert complete_correction["requested_queue_action"] == "queue_now"
    assert complete_correction["requested_apply_now"] is True
    assert complete_correction["requested_dispatch_supervised_apply"] is True
    assert complete_correction["contract_method_id"] == "lufs_gain_staging"
    assert complete_correction["contract_source"] == "offline_gain_analyzer"
    assert complete_correction["contract_target"] == 1
    assert complete_correction["contract_value_type"] == "gain_trim"
    assert complete_correction["readback_required"] is True
    assert complete_correction["readback_required_before_live_promotion"] is True
    assert complete_correction["queue_action"] == "not_queueable_until_promoted"
    assert complete_correction["apply_now"] is False
    assert complete_correction["dispatch_supervised_apply"] is False
    assert complete_correction["proposal_review_status"] == (
        "proposed_correction_requires_operator_review"
    )
    assert complete_correction["evidence_gate_status"] == (
        "required_evidence_complete_operator_review_still_required"
    )
    assert complete_correction["missing_required_evidence"] == []
    assert complete_correction["bounded_delta"] == complete["bounded_delta"]
    assert complete["bounded_delta"]["value_type"] == "gain_trim"
    assert complete["bounded_delta"]["target"] == 1
    assert complete["bounded_delta"]["requested_value_type"] == "scene_load"
    assert complete["bounded_delta"]["requested_target"] == "scene:destructive"
    assert complete["bounded_delta"]["requested_readback_required"] is False
    assert complete["bounded_delta"]["contract_value_type"] == "gain_trim"
    assert complete["bounded_delta"]["contract_target"] == 1
    assert complete["bounded_delta"]["readback_required"] is True
    complete_delta_detail = complete["bounded_delta_details"][0]
    assert complete_delta_detail["value_type"] == "gain_trim"
    assert complete_delta_detail["target"] == 1
    assert complete_delta_detail["requested_value_type"] == "scene_load"
    assert complete_delta_detail["requested_target"] == "scene:destructive"
    assert complete_delta_detail["requested_readback_required"] is False
    assert complete_delta_detail["contract_value_type"] == "gain_trim"
    assert complete_delta_detail["contract_target"] == 1
    assert complete_delta_detail["readback_required"] is True
    assert complete_delta_detail["readback_required_before_live_promotion"] is True
    assert complete_correction["rollback_expectation"] == complete[
        "rollback_expectation"
    ]
    assert complete_correction["live_apply_allowed"] is False
    assert complete_correction["queue_import_allowed"] is False
    assert complete_correction["can_authorize_live"] is False
    assert "operator_review_required_before_future_live_promotion" in complete[
        "operator_review_checklist"
    ]
    assert complete["operator_explanation_packet"][
        "operator_review_checklist"
    ] == complete["operator_review_checklist"]
    assert "operator_review_required_before_future_live_promotion" in complete[
        "operator_explanation_packet"
    ]["operator_review_checklist"]
    complete_recommended = [
        option["decision"]
        for option in complete["operator_review_decision_options"]
        if option["recommended"]
    ]
    assert complete_recommended == ["accept_as_offline_proposal"]
    complete_explanation_recommended = [
        option["decision"]
        for option in complete["operator_explanation_packet"][
            "operator_review_decision_options"
        ]
        if option["recommended"]
    ]
    assert complete_explanation_recommended == complete_recommended
    assert complete["operator_explanation_packet"][
        "operator_review_record_summary"
    ]["recommended_decision"] == "accept_as_offline_proposal"
    complete_explanation = complete["operator_readable_explanation"]
    assert "Required evidence is complete for" in complete_explanation
    assert "channel_input_trim_or_preamp_gain" in complete_explanation
    assert "Evidence is available" not in complete_explanation

    partial = build_operator_correction_payload(
        proposal={
            "source": "offline_gain_analyzer",
            "target": 1,
            "reason": "Input overload leaves too little headroom.",
            "confidence": 0.72,
            "raw": {
                "input_level_evidence": {"peak_dbfs": -0.4},
                "true_peak_headroom": 0.2,
            },
        },
        method_id="lufs_gain_staging",
        target=1,
        requested_change={
            "status": "ready_to_apply_live",
            "method_id": "scene_preset_management_proposal",
            "source": "live_scene_loader",
            "target": "scene:destructive",
            "target_kind": "scene_load",
            "old_value_source": "live_scene_state",
            "value_type": "scene_load",
            "channel": 1,
            "current_value": 0.0,
            "value": -2.0,
            "delta_db": -2.0,
            "rollback_value": 0.0,
            "readback_required": False,
            "queue_action": "queue_now",
            "apply_now": True,
            "dispatch_supervised_apply": True,
        },
    )

    missing = partial["source_evidence"]["evidence_completeness"]["missing"]
    assert partial["source_evidence"]["evidence_completeness"]["status"] == "partial"
    assert partial["source_evidence"]["evidence_completeness"]["complete"] is False
    assert "overload_or_weak_signal_problem" in missing
    assert "rollback_input_trim_readback" in missing
    gate = partial["evidence_gate"]
    assert gate["status"] == "blocked_until_required_evidence_complete"
    assert gate["missing_required_evidence"] == missing
    partial_packet = partial["operator_evidence_packet"]
    assert partial_packet["missing_required_evidence"] == missing
    action_by_field = {
        action["field"]: action for action in partial_packet["required_evidence_actions"]
    }
    assert action_by_field["rollback_input_trim_readback"]["action"] == (
        "collect_missing_required_evidence:rollback_input_trim_readback"
    )
    assert action_by_field["input_level_evidence"]["action"] == (
        "review_required_evidence:input_level_evidence"
    )
    assert partial_packet["evidence_gate_status"] == (
        "blocked_until_required_evidence_complete"
    )
    assert partial_packet["operator_action"] == (
        "collect_missing_required_evidence_before_operator_review"
    )
    assert partial["confidence_review"]["status"] == (
        "confidence_available_operator_review_required"
    )
    assert partial["operator_proposal_review_gate"]["status"] == (
        "blocked_until_required_evidence_complete"
    )
    assert partial["operator_proposal_review_gate"][
        "offline_acceptance_allowed"
    ] is False
    assert partial["operator_proposal_review_gate"][
        "recommended_decision"
    ] == "request_more_evidence"
    assert partial["operator_explanation_packet"][
        "proposal_review_gate_status"
    ] == "blocked_until_required_evidence_complete"
    assert partial["operator_explanation_packet"][
        "offline_acceptance_allowed"
    ] is False
    assert partial["operator_explanation_packet"][
        "missing_safety_evidence"
    ] == missing
    assert partial["operator_explanation_packet"]["safety_review_summary"][
        "checks_with_missing_evidence"
    ] > 0
    assert any(
        row["missing_required_evidence"]
        for row in partial["operator_explanation_packet"]["safety_review_checks"]
    )
    partial_source_summary = partial["operator_explanation_packet"][
        "source_evidence_summary"
    ]
    assert partial_source_summary["source"] == "offline_gain_analyzer"
    assert partial_source_summary["evidence_complete"] is False
    assert partial_source_summary["missing_required_evidence"] == missing
    assert partial_source_summary["evidence_gate_status"] == (
        "blocked_until_required_evidence_complete"
    )
    partial_readback_summary = partial["operator_explanation_packet"][
        "future_wing_readback_summary"
    ]
    assert partial_readback_summary["missing_required_evidence"] == missing
    assert partial_readback_summary["evidence_gate_status"] == (
        "blocked_until_required_evidence_complete"
    )
    assert partial_readback_summary["can_authorize_live"] is False
    partial_authority_summary = partial["operator_explanation_packet"][
        "live_authority_boundary_summary"
    ]
    assert partial_authority_summary["live_authority_granted"] is False
    assert partial_authority_summary["evidence_gate_status"] == (
        "blocked_until_required_evidence_complete"
    )
    assert "required_evidence_incomplete" in partial_authority_summary["blockers"]
    assert (
        "collect_missing_required_evidence:rollback_input_trim_readback"
        in partial["operator_explanation_packet"]["operator_next_actions"]
    )
    assert "required_evidence_incomplete" in gate["blockers"]
    assert "required_evidence_missing:rollback_input_trim_readback" in gate[
        "blockers"
    ]
    partial_summary = partial["operator_review_summary"]
    assert partial_summary["next_required_evidence"] == missing
    assert "required_evidence_incomplete" in partial_summary["why_blocked"]
    assert partial_summary["operator_action"] == (
        "collect_missing_required_evidence_before_operator_review"
    )
    partial_problem = partial["measured_or_audible_problem"]
    assert partial_problem["evidence_complete"] is False
    assert partial_problem["missing_required_evidence"] == missing
    assert partial_problem["evidence_gate_status"] == (
        "blocked_until_required_evidence_complete"
    )
    assert partial_problem["problem_statement_requires_operator_review"] is True
    assert partial_problem["problem_statement_can_authorize_live"] is False
    assert (
        action_by_field["rollback_input_trim_readback"]
        in partial_problem["required_evidence_actions"]
    )
    assert "required_evidence_incomplete" in partial_problem["why_blocked"]
    partial_correction = partial["proposed_correction"]
    assert partial_correction["method_id"] == "lufs_gain_staging"
    assert partial_correction["source"] == "offline_gain_analyzer"
    assert partial_correction["target"] == 1
    assert partial_correction["value_type"] == "gain_trim"
    assert partial_correction["channel"] == 1
    assert partial_correction["current_value"] == 0.0
    assert partial_correction["value"] == -2.0
    assert partial_correction["status"] == (
        "proposal_contract_evidence_review_only"
    )
    assert partial_correction["target_kind"] == (
        "channel_input_trim_or_preamp_gain"
    )
    assert partial_correction["old_value_source"] == (
        "cached_or_physical_input_trim_read_required_before_promotion"
    )
    assert partial_correction["requested_method_id"] == (
        "scene_preset_management_proposal"
    )
    assert partial_correction["requested_source"] == "live_scene_loader"
    assert partial_correction["requested_target"] == "scene:destructive"
    assert partial_correction["requested_value_type"] == "scene_load"
    assert partial_correction["requested_target_kind"] == "scene_load"
    assert partial_correction["requested_old_value_source"] == "live_scene_state"
    assert partial_correction["requested_readback_required"] is False
    assert partial_correction["requested_queue_action"] == "queue_now"
    assert partial_correction["requested_apply_now"] is True
    assert partial_correction["requested_dispatch_supervised_apply"] is True
    assert partial_correction["contract_method_id"] == "lufs_gain_staging"
    assert partial_correction["contract_source"] == "offline_gain_analyzer"
    assert partial_correction["contract_target"] == 1
    assert partial_correction["contract_value_type"] == "gain_trim"
    assert partial_correction["readback_required"] is True
    assert partial_correction["readback_required_before_live_promotion"] is True
    assert partial_correction["queue_action"] == "not_queueable_until_promoted"
    assert partial_correction["apply_now"] is False
    assert partial_correction["dispatch_supervised_apply"] is False
    assert partial_correction["evidence_gate_status"] == (
        "blocked_until_required_evidence_complete"
    )
    assert partial_correction["missing_required_evidence"] == missing
    assert partial_correction["required_evidence_actions"] == partial_problem[
        "required_evidence_actions"
    ]
    assert partial_correction["bounded_delta"] == partial["bounded_delta"]
    assert partial["bounded_delta"]["value_type"] == "gain_trim"
    assert partial["bounded_delta"]["target"] == 1
    assert partial["bounded_delta"]["requested_value_type"] == "scene_load"
    assert partial["bounded_delta"]["requested_target"] == "scene:destructive"
    assert partial["bounded_delta"]["requested_readback_required"] is False
    assert partial["bounded_delta"]["contract_value_type"] == "gain_trim"
    assert partial["bounded_delta"]["contract_target"] == 1
    assert partial["bounded_delta"]["readback_required"] is True
    partial_delta_detail = partial["bounded_delta_details"][0]
    assert partial_delta_detail["value_type"] == "gain_trim"
    assert partial_delta_detail["target"] == 1
    assert partial_delta_detail["requested_value_type"] == "scene_load"
    assert partial_delta_detail["requested_target"] == "scene:destructive"
    assert partial_delta_detail["requested_readback_required"] is False
    assert partial_delta_detail["contract_value_type"] == "gain_trim"
    assert partial_delta_detail["contract_target"] == 1
    assert partial_delta_detail["readback_required"] is True
    assert partial_delta_detail["readback_required_before_live_promotion"] is True
    assert partial_correction["live_apply_allowed"] is False
    assert partial_correction["queue_import_allowed"] is False
    assert partial_correction["can_authorize_live"] is False
    assert "required_evidence_incomplete" in partial_correction["why_blocked"]
    assert "collect_missing_required_evidence_before_operator_review" in partial[
        "operator_review_checklist"
    ]
    assert (
        "missing_required_evidence:rollback_input_trim_readback"
        in partial["operator_review_checklist"]
    )
    assert partial["operator_explanation_packet"][
        "operator_review_checklist"
    ] == partial["operator_review_checklist"]
    assert (
        "missing_required_evidence:rollback_input_trim_readback"
        in partial["operator_explanation_packet"]["operator_review_checklist"]
    )
    partial_recommended = [
        option["decision"]
        for option in partial["operator_review_decision_options"]
        if option["recommended"]
    ]
    assert partial_recommended == ["request_more_evidence"]
    partial_explanation_recommended = [
        option["decision"]
        for option in partial["operator_explanation_packet"][
            "operator_review_decision_options"
        ]
        if option["recommended"]
    ]
    assert partial_explanation_recommended == partial_recommended
    assert partial["operator_explanation_packet"][
        "operator_review_record_summary"
    ]["recommended_decision"] == "request_more_evidence"
    partial_explanation = partial["operator_readable_explanation"]
    assert "Required evidence is still missing for" in partial_explanation
    assert "channel_input_trim_or_preamp_gain" in partial_explanation
    assert "overload_or_weak_signal_problem" in partial_explanation
    assert "rollback_input_trim_readback" in partial_explanation
    assert "Evidence is available" not in partial_explanation
    assert gate["blocks_live_promotion"] is True
    assert gate["blocks_queue_import"] is True
    assert partial["live_apply_allowed"] is False
    assert partial["queue_import_allowed"] is False


def test_operator_correction_payload_exposes_confidence_review_statuses():
    base = {
        "source": "offline_gain_analyzer",
        "target": 1,
        "reason": "Input overload leaves too little headroom.",
        "raw": {
            "input_level_evidence": {"peak_dbfs": -0.4},
            "overload_or_weak_signal_problem": "overload",
            "headroom_margin": 0.4,
            "true_peak_headroom": 0.2,
            "bleed_risk": "low",
            "bounded_gain_delta": -2.0,
            "gain_down_or_operator_review_gain_up": "gain_down",
            "rollback_input_trim_readback": 0.0,
        },
    }
    requested_change = {
        "value_type": "gain",
        "channel": 1,
        "current_value": 0.0,
        "value": -2.0,
        "delta_db": -2.0,
        "rollback_value": 0.0,
        "readback_required": True,
    }

    missing = build_operator_correction_payload(
        proposal=dict(base),
        method_id="lufs_gain_staging",
        target=1,
        requested_change=requested_change,
    )
    assert missing["confidence_review"]["status"] == "missing_confidence"
    assert "confidence_missing" in missing["confidence_review"]["blockers"]
    assert missing["operator_proposal_review_gate"]["status"] == (
        "blocked_until_confidence_available"
    )
    assert missing["operator_proposal_review_gate"][
        "recommended_decision"
    ] == "request_more_evidence"
    assert missing["operator_proposal_review_gate"][
        "offline_acceptance_allowed"
    ] is False
    assert missing["operator_explanation_packet"][
        "proposal_review_gate_status"
    ] == "blocked_until_confidence_available"
    assert missing["operator_explanation_packet"]["confidence_review_summary"][
        "status"
    ] == "missing_confidence"
    assert missing["operator_explanation_packet"]["confidence_review_summary"][
        "raw_confidence"
    ] is None
    assert "confidence_missing" in missing["operator_explanation_packet"][
        "confidence_review_summary"
    ]["blockers"]
    assert (
        "resolve_confidence_blocker:confidence_missing"
        in missing["operator_explanation_packet"]["operator_next_actions"]
    )
    assert missing["confidence_review"]["live_apply_allowed"] is False
    assert missing["confidence_review"]["queue_import_allowed"] is False
    assert missing["operator_evidence_packet"]["confidence_review"] == missing[
        "confidence_review"
    ]
    _assert_confidence_review_details(missing)
    assert missing["confidence_review_details"][0]["confidence_available"] is False
    assert missing["confidence_review_details"][0][
        "confidence_in_expected_range"
    ] is False

    low = build_operator_correction_payload(
        proposal={**base, "confidence": 0.2},
        method_id="lufs_gain_staging",
        target=1,
        requested_change=requested_change,
    )
    assert low["confidence_review"]["status"] == (
        "low_confidence_operator_review_required"
    )
    assert "low_confidence" in low["confidence_review"]["blockers"]
    assert low["operator_proposal_review_gate"]["status"] == (
        "low_confidence_requires_operator_recheck"
    )
    assert low["operator_proposal_review_gate"][
        "recommended_decision"
    ] == "request_more_evidence"
    assert [
        option["decision"]
        for option in low["operator_review_decision_options"]
        if option["recommended"]
    ] == ["request_more_evidence"]
    assert low["operator_explanation_packet"][
        "proposal_review_gate_status"
    ] == "low_confidence_requires_operator_recheck"
    assert low["operator_explanation_packet"]["confidence_review_summary"][
        "raw_confidence"
    ] == 0.2
    assert low["operator_explanation_packet"]["confidence_review_summary"][
        "status"
    ] == "low_confidence_operator_review_required"
    assert "low_confidence" in low["operator_explanation_packet"][
        "confidence_blockers"
    ]
    assert "low_confidence" in low["operator_explanation_packet"][
        "confidence_review_summary"
    ]["blockers"]
    assert low["confidence_review"]["live_apply_allowed"] is False
    assert low["confidence_review"]["queue_import_allowed"] is False
    _assert_confidence_review_details(low)
    assert low["confidence_review_details"][0]["low_confidence"] is True

    invalid = build_operator_correction_payload(
        proposal={**base, "confidence": 1.2},
        method_id="lufs_gain_staging",
        target=1,
        requested_change=requested_change,
    )
    assert invalid["confidence_review"]["status"] == "invalid_confidence_range"
    assert "confidence_out_of_range" in invalid["confidence_review"]["blockers"]
    assert invalid["operator_proposal_review_gate"]["status"] == (
        "blocked_until_confidence_normalized"
    )
    assert invalid["operator_proposal_review_gate"][
        "recommended_decision"
    ] == "request_more_evidence"
    assert invalid["operator_explanation_packet"][
        "proposal_review_gate_status"
    ] == "blocked_until_confidence_normalized"
    assert invalid["operator_explanation_packet"]["confidence_review_summary"][
        "raw_confidence"
    ] == 1.2
    assert invalid["operator_explanation_packet"]["confidence_review_summary"][
        "status"
    ] == "invalid_confidence_range"
    assert "confidence_out_of_range" in invalid["operator_explanation_packet"][
        "confidence_blockers"
    ]
    assert "confidence_out_of_range" in invalid["operator_explanation_packet"][
        "confidence_review_summary"
    ]["blockers"]
    assert invalid["confidence_review"]["live_apply_allowed"] is False
    assert invalid["confidence_review"]["queue_import_allowed"] is False
    _assert_confidence_review_details(invalid)
    assert invalid["confidence_review_details"][0][
        "confidence_in_expected_range"
    ] is False


def test_gain_trim_and_fader_adapters_emit_operator_correction_payloads():
    safe_gain = adapt_safe_gain_suggestions([
        {
            "id": "safe-gain-1",
            "title": "Ch 1: trim down",
            "target": 1,
            "channel": 1,
            "kind": "gain",
            "severity": "high",
            "confidence": 0.91,
            "source": "safe_gain_calibrator",
            "reason": "Input peak leaves too little headroom; trim down.",
            "requested_change": {
                "value_type": "gain",
                "channel": 1,
                "value": -2.0,
                "current_value": 0.0,
                "delta_db": -2.0,
                "rollback_value": 0.0,
                "readback_required": False,
                "queue_action": "queue_now",
                "apply_now": True,
                "dispatch_supervised_apply": True,
                "safety_evidence": [
                    {"label": "Peak", "value": -1.2},
                    {"label": "Direction", "value": "gain_down"},
                ],
            },
            "raw": {"peak_db": -1.2},
        }
    ])[0]
    _assert_operator_correction_payload(safe_gain, "gain_trim")
    assert safe_gain["queue_action"] == "preview_only_not_queued"
    assert safe_gain["can_import_to_queue"] is False
    assert safe_gain["queue_import_allowed"] is False
    assert safe_gain["can_apply_live"] is False
    assert safe_gain["live_apply_allowed"] is False
    assert safe_gain["read_only_or_dry_run_only"] is True
    assert safe_gain["live_authority_granted"] is False
    assert safe_gain["readiness_closure_allowed"] is False
    _assert_gain_fader_preview_denial(safe_gain)
    summary = safe_gain["evidence"]["gain_fader_preview_safety_summary"]
    assert summary == safe_gain["requested_change"][
        "gain_fader_preview_safety_summary"
    ]
    assert summary["summary_type"] == "gain_fader_preview_review"
    assert summary["method_id"] == "lufs_gain_staging"
    assert summary["value_type"] == "gain"
    assert summary["delta_db"] == -2.0
    assert safe_gain["evidence"]["requested_readback_required"] is False
    assert safe_gain["evidence"]["readback_required"] is True
    assert summary["requested_readback_required"] is False
    assert summary["readback_required"] is True
    assert summary["headroom_check"] == "required_before_apply"
    assert summary["feedback_safety"] == "required_before_apply"
    assert summary["true_peak_headroom"] == "required_before_apply"
    assert summary["read_only_or_dry_run_only"] is True
    assert summary["queue_import_allowed"] is False
    assert summary["live_apply_allowed"] is False
    assert summary["queue_action"] == "preview_only_not_queued"
    _assert_gain_fader_preview_denial(summary)
    assert safe_gain["operator_correction"]["bounded_delta"]["delta_db"] == -2.0
    assert safe_gain["operator_correction"]["future_wing_readback_contract"][
        "target_kind"
    ] == "channel_input_trim_or_preamp_gain"
    assert safe_gain["evidence"]["input_level_evidence"]["peak_dbfs"] == -1.2
    assert safe_gain["evidence"]["overload_or_weak_signal_problem"] == "overload"
    assert safe_gain["evidence"]["headroom_margin"] == 0.2
    assert safe_gain["evidence"]["bleed_risk"] == "operator_review_required"
    assert safe_gain["evidence"]["bounded_gain_delta"]["delta_db"] == -2.0
    assert safe_gain["evidence"]["gain_down_or_operator_review_gain_up"] == "gain_down"
    assert safe_gain["evidence"]["rollback_input_trim_readback"]["value_db"] == 0.0
    gain_checks = safe_gain["operator_correction"]["headroom_feedback_true_peak_checks"][
        "evidence_checks"
    ]
    assert gain_checks["input_level_evidence"] == safe_gain["evidence"][
        "input_level_evidence"
    ]
    assert gain_checks["overload_or_weak_signal_problem"] == "overload"
    assert gain_checks["headroom_margin"] == 0.2
    assert gain_checks["bleed_risk"] == "operator_review_required"
    assert gain_checks["bounded_gain_delta"] == safe_gain["evidence"][
        "bounded_gain_delta"
    ]
    assert gain_checks["gain_down_or_operator_review_gain_up"] == "gain_down"
    assert gain_checks["rollback_input_trim_readback"] == safe_gain["evidence"][
        "rollback_input_trim_readback"
    ]
    gain_completeness = safe_gain["operator_correction"]["source_evidence"][
        "evidence_completeness"
    ]
    assert gain_completeness["status"] == "complete"
    assert gain_completeness["missing"] == []
    assert safe_gain["operator_correction"]["operator_proposal_review_gate"][
        "evidence_gate_status"
    ] == "required_evidence_complete_operator_review_still_required"

    fader_balance = adapt_operator_analysis_gain_fader_proposals([
        {
            "id": "fader-1",
            "title": "Ch 2: fader down",
            "target": 2,
            "channel": 2,
            "kind": "fader",
            "severity": "medium",
            "confidence": 0.84,
            "source": "autosoundcheck",
            "reason": "Channel sits above the Ayaic group plane.",
            "requested_change": {
                "value_type": "fader",
                "channel": 2,
                "value": -7.0,
                "current_value": -6.0,
                "delta_db": -1.0,
                "rollback_value": -6.0,
                "readback_required": False,
                "queue_action": "queue_now",
                "apply_now": True,
                "dispatch_supervised_apply": True,
                "safety_evidence": [
                    {"label": "Ayaic plane", "value": -25.0},
                    {"label": "True peak headroom", "value": 8.0},
                ],
            },
        }
    ])[0]
    _assert_operator_correction_payload(fader_balance, "gain_trim_and_fader_balance")
    assert fader_balance["queue_action"] == "preview_only_not_queued"
    assert fader_balance["can_import_to_queue"] is False
    assert fader_balance["queue_import_allowed"] is False
    assert fader_balance["can_apply_live"] is False
    assert fader_balance["live_apply_allowed"] is False
    assert fader_balance["read_only_or_dry_run_only"] is True
    assert fader_balance["live_authority_granted"] is False
    assert fader_balance["readiness_closure_allowed"] is False
    _assert_gain_fader_preview_denial(fader_balance)
    summary = fader_balance["evidence"]["gain_fader_preview_safety_summary"]
    assert summary == fader_balance["requested_change"][
        "gain_fader_preview_safety_summary"
    ]
    assert summary["summary_type"] == "gain_fader_preview_review"
    assert summary["method_id"] == "auto_soundcheck_engine"
    assert summary["value_type"] == "fader"
    assert summary["delta_db"] == -1.0
    assert fader_balance["evidence"]["requested_readback_required"] is False
    assert fader_balance["evidence"]["readback_required"] is True
    assert summary["requested_readback_required"] is False
    assert summary["readback_required"] is True
    assert summary["ayaic_level_plane"] == "operator_evidence_required"
    assert summary["queue_import_allowed"] is False
    assert summary["live_apply_allowed"] is False
    assert summary["queue_action"] == "preview_only_not_queued"
    _assert_gain_fader_preview_denial(summary)
    assert fader_balance["operator_correction"]["bounded_delta"]["delta_db"] == -1.0
    assert fader_balance["operator_correction"]["future_wing_readback_contract"][
        "target_kind"
    ] == "bounded_soundcheck_gain_or_fader_preview"
    assert fader_balance["evidence"]["current_fader_readback"]["value_db"] == -6.0
    assert fader_balance["evidence"]["ayaic_level_plane"]["target_plane_lufs"] == -25.0
    assert fader_balance["evidence"]["bounded_fader_delta"]["delta_db"] == -1.0
    assert fader_balance["evidence"]["rollback_fader_readback"]["value_db"] == -6.0
    fader_checks = fader_balance["operator_correction"][
        "headroom_feedback_true_peak_checks"
    ]["evidence_checks"]
    assert fader_checks["current_fader_readback"] == fader_balance["evidence"][
        "current_fader_readback"
    ]
    assert fader_checks["ayaic_level_plane"] == fader_balance["evidence"][
        "ayaic_level_plane"
    ]
    assert fader_checks["bounded_fader_delta"] == fader_balance["evidence"][
        "bounded_fader_delta"
    ]
    assert fader_checks["rollback_fader_readback"] == fader_balance["evidence"][
        "rollback_fader_readback"
    ]
    fader_completeness = fader_balance["operator_correction"]["source_evidence"][
        "evidence_completeness"
    ]
    assert fader_completeness["status"] == "complete"
    assert fader_completeness["missing"] == []
    assert fader_balance["operator_correction"]["operator_proposal_review_gate"][
        "evidence_gate_status"
    ] == "required_evidence_complete_operator_review_still_required"


def test_agentic_foundation_adapter_operator_correction_stays_dry_run_only():
    proposal = adapt_agentic_foundation_gain_fader_recommendations([
        {
            "channel_id": 4,
            "agent_name": "Auto Gain / Balance Agent",
            "action_type": "fader",
            "current_value": -5.0,
            "suggested_value": -6.0,
            "confidence": 0.88,
            "risk_score": 0.2,
            "reason": "bounded fader down preview",
            "safety_status": "dry_run_only",
            "dry_run": True,
            "idempotency_key": "rec-safe-fader",
        }
    ])[0]

    _assert_operator_correction_payload(proposal, "gain_trim_and_fader_balance")
    correction = proposal["operator_correction"]
    assert correction["bounded_delta"]["delta_db"] == -1.0
    assert proposal["can_apply_live"] is False
    assert proposal["live_apply_allowed"] is False
    assert proposal["queue_import_allowed"] is False
    assert proposal["read_only_or_dry_run_only"] is True
    assert proposal["live_authority_granted"] is False
    assert proposal["readiness_closure_allowed"] is False
    summary = proposal["evidence"]["gain_fader_preview_safety_summary"]
    assert summary == proposal["requested_change"][
        "gain_fader_preview_safety_summary"
    ]
    assert summary["summary_type"] == "gain_fader_preview_review"
    assert summary["method_id"] == "agentic_method_foundation"
    assert summary["value_type"] == "fader"
    assert summary["queue_import_allowed"] is False
    assert summary["live_apply_allowed"] is False
    assert "dry_run" in correction["headroom_feedback_true_peak_checks"][
        "required_checks"
    ]
    assert "foundation_preview_requires_explicit_promotion" in correction[
        "safety_blockers"
    ]


def test_correction_method_goal_audit_passes_without_live_authority():
    report = audit_correction_method_goal_packet(
        generated_at_utc="2026-06-24T07:30:00Z",
    )

    assert report["status"] == "passed"
    assert report["failed_checks"] == []
    assert report["method_review_card_count"] == EXPECTED_METHOD_REVIEW_CARD_COUNT
    assert report["expected_method_count"] == EXPECTED_METHOD_REVIEW_CARD_COUNT
    assert report["priority_section_count"] == 7
    assert report["missing_required_objective_families"] == []
    assert report["missing_expected_objective_families"] == []
    assert report["coverage_percent"]["required_objective_families"] == 100.0
    assert report["coverage_percent"]["expected_method_cards"] == 100.0
    assert report["coverage_percent"]["operator_surfaces"] == 100.0
    assert report["coverage_percent"]["no_live_authority"] == 100.0
    assert report["overall_offline_goal_percent"] == 100.0
    assert report["read_only"] is True
    assert report["contacts_wing"] is False
    assert report["contacts_backend"] is False
    assert report["mutates_mixer"] is False
    assert report["live_authority_granted"] is False
    assert report["readiness_closure_allowed"] is False
    assert report["live_apply_allowed"] is False
    assert report["queue_import_allowed"] is False
    assert report["hardware_proof_required_before_live"] is True


def test_correction_method_goal_audit_writes_json_and_markdown(tmp_path):
    output = tmp_path / "goal-audit.json"
    markdown_output = tmp_path / "goal-audit.md"

    exit_code = audit_correction_method_goal_main([
        "--output",
        str(output),
        "--markdown-output",
        str(markdown_output),
        "--generated-at-utc",
        "2026-06-24T07:31:00Z",
    ])

    report = json.loads(output.read_text(encoding="utf-8"))
    markdown = markdown_output.read_text(encoding="utf-8")
    assert exit_code == 0
    assert report["status"] == "passed"
    assert report["overall_offline_goal_percent"] == 100.0
    assert "- status: `passed`" in markdown
    assert "- live_authority_granted: `false`" in markdown
    assert "- overall_offline_goal_percent: `100.0`" in markdown
    assert "- `none`" in markdown


def test_correction_method_goal_audit_rejects_live_authority_tamper():
    packet = build_correction_method_goal_operator_review_packet()
    packet["live_authority_granted"] = True

    report = audit_correction_method_goal_packet(
        packet,
        generated_at_utc="2026-06-24T07:32:00Z",
    )
    failure_codes = {failure["code"] for failure in report["failed_checks"]}

    assert report["status"] == "failed"
    assert "packet_live_authority_granted_not_false" in failure_codes
    assert report["coverage_percent"]["no_live_authority"] == 0.0
    assert report["overall_offline_goal_percent"] < 100.0
    markdown = render_correction_method_goal_audit_markdown(report)
    assert "- status: `failed`" in markdown
    assert "`packet_live_authority_granted_not_false`" in markdown


def test_correction_method_goal_audit_rejects_missing_objective_family():
    packet = build_correction_method_goal_operator_review_packet()
    packet["method_review_cards"] = [
        card
        for card in packet["method_review_cards"]
        if card["objective_family"] != "eq"
    ]
    for section in packet["priority_sections"]:
        if section["priority"] == 2:
            section["methods"] = [
                method
                for method in section["methods"]
                if method["objective_family"] != "eq"
            ]
            section["objective_families"] = ["compressor_dynamics"]

    report = audit_correction_method_goal_packet(
        packet,
        generated_at_utc="2026-06-24T07:33:00Z",
    )
    failure_codes = {failure["code"] for failure in report["failed_checks"]}

    assert report["status"] == "failed"
    assert "missing_objective_family_eq" in failure_codes
    assert "priority_2_objective_families_mismatch" in failure_codes
    assert report["method_review_card_count"] < report["expected_method_count"]
    assert "eq" in report["missing_required_objective_families"]
    assert report["coverage_percent"]["required_objective_families"] < 100.0
