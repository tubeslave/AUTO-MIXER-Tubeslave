#!/usr/bin/env python3
"""Verify correction-method operator review export artifacts."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from tools.audit_correction_method_goal import (  # noqa: E402
    audit_correction_method_goal_packet,
)

EXPECTED_MANIFEST_SCHEMA = (
    "automixer.correction_method_operator_review_export_manifest.v1"
)
EXPECTED_PACKET_SCHEMA = (
    "automixer.correction_method_goal_operator_review_packet.v1"
)
EXPECTED_GOAL_AUDIT_SCHEMA = "automixer.correction_method_goal_audit.v1"
EXPECTED_DECISION_TEMPLATE_SCHEMA = (
    "automixer.correction_method_operator_decision_template.v1"
)
EXPECTED_DECISION_RECORD_SCHEMA = (
    "automixer.correction_method_operator_decision_record.v1"
)
EXPECTED_MANIFEST_STATUS = "operator_review_export_ready_proposal_only"
EXPECTED_PACKET_STATUS = "operator_review_packet_ready_proposal_only"
EXPECTED_GOAL_AUDIT_STATUS = "passed"
EXPECTED_DECISION_TEMPLATE_STATUS = (
    "operator_review_template_pending_proposal_only"
)
ALLOWED_DECISION_RECORD_STATUSES = (
    EXPECTED_DECISION_TEMPLATE_STATUS,
    "operator_decisions_recorded_proposal_only",
    "operator_decisions_validated_proposal_only",
)
REQUIRED_OFFLINE_REVIEW_DECISIONS = (
    "request_more_evidence",
    "accept_as_offline_proposal",
    "reject_proposal",
    "defer_until_real_wing_proof",
)
EXPECTED_ARTIFACTS = {
    "packet_json": "packet.json",
    "operator_review_markdown": "operator-review.md",
    "operator_decision_template": "operator-decision-template.json",
    "goal_audit_json": "goal-audit.json",
    "goal_audit_markdown": "goal-audit.md",
}
EXPECTED_VERIFICATION_REPORT_REFERENCE = "verification-report.json"
FALSE_AUTHORITY_FLAGS = (
    "contacts_wing",
    "contacts_backend",
    "queries_console",
    "mutates_mixer",
    "mutates_console_state",
    "live_write",
    "live_authority_granted",
    "readiness_closure_allowed",
    "live_apply_allowed",
    "queue_import_allowed",
)
TRUE_REVIEW_FLAGS = (
    "read_only_or_dry_run_only",
    "operator_review_required",
    "future_wing_readback_required",
    "hardware_proof_required_before_live",
)
REQUIRED_FUTURE_PROOFS = (
    "real_wing_readonly_connection_artifact",
    "physical_console_readback_for_old_and_proposed_values",
    "operator_review_acceptance_record",
    "rollback_readback_proof",
    "emergency_stop_proof",
    "connection_loss_timeout_fail_closed_proof",
    "safety_kernel_trace_for_target_action",
)
REQUIRED_FUTURE_WING_READBACK_CONTRACT_SUMMARY_KEYS = (
    "target_kind",
    "old_value_source",
    "readback_targets",
    "required_evidence_fields",
    "operator_next_step",
    "denial_reason",
)
REQUIRED_FUTURE_WING_READBACK_OPERATOR_NEXT_STEP = (
    "capture_real_wing_readback_and_rollback_proof_before_live_promotion"
)
REQUIRED_FUTURE_WING_READBACK_DENIAL_REASON = (
    "real_wing_hardware_readback_rollback_and_operator_review_required"
)
ALLOWED_BLOCKED_LIVE_ACTIONS = (
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
REQUIRED_LIVE_AUTHORITY_BOUNDARY_SUMMARY = (
    "allowed_current_live_write_kinds=fader,gain",
    "can_authorize_live=false",
    "denial_reason=real_wing_hardware_readback_rollback_and_operator_review_required",
    "live_apply_allowed=false",
    "live_authority_granted=false",
    "queue_import_allowed=false",
    "readiness_closure_allowed=false",
)
REQUIRED_OPERATOR_WHY_SAFE_SUMMARY_KEYS = (
    "read_only_or_dry_run_only",
    "live_apply_allowed",
    "queue_import_allowed",
    "operator_review_required",
    "bounded_delta",
    "required_safety_checks",
)
REQUIRED_OPERATOR_WHY_BLOCKED_SUMMARY_KEYS = (
    "denial_reason",
    "future_wing_readback_required",
    "hardware_proof_required_before_live",
    "readiness_closure_allowed",
    "blocked_live_actions",
    "future_live_required_proofs",
)
ALLOWED_REQUIRED_SAFETY_CHECKS = (
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
ALLOWED_BOUNDED_DELTA_UNITS = (
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
ALLOWED_BOUNDED_DELTA_POLICIES = (
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
ALLOWED_FUTURE_WING_READBACK_TARGETS = (
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
ALLOWED_ROLLBACK_EXPECTATIONS = (
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
REQUIRED_OPERATOR_PROBLEM_REVIEW_SUMMARY = (
    "source_evidence=required_from_offline_proposal_payload",
    "source_evidence_details=operator_must_review_before_acceptance",
    "measured_or_audible_problem=required_operator_visible_problem_statement",
    "proposed_correction=required_operator_visible_correction_statement",
    "operator_action=confirm_problem_statement_matches_source_evidence_and_correction",
    "live_boundary=offline_only_until_real_wing_readback_and_rollback_proof",
)
REQUIRED_OPERATOR_METHOD_REVIEW_FOCUS_KEYS = (
    "method_id",
    "priority",
    "priority_label",
    "objective_family",
    "required_evidence",
    "required_safety_checks",
    "bounded_delta",
    "rollback_expectation",
    "future_wing_readback_targets",
    "blocked_live_actions",
    "live_boundary",
)
REQUIRED_METHOD_CARD_TEXT_FIELDS = (
    "method_label",
    "objective_family",
    "priority_label",
    "priority_review_scope",
    "domain",
    "status",
    "review_status",
    "bounded_delta_unit",
    "bounded_delta_policy",
    "rollback_expectation",
    "operator_question",
    "operator_next_step",
    "denial_reason",
)
REQUIRED_METHOD_CARD_LIST_FIELDS = (
    "required_evidence",
    "required_safety_checks",
    "required_operator_surfaces",
    "operator_payload_required_fields",
    "operator_evidence_required_fields",
    "operator_problem_review_summary",
    "operator_method_review_focus",
    "why_safe_summary",
    "why_blocked_summary",
    "goal_requirement_details",
    "method_specific_review_details",
    "operator_actionable_explanation_details",
    "evidence_problem_correction_chain_details",
    "safety_blocker_details",
    "headroom_feedback_true_peak_check_details",
    "test_evidence_references",
    "what_operator_must_confirm",
    "required_confirmation_details",
    "operator_acceptance_gate_details",
    "operator_review_checklist",
    "operator_review_checklist_details",
    "operator_review_decision_option_details",
    "operator_review_record_required_fields",
    "operator_review_record_field_details",
    "operator_next_action_details",
    "operator_review_decisions",
    "future_wing_readback_targets",
    "future_wing_readback_contract_summary",
    "future_live_required_proofs",
    "future_live_requirement_details",
    "blocked_live_actions",
    "live_authority_boundary_summary",
)
REQUIRED_METHOD_CARD_CONTRACT_FIELDS = (
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
REQUIRED_METHOD_CARD_CONTRACT_SOURCE_FIELDS = {
    "test_evidence": ("test_evidence_references",),
}
ABSTRACT_METHOD_CARD_CONTRACT_SOURCE_FIELDS = (
    "operator_review_contract_tests",
    "export_verification_tests",
    "no_wing_release_gate_tests",
)
PRIORITY_METHOD_CARD_SCALAR_MIRROR_FIELDS = (
    REQUIRED_METHOD_CARD_TEXT_FIELDS
    + REQUIRED_METHOD_CARD_CONTRACT_FIELDS
    + ("priority", "review_scope")
)
PRIORITY_METHOD_CARD_LIST_MIRROR_FIELDS = REQUIRED_METHOD_CARD_LIST_FIELDS + (
    "missing_required_operator_surfaces",
)
PRIORITY_METHOD_CARD_FLAG_MIRROR_FIELDS = (
    "read_only_or_dry_run_only",
    "live_authority_granted",
    "readiness_closure_allowed",
    "live_apply_allowed",
    "queue_import_allowed",
    "contacts_wing",
    "contacts_backend",
    "mutates_mixer",
)
PRIORITY_SECTION_LIST_AGGREGATE_FIELDS = (
    "required_evidence",
    "required_safety_checks",
    "test_evidence_references",
    "operator_payload_required_fields",
    "operator_evidence_required_fields",
    "operator_problem_review_summary",
    "operator_method_review_focus",
    "why_safe_summary",
    "why_blocked_summary",
    "goal_requirement_details",
    "method_specific_review_details",
    "operator_actionable_explanation_details",
    "evidence_problem_correction_chain_details",
    "safety_blocker_details",
    "headroom_feedback_true_peak_check_details",
    "future_wing_readback_targets",
    "future_wing_readback_contract_summary",
    "future_live_required_proofs",
    "future_live_requirement_details",
    "blocked_live_actions",
    "live_authority_boundary_summary",
    "operator_review_record_required_fields",
    "required_confirmation_details",
    "operator_acceptance_gate_details",
    "operator_review_decision_option_details",
    "operator_review_checklist_details",
    "operator_review_record_field_details",
    "operator_next_action_details",
)
PRIORITY_SECTION_SCALAR_AGGREGATE_FIELDS = (
    ("bounded_delta_policies", "bounded_delta_policy"),
    ("rollback_expectations", "rollback_expectation"),
)
PRIORITY_SECTION_EMPTY_AUTHORITY_LIST_FIELDS = (
    "missing_required_operator_surface_method_ids",
    "live_authority_method_ids",
    "live_apply_method_ids",
    "queueable_method_ids",
)
PRIORITY_SECTION_FALSE_FLAGS = (
    "contacts_wing",
    "contacts_backend",
    "mutates_mixer",
    "live_authority_granted",
    "readiness_closure_allowed",
    "live_apply_allowed",
    "queue_import_allowed",
)
PRIORITY_SECTION_TRUE_FLAGS = (
    "read_only_or_dry_run_only",
    "operator_review_required",
    "future_wing_readback_required",
    "hardware_proof_required_before_live",
)
REQUIRED_METHOD_CARD_CONFIRMATIONS = (
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
REQUIRED_OPERATOR_REVIEW_RECORD_FIELDS = (
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
REQUIRED_BOUNDED_DELTA_ACKNOWLEDGEMENT = (
    "operator_confirms_bounded_delta_reviewed_before_offline_acceptance"
)
REQUIRED_MEASURED_PROBLEM_ACKNOWLEDGEMENT = (
    "operator_confirms_measured_or_audible_problem_reviewed_before_offline_acceptance"
)
REQUIRED_PROPOSED_CORRECTION_ACKNOWLEDGEMENT = (
    "operator_confirms_proposed_correction_reviewed_before_offline_acceptance"
)
REQUIRED_SOURCE_EVIDENCE_ACKNOWLEDGEMENT = (
    "operator_confirms_source_evidence_reviewed_before_offline_acceptance"
)
REQUIRED_TEST_EVIDENCE_ACKNOWLEDGEMENT = (
    "operator_confirms_test_evidence_reviewed_before_offline_acceptance"
)
REQUIRED_VERIFICATION_REPORT_ACKNOWLEDGEMENT = (
    "operator_confirms_verification_report_reviewed_before_offline_acceptance"
)
REQUIRED_CONFIDENCE_ACKNOWLEDGEMENT = (
    "operator_confirms_confidence_reviewed_before_offline_acceptance"
)
REQUIRED_OPERATOR_EXPLANATION_ACKNOWLEDGEMENT = (
    "operator_confirms_operator_explanation_reviewed_before_offline_acceptance"
)
REQUIRED_FUTURE_WING_READBACK_ACKNOWLEDGEMENT = (
    "operator_confirms_future_wing_readback_contract_reviewed_before_offline_acceptance"
)
REQUIRED_HEADROOM_FEEDBACK_TRUE_PEAK_ACKNOWLEDGEMENT = (
    "operator_confirms_headroom_feedback_true_peak_checks_reviewed_before_offline_acceptance"
)
REQUIRED_ROLLBACK_READBACK_ACKNOWLEDGEMENT = (
    "operator_confirms_rollback_and_readback_expectations_reviewed_before_offline_acceptance"
)
REQUIRED_SAFETY_BLOCKERS_ACKNOWLEDGEMENT = (
    "operator_confirms_safety_blockers_reviewed_before_offline_acceptance"
)
REQUIRED_OPERATOR_CONFIRMATION_ACKNOWLEDGEMENT = (
    "operator_confirms_all_required_confirmations_reviewed_before_offline_acceptance"
)
REQUIRED_OFFLINE_ACCEPTANCE_BOUNDARY_ACKNOWLEDGEMENT = (
    "offline_acceptance_does_not_authorize_live_apply_queue_import_or_readiness_closure"
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
REQUIRED_OPERATOR_SURFACES = (
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
)
FORBIDDEN_MARKDOWN_TRUE_FLAGS = (
    "live_authority_granted: `true`",
    "live_apply_allowed: `true`",
    "queue_import_allowed: `true`",
    "contacts_wing: `true`",
    "contacts_backend: `true`",
    "mutates_mixer: `true`",
)
EXPECTED_PRIORITY_REVIEW_SUMMARIES = (
    (
        "first_priority_review_summary",
        "first_priority",
        "First",
        1,
        "Gain / Trim and Fader / Balance",
    ),
    (
        "second_priority_review_summary",
        "second_priority",
        "Second",
        2,
        "EQ and Compressor",
    ),
    (
        "third_priority_review_summary",
        "third_priority",
        "Third",
        3,
        "Gate / Expander / De-esser",
    ),
    (
        "fourth_priority_review_summary",
        "fourth_priority",
        "Fourth",
        4,
        "FX / Sends",
    ),
    (
        "fifth_priority_review_summary",
        "fifth_priority",
        "Fifth",
        5,
        "Pan / Stereo",
    ),
    (
        "sixth_priority_review_summary",
        "sixth_priority",
        "Sixth",
        6,
        "Routing / Groups / Scenes",
    ),
    (
        "seventh_priority_review_summary",
        "seventh_priority",
        "Seventh",
        7,
        "Master / Output",
    ),
)
MARKDOWN_SUMMARY_MIRROR_KEYS = (
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
)


def _failure(code: str, detail: str) -> Dict[str, str]:
    return {"code": code, "detail": detail}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json_object(
    path: Path,
    *,
    label: str,
    failures: List[Dict[str, str]],
) -> Dict[str, Any]:
    if not path.exists():
        failures.append(_failure(f"missing_{label}", f"{path} does not exist"))
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        failures.append(_failure(f"invalid_{label}_json", str(exc)))
        return {}
    if not isinstance(payload, dict):
        failures.append(
            _failure(f"invalid_{label}_shape", f"{label} must be a JSON object")
        )
        return {}
    return payload


def _artifact_rows(
    manifest: Mapping[str, Any],
    failures: List[Dict[str, str]],
) -> Dict[str, Mapping[str, Any]]:
    raw_rows = manifest.get("artifacts")
    if not isinstance(raw_rows, list):
        failures.append(
            _failure("manifest_artifacts_not_list", "artifacts must be a list")
        )
        return {}
    rows: Dict[str, Mapping[str, Any]] = {}
    for row in raw_rows:
        if not isinstance(row, dict):
            failures.append(
                _failure("manifest_artifact_row_not_object", repr(row))
            )
            continue
        artifact_id = str(row.get("artifact_id") or "")
        if not artifact_id:
            failures.append(_failure("manifest_artifact_missing_id", repr(row)))
            continue
        if artifact_id in rows:
            failures.append(
                _failure("manifest_artifact_duplicate_id", artifact_id)
            )
            continue
        rows[artifact_id] = row
    return rows


def _check_equal(
    failures: List[Dict[str, str]],
    *,
    code: str,
    actual: Any,
    expected: Any,
) -> None:
    if actual != expected:
        failures.append(
            _failure(code, f"expected {expected!r}, got {actual!r}")
        )


def _check_false_flags(
    failures: List[Dict[str, str]],
    *,
    payload: Mapping[str, Any],
    label: str,
    flags: Iterable[str],
) -> None:
    for flag in flags:
        if payload.get(flag) is not False:
            failures.append(
                _failure(f"{label}_{flag}_not_false", f"{flag}={payload.get(flag)!r}")
            )


def _check_true_flags(
    failures: List[Dict[str, str]],
    *,
    payload: Mapping[str, Any],
    label: str,
    flags: Iterable[str],
) -> None:
    for flag in flags:
        if payload.get(flag) is not True:
            failures.append(
                _failure(f"{label}_{flag}_not_true", f"{flag}={payload.get(flag)!r}")
            )


def _sorted_str_items(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return sorted(str(item) for item in value)


def _expected_operator_review_record_field_details(
    required_fields: Any,
) -> List[str]:
    return [
        (
            f"{field}:source={source};type={field_type};"
            f"action=fill_operator_review_record_field:{field}"
        )
        for field in _sorted_str_items(required_fields)
        for source, field_type in [
            OPERATOR_REVIEW_RECORD_FIELD_PROFILES.get(
                field,
                ("operator_review_record_template", "text"),
            )
        ]
    ]


def _expected_operator_review_checklist_details(
    checklist_items: Any,
) -> List[str]:
    details: List[str] = []
    for item in _sorted_str_items(checklist_items):
        if item.startswith("missing_required_evidence:"):
            source = "required_evidence_status"
            action_type = "collect_missing_required_evidence"
            acknowledgement_fields = "required_confirmations_acknowledgement"
        elif item.startswith("confidence_blocker:"):
            source = "confidence_review"
            action_type = "resolve_confidence_blocker"
            acknowledgement_fields = "confidence_acknowledgement"
        else:
            source, action_type, acknowledgement_fields = (
                OPERATOR_REVIEW_CHECKLIST_DETAIL_PROFILES.get(
                    item,
                    (
                        "operator_review_checklist",
                        "operator_review_check",
                        "required_confirmations_acknowledgement",
                    ),
                )
            )
        details.append(
            (
                f"{item}:source={source};action_type={action_type};"
                f"acknowledgement_fields={acknowledgement_fields};"
                f"action=review_operator_checklist_item:{item}"
            )
        )
    return details


def _expected_required_confirmation_details(
    confirmation_items: Any,
) -> List[str]:
    details: List[str] = []
    for item in _sorted_str_items(confirmation_items):
        source, action_type, acknowledgement_fields = (
            REQUIRED_CONFIRMATION_DETAIL_PROFILES.get(
                item,
                (
                    "what_operator_must_confirm",
                    "confirm_required_operator_condition",
                    "required_confirmations_acknowledgement",
                ),
            )
        )
        details.append(
            (
                f"{item}:source={source};action_type={action_type};"
                f"acknowledgement_fields={acknowledgement_fields};"
                f"action=confirm_operator_required_condition:{item}"
            )
        )
    return details


def _expected_operator_acceptance_gate_details(
    confirmation_items: Any,
) -> List[str]:
    details: List[str] = []
    for item in _sorted_str_items(confirmation_items):
        source, action_type, acknowledgement_fields = (
            REQUIRED_CONFIRMATION_DETAIL_PROFILES.get(
                item,
                (
                    "what_operator_must_confirm",
                    "confirm_required_operator_condition",
                    "required_confirmations_acknowledgement",
                ),
            )
        )
        details.append(
            (
                f"{item}:source={source};action_type={action_type};"
                f"acknowledgement_fields={acknowledgement_fields};"
                f"action=confirm_operator_required_condition:{item};"
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


def _expected_operator_review_decision_option_details(
    decision_items: Any,
) -> List[str]:
    details: List[str] = []
    for decision in _sorted_str_items(decision_items):
        effect, action_type, operator_action, blocks_offline = (
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
                f"{decision}:effect={effect};action_type={action_type};"
                f"operator_action={operator_action};"
                f"blocks_offline_acceptance={str(bool(blocks_offline)).lower()};"
                "live_apply_allowed=false;queue_import_allowed=false;"
                "requires_real_wing_proof_before_live=true"
            )
        )
    return details


def _expected_operator_next_action_details(card: Mapping[str, Any]) -> List[str]:
    def _joined(values: Any) -> str:
        items = _sorted_str_items(values)
        return ",".join(items) if items else "none"

    def _detail(
        action: str,
        *,
        source: str,
        action_type: str,
        status: str,
        required_before_operator_acceptance: bool,
        fields: Any = (),
        readback_targets: Any = (),
        proofs: Any = (),
        blocked_actions: Any = (),
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
            fields=card.get("operator_evidence_required_fields"),
        ),
        _detail(
            "record_operator_decision_before_promotion",
            source="operator_review_record_template",
            action_type="record_operator_review_decision",
            status="operator_review_required",
            required_before_operator_acceptance=True,
            fields=card.get("operator_review_record_required_fields"),
        ),
        _detail(
            "collect_future_live_proofs_before_live_promotion",
            source="future_live_promotion_requirements",
            action_type="collect_future_live_proofs",
            status="pending_future_live_proof",
            required_before_operator_acceptance=False,
            proofs=card.get("future_live_required_proofs"),
        ),
        _detail(
            "do_not_queue_or_apply_live",
            source="live_authority_boundary",
            action_type="hold_live_and_queue_authority",
            status="live_authority_denied_until_real_wing_proof",
            required_before_operator_acceptance=True,
            blocked_actions=card.get("blocked_live_actions"),
        ),
    ]
    operator_next_step = str(card.get("operator_next_step") or "")
    if operator_next_step:
        details.append(
            _detail(
                operator_next_step,
                source="future_wing_readback_contract",
                action_type="capture_future_wing_readback",
                status="pending_future_wing_readback",
                required_before_operator_acceptance=False,
                readback_targets=card.get("future_wing_readback_targets"),
            )
        )
    return sorted(dict.fromkeys(details))


def _expected_future_live_requirement_details(card: Mapping[str, Any]) -> List[str]:
    def _joined(values: Any) -> str:
        items = _sorted_str_items(values)
        return ",".join(items) if items else "none"

    blocked = card.get("blocked_live_actions")
    details: List[str] = []

    for proof in _sorted_str_items(card.get("future_live_required_proofs")):
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

    for target in _sorted_str_items(card.get("future_wing_readback_targets")):
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


def _expected_safety_blocker_details(card: Mapping[str, Any]) -> List[str]:
    reason = str(card.get("denial_reason") or "real_wing_hardware_proof_required")
    details: List[str] = []

    for check in _sorted_str_items(card.get("required_safety_checks")):
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

    for action in _sorted_str_items(card.get("blocked_live_actions")):
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

    for surface in _sorted_str_items(
        card.get("missing_required_operator_surfaces")
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


def _expected_headroom_feedback_true_peak_check_details(
    card: Mapping[str, Any]
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

    reason = str(card.get("denial_reason") or "real_wing_hardware_proof_required")
    details: List[str] = []
    for check in _sorted_str_items(card.get("required_safety_checks")):
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


def _expected_evidence_problem_correction_chain_details(
    card: Mapping[str, Any]
) -> List[str]:
    reason = str(card.get("denial_reason") or "real_wing_hardware_proof_required")

    def _required(field: str) -> bool:
        contract = card.get(field)
        return isinstance(contract, Mapping) and contract.get("required") is True

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
            required=_required("source_evidence"),
        ),
        _detail(
            "source_evidence_details",
            chain_step="2_source_evidence_details",
            source_fields=["source_evidence_details"],
            operator_action="review_source_evidence_details_before_acceptance",
            required=_required("source_evidence_details"),
        ),
        _detail(
            "measured_or_audible_problem",
            chain_step="3_measured_or_audible_problem",
            source_fields=["measured_or_audible_problem"],
            operator_action="confirm_problem_statement_matches_source_evidence",
            required=_required("measured_or_audible_problem"),
        ),
        _detail(
            "proposed_correction",
            chain_step="4_proposed_correction",
            source_fields=["proposed_correction"],
            operator_action="confirm_proposed_correction_matches_the_problem",
            required=_required("proposed_correction"),
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
            required=_required("confidence"),
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
            required=_required("bounded_delta"),
        ),
    ]
    return sorted(details)


def _expected_goal_requirement_details(card: Mapping[str, Any]) -> List[str]:
    reason = str(card.get("denial_reason") or "real_wing_hardware_proof_required")

    def _required(field: str) -> bool:
        contract = card.get(field)
        return isinstance(contract, Mapping) and contract.get("required") is True

    def _covered(value: Any) -> str:
        return "covered" if bool(value) else "missing"

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
            covered=_required("source_evidence")
            and _required("source_evidence_details"),
        ),
        _detail(
            "measured_or_audible_problem",
            source_fields=["measured_or_audible_problem"],
            operator_action="confirm_problem_statement_before_acceptance",
            covered=_required("measured_or_audible_problem"),
        ),
        _detail(
            "proposed_correction",
            source_fields=["proposed_correction"],
            operator_action="confirm_proposed_correction_before_acceptance",
            covered=_required("proposed_correction"),
        ),
        _detail(
            "confidence",
            source_fields=[
                "confidence",
                "confidence_review",
                "confidence_review_details",
            ],
            operator_action="review_confidence_before_acceptance",
            covered=_required("confidence"),
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
            covered=_required("bounded_delta"),
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
            covered=_required("safety_blockers")
            and bool(_sorted_str_items(card.get("safety_blocker_details")))
            and bool(_sorted_str_items(card.get("required_safety_checks"))),
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
            covered=_required("headroom_feedback_true_peak_checks")
            and bool(
                _sorted_str_items(card.get("headroom_feedback_true_peak_check_details"))
            ),
        ),
        _detail(
            "operator_readable_explanation",
            source_fields=[
                "operator_readable_explanation",
                "operator_explanation_packet",
            ],
            operator_action="confirm_operator_explanation_before_acceptance",
            covered=_required("operator_readable_explanation"),
        ),
        _detail(
            "rollback_expectation",
            source_fields=[
                "rollback_expectation",
                "rollback_readback_contract",
                "rollback_readback_details",
            ],
            operator_action="confirm_rollback_expectation_before_acceptance",
            covered=bool(card.get("rollback_expectation"))
            and _required("rollback_readback_contract"),
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
            covered=_required("future_wing_readback_contract")
            and bool(_sorted_str_items(card.get("future_wing_readback_targets")))
            and bool(
                _sorted_str_items(card.get("future_wing_readback_contract_summary"))
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
            covered=_required("test_evidence")
            and bool(_sorted_str_items(card.get("test_evidence_references"))),
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


def _expected_method_specific_review_details(
    card: Mapping[str, Any]
) -> List[str]:
    family = str(card.get("objective_family") or "supporting_context")
    priority = card.get("priority")
    priority_value = "none" if priority is None else str(priority)
    profile = METHOD_SPECIFIC_REVIEW_PROFILES.get(
        family,
        DEFAULT_METHOD_SPECIFIC_REVIEW_PROFILE,
    )

    def _joined(values: Any) -> str:
        items = _sorted_str_items(values)
        return ",".join(items) if items else "none"

    def _detail(name: str, focus_value: str, *, source_fields: Iterable[str]) -> str:
        fields = ",".join(str(field) for field in source_fields)
        return (
            f"{name}:source=correction_method_goal.method_specific_review;"
            f"method_id={card.get('method_id')};"
            f"objective_family={family};"
            f"priority={priority_value};"
            f"priority_label={card.get('priority_label')};"
            f"focus={focus_value};"
            f"source_fields={fields};"
            f"required_evidence={_joined(card.get('required_evidence'))};"
            f"required_safety_checks={_joined(card.get('required_safety_checks'))};"
            f"bounded_delta={card.get('bounded_delta_unit')}:"
            f"{card.get('bounded_delta_policy')};"
            f"rollback_expectation={card.get('rollback_expectation')};"
            "future_wing_readback_targets="
            f"{_joined(card.get('future_wing_readback_targets'))};"
            f"blocked_live_actions={_joined(card.get('blocked_live_actions'))};"
            f"operator_action=review_method_specific_{name}:{family};"
            "acceptance_scope=offline_proposal_only;"
            "required_before_operator_acceptance=true;"
            "required_before_live_promotion=true;"
            "required_before_queue_import=true;"
            f"denial_reason={card.get('denial_reason') or 'real_wing_hardware_proof_required'};"
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


def _expected_operator_actionable_explanation_details(
    card: Mapping[str, Any]
) -> List[str]:
    family = str(card.get("objective_family") or "supporting_context")
    priority = card.get("priority")
    priority_value = "none" if priority is None else str(priority)
    profile = METHOD_SPECIFIC_REVIEW_PROFILES.get(
        family,
        DEFAULT_METHOD_SPECIFIC_REVIEW_PROFILE,
    )

    def _joined(values: Any) -> str:
        items = _sorted_str_items(values)
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
            f"method_id={card.get('method_id')};"
            f"objective_family={family};"
            f"priority={priority_value};"
            f"priority_label={card.get('priority_label')};"
            f"explanation={explanation};"
            f"source_fields={fields};"
            f"required_evidence={_joined(card.get('required_evidence'))};"
            f"required_safety_checks={_joined(card.get('required_safety_checks'))};"
            f"bounded_delta={card.get('bounded_delta_unit')}:"
            f"{card.get('bounded_delta_policy')};"
            f"rollback_expectation={card.get('rollback_expectation')};"
            "future_wing_readback_targets="
            f"{_joined(card.get('future_wing_readback_targets'))};"
            f"blocked_live_actions={_joined(card.get('blocked_live_actions'))};"
            f"operator_action={operator_action};"
            "acceptance_scope=offline_proposal_only;"
            "required_before_operator_acceptance=true;"
            "required_before_live_promotion=true;"
            "required_before_queue_import=true;"
            f"denial_reason={card.get('denial_reason') or 'real_wing_hardware_proof_required'};"
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


def _check_future_live_required_proofs(
    failures: List[Dict[str, str]],
    *,
    code_prefix: str,
    value: Any,
) -> None:
    if not isinstance(value, list):
        failures.append(
            _failure(
                f"{code_prefix}_future_live_required_proofs_not_list",
                repr(value),
            )
        )
        value = []

    valid_items: List[str] = []
    invalid_items: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            invalid_items.append(repr(item))
            continue
        valid_items.append(item)

    if invalid_items:
        failures.append(
            _failure(
                f"{code_prefix}_invalid_future_live_required_proofs",
                repr(sorted(invalid_items)),
            )
        )

    required = set(REQUIRED_FUTURE_PROOFS)
    missing_future_proofs = sorted(
        proof for proof in REQUIRED_FUTURE_PROOFS if proof not in valid_items
    )
    if missing_future_proofs:
        failures.append(
            _failure(
                f"{code_prefix}_missing_future_live_required_proofs",
                repr(missing_future_proofs),
            )
        )

    unknown_future_proofs = sorted(
        {proof for proof in valid_items if proof not in required}
    )
    if unknown_future_proofs:
        failures.append(
            _failure(
                f"{code_prefix}_unknown_future_live_required_proofs",
                repr(unknown_future_proofs),
            )
        )

    duplicate_future_proofs = sorted(
        {proof for proof in valid_items if valid_items.count(proof) > 1}
    )
    if duplicate_future_proofs:
        failures.append(
            _failure(
                f"{code_prefix}_duplicate_future_live_required_proofs",
                repr(duplicate_future_proofs),
            )
        )


def _check_future_wing_readback_contract_summary(
    failures: List[Dict[str, str]],
    *,
    code_prefix: str,
    value: Any,
    allow_duplicate_keys: bool = False,
) -> List[str]:
    field_name = "future_wing_readback_contract_summary"
    if not isinstance(value, list):
        failures.append(
            _failure(
                f"{code_prefix}_{field_name}_not_list",
                repr(value),
            )
        )
        value = []

    valid_items: List[str] = []
    invalid_items: List[str] = []
    parsed_items: List[tuple[str, str, str]] = []
    for item in value:
        if not isinstance(item, str) or not item or "=" not in item:
            invalid_items.append(repr(item))
            continue
        key, item_value = item.split("=", 1)
        if not key or not item_value:
            invalid_items.append(repr(item))
            continue
        valid_items.append(item)
        parsed_items.append((key, item_value, item))

    if invalid_items:
        failures.append(
            _failure(
                f"{code_prefix}_invalid_{field_name}",
                repr(sorted(invalid_items)),
            )
        )

    keys = [key for key, _, _ in parsed_items]
    required = set(REQUIRED_FUTURE_WING_READBACK_CONTRACT_SUMMARY_KEYS)
    missing_keys = sorted(
        key
        for key in REQUIRED_FUTURE_WING_READBACK_CONTRACT_SUMMARY_KEYS
        if key not in keys
    )
    if missing_keys:
        failures.append(
            _failure(
                f"{code_prefix}_missing_{field_name}",
                repr(missing_keys),
            )
        )

    unknown_keys = sorted({key for key in keys if key not in required})
    if unknown_keys:
        failures.append(
            _failure(
                f"{code_prefix}_unknown_{field_name}",
                repr(unknown_keys),
            )
        )

    duplicate_items = sorted(
        {item for item in valid_items if valid_items.count(item) > 1}
    )
    duplicate_keys = (
        []
        if allow_duplicate_keys
        else sorted({key for key in keys if keys.count(key) > 1})
    )
    if duplicate_items or duplicate_keys:
        failures.append(
            _failure(
                f"{code_prefix}_duplicate_{field_name}",
                repr(
                    {
                        "items": duplicate_items,
                        "keys": duplicate_keys,
                    }
                ),
            )
        )

    invalid_operator_next_steps = sorted(
        {
            item_value
            for key, item_value, _ in parsed_items
            if key == "operator_next_step"
            and item_value != REQUIRED_FUTURE_WING_READBACK_OPERATOR_NEXT_STEP
        }
    )
    if invalid_operator_next_steps:
        failures.append(
            _failure(
                f"{code_prefix}_invalid_future_wing_readback_operator_next_step",
                repr(invalid_operator_next_steps),
            )
        )

    invalid_denial_reasons = sorted(
        {
            item_value
            for key, item_value, _ in parsed_items
            if key == "denial_reason"
            and item_value != REQUIRED_FUTURE_WING_READBACK_DENIAL_REASON
        }
    )
    if invalid_denial_reasons:
        failures.append(
            _failure(
                f"{code_prefix}_invalid_future_wing_readback_denial_reason",
                repr(invalid_denial_reasons),
            )
        )

    return valid_items


def _check_required_safety_checks(
    failures: List[Dict[str, str]],
    *,
    code_prefix: str,
    value: Any,
) -> List[str]:
    field_name = "required_safety_checks"
    if not isinstance(value, list):
        failures.append(
            _failure(
                f"{code_prefix}_{field_name}_not_list",
                repr(value),
            )
        )
        value = []

    valid_items: List[str] = []
    invalid_items: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            invalid_items.append(repr(item))
            continue
        valid_items.append(item)

    if invalid_items:
        failures.append(
            _failure(
                f"{code_prefix}_invalid_{field_name}",
                repr(sorted(invalid_items)),
            )
        )

    allowed = set(ALLOWED_REQUIRED_SAFETY_CHECKS)
    unknown_checks = sorted(
        {item for item in valid_items if item not in allowed}
    )
    if unknown_checks:
        failures.append(
            _failure(
                f"{code_prefix}_unknown_{field_name}",
                repr(unknown_checks),
            )
        )

    duplicate_checks = sorted(
        {item for item in valid_items if valid_items.count(item) > 1}
    )
    if duplicate_checks:
        failures.append(
            _failure(
                f"{code_prefix}_duplicate_{field_name}",
                repr(duplicate_checks),
            )
        )

    return valid_items


def _check_bounded_delta_contract(
    failures: List[Dict[str, str]],
    *,
    code_prefix: str,
    bounded_delta_unit: Any,
    bounded_delta_policy: Any,
) -> None:
    if not isinstance(bounded_delta_unit, str) or not bounded_delta_unit:
        failures.append(
            _failure(
                f"{code_prefix}_invalid_bounded_delta_unit",
                repr(bounded_delta_unit),
            )
        )
    elif bounded_delta_unit not in set(ALLOWED_BOUNDED_DELTA_UNITS):
        failures.append(
            _failure(
                f"{code_prefix}_unknown_bounded_delta_unit",
                repr(bounded_delta_unit),
            )
        )

    if not isinstance(bounded_delta_policy, str) or not bounded_delta_policy:
        failures.append(
            _failure(
                f"{code_prefix}_invalid_bounded_delta_policy",
                repr(bounded_delta_policy),
            )
        )
    elif bounded_delta_policy not in set(ALLOWED_BOUNDED_DELTA_POLICIES):
        failures.append(
            _failure(
                f"{code_prefix}_unknown_bounded_delta_policy",
                repr(bounded_delta_policy),
            )
        )


def _check_bounded_delta_policies(
    failures: List[Dict[str, str]],
    *,
    code_prefix: str,
    value: Any,
) -> List[str]:
    field_name = "bounded_delta_policies"
    if not isinstance(value, list):
        failures.append(
            _failure(
                f"{code_prefix}_{field_name}_not_list",
                repr(value),
            )
        )
        value = []

    valid_items: List[str] = []
    invalid_items: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            invalid_items.append(repr(item))
            continue
        valid_items.append(item)

    if invalid_items:
        failures.append(
            _failure(
                f"{code_prefix}_invalid_{field_name}",
                repr(sorted(invalid_items)),
            )
        )

    allowed = set(ALLOWED_BOUNDED_DELTA_POLICIES)
    unknown_policies = sorted(
        {item for item in valid_items if item not in allowed}
    )
    if unknown_policies:
        failures.append(
            _failure(
                f"{code_prefix}_unknown_{field_name}",
                repr(unknown_policies),
            )
        )

    duplicate_policies = sorted(
        {item for item in valid_items if valid_items.count(item) > 1}
    )
    if duplicate_policies:
        failures.append(
            _failure(
                f"{code_prefix}_duplicate_{field_name}",
                repr(duplicate_policies),
            )
        )

    return valid_items


def _check_future_wing_readback_targets(
    failures: List[Dict[str, str]],
    *,
    code_prefix: str,
    value: Any,
) -> List[str]:
    field_name = "future_wing_readback_targets"
    if not isinstance(value, list):
        failures.append(
            _failure(
                f"{code_prefix}_{field_name}_not_list",
                repr(value),
            )
        )
        value = []

    valid_items: List[str] = []
    invalid_items: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            invalid_items.append(repr(item))
            continue
        valid_items.append(item)

    if invalid_items:
        failures.append(
            _failure(
                f"{code_prefix}_invalid_{field_name}",
                repr(sorted(invalid_items)),
            )
        )

    allowed = set(ALLOWED_FUTURE_WING_READBACK_TARGETS)
    unknown_targets = sorted(
        {item for item in valid_items if item not in allowed}
    )
    if unknown_targets:
        failures.append(
            _failure(
                f"{code_prefix}_unknown_{field_name}",
                repr(unknown_targets),
            )
        )

    duplicate_targets = sorted(
        {item for item in valid_items if valid_items.count(item) > 1}
    )
    if duplicate_targets:
        failures.append(
            _failure(
                f"{code_prefix}_duplicate_{field_name}",
                repr(duplicate_targets),
            )
        )

    return valid_items


def _check_rollback_expectation(
    failures: List[Dict[str, str]],
    *,
    code_prefix: str,
    value: Any,
) -> None:
    field_name = "rollback_expectation"
    if not isinstance(value, str) or not value:
        failures.append(
            _failure(
                f"{code_prefix}_invalid_{field_name}",
                repr(value),
            )
        )
    elif value not in set(ALLOWED_ROLLBACK_EXPECTATIONS):
        failures.append(
            _failure(
                f"{code_prefix}_unknown_{field_name}",
                repr(value),
            )
        )


def _check_rollback_expectations(
    failures: List[Dict[str, str]],
    *,
    code_prefix: str,
    value: Any,
) -> List[str]:
    field_name = "rollback_expectations"
    if not isinstance(value, list):
        failures.append(
            _failure(
                f"{code_prefix}_{field_name}_not_list",
                repr(value),
            )
        )
        value = []

    valid_items: List[str] = []
    invalid_items: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            invalid_items.append(repr(item))
            continue
        valid_items.append(item)

    if invalid_items:
        failures.append(
            _failure(
                f"{code_prefix}_invalid_{field_name}",
                repr(sorted(invalid_items)),
            )
        )

    allowed = set(ALLOWED_ROLLBACK_EXPECTATIONS)
    unknown_expectations = sorted(
        {item for item in valid_items if item not in allowed}
    )
    if unknown_expectations:
        failures.append(
            _failure(
                f"{code_prefix}_unknown_{field_name}",
                repr(unknown_expectations),
            )
        )

    duplicate_expectations = sorted(
        {item for item in valid_items if valid_items.count(item) > 1}
    )
    if duplicate_expectations:
        failures.append(
            _failure(
                f"{code_prefix}_duplicate_{field_name}",
                repr(duplicate_expectations),
            )
        )

    return valid_items


def _check_blocked_live_actions(
    failures: List[Dict[str, str]],
    *,
    code_prefix: str,
    value: Any,
) -> List[str]:
    field_name = "blocked_live_actions"
    if not isinstance(value, list):
        failures.append(
            _failure(
                f"{code_prefix}_{field_name}_not_list",
                repr(value),
            )
        )
        value = []

    valid_items: List[str] = []
    invalid_items: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            invalid_items.append(repr(item))
            continue
        valid_items.append(item)

    if invalid_items:
        failures.append(
            _failure(
                f"{code_prefix}_invalid_{field_name}",
                repr(sorted(invalid_items)),
            )
        )

    allowed = set(ALLOWED_BLOCKED_LIVE_ACTIONS)
    unknown_actions = sorted(
        {action for action in valid_items if action not in allowed}
    )
    if unknown_actions:
        failures.append(
            _failure(
                f"{code_prefix}_unknown_{field_name}",
                repr(unknown_actions),
            )
        )

    duplicate_actions = sorted(
        {action for action in valid_items if valid_items.count(action) > 1}
    )
    if duplicate_actions:
        failures.append(
            _failure(
                f"{code_prefix}_duplicate_{field_name}",
                repr(duplicate_actions),
            )
        )

    return valid_items


def _check_live_authority_boundary_summary(
    failures: List[Dict[str, str]],
    *,
    code_prefix: str,
    value: Any,
) -> List[str]:
    field_name = "live_authority_boundary_summary"
    if not isinstance(value, list):
        failures.append(
            _failure(
                f"{code_prefix}_{field_name}_not_list",
                repr(value),
            )
        )
        value = []

    valid_items: List[str] = []
    invalid_items: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            invalid_items.append(repr(item))
            continue
        valid_items.append(item)

    if invalid_items:
        failures.append(
            _failure(
                f"{code_prefix}_invalid_{field_name}",
                repr(sorted(invalid_items)),
            )
        )

    required = set(REQUIRED_LIVE_AUTHORITY_BOUNDARY_SUMMARY)
    missing_items = sorted(
        item
        for item in REQUIRED_LIVE_AUTHORITY_BOUNDARY_SUMMARY
        if item not in valid_items
    )
    if missing_items:
        failures.append(
            _failure(
                f"{code_prefix}_missing_{field_name}",
                repr(missing_items),
            )
        )

    unknown_items = sorted(
        {item for item in valid_items if item not in required}
    )
    if unknown_items:
        failures.append(
            _failure(
                f"{code_prefix}_unknown_{field_name}",
                repr(unknown_items),
            )
        )

    duplicate_items = sorted(
        {item for item in valid_items if valid_items.count(item) > 1}
    )
    if duplicate_items:
        failures.append(
            _failure(
                f"{code_prefix}_duplicate_{field_name}",
                repr(duplicate_items),
            )
        )

    return valid_items


def _check_offline_review_decisions(
    failures: List[Dict[str, str]],
    *,
    code_prefix: str,
    field_name: str,
    value: Any,
) -> List[str]:
    if not isinstance(value, list):
        failures.append(
            _failure(
                f"{code_prefix}_{field_name}_not_list",
                repr(value),
            )
        )
        value = []

    valid_items: List[str] = []
    invalid_items: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            invalid_items.append(repr(item))
            continue
        valid_items.append(item)

    if invalid_items:
        failures.append(
            _failure(
                f"{code_prefix}_invalid_{field_name}",
                repr(sorted(invalid_items)),
            )
        )

    required = set(REQUIRED_OFFLINE_REVIEW_DECISIONS)
    missing_decisions = sorted(
        decision
        for decision in REQUIRED_OFFLINE_REVIEW_DECISIONS
        if decision not in valid_items
    )
    if missing_decisions:
        failures.append(
            _failure(
                f"{code_prefix}_missing_{field_name}",
                repr(missing_decisions),
            )
        )

    unknown_decisions = sorted(
        {decision for decision in valid_items if decision not in required}
    )
    if unknown_decisions:
        failures.append(
            _failure(
                f"{code_prefix}_unknown_{field_name}",
                repr(unknown_decisions),
            )
        )

    duplicate_decisions = sorted(
        {decision for decision in valid_items if valid_items.count(decision) > 1}
    )
    if duplicate_decisions:
        failures.append(
            _failure(
                f"{code_prefix}_duplicate_{field_name}",
                repr(duplicate_decisions),
            )
        )

    return valid_items


def _check_operator_confirmations(
    failures: List[Dict[str, str]],
    *,
    code_prefix: str,
    field_name: str,
    value: Any,
) -> List[str]:
    if not isinstance(value, list):
        failures.append(
            _failure(
                f"{code_prefix}_{field_name}_not_list",
                repr(value),
            )
        )
        value = []

    valid_items: List[str] = []
    invalid_items: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            invalid_items.append(repr(item))
            continue
        valid_items.append(item)

    if invalid_items:
        failures.append(
            _failure(
                f"{code_prefix}_invalid_{field_name}",
                repr(sorted(invalid_items)),
            )
        )

    required = set(REQUIRED_METHOD_CARD_CONFIRMATIONS)
    missing_confirmations = sorted(
        confirmation
        for confirmation in REQUIRED_METHOD_CARD_CONFIRMATIONS
        if confirmation not in valid_items
    )
    if missing_confirmations:
        failures.append(
            _failure(
                f"{code_prefix}_missing_{field_name}",
                repr(missing_confirmations),
            )
        )

    unknown_confirmations = sorted(
        {confirmation for confirmation in valid_items if confirmation not in required}
    )
    if unknown_confirmations:
        failures.append(
            _failure(
                f"{code_prefix}_unknown_{field_name}",
                repr(unknown_confirmations),
            )
        )

    duplicate_confirmations = sorted(
        {
            confirmation
            for confirmation in valid_items
            if valid_items.count(confirmation) > 1
        }
    )
    if duplicate_confirmations:
        failures.append(
            _failure(
                f"{code_prefix}_duplicate_{field_name}",
                repr(duplicate_confirmations),
            )
        )

    return valid_items


def _check_operator_review_record_required_fields(
    failures: List[Dict[str, str]],
    *,
    code_prefix: str,
    value: Any,
) -> List[str]:
    field_name = "operator_review_record_required_fields"
    if not isinstance(value, list):
        failures.append(
            _failure(
                f"{code_prefix}_{field_name}_not_list",
                repr(value),
            )
        )
        value = []

    valid_items: List[str] = []
    invalid_items: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            invalid_items.append(repr(item))
            continue
        valid_items.append(item)

    if invalid_items:
        failures.append(
            _failure(
                f"{code_prefix}_invalid_{field_name}",
                repr(sorted(invalid_items)),
            )
        )

    required = set(REQUIRED_OPERATOR_REVIEW_RECORD_FIELDS)
    missing_fields = sorted(
        field
        for field in REQUIRED_OPERATOR_REVIEW_RECORD_FIELDS
        if field not in valid_items
    )
    if missing_fields:
        failures.append(
            _failure(
                f"{code_prefix}_missing_{field_name}",
                repr(missing_fields),
            )
        )

    unknown_fields = sorted(
        {field for field in valid_items if field not in required}
    )
    if unknown_fields:
        failures.append(
            _failure(
                f"{code_prefix}_unknown_{field_name}",
                repr(unknown_fields),
            )
        )

    duplicate_fields = sorted(
        {field for field in valid_items if valid_items.count(field) > 1}
    )
    if duplicate_fields:
        failures.append(
            _failure(
                f"{code_prefix}_duplicate_{field_name}",
                repr(duplicate_fields),
            )
        )

    return valid_items


def _check_pending_operator_review_record_fields(
    failures: List[Dict[str, str]],
    *,
    code_prefix: str,
    payload: Mapping[str, Any],
) -> None:
    for field in REQUIRED_OPERATOR_REVIEW_RECORD_FIELDS:
        if field == "decision":
            continue
        if field not in payload:
            failures.append(
                _failure(
                    f"{code_prefix}_{field}_missing",
                    "pending operator review template must expose this field",
                )
            )
            continue
        if payload.get(field) != "":
            failures.append(
                _failure(
                    f"{code_prefix}_{field}_not_empty",
                    repr(payload.get(field)),
                )
            )


def _check_completed_operator_review_record_fields(
    failures: List[Dict[str, str]],
    *,
    code_prefix: str,
    payload: Mapping[str, Any],
) -> None:
    decision = payload.get("operator_decision")
    if not isinstance(decision, str) or not decision.strip():
        failures.append(
            _failure(
                f"{code_prefix}_decision_record_decision_missing",
                repr(decision),
            )
        )
    for field in REQUIRED_OPERATOR_REVIEW_RECORD_FIELDS:
        if field == "decision":
            continue
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            failures.append(
                _failure(
                    f"{code_prefix}_decision_record_{field}_missing",
                    repr(value),
                )
            )
    reviewed_at = payload.get("reviewed_at_utc")
    if isinstance(reviewed_at, str) and reviewed_at.strip():
        if "T" not in reviewed_at or not reviewed_at.endswith("Z"):
            failures.append(
                _failure(
                    f"{code_prefix}_decision_record_reviewed_at_utc_invalid",
                    repr(reviewed_at),
                )
            )
    acknowledgement = payload.get("offline_acceptance_boundary_acknowledgement")
    if (
        isinstance(acknowledgement, str)
        and acknowledgement.strip()
        and acknowledgement != REQUIRED_OFFLINE_ACCEPTANCE_BOUNDARY_ACKNOWLEDGEMENT
    ):
        failures.append(
            _failure(
                f"{code_prefix}_decision_record_offline_acceptance_boundary_acknowledgement_invalid",
                repr(acknowledgement),
            )
        )
    source_evidence_acknowledgement = payload.get("source_evidence_acknowledgement")
    if (
        isinstance(source_evidence_acknowledgement, str)
        and source_evidence_acknowledgement.strip()
        and source_evidence_acknowledgement
        != REQUIRED_SOURCE_EVIDENCE_ACKNOWLEDGEMENT
    ):
        failures.append(
            _failure(
                f"{code_prefix}_decision_record_source_evidence_acknowledgement_invalid",
                repr(source_evidence_acknowledgement),
            )
        )
    test_evidence_acknowledgement = payload.get("test_evidence_acknowledgement")
    if (
        isinstance(test_evidence_acknowledgement, str)
        and test_evidence_acknowledgement.strip()
        and test_evidence_acknowledgement
        != REQUIRED_TEST_EVIDENCE_ACKNOWLEDGEMENT
    ):
        failures.append(
            _failure(
                f"{code_prefix}_decision_record_test_evidence_acknowledgement_invalid",
                repr(test_evidence_acknowledgement),
            )
        )
    verification_report_acknowledgement = payload.get(
        "verification_report_acknowledgement"
    )
    if (
        isinstance(verification_report_acknowledgement, str)
        and verification_report_acknowledgement.strip()
        and verification_report_acknowledgement
        != REQUIRED_VERIFICATION_REPORT_ACKNOWLEDGEMENT
    ):
        failures.append(
            _failure(
                f"{code_prefix}_decision_record_verification_report_acknowledgement_invalid",
                repr(verification_report_acknowledgement),
            )
        )
    measured_problem_acknowledgement = payload.get(
        "measured_problem_acknowledgement"
    )
    if (
        isinstance(measured_problem_acknowledgement, str)
        and measured_problem_acknowledgement.strip()
        and measured_problem_acknowledgement
        != REQUIRED_MEASURED_PROBLEM_ACKNOWLEDGEMENT
    ):
        failures.append(
            _failure(
                f"{code_prefix}_decision_record_measured_problem_acknowledgement_invalid",
                repr(measured_problem_acknowledgement),
            )
        )
    proposed_correction_acknowledgement = payload.get(
        "proposed_correction_acknowledgement"
    )
    if (
        isinstance(proposed_correction_acknowledgement, str)
        and proposed_correction_acknowledgement.strip()
        and proposed_correction_acknowledgement
        != REQUIRED_PROPOSED_CORRECTION_ACKNOWLEDGEMENT
    ):
        failures.append(
            _failure(
                f"{code_prefix}_decision_record_proposed_correction_acknowledgement_invalid",
                repr(proposed_correction_acknowledgement),
            )
        )
    bounded_delta_acknowledgement = payload.get("bounded_delta_acknowledgement")
    if (
        isinstance(bounded_delta_acknowledgement, str)
        and bounded_delta_acknowledgement.strip()
        and bounded_delta_acknowledgement
        != REQUIRED_BOUNDED_DELTA_ACKNOWLEDGEMENT
    ):
        failures.append(
            _failure(
                f"{code_prefix}_decision_record_bounded_delta_acknowledgement_invalid",
                repr(bounded_delta_acknowledgement),
            )
        )
    confidence_acknowledgement = payload.get("confidence_acknowledgement")
    if (
        isinstance(confidence_acknowledgement, str)
        and confidence_acknowledgement.strip()
        and confidence_acknowledgement != REQUIRED_CONFIDENCE_ACKNOWLEDGEMENT
    ):
        failures.append(
            _failure(
                f"{code_prefix}_decision_record_confidence_acknowledgement_invalid",
                repr(confidence_acknowledgement),
            )
        )
    operator_explanation_acknowledgement = payload.get(
        "operator_explanation_acknowledgement"
    )
    if (
        isinstance(operator_explanation_acknowledgement, str)
        and operator_explanation_acknowledgement.strip()
        and operator_explanation_acknowledgement
        != REQUIRED_OPERATOR_EXPLANATION_ACKNOWLEDGEMENT
    ):
        failures.append(
            _failure(
                f"{code_prefix}_decision_record_operator_explanation_acknowledgement_invalid",
                repr(operator_explanation_acknowledgement),
            )
        )
    future_wing_readback_acknowledgement = payload.get(
        "future_wing_readback_acknowledgement"
    )
    if (
        isinstance(future_wing_readback_acknowledgement, str)
        and future_wing_readback_acknowledgement.strip()
        and future_wing_readback_acknowledgement
        != REQUIRED_FUTURE_WING_READBACK_ACKNOWLEDGEMENT
    ):
        failures.append(
            _failure(
                f"{code_prefix}_decision_record_future_wing_readback_acknowledgement_invalid",
                repr(future_wing_readback_acknowledgement),
            )
        )
    confirmation_acknowledgement = payload.get(
        "required_confirmations_acknowledgement"
    )
    if (
        isinstance(confirmation_acknowledgement, str)
        and confirmation_acknowledgement.strip()
        and confirmation_acknowledgement
        != REQUIRED_OPERATOR_CONFIRMATION_ACKNOWLEDGEMENT
    ):
        failures.append(
            _failure(
                f"{code_prefix}_decision_record_required_confirmations_acknowledgement_invalid",
                repr(confirmation_acknowledgement),
            )
        )
    safety_blockers_acknowledgement = payload.get(
        "safety_blockers_acknowledgement"
    )
    if (
        isinstance(safety_blockers_acknowledgement, str)
        and safety_blockers_acknowledgement.strip()
        and safety_blockers_acknowledgement
        != REQUIRED_SAFETY_BLOCKERS_ACKNOWLEDGEMENT
    ):
        failures.append(
            _failure(
                f"{code_prefix}_decision_record_safety_blockers_acknowledgement_invalid",
                repr(safety_blockers_acknowledgement),
            )
        )
    rollback_readback_acknowledgement = payload.get(
        "rollback_readback_acknowledgement"
    )
    if (
        isinstance(rollback_readback_acknowledgement, str)
        and rollback_readback_acknowledgement.strip()
        and rollback_readback_acknowledgement
        != REQUIRED_ROLLBACK_READBACK_ACKNOWLEDGEMENT
    ):
        failures.append(
            _failure(
                f"{code_prefix}_decision_record_rollback_readback_acknowledgement_invalid",
                repr(rollback_readback_acknowledgement),
            )
        )
    headroom_feedback_true_peak_acknowledgement = payload.get(
        "headroom_feedback_true_peak_acknowledgement"
    )
    if (
        isinstance(headroom_feedback_true_peak_acknowledgement, str)
        and headroom_feedback_true_peak_acknowledgement.strip()
        and headroom_feedback_true_peak_acknowledgement
        != REQUIRED_HEADROOM_FEEDBACK_TRUE_PEAK_ACKNOWLEDGEMENT
    ):
        failures.append(
            _failure(
                f"{code_prefix}_decision_record_headroom_feedback_true_peak_acknowledgement_invalid",
                repr(headroom_feedback_true_peak_acknowledgement),
            )
        )


def _completed_operator_review_record_field_count(payload: Mapping[str, Any]) -> int:
    count = 0
    for field in REQUIRED_OPERATOR_REVIEW_RECORD_FIELDS:
        if field == "decision":
            value = payload.get("operator_decision")
        else:
            value = payload.get(field)
        if isinstance(value, str) and value.strip():
            count += 1
    return count


def _split_operator_reference_list(value: Any) -> List[str]:
    if not isinstance(value, str):
        return []
    return [item.strip() for item in value.split(";") if item.strip()]


def _check_operator_review_reference_bindings(
    failures: List[Dict[str, str]],
    *,
    code_prefix: str,
    payload: Mapping[str, Any],
    expected_evidence_packet_reference: Any,
    expected_source_material_reference: Any,
    expected_test_evidence_references: Any,
    expected_verification_report_reference: Any,
) -> None:
    if (
        not isinstance(expected_evidence_packet_reference, str)
        or not expected_evidence_packet_reference.strip()
    ):
        failures.append(
            _failure(
                f"{code_prefix}_expected_evidence_packet_reference_missing",
                repr(expected_evidence_packet_reference),
            )
        )
    else:
        _check_equal(
            failures,
            code=f"{code_prefix}_evidence_packet_reference_mismatch",
            actual=payload.get("evidence_packet_reference"),
            expected=expected_evidence_packet_reference,
        )

    if (
        not isinstance(expected_source_material_reference, str)
        or not expected_source_material_reference.strip()
    ):
        failures.append(
            _failure(
                f"{code_prefix}_expected_source_material_reference_missing",
                repr(expected_source_material_reference),
            )
        )
    else:
        _check_equal(
            failures,
            code=f"{code_prefix}_source_material_reference_mismatch",
            actual=payload.get("source_material_reference"),
            expected=expected_source_material_reference,
        )

    if (
        not isinstance(expected_verification_report_reference, str)
        or not expected_verification_report_reference.strip()
    ):
        failures.append(
            _failure(
                f"{code_prefix}_expected_verification_report_reference_missing",
                repr(expected_verification_report_reference),
            )
        )
    else:
        _check_equal(
            failures,
            code=f"{code_prefix}_verification_report_reference_mismatch",
            actual=payload.get("verification_report_reference"),
            expected=expected_verification_report_reference,
        )

    if (
        not isinstance(expected_test_evidence_references, list)
        or not expected_test_evidence_references
    ):
        failures.append(
            _failure(
                f"{code_prefix}_expected_test_evidence_references_missing",
                repr(expected_test_evidence_references),
            )
        )
        return
    _check_equal(
        failures,
        code=f"{code_prefix}_test_evidence_reference_mismatch",
        actual=sorted(_split_operator_reference_list(
            payload.get("test_evidence_reference")
        )),
        expected=sorted(str(item) for item in expected_test_evidence_references),
    )


def _test_evidence_reference_path(reference: Any) -> Optional[str]:
    if not isinstance(reference, str):
        return None
    value = reference.strip()
    if not value:
        return None
    path_text, separator, node_id = value.partition("::")
    if not path_text or (separator and not node_id):
        return None
    return path_text


def _pytest_node_part(value: str) -> str:
    return value.split("[", 1)[0]


@lru_cache(maxsize=None)
def _python_test_node_index(path: Path) -> Dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = set()
    classes: Dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.add(node.name)
            continue
        if isinstance(node, ast.ClassDef):
            classes[node.name] = {
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    return {"functions": functions, "classes": classes}


def _pytest_node_exists(path: Path, node_id: str) -> bool:
    parts = [_pytest_node_part(part) for part in node_id.split("::")]
    if not parts or any(not part for part in parts):
        return False
    index = _python_test_node_index(path)
    if len(parts) == 1:
        return parts[0] in index["functions"]
    if len(parts) == 2:
        return parts[1] in index["classes"].get(parts[0], set())
    return False


def _check_test_evidence_references(
    failures: List[Dict[str, str]],
    *,
    method_id: str,
    card: Mapping[str, Any],
) -> None:
    references = card.get("test_evidence_references")
    references = references if isinstance(references, list) else []
    has_concrete_pytest_node = False
    for index, reference in enumerate(references):
        code_prefix = f"method_{method_id}_test_evidence_reference_{index}"
        path_text = _test_evidence_reference_path(reference)
        if path_text is None:
            failures.append(
                _failure(f"{code_prefix}_invalid", f"reference={reference!r}")
            )
            continue

        path = Path(path_text)
        if path.is_absolute() or ".." in path.parts:
            failures.append(
                _failure(
                    f"{code_prefix}_unsafe_path",
                    f"reference path must stay inside tests/: {reference!r}",
                )
            )
            continue
        if (
            path.parts[:1] != ("tests",)
            or not path.name.startswith("test_")
            or path.suffix != ".py"
        ):
            failures.append(
                _failure(
                    f"{code_prefix}_invalid_path",
                    f"reference path must be tests/test_*.py: {reference!r}",
                )
            )
            continue
        if not (REPO_ROOT / path).is_file():
            failures.append(
                _failure(
                    f"{code_prefix}_path_missing",
                    f"referenced test file does not exist: {reference!r}",
                )
            )
            continue
        _, separator, node_id = str(reference).strip().partition("::")
        if not separator:
            continue
        if not _pytest_node_exists(REPO_ROOT / path, node_id):
            failures.append(
                _failure(
                    f"{code_prefix}_node_missing",
                    f"referenced pytest node does not exist: {reference!r}",
                )
            )
            continue
        has_concrete_pytest_node = True
    if references and not has_concrete_pytest_node:
        failures.append(
            _failure(
                f"method_{method_id}_test_evidence_references_missing_pytest_node",
                "at least one test_evidence_references item must name a pytest node",
            )
        )


def _markdown_code(value: Any) -> str:
    text = "none" if value in (None, "") else str(value)
    escaped = text.replace("`", "\\`")
    return f"`{escaped}`"


def _markdown_bool(value: Any) -> str:
    return _markdown_code("true" if value is True else "false")


def _markdown_items(values: Any) -> str:
    if not values:
        return "`none`"
    return ", ".join(_markdown_code(value) for value in values)


def _markdown_authority_lines(payload: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    return (
        (
            "read_only_or_dry_run_only",
            (
                "- read_only_or_dry_run_only: "
                f"{_markdown_bool(payload.get('read_only_or_dry_run_only'))}"
            ),
        ),
        (
            "operator_review_required",
            (
                "- operator_review_required: "
                f"{_markdown_bool(payload.get('operator_review_required'))}"
            ),
        ),
        (
            "future_wing_readback_required",
            (
                "- future_wing_readback_required: "
                f"{_markdown_bool(payload.get('future_wing_readback_required'))}"
            ),
        ),
        (
            "hardware_proof_required_before_live",
            (
                "- hardware_proof_required_before_live: "
                f"{_markdown_bool(payload.get('hardware_proof_required_before_live'))}"
            ),
        ),
        (
            "blocks_live_promotion",
            (
                "- blocks_live_promotion: "
                f"{_markdown_bool(payload.get('blocks_live_promotion'))}"
            ),
        ),
        (
            "blocks_queue_import",
            (
                "- blocks_queue_import: "
                f"{_markdown_bool(payload.get('blocks_queue_import'))}"
            ),
        ),
        (
            "live_authority_granted",
            (
                "- live_authority_granted: "
                f"{_markdown_bool(payload.get('live_authority_granted'))}"
            ),
        ),
        (
            "full_autonomous_ready",
            (
                "- full_autonomous_ready: "
                f"{_markdown_bool(payload.get('full_autonomous_ready'))}"
            ),
        ),
        (
            "readiness_closure_allowed",
            (
                "- readiness_closure_allowed: "
                f"{_markdown_bool(payload.get('readiness_closure_allowed'))}"
            ),
        ),
        (
            "live_apply_allowed",
            f"- live_apply_allowed: {_markdown_bool(payload.get('live_apply_allowed'))}",
        ),
        (
            "queue_import_allowed",
            (
                "- queue_import_allowed: "
                f"{_markdown_bool(payload.get('queue_import_allowed'))}"
            ),
        ),
        (
            "can_apply_live",
            f"- can_apply_live: {_markdown_bool(payload.get('can_apply_live'))}",
        ),
        (
            "can_import_to_queue",
            (
                "- can_import_to_queue: "
                f"{_markdown_bool(payload.get('can_import_to_queue'))}"
            ),
        ),
        (
            "can_authorize_live",
            (
                "- can_authorize_live: "
                f"{_markdown_bool(payload.get('can_authorize_live'))}"
            ),
        ),
        (
            "can_close_readiness",
            (
                "- can_close_readiness: "
                f"{_markdown_bool(payload.get('can_close_readiness'))}"
            ),
        ),
        (
            "queue_action",
            f"- queue_action: {_markdown_code(payload.get('queue_action'))}",
        ),
        (
            "contacts_wing",
            f"- contacts_wing: {_markdown_bool(payload.get('contacts_wing'))}",
        ),
        (
            "contacts_backend",
            f"- contacts_backend: {_markdown_bool(payload.get('contacts_backend'))}",
        ),
        (
            "mutates_mixer",
            f"- mutates_mixer: {_markdown_bool(payload.get('mutates_mixer'))}",
        ),
    )


def _markdown_card_block(markdown: str, heading: str) -> str:
    start = markdown.find(heading)
    if start == -1:
        return ""
    next_heading = markdown.find("\n#### ", start + len(heading))
    if next_heading == -1:
        next_heading = len(markdown)
    return markdown[start:next_heading]


def _markdown_h2_block(markdown: str, heading: str) -> str:
    start = markdown.find(heading)
    if start == -1:
        return ""
    next_heading = markdown.find("\n## ", start + len(heading))
    if next_heading == -1:
        next_heading = len(markdown)
    return markdown[start:next_heading]


def _markdown_h2_intro_block(markdown: str, heading: str) -> str:
    start = markdown.find(heading)
    if start == -1:
        return ""
    after_heading = start + len(heading)
    next_h2 = markdown.find("\n## ", after_heading)
    next_h3 = markdown.find("\n### ", after_heading)
    candidates = [index for index in (next_h2, next_h3) if index != -1]
    next_heading = min(candidates) if candidates else len(markdown)
    return markdown[start:next_heading]


def _markdown_h3_block(markdown: str, heading: str) -> str:
    start = markdown.find(heading)
    if start == -1:
        return ""
    after_heading = start + len(heading)
    next_h2 = markdown.find("\n## ", after_heading)
    next_h3 = markdown.find("\n### ", after_heading)
    candidates = [index for index in (next_h2, next_h3) if index != -1]
    next_heading = min(candidates) if candidates else len(markdown)
    return markdown[start:next_heading]


def _priority_section_method_ids(section: Mapping[str, Any]) -> List[Any]:
    methods = section.get("methods")
    if not isinstance(methods, list):
        return []
    return [
        method.get("method_id")
        for method in methods
        if isinstance(method, Mapping)
    ]


def _priority_section_method_items(
    section: Mapping[str, Any],
    key: str,
) -> List[str]:
    methods = section.get("methods")
    if not isinstance(methods, list):
        return []
    return sorted(
        {
            str(item)
            for method in methods
            if isinstance(method, Mapping)
            for item in (method.get(key) or [])
        }
    )


def _priority_section_method_scalar_items(
    section: Mapping[str, Any],
    key: str,
) -> List[str]:
    methods = section.get("methods")
    if not isinstance(methods, list):
        return []
    return sorted(
        {
            str(method.get(key))
            for method in methods
            if isinstance(method, Mapping) and method.get(key)
        }
    )


def _priority_section_method_contract_items(
    section: Mapping[str, Any],
    key: str,
) -> List[str]:
    methods = section.get("methods")
    if not isinstance(methods, list):
        return []
    values: List[str] = []
    for method in methods:
        if not isinstance(method, Mapping):
            continue
        contract = method.get(key)
        if not isinstance(contract, Mapping):
            continue
        source_fields = contract.get("source_fields")
        if isinstance(source_fields, list):
            source_field_value = ",".join(str(item) for item in source_fields)
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


def _operator_visible_explanation_contract_items(card: Mapping[str, Any]) -> List[str]:
    contract = card.get("operator_visible_explanation_contract")
    if not isinstance(contract, Mapping):
        return []
    return [
        f"{key}={str(value).lower()}"
        for key, value in sorted(contract.items())
    ]


def _check_operator_visible_explanation_contract(
    failures: List[Dict[str, str]],
    *,
    code_prefix: str,
    value: Any,
) -> None:
    if not isinstance(value, Mapping):
        failures.append(
            _failure(
                f"{code_prefix}_operator_visible_explanation_contract_not_object",
                repr(value),
            )
        )
        return
    for flag in REQUIRED_OPERATOR_VISIBLE_CONTRACT_FLAGS:
        if value.get(flag) is not True:
            failures.append(
                _failure(
                    (
                        f"{code_prefix}_operator_visible_explanation_contract_"
                        f"{flag}_not_true"
                    ),
                    f"{flag}={value.get(flag)!r}",
                )
            )


def _operator_review_contract_items(
    card: Mapping[str, Any],
    field: str,
) -> List[str]:
    contract = card.get(field)
    if not isinstance(contract, Mapping):
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


def _expected_operator_method_review_focus(card: Mapping[str, Any]) -> List[str]:
    def _csv(key: str) -> str:
        values = card.get(key)
        if not isinstance(values, list):
            return ""
        return ",".join(str(value) for value in values)

    priority = card.get("priority")
    priority_value = "none" if priority is None else str(priority)
    return [
        f"method_id={card.get('method_id')}",
        f"priority={priority_value}",
        f"priority_label={card.get('priority_label')}",
        f"objective_family={card.get('objective_family')}",
        f"required_evidence={_csv('required_evidence')}",
        f"required_safety_checks={_csv('required_safety_checks')}",
        (
            "bounded_delta="
            f"{card.get('bounded_delta_unit')}:{card.get('bounded_delta_policy')}"
        ),
        f"rollback_expectation={card.get('rollback_expectation')}",
        f"future_wing_readback_targets={_csv('future_wing_readback_targets')}",
        f"blocked_live_actions={_csv('blocked_live_actions')}",
        "live_boundary=offline_only_no_queue_no_live_apply_until_real_wing_proof",
    ]


def _expected_operator_why_safe_summary(card: Mapping[str, Any]) -> List[str]:
    def _csv(key: str) -> str:
        values = card.get(key)
        if not isinstance(values, list):
            return ""
        return ",".join(str(value) for value in values)

    return [
        "read_only_or_dry_run_only=true",
        "live_apply_allowed=false",
        "queue_import_allowed=false",
        "operator_review_required=true",
        (
            "bounded_delta="
            f"{card.get('bounded_delta_unit')}:{card.get('bounded_delta_policy')}"
        ),
        f"required_safety_checks={_csv('required_safety_checks')}",
    ]


def _expected_operator_why_blocked_summary(card: Mapping[str, Any]) -> List[str]:
    def _csv(key: str) -> str:
        values = card.get(key)
        if not isinstance(values, list):
            return ""
        return ",".join(str(value) for value in values)

    return [
        f"denial_reason={card.get('denial_reason')}",
        "future_wing_readback_required=true",
        "hardware_proof_required_before_live=true",
        "readiness_closure_allowed=false",
        f"blocked_live_actions={_csv('blocked_live_actions')}",
        f"future_live_required_proofs={_csv('future_live_required_proofs')}",
    ]


def _summary_keys(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).split("=", 1)[0] for item in value]


def _check_method_card_mirror(
    failures: List[Dict[str, str]],
    *,
    actual_card: Mapping[str, Any],
    expected_card: Mapping[str, Any],
    label: str,
) -> None:
    for field in PRIORITY_METHOD_CARD_SCALAR_MIRROR_FIELDS:
        _check_equal(
            failures,
            code=f"{label}_{field}_mismatch",
            actual=actual_card.get(field),
            expected=expected_card.get(field),
        )
    for field in PRIORITY_METHOD_CARD_LIST_MIRROR_FIELDS:
        _check_equal(
            failures,
            code=f"{label}_{field}_mismatch",
            actual=_sorted_str_items(actual_card.get(field)),
            expected=_sorted_str_items(expected_card.get(field)),
        )
    for field in PRIORITY_METHOD_CARD_FLAG_MIRROR_FIELDS:
        _check_equal(
            failures,
            code=f"{label}_{field}_mismatch",
            actual=actual_card.get(field),
            expected=expected_card.get(field),
        )
    _check_equal(
        failures,
        code=f"{label}_operator_visible_explanation_contract_mismatch",
        actual=actual_card.get("operator_visible_explanation_contract"),
        expected=expected_card.get("operator_visible_explanation_contract"),
    )


def verify_correction_method_operator_review_export(
    run_dir: Path,
) -> Dict[str, Any]:
    """Verify a local operator-review export bundle without external contact."""

    run_dir = Path(run_dir)
    failures: List[Dict[str, str]] = []
    manifest_path = run_dir / "artifact-manifest.json"
    packet_path = run_dir / "packet.json"
    markdown_path = run_dir / "operator-review.md"
    decision_template_path = run_dir / "operator-decision-template.json"
    goal_audit_path = run_dir / "goal-audit.json"
    goal_audit_markdown_path = run_dir / "goal-audit.md"

    manifest = _load_json_object(
        manifest_path, label="manifest", failures=failures
    )
    packet = _load_json_object(packet_path, label="packet", failures=failures)
    decision_template = _load_json_object(
        decision_template_path,
        label="operator_decision_template",
        failures=failures,
    )
    goal_audit = _load_json_object(
        goal_audit_path,
        label="goal_audit",
        failures=failures,
    )
    markdown = ""
    if not markdown_path.exists():
        failures.append(
            _failure("missing_operator_review_markdown", str(markdown_path))
        )
    else:
        markdown = markdown_path.read_text(encoding="utf-8")
    goal_audit_markdown = ""
    if not goal_audit_markdown_path.exists():
        failures.append(
            _failure("missing_goal_audit_markdown", str(goal_audit_markdown_path))
        )
    else:
        goal_audit_markdown = goal_audit_markdown_path.read_text(encoding="utf-8")

    if manifest:
        _check_equal(
            failures,
            code="manifest_schema_mismatch",
            actual=manifest.get("schema"),
            expected=EXPECTED_MANIFEST_SCHEMA,
        )
        _check_equal(
            failures,
            code="manifest_status_mismatch",
            actual=manifest.get("status"),
            expected=EXPECTED_MANIFEST_STATUS,
        )
        _check_equal(
            failures,
            code="manifest_packet_schema_mismatch",
            actual=manifest.get("packet_schema"),
            expected=EXPECTED_PACKET_SCHEMA,
        )
        _check_equal(
            failures,
            code="manifest_packet_status_mismatch",
            actual=manifest.get("packet_status"),
            expected=EXPECTED_PACKET_STATUS,
        )
        _check_equal(
            failures,
            code="manifest_goal_audit_schema_mismatch",
            actual=manifest.get("goal_audit_schema"),
            expected=EXPECTED_GOAL_AUDIT_SCHEMA,
        )
        _check_equal(
            failures,
            code="manifest_goal_audit_status_mismatch",
            actual=manifest.get("goal_audit_status"),
            expected=EXPECTED_GOAL_AUDIT_STATUS,
        )
        _check_equal(
            failures,
            code="manifest_decision_template_schema_mismatch",
            actual=manifest.get("operator_decision_template_schema"),
            expected=EXPECTED_DECISION_TEMPLATE_SCHEMA,
        )
        _check_equal(
            failures,
            code="manifest_decision_template_status_mismatch",
            actual=manifest.get("operator_decision_template_status"),
            expected=EXPECTED_DECISION_TEMPLATE_STATUS,
        )
        _check_false_flags(
            failures,
            payload=manifest,
            label="manifest",
            flags=FALSE_AUTHORITY_FLAGS,
        )
        _check_true_flags(
            failures,
            payload=manifest,
            label="manifest",
            flags=TRUE_REVIEW_FLAGS + ("read_only",),
        )

        rows = _artifact_rows(manifest, failures)
        _check_equal(
            failures,
            code="manifest_artifact_ids_mismatch",
            actual=sorted(rows),
            expected=sorted(EXPECTED_ARTIFACTS),
        )
        for artifact_id, expected_name in EXPECTED_ARTIFACTS.items():
            row = rows.get(artifact_id)
            if not row:
                continue
            _check_equal(
                failures,
                code=f"{artifact_id}_path_mismatch",
                actual=row.get("path"),
                expected=expected_name,
            )
            path = run_dir / expected_name
            if not path.exists():
                failures.append(_failure(f"{artifact_id}_missing_file", str(path)))
                continue
            _check_equal(
                failures,
                code=f"{artifact_id}_sha256_mismatch",
                actual=row.get("sha256"),
                expected=_sha256_file(path),
            )
            _check_equal(
                failures,
                code=f"{artifact_id}_byte_count_mismatch",
                actual=row.get("bytes"),
                expected=path.stat().st_size,
            )

        if goal_audit:
            _check_equal(
                failures,
                code="goal_audit_schema_mismatch",
                actual=goal_audit.get("schema"),
                expected=EXPECTED_GOAL_AUDIT_SCHEMA,
            )
            _check_equal(
                failures,
                code="goal_audit_status_mismatch",
                actual=goal_audit.get("status"),
                expected=EXPECTED_GOAL_AUDIT_STATUS,
            )
            _check_equal(
                failures,
                code="goal_audit_failed_checks_not_empty",
                actual=goal_audit.get("failed_checks"),
                expected=[],
            )
            _check_equal(
                failures,
                code="manifest_goal_audit_failed_check_count_mismatch",
                actual=manifest.get("goal_audit_failed_check_count"),
                expected=len(goal_audit.get("failed_checks") or []),
            )
            _check_false_flags(
                failures,
                payload=goal_audit,
                label="goal_audit",
                flags=(
                    "contacts_wing",
                    "contacts_backend",
                    "mutates_mixer",
                    "live_authority_granted",
                    "readiness_closure_allowed",
                    "live_apply_allowed",
                    "queue_import_allowed",
                ),
            )
            _check_true_flags(
                failures,
                payload=goal_audit,
                label="goal_audit",
                flags=(
                    "read_only",
                    "operator_review_required",
                    "future_wing_readback_required",
                    "hardware_proof_required_before_live",
                    "blocks_live_promotion",
                    "blocks_queue_import",
                ),
            )
            goal_audit_summary = manifest.get("goal_audit_summary")
            if not isinstance(goal_audit_summary, dict):
                failures.append(
                    _failure(
                        "manifest_goal_audit_summary_not_object",
                        repr(goal_audit_summary),
                    )
                )
                goal_audit_summary = {}
            for key in (
                "overall_offline_goal_percent",
                "method_review_card_count",
                "expected_method_count",
                "priority_section_count",
                "expected_priority_section_count",
                "missing_required_objective_families",
                "missing_expected_objective_families",
                "coverage_percent",
                "no_live_authority",
                "contacts_wing",
                "contacts_backend",
                "mutates_mixer",
                "live_authority_granted",
                "readiness_closure_allowed",
                "live_apply_allowed",
                "queue_import_allowed",
                "hardware_proof_required_before_live",
            ):
                _check_equal(
                    failures,
                    code=f"manifest_goal_audit_summary_{key}_mismatch",
                    actual=goal_audit_summary.get(key),
                    expected=goal_audit.get(key),
                )
            _check_false_flags(
                failures,
                payload=goal_audit_summary,
                label="manifest_goal_audit_summary",
                flags=(
                    "contacts_wing",
                    "contacts_backend",
                    "mutates_mixer",
                    "live_authority_granted",
                    "readiness_closure_allowed",
                    "live_apply_allowed",
                    "queue_import_allowed",
                ),
            )
            _check_equal(
                failures,
                code="manifest_goal_audit_summary_no_live_authority_mismatch",
                actual=goal_audit_summary.get("no_live_authority"),
                expected=True,
            )

        summary = manifest.get("summary") if isinstance(manifest.get("summary"), dict) else {}
        _check_equal(
            failures,
            code="manifest_summary_priority_section_count_mismatch",
            actual=summary.get("priority_section_count"),
            expected=7,
        )
        packet_summary = (
            packet.get("summary") if isinstance(packet.get("summary"), dict) else {}
        )
        _check_equal(
            failures,
            code="manifest_denial_reason_mismatch",
            actual=manifest.get("denial_reason"),
            expected=packet.get("denial_reason"),
        )
        _check_equal(
            failures,
            code="manifest_operator_next_step_mismatch",
            actual=manifest.get("operator_next_step"),
            expected=packet.get("operator_next_step"),
        )
        for key in (
            "required_objective_family_count",
            "covered_objective_family_count",
            "method_review_card_count",
            "supporting_method_review_card_count",
            "operator_review_contract_coverage",
            "blocked_by",
        ):
            _check_equal(
                failures,
                code=f"manifest_summary_{key}_mismatch",
                actual=summary.get(key),
                expected=packet_summary.get(key),
            )
        for (
            _packet_key,
            summary_prefix,
            _ordinal_label,
            _expected_priority,
            _expected_label,
        ) in EXPECTED_PRIORITY_REVIEW_SUMMARIES:
            for suffix in (
                "method_count",
                "method_ids",
                "applicable_safety_checks",
                "future_wing_readback_target_count",
                "blocked_live_action_count",
            ):
                summary_key = f"{summary_prefix}_{suffix}"
                _check_equal(
                    failures,
                    code=f"manifest_summary_{summary_key}_mismatch",
                    actual=summary.get(summary_key),
                    expected=packet_summary.get(summary_key),
                )
        for key in (
            "live_authority_method_ids",
            "live_apply_method_ids",
            "queueable_method_ids",
            "missing_required_operator_surface_method_ids",
        ):
            _check_equal(
                failures,
                code=f"manifest_summary_{key}_not_empty",
                actual=summary.get(key),
                expected=[],
            )

        decision_template_summary = manifest.get(
            "operator_decision_template_summary"
        )
        if not isinstance(decision_template_summary, dict):
            failures.append(
                _failure(
                    "manifest_decision_template_summary_not_object",
                    repr(decision_template_summary),
                )
            )
        else:
            expected_priority_labels = [
                row[4] for row in EXPECTED_PRIORITY_REVIEW_SUMMARIES
            ]
            _check_equal(
                failures,
                code="manifest_decision_template_summary_method_count_mismatch",
                actual=decision_template_summary.get("method_decision_count"),
                expected=decision_template.get("method_decision_count"),
            )
            _check_equal(
                failures,
                code="manifest_decision_template_summary_priority_count_mismatch",
                actual=decision_template_summary.get(
                    "priority_decision_section_count"
                ),
                expected=len(EXPECTED_PRIORITY_REVIEW_SUMMARIES),
            )
            _check_equal(
                failures,
                code="manifest_decision_template_summary_priority_labels_mismatch",
                actual=decision_template_summary.get("priority_decision_labels"),
                expected=expected_priority_labels,
            )
            priority_sections = decision_template.get(
                "priority_decision_sections"
            )
            if isinstance(priority_sections, list):
                expected_method_counts = {
                    str(section.get("priority")): section.get(
                        "method_decision_count"
                    )
                    for section in priority_sections
                    if isinstance(section, dict)
                }
                _check_equal(
                    failures,
                    code=(
                        "manifest_decision_template_summary_priority_"
                        "method_counts_mismatch"
                    ),
                    actual=decision_template_summary.get(
                        "priority_method_decision_counts"
                    ),
                    expected=expected_method_counts,
                )
                for summary_key, section_field in (
                    (
                        "priority_required_evidence_counts",
                        "required_evidence",
                    ),
                    (
                        "priority_required_safety_check_counts",
                        "required_safety_checks",
                    ),
                    (
                        "priority_source_evidence_counts",
                        "source_evidence",
                    ),
                    (
                        "priority_source_evidence_detail_counts",
                        "source_evidence_details",
                    ),
                    (
                        "priority_measured_or_audible_problem_counts",
                        "measured_or_audible_problem",
                    ),
                    (
                        "priority_proposed_correction_counts",
                        "proposed_correction",
                    ),
                    (
                        "priority_confidence_counts",
                        "confidence",
                    ),
                    (
                        "priority_bounded_delta_counts",
                        "bounded_delta",
                    ),
                    (
                        "priority_safety_blocker_counts",
                        "safety_blockers",
                    ),
                    (
                        "priority_goal_requirement_detail_counts",
                        "goal_requirement_details",
                    ),
                    (
                        "priority_method_specific_review_detail_counts",
                        "method_specific_review_details",
                    ),
                    (
                        "priority_operator_actionable_explanation_detail_counts",
                        "operator_actionable_explanation_details",
                    ),
                    (
                        "priority_evidence_problem_correction_chain_detail_counts",
                        "evidence_problem_correction_chain_details",
                    ),
                    (
                        "priority_safety_blocker_detail_counts",
                        "safety_blocker_details",
                    ),
                    (
                        "priority_headroom_feedback_true_peak_check_counts",
                        "headroom_feedback_true_peak_checks",
                    ),
                    (
                        "priority_headroom_feedback_true_peak_check_detail_counts",
                        "headroom_feedback_true_peak_check_details",
                    ),
                    (
                        "priority_operator_readable_explanation_counts",
                        "operator_readable_explanation",
                    ),
                    (
                        "priority_rollback_readback_contract_counts",
                        "rollback_readback_contract",
                    ),
                    (
                        "priority_future_wing_readback_contract_counts",
                        "future_wing_readback_contract",
                    ),
                    (
                        "priority_test_evidence_counts",
                        "test_evidence",
                    ),
                    (
                        "priority_test_evidence_reference_counts",
                        "test_evidence_references",
                    ),
                    (
                        "priority_bounded_delta_policy_counts",
                        "bounded_delta_policies",
                    ),
                    (
                        "priority_rollback_expectation_counts",
                        "rollback_expectations",
                    ),
                    (
                        "priority_operator_payload_required_field_counts",
                        "operator_payload_required_fields",
                    ),
                    (
                        "priority_operator_evidence_required_field_counts",
                        "operator_evidence_required_fields",
                    ),
                    (
                        "priority_operator_problem_review_summary_counts",
                        "operator_problem_review_summary",
                    ),
                    (
                        "priority_operator_method_review_focus_counts",
                        "operator_method_review_focus",
                    ),
                    (
                        "priority_why_safe_summary_counts",
                        "why_safe_summary",
                    ),
                    (
                        "priority_why_blocked_summary_counts",
                        "why_blocked_summary",
                    ),
                    (
                        "priority_operator_visible_explanation_contract_counts",
                        "operator_visible_explanation_contract",
                    ),
                    (
                        "priority_future_wing_readback_contract_summary_counts",
                        "future_wing_readback_contract_summary",
                    ),
                    (
                        "priority_future_live_requirement_detail_counts",
                        "future_live_requirement_details",
                    ),
                    (
                        "priority_live_authority_boundary_summary_counts",
                        "live_authority_boundary_summary",
                    ),
                    (
                        "priority_operator_review_record_required_field_counts",
                        "operator_review_record_required_fields",
                    ),
                    (
                        "priority_required_confirmation_detail_counts",
                        "required_confirmation_details",
                    ),
                    (
                        "priority_operator_acceptance_gate_detail_counts",
                        "operator_acceptance_gate_details",
                    ),
                    (
                        "priority_operator_review_decision_option_detail_counts",
                        "operator_review_decision_option_details",
                    ),
                    (
                        "priority_operator_review_checklist_detail_counts",
                        "operator_review_checklist_details",
                    ),
                    (
                        "priority_operator_review_record_field_detail_counts",
                        "operator_review_record_field_details",
                    ),
                    (
                        "priority_operator_next_action_detail_counts",
                        "operator_next_action_details",
                    ),
                ):
                    expected_counts = {
                        str(section.get("priority")): len(
                            section.get(section_field) or []
                        )
                        for section in priority_sections
                        if isinstance(section, dict)
                    }
                    _check_equal(
                        failures,
                        code=(
                            "manifest_decision_template_summary_"
                            f"{summary_key}_mismatch"
                        ),
                        actual=decision_template_summary.get(summary_key),
                        expected=expected_counts,
                    )
            _check_equal(
                failures,
                code=(
                    "manifest_decision_template_summary_completion_"
                    "requirements_mismatch"
                ),
                actual=decision_template_summary.get(
                    "operator_completion_requires"
                ),
                expected=decision_template.get("operator_completion_requires"),
            )
            _check_equal(
                failures,
                code="manifest_decision_template_summary_denial_reason_mismatch",
                actual=decision_template_summary.get("denial_reason"),
                expected=decision_template.get("denial_reason"),
            )
            _check_false_flags(
                failures,
                payload=decision_template_summary,
                label="manifest_decision_template_summary",
                flags=FALSE_AUTHORITY_FLAGS + ("can_authorize_live",),
            )
            _check_true_flags(
                failures,
                payload=decision_template_summary,
                label="manifest_decision_template_summary",
                flags=TRUE_REVIEW_FLAGS + ("read_only",),
            )

    method_ids: List[str] = []
    cards_by_method_id: Dict[str, Mapping[str, Any]] = {}
    verified_priority_labels: List[str] = []
    priority_method_ids_by_priority: Dict[int, List[str]] = {}
    section_by_priority: Dict[int, Mapping[str, Any]] = {}
    expected_priority_by_method_id: Dict[str, int] = {}
    expected_priority_label_by_method_id: Dict[str, str] = {}
    verified_priority_decision_labels: List[str] = []
    if packet:
        _check_equal(
            failures,
            code="packet_schema_mismatch",
            actual=packet.get("schema"),
            expected=EXPECTED_PACKET_SCHEMA,
        )
        _check_equal(
            failures,
            code="packet_status_mismatch",
            actual=packet.get("status"),
            expected=EXPECTED_PACKET_STATUS,
        )
        _check_false_flags(
            failures,
            payload=packet,
            label="packet",
            flags=(
                "contacts_wing",
                "contacts_backend",
                "mutates_mixer",
                "live_authority_granted",
                "readiness_closure_allowed",
                "live_apply_allowed",
                "queue_import_allowed",
            ),
        )
        _check_true_flags(
            failures,
            payload=packet,
            label="packet",
            flags=TRUE_REVIEW_FLAGS,
        )
        _check_equal(
            failures,
            code="packet_required_operator_surfaces_mismatch",
            actual=_sorted_str_items(packet.get("required_operator_surfaces")),
            expected=sorted(REQUIRED_OPERATOR_SURFACES),
        )
        if goal_audit:
            expected_goal_audit = audit_correction_method_goal_packet(
                packet,
                pipeline_id=str(packet.get("pipeline_id") or ""),
                generated_at_utc=str(goal_audit.get("generated_at_utc") or ""),
            )
            for key in (
                "schema",
                "status",
                "failed_checks",
                "method_review_card_count",
                "expected_method_count",
                "missing_expected_method_ids",
                "extra_method_ids",
                "priority_section_count",
                "expected_priority_section_count",
                "required_objective_families",
                "covered_required_objective_families",
                "missing_required_objective_families",
                "expected_objective_families",
                "covered_expected_objective_families",
                "missing_expected_objective_families",
                "supporting_objective_families",
                "required_operator_surfaces",
                "required_method_review_fields",
                "required_method_detail_list_fields",
                "coverage_percent",
                "overall_offline_goal_percent",
                "denial_reason",
                "operator_next_step",
                "read_only",
                "contacts_wing",
                "contacts_backend",
                "mutates_mixer",
                "live_authority_granted",
                "readiness_closure_allowed",
                "live_apply_allowed",
                "queue_import_allowed",
                "hardware_proof_required_before_live",
                "operator_review_required",
                "future_wing_readback_required",
                "blocks_live_promotion",
                "blocks_queue_import",
                "no_live_authority",
            ):
                _check_equal(
                    failures,
                    code=f"goal_audit_{key}_mismatch",
                    actual=goal_audit.get(key),
                    expected=expected_goal_audit.get(key),
                )

        priority_sections = packet.get("priority_sections")
        if not isinstance(priority_sections, list):
            failures.append(
                _failure("packet_priority_sections_not_list", repr(priority_sections))
            )
        else:
            _check_equal(
                failures,
                code="packet_priority_section_count_mismatch",
                actual=len(priority_sections),
                expected=len(EXPECTED_PRIORITY_REVIEW_SUMMARIES),
            )
            for section in priority_sections:
                if not isinstance(section, dict):
                    failures.append(
                        _failure("packet_priority_section_not_object", repr(section))
                    )
                    continue
                priority = section.get("priority")
                if not isinstance(priority, int):
                    failures.append(
                        _failure(
                            "packet_priority_section_priority_not_int",
                            repr(priority),
                        )
                    )
                    continue
                if priority in section_by_priority:
                    failures.append(
                        _failure(
                            "packet_priority_section_duplicate_priority",
                            str(priority),
                        )
                    )
                    continue
                section_by_priority[priority] = section
            for priority, section in section_by_priority.items():
                label = str(section.get("label") or "")
                for raw_method_id in _priority_section_method_ids(section):
                    method_id = str(raw_method_id or "")
                    if not method_id:
                        continue
                    if method_id in expected_priority_by_method_id:
                        failures.append(
                            _failure(
                                "packet_method_card_duplicate_priority_assignment",
                                method_id,
                            )
                        )
                        continue
                    expected_priority_by_method_id[method_id] = priority
                    expected_priority_label_by_method_id[method_id] = label

        cards = packet.get("method_review_cards")
        if not isinstance(cards, list) or not cards:
            failures.append(
                _failure("packet_method_review_cards_missing", repr(cards))
            )
            cards = []
        for card in cards:
            if not isinstance(card, dict):
                failures.append(_failure("packet_method_card_not_object", repr(card)))
                continue
            method_id = str(card.get("method_id") or "")
            if not method_id:
                failures.append(_failure("packet_method_card_missing_id", repr(card)))
                continue
            method_ids.append(method_id)
            if method_id in cards_by_method_id:
                failures.append(
                    _failure("packet_method_card_duplicate_id", method_id)
                )
                continue
            cards_by_method_id[method_id] = card
            if method_id in expected_priority_by_method_id:
                _check_equal(
                    failures,
                    code=f"method_{method_id}_priority_mismatch",
                    actual=card.get("priority"),
                    expected=expected_priority_by_method_id[method_id],
                )
                _check_equal(
                    failures,
                    code=f"method_{method_id}_priority_label_mismatch",
                    actual=card.get("priority_label"),
                    expected=expected_priority_label_by_method_id[method_id],
                )
                _check_equal(
                    failures,
                    code=f"method_{method_id}_priority_review_scope_mismatch",
                    actual=card.get("priority_review_scope"),
                    expected="required_priority",
                )
            else:
                _check_equal(
                    failures,
                    code=f"method_{method_id}_supporting_priority_mismatch",
                    actual=card.get("priority"),
                    expected=None,
                )
                _check_equal(
                    failures,
                    code=f"method_{method_id}_supporting_priority_label_mismatch",
                    actual=card.get("priority_label"),
                    expected="Supporting / Context",
                )
                _check_equal(
                    failures,
                    code=(
                        f"method_{method_id}_"
                        "supporting_priority_review_scope_mismatch"
                    ),
                    actual=card.get("priority_review_scope"),
                    expected="supporting_context",
                )
            _check_false_flags(
                failures,
                payload=card,
                label=f"method_{method_id}",
                flags=(
                    "contacts_wing",
                    "contacts_backend",
                    "mutates_mixer",
                    "live_authority_granted",
                    "readiness_closure_allowed",
                    "live_apply_allowed",
                    "queue_import_allowed",
                ),
            )
            if card.get("review_scope") != "offline_proposal_only":
                failures.append(
                    _failure(
                        f"method_{method_id}_review_scope_mismatch",
                        repr(card.get("review_scope")),
                    )
                )
            for field in REQUIRED_METHOD_CARD_TEXT_FIELDS:
                if not card.get(field):
                    failures.append(
                        _failure(
                            f"method_{method_id}_missing_{field}",
                            f"{field}={card.get(field)!r}",
                        )
                    )
            for field in REQUIRED_METHOD_CARD_LIST_FIELDS:
                if not card.get(field):
                    failures.append(
                        _failure(
                            f"method_{method_id}_missing_{field}",
                            f"{field}={card.get(field)!r}",
                        )
                    )
            _check_equal(
                failures,
                code=f"method_{method_id}_required_confirmation_details_mismatch",
                actual=_sorted_str_items(card.get("required_confirmation_details")),
                expected=sorted(
                    _expected_required_confirmation_details(
                        card.get("what_operator_must_confirm")
                    )
                ),
            )
            _check_equal(
                failures,
                code=f"method_{method_id}_operator_acceptance_gate_details_mismatch",
                actual=_sorted_str_items(
                    card.get("operator_acceptance_gate_details")
                ),
                expected=sorted(
                    _expected_operator_acceptance_gate_details(
                        card.get("what_operator_must_confirm")
                    )
                ),
            )
            _check_equal(
                failures,
                code=(
                    f"method_{method_id}_"
                    "operator_review_decision_option_details_mismatch"
                ),
                actual=_sorted_str_items(
                    card.get("operator_review_decision_option_details")
                ),
                expected=sorted(
                    _expected_operator_review_decision_option_details(
                        card.get("operator_review_decisions")
                    )
                ),
            )
            _check_equal(
                failures,
                code=f"method_{method_id}_operator_problem_review_summary_mismatch",
                actual=_sorted_str_items(card.get("operator_problem_review_summary")),
                expected=sorted(REQUIRED_OPERATOR_PROBLEM_REVIEW_SUMMARY),
            )
            _check_equal(
                failures,
                code=f"method_{method_id}_operator_method_review_focus_mismatch",
                actual=list(card.get("operator_method_review_focus") or []),
                expected=_expected_operator_method_review_focus(card),
            )
            focus_keys = [
                str(item).split("=", 1)[0]
                for item in (card.get("operator_method_review_focus") or [])
            ]
            _check_equal(
                failures,
                code=f"method_{method_id}_operator_method_review_focus_keys_mismatch",
                actual=focus_keys,
                expected=list(REQUIRED_OPERATOR_METHOD_REVIEW_FOCUS_KEYS),
            )
            _check_equal(
                failures,
                code=f"method_{method_id}_why_safe_summary_mismatch",
                actual=list(card.get("why_safe_summary") or []),
                expected=_expected_operator_why_safe_summary(card),
            )
            _check_equal(
                failures,
                code=f"method_{method_id}_why_safe_summary_keys_mismatch",
                actual=_summary_keys(card.get("why_safe_summary")),
                expected=list(REQUIRED_OPERATOR_WHY_SAFE_SUMMARY_KEYS),
            )
            _check_equal(
                failures,
                code=f"method_{method_id}_why_blocked_summary_mismatch",
                actual=list(card.get("why_blocked_summary") or []),
                expected=_expected_operator_why_blocked_summary(card),
            )
            _check_equal(
                failures,
                code=f"method_{method_id}_why_blocked_summary_keys_mismatch",
                actual=_summary_keys(card.get("why_blocked_summary")),
                expected=list(REQUIRED_OPERATOR_WHY_BLOCKED_SUMMARY_KEYS),
            )
            _check_equal(
                failures,
                code=f"method_{method_id}_goal_requirement_details_mismatch",
                actual=_sorted_str_items(card.get("goal_requirement_details")),
                expected=_expected_goal_requirement_details(card),
            )
            _check_equal(
                failures,
                code=f"method_{method_id}_method_specific_review_details_mismatch",
                actual=_sorted_str_items(card.get("method_specific_review_details")),
                expected=_expected_method_specific_review_details(card),
            )
            _check_equal(
                failures,
                code=(
                    f"method_{method_id}_"
                    "operator_actionable_explanation_details_mismatch"
                ),
                actual=_sorted_str_items(
                    card.get("operator_actionable_explanation_details")
                ),
                expected=_expected_operator_actionable_explanation_details(card),
            )
            _check_test_evidence_references(
                failures,
                method_id=method_id,
                card=card,
            )
            _check_equal(
                failures,
                code=f"method_{method_id}_operator_review_record_field_details_mismatch",
                actual=_sorted_str_items(
                    card.get("operator_review_record_field_details")
                ),
                expected=sorted(
                    _expected_operator_review_record_field_details(
                        card.get("operator_review_record_required_fields")
                    )
                ),
            )
            _check_equal(
                failures,
                code=f"method_{method_id}_operator_review_checklist_details_mismatch",
                actual=_sorted_str_items(card.get("operator_review_checklist_details")),
                expected=sorted(
                    _expected_operator_review_checklist_details(
                        card.get("operator_review_checklist")
                    )
                ),
            )
            _check_equal(
                failures,
                code=f"method_{method_id}_operator_next_action_details_mismatch",
                actual=_sorted_str_items(card.get("operator_next_action_details")),
                expected=_expected_operator_next_action_details(card),
            )
            _check_equal(
                failures,
                code=f"method_{method_id}_future_live_requirement_details_mismatch",
                actual=_sorted_str_items(card.get("future_live_requirement_details")),
                expected=_expected_future_live_requirement_details(card),
            )
            _check_equal(
                failures,
                code=f"method_{method_id}_safety_blocker_details_mismatch",
                actual=_sorted_str_items(card.get("safety_blocker_details")),
                expected=_expected_safety_blocker_details(card),
            )
            _check_equal(
                failures,
                code=(
                    f"method_{method_id}_"
                    "evidence_problem_correction_chain_details_mismatch"
                ),
                actual=_sorted_str_items(
                    card.get("evidence_problem_correction_chain_details")
                ),
                expected=_expected_evidence_problem_correction_chain_details(card),
            )
            _check_equal(
                failures,
                code=(
                    f"method_{method_id}_"
                    "headroom_feedback_true_peak_check_details_mismatch"
                ),
                actual=_sorted_str_items(
                    card.get("headroom_feedback_true_peak_check_details")
                ),
                expected=_expected_headroom_feedback_true_peak_check_details(card),
            )
            for field in REQUIRED_METHOD_CARD_CONTRACT_FIELDS:
                contract = card.get(field)
                if not isinstance(contract, Mapping):
                    failures.append(
                        _failure(
                            f"method_{method_id}_{field}_contract_not_object",
                            repr(contract),
                        )
                    )
                    continue
                if contract.get("required") is not True:
                    failures.append(
                        _failure(
                            f"method_{method_id}_{field}_contract_required_not_true",
                            repr(contract.get("required")),
                        )
                    )
                if contract.get("review_status") != (
                    "required_from_offline_proposal_payload"
                ):
                    failures.append(
                        _failure(
                            f"method_{method_id}_{field}_contract_review_status_mismatch",
                            repr(contract.get("review_status")),
                        )
                    )
                source_fields = contract.get("source_fields")
                if not isinstance(source_fields, list) or not source_fields:
                    failures.append(
                        _failure(
                            f"method_{method_id}_{field}_contract_source_fields_missing",
                            repr(source_fields),
                        )
                    )
                    source_fields = []
                available_source_fields = {
                    str(key) for key in card.keys()
                } | set(_sorted_str_items(card.get("operator_payload_required_fields")))
                available_source_fields |= set(
                    _sorted_str_items(card.get("operator_evidence_required_fields"))
                )
                available_source_fields |= set(
                    ABSTRACT_METHOD_CARD_CONTRACT_SOURCE_FIELDS
                )
                for source_field in source_fields:
                    if not isinstance(source_field, str) or not source_field:
                        failures.append(
                            _failure(
                                (
                                    f"method_{method_id}_{field}_contract_"
                                    "source_field_invalid"
                                ),
                                repr(source_field),
                            )
                        )
                        continue
                    if source_field not in available_source_fields:
                        failures.append(
                            _failure(
                                (
                                    f"method_{method_id}_{field}_contract_"
                                    "source_field_unbound"
                                ),
                                repr(source_field),
                            )
                        )
                for required_source_field in (
                    REQUIRED_METHOD_CARD_CONTRACT_SOURCE_FIELDS.get(field, ())
                ):
                    if required_source_field not in source_fields:
                        failures.append(
                            _failure(
                                (
                                    f"method_{method_id}_{field}_contract_"
                                    f"source_field_{required_source_field}_missing"
                                ),
                                repr(source_fields),
                            )
                        )
                if not contract.get("operator_action"):
                    failures.append(
                        _failure(
                            f"method_{method_id}_{field}_contract_operator_action_missing",
                            repr(contract.get("operator_action")),
                        )
                    )
            _check_equal(
                failures,
                code=f"method_{method_id}_required_operator_surfaces_mismatch",
                actual=_sorted_str_items(card.get("required_operator_surfaces")),
                expected=sorted(REQUIRED_OPERATOR_SURFACES),
            )
            _check_equal(
                failures,
                code=(
                    f"method_{method_id}_"
                    "missing_required_operator_surfaces_not_empty"
                ),
                actual=_sorted_str_items(
                    card.get("missing_required_operator_surfaces")
                ),
                expected=[],
            )
            _check_operator_confirmations(
                failures,
                code_prefix=f"method_{method_id}",
                field_name="operator_confirmations",
                value=card.get("what_operator_must_confirm"),
            )
            _check_operator_review_record_required_fields(
                failures,
                code_prefix=f"method_{method_id}",
                value=card.get("operator_review_record_required_fields"),
            )
            _check_required_safety_checks(
                failures,
                code_prefix=f"method_{method_id}",
                value=card.get("required_safety_checks"),
            )
            _check_bounded_delta_contract(
                failures,
                code_prefix=f"method_{method_id}",
                bounded_delta_unit=card.get("bounded_delta_unit"),
                bounded_delta_policy=card.get("bounded_delta_policy"),
            )
            _check_future_wing_readback_targets(
                failures,
                code_prefix=f"method_{method_id}",
                value=card.get("future_wing_readback_targets"),
            )
            _check_rollback_expectation(
                failures,
                code_prefix=f"method_{method_id}",
                value=card.get("rollback_expectation"),
            )
            _check_offline_review_decisions(
                failures,
                code_prefix=f"method_{method_id}",
                field_name="operator_review_decisions",
                value=card.get("operator_review_decisions"),
            )
            _check_future_wing_readback_contract_summary(
                failures,
                code_prefix=f"method_{method_id}",
                value=card.get("future_wing_readback_contract_summary"),
            )
            _check_future_live_required_proofs(
                failures,
                code_prefix=f"method_{method_id}",
                value=card.get("future_live_required_proofs"),
            )
            _check_blocked_live_actions(
                failures,
                code_prefix=f"method_{method_id}",
                value=card.get("blocked_live_actions"),
            )
            _check_live_authority_boundary_summary(
                failures,
                code_prefix=f"method_{method_id}",
                value=card.get("live_authority_boundary_summary"),
            )
            operator_visible_contract = card.get(
                "operator_visible_explanation_contract"
            )
            if not isinstance(operator_visible_contract, dict):
                failures.append(
                    _failure(
                        (
                            f"method_{method_id}_"
                            "operator_visible_explanation_contract_not_object"
                        ),
                        repr(operator_visible_contract),
                    )
                )
                operator_visible_contract = {}
            for flag in REQUIRED_OPERATOR_VISIBLE_CONTRACT_FLAGS:
                if operator_visible_contract.get(flag) is not True:
                    failures.append(
                        _failure(
                            (
                                f"method_{method_id}_"
                                f"operator_visible_explanation_contract_{flag}_"
                                "not_true"
                            ),
                            f"{flag}={operator_visible_contract.get(flag)!r}",
                        )
                    )

        packet_summary = (
            packet.get("summary") if isinstance(packet.get("summary"), dict) else {}
        )
        _check_equal(
            failures,
            code="packet_summary_method_review_card_count_mismatch",
            actual=packet_summary.get("method_review_card_count"),
            expected=len(method_ids),
        )
        _check_equal(
            failures,
            code="packet_summary_required_objective_family_count_mismatch",
            actual=packet_summary.get("required_objective_family_count"),
            expected=len(packet.get("required_objective_families") or []),
        )
        _check_equal(
            failures,
            code="packet_summary_covered_objective_family_count_mismatch",
            actual=packet_summary.get("covered_objective_family_count"),
            expected=len(
                set(str(item) for item in packet.get("required_objective_families") or [])
                & {
                    str(card.get("objective_family"))
                    for card in cards
                    if isinstance(card, dict) and card.get("objective_family")
                }
            ),
        )
        supporting_cards = packet.get("supporting_method_cards")
        supporting_method_ids: List[str] = []
        if not isinstance(supporting_cards, list):
            failures.append(
                _failure(
                    "packet_supporting_method_cards_not_list",
                    repr(supporting_cards),
                )
            )
            supporting_cards = []
        for supporting_card in supporting_cards:
            if not isinstance(supporting_card, dict):
                failures.append(
                    _failure(
                        "packet_supporting_method_card_not_object",
                        repr(supporting_card),
                    )
                )
                continue
            method_id = str(supporting_card.get("method_id") or "")
            if not method_id:
                failures.append(
                    _failure(
                        "packet_supporting_method_card_missing_id",
                        repr(supporting_card),
                    )
                )
                continue
            if method_id in supporting_method_ids:
                failures.append(
                    _failure(
                        "packet_supporting_method_card_duplicate_id",
                        method_id,
                    )
                )
                continue
            supporting_method_ids.append(method_id)
            expected_card = cards_by_method_id.get(method_id)
            if expected_card is None:
                failures.append(
                    _failure(
                        (
                            f"packet_supporting_method_{method_id}_"
                            "missing_top_level_card"
                        ),
                        repr(supporting_card),
                    )
                )
                continue
            _check_method_card_mirror(
                failures,
                actual_card=supporting_card,
                expected_card=expected_card,
                label=f"packet_supporting_method_{method_id}",
            )
        _check_equal(
            failures,
            code="packet_supporting_method_card_count_mismatch",
            actual=len(supporting_method_ids),
            expected=packet_summary.get("supporting_method_review_card_count"),
        )
        contract_coverage = packet_summary.get("operator_review_contract_coverage")
        if not isinstance(contract_coverage, Mapping):
            failures.append(
                _failure(
                    "packet_summary_operator_review_contract_coverage_not_object",
                    repr(contract_coverage),
                )
            )
            contract_coverage = {}
        priority_section_values = [
            section
            for section in section_by_priority.values()
            if isinstance(section, Mapping)
        ]
        for field in REQUIRED_METHOD_CARD_CONTRACT_FIELDS:
            row = contract_coverage.get(field)
            if not isinstance(row, Mapping):
                failures.append(
                    _failure(
                        f"packet_summary_{field}_contract_coverage_not_object",
                        repr(row),
                    )
                )
                continue
            expected_method_count = sum(
                1
                for card in cards
                if isinstance(card, Mapping)
                and isinstance(card.get(field), Mapping)
            )
            expected_required_method_count = sum(
                1
                for card in cards
                if isinstance(card, Mapping)
                and isinstance(card.get(field), Mapping)
                and card[field].get("required") is True
            )
            expected_priority_section_count = sum(
                1 for section in priority_section_values if section.get(field)
            )
            expected_priority_section_item_count = sum(
                len(section.get(field) or []) for section in priority_section_values
            )
            expected_covered = (
                expected_method_count == len(method_ids)
                and expected_required_method_count == len(method_ids)
                and expected_priority_section_count == len(priority_section_values)
            )
            for metric, expected_value in (
                ("method_card_count", expected_method_count),
                ("required_method_card_count", expected_required_method_count),
                ("priority_section_count", expected_priority_section_count),
                (
                    "priority_section_contract_item_count",
                    expected_priority_section_item_count,
                ),
                ("covered", expected_covered),
            ):
                _check_equal(
                    failures,
                    code=(
                        f"packet_summary_{field}_contract_coverage_"
                        f"{metric}_mismatch"
                    ),
                    actual=row.get(metric),
                    expected=expected_value,
                )
        for key in (
            "live_authority_method_ids",
            "live_apply_method_ids",
            "queueable_method_ids",
            "missing_required_operator_surface_method_ids",
        ):
            _check_equal(
                failures,
                code=f"packet_summary_{key}_not_empty",
                actual=packet_summary.get(key),
                expected=[],
            )

        for (
            packet_key,
            summary_prefix,
            ordinal_label,
            expected_priority,
            expected_label,
        ) in EXPECTED_PRIORITY_REVIEW_SUMMARIES:
            priority_row = packet.get(packet_key)
            if not isinstance(priority_row, dict):
                failures.append(
                    _failure(f"packet_missing_{packet_key}", repr(priority_row))
                )
                continue
            verified_priority_labels.append(str(priority_row.get("label") or ""))
            _check_equal(
                failures,
                code=f"{packet_key}_priority_mismatch",
                actual=priority_row.get("priority"),
                expected=expected_priority,
            )
            _check_equal(
                failures,
                code=f"{packet_key}_label_mismatch",
                actual=priority_row.get("label"),
                expected=expected_label,
            )
            _check_equal(
                failures,
                code=f"{packet_key}_covered_not_true",
                actual=priority_row.get("covered"),
                expected=True,
            )
            _check_true_flags(
                failures,
                payload=priority_row,
                label=packet_key,
                flags=TRUE_REVIEW_FLAGS,
            )
            _check_false_flags(
                failures,
                payload=priority_row,
                label=packet_key,
                flags=(
                    "contacts_wing",
                    "contacts_backend",
                    "mutates_mixer",
                    "live_authority_granted",
                    "readiness_closure_allowed",
                    "live_apply_allowed",
                    "queue_import_allowed",
                ),
            )
            priority_method_ids = priority_row.get("method_ids")
            if not isinstance(priority_method_ids, list) or not priority_method_ids:
                failures.append(
                    _failure(
                        f"{packet_key}_method_ids_missing",
                        repr(priority_method_ids),
                    )
                )
                priority_method_ids = []
            priority_method_ids_by_priority[expected_priority] = [
                str(method_id) for method_id in priority_method_ids
            ]
            if not priority_row.get("applicable_safety_checks"):
                failures.append(
                    _failure(
                        f"{packet_key}_missing_applicable_safety_checks",
                        packet_key,
                    )
                )
            if not priority_row.get("future_wing_readback_targets"):
                failures.append(
                    _failure(
                        f"{packet_key}_missing_future_wing_readback_targets",
                        packet_key,
                    )
                )
            _check_future_wing_readback_targets(
                failures,
                code_prefix=packet_key,
                value=priority_row.get("future_wing_readback_targets"),
            )
            if not priority_row.get("blocked_live_actions"):
                failures.append(
                    _failure(f"{packet_key}_missing_blocked_live_actions", packet_key)
                )

            section = section_by_priority.get(expected_priority)
            if not section:
                failures.append(
                    _failure(
                        f"packet_missing_priority_section_{expected_priority}",
                        expected_label,
                    )
                )
            else:
                _check_equal(
                    failures,
                    code=f"{packet_key}_section_label_mismatch",
                    actual=section.get("label"),
                    expected=expected_label,
                )
                _check_equal(
                    failures,
                    code=f"{packet_key}_section_method_count_mismatch",
                    actual=section.get("method_count"),
                    expected=priority_row.get("method_count"),
                )
                section_methods = section.get("methods")
                if not isinstance(section_methods, list):
                    failures.append(
                        _failure(
                            f"{packet_key}_section_methods_not_list",
                            repr(section_methods),
                        )
                    )
                else:
                    section_method_ids = sorted(
                        str(method.get("method_id") or "")
                        for method in section_methods
                        if isinstance(method, dict)
                    )
                    _check_equal(
                        failures,
                        code=f"{packet_key}_section_method_ids_mismatch",
                        actual=section_method_ids,
                        expected=sorted(str(item) for item in priority_method_ids),
                    )
                    for method in section_methods:
                        if not isinstance(method, dict):
                            failures.append(
                                _failure(
                                    (
                                        "packet_priority_section_"
                                        f"{expected_priority}_method_not_object"
                                    ),
                                    repr(method),
                                )
                            )
                            continue
                        method_id = str(method.get("method_id") or "")
                        expected_card = cards_by_method_id.get(method_id)
                        if expected_card is None:
                            failures.append(
                                _failure(
                                    (
                                        "packet_priority_section_"
                                        f"{expected_priority}_method_"
                                        f"{method_id or 'missing'}_"
                                        "missing_top_level_card"
                                    ),
                                    repr(method),
                                )
                            )
                            continue
                        mirror_label = (
                            f"packet_priority_section_{expected_priority}_"
                            f"method_{method_id}"
                        )
                        _check_method_card_mirror(
                            failures,
                            actual_card=method,
                            expected_card=expected_card,
                            label=mirror_label,
                        )
                    for field in PRIORITY_SECTION_LIST_AGGREGATE_FIELDS:
                        _check_equal(
                            failures,
                            code=(
                                f"packet_priority_section_{expected_priority}_"
                                f"{field}_mismatch"
                            ),
                            actual=_sorted_str_items(section.get(field)),
                            expected=_priority_section_method_items(section, field),
                        )
                    _check_future_wing_readback_contract_summary(
                        failures,
                        code_prefix=(
                            f"packet_priority_section_{expected_priority}"
                        ),
                        value=section.get(
                            "future_wing_readback_contract_summary"
                        ),
                        allow_duplicate_keys=True,
                    )
                    _check_required_safety_checks(
                        failures,
                        code_prefix=(
                            f"packet_priority_section_{expected_priority}"
                        ),
                        value=section.get("required_safety_checks"),
                    )
                    _check_bounded_delta_policies(
                        failures,
                        code_prefix=(
                            f"packet_priority_section_{expected_priority}"
                        ),
                        value=section.get("bounded_delta_policies"),
                    )
                    _check_future_wing_readback_targets(
                        failures,
                        code_prefix=(
                            f"packet_priority_section_{expected_priority}"
                        ),
                        value=section.get("future_wing_readback_targets"),
                    )
                    _check_rollback_expectations(
                        failures,
                        code_prefix=(
                            f"packet_priority_section_{expected_priority}"
                        ),
                        value=section.get("rollback_expectations"),
                    )
                    for field in REQUIRED_METHOD_CARD_CONTRACT_FIELDS:
                        _check_equal(
                            failures,
                            code=(
                                f"packet_priority_section_{expected_priority}_"
                                f"{field}_mismatch"
                            ),
                            actual=_sorted_str_items(section.get(field)),
                            expected=_priority_section_method_contract_items(
                                section,
                                field,
                            ),
                        )
                    for (
                        section_field,
                        method_field,
                    ) in PRIORITY_SECTION_SCALAR_AGGREGATE_FIELDS:
                        _check_equal(
                            failures,
                            code=(
                                f"packet_priority_section_{expected_priority}_"
                                f"{section_field}_mismatch"
                            ),
                            actual=_sorted_str_items(section.get(section_field)),
                            expected=_priority_section_method_scalar_items(
                                section,
                                method_field,
                            ),
                        )
                    for field in PRIORITY_SECTION_EMPTY_AUTHORITY_LIST_FIELDS:
                        _check_equal(
                            failures,
                            code=(
                                f"packet_priority_section_{expected_priority}_"
                                f"{field}_not_empty"
                            ),
                            actual=_sorted_str_items(section.get(field)),
                            expected=[],
                        )
                    _check_false_flags(
                        failures,
                        payload=section,
                        label=f"packet_priority_section_{expected_priority}",
                        flags=PRIORITY_SECTION_FALSE_FLAGS,
                    )
                    _check_true_flags(
                        failures,
                        payload=section,
                        label=f"packet_priority_section_{expected_priority}",
                        flags=PRIORITY_SECTION_TRUE_FLAGS,
                    )
                    _check_equal(
                        failures,
                        code=(
                            f"packet_priority_section_{expected_priority}_"
                            "operator_next_step_mismatch"
                        ),
                        actual=section.get("operator_next_step"),
                        expected=priority_row.get("operator_next_step"),
                    )

            for suffix, expected_value in (
                ("method_count", priority_row.get("method_count")),
                ("method_ids", priority_method_ids),
                (
                    "applicable_safety_checks",
                    priority_row.get("applicable_safety_checks"),
                ),
                (
                    "future_wing_readback_target_count",
                    len(priority_row.get("future_wing_readback_targets") or []),
                ),
                (
                    "blocked_live_action_count",
                    len(priority_row.get("blocked_live_actions") or []),
                ),
            ):
                _check_equal(
                    failures,
                    code=f"packet_summary_{summary_prefix}_{suffix}_mismatch",
                    actual=packet_summary.get(f"{summary_prefix}_{suffix}"),
                    expected=expected_value,
                )

    if decision_template:
        _check_equal(
            failures,
            code="decision_template_schema_mismatch",
            actual=decision_template.get("schema"),
            expected=EXPECTED_DECISION_TEMPLATE_SCHEMA,
        )
        _check_equal(
            failures,
            code="decision_template_status_mismatch",
            actual=decision_template.get("status"),
            expected=EXPECTED_DECISION_TEMPLATE_STATUS,
        )
        _check_false_flags(
            failures,
            payload=decision_template,
            label="decision_template",
            flags=FALSE_AUTHORITY_FLAGS + ("can_authorize_live",),
        )
        _check_true_flags(
            failures,
            payload=decision_template,
            label="decision_template",
            flags=TRUE_REVIEW_FLAGS + ("read_only",),
        )
        _check_equal(
            failures,
            code="decision_template_method_count_mismatch",
            actual=decision_template.get("method_decision_count"),
            expected=len(method_ids),
        )
        decisions = decision_template.get("method_decisions")
        if not isinstance(decisions, list):
            failures.append(
                _failure("decision_template_method_decisions_not_list", repr(decisions))
            )
            decisions = []
        decision_method_ids = []
        for row in decisions:
            if not isinstance(row, dict):
                failures.append(
                    _failure("decision_template_method_decision_not_object", repr(row))
                )
                continue
            method_id = str(row.get("method_id") or "")
            decision_method_ids.append(method_id)
            if row.get("operator_decision") is not None:
                failures.append(
                    _failure(
                        f"decision_template_{method_id}_decision_not_pending",
                        repr(row.get("operator_decision")),
                    )
                )
            if row.get("review_scope") != "offline_proposal_only":
                failures.append(
                    _failure(
                        f"decision_template_{method_id}_review_scope_mismatch",
                        repr(row.get("review_scope")),
                    )
                )
            if row.get("review_status") != "operator_review_pending":
                failures.append(
                    _failure(
                        f"decision_template_{method_id}_review_status_mismatch",
                        repr(row.get("review_status")),
                    )
                )
            if row.get("operator_notes") != "":
                failures.append(
                    _failure(
                        f"decision_template_{method_id}_operator_notes_not_empty",
                        repr(row.get("operator_notes")),
                    )
                )
            _check_pending_operator_review_record_fields(
                failures,
                code_prefix=f"decision_template_{method_id}",
                payload=row,
            )
            _check_equal(
                failures,
                code=(
                    f"decision_template_{method_id}_"
                    "expected_evidence_packet_reference_mismatch"
                ),
                actual=row.get("expected_evidence_packet_reference"),
                expected=f"packet.json#method_review_cards/{method_id}",
            )
            _check_equal(
                failures,
                code=(
                    f"decision_template_{method_id}_"
                    "expected_source_material_reference_mismatch"
                ),
                actual=row.get("expected_source_material_reference"),
                expected=(
                    f"packet.json#method_review_cards/{method_id}/source_evidence"
                ),
            )
            _check_equal(
                failures,
                code=(
                    f"decision_template_{method_id}_"
                    "expected_verification_report_reference_mismatch"
                ),
                actual=row.get("expected_verification_report_reference"),
                expected=EXPECTED_VERIFICATION_REPORT_REFERENCE,
            )
            _check_false_flags(
                failures,
                payload=row,
                label=f"decision_template_{method_id}",
                flags=(
                    "contacts_wing",
                    "contacts_backend",
                    "queries_console",
                    "mutates_mixer",
                    "mutates_console_state",
                    "live_write",
                    "live_authority_granted",
                    "readiness_closure_allowed",
                    "live_apply_allowed",
                    "queue_import_allowed",
                    "can_authorize_live",
                ),
            )
            _check_true_flags(
                failures,
                payload=row,
                label=f"decision_template_{method_id}",
                flags=TRUE_REVIEW_FLAGS,
            )
            allowed_decisions = _check_offline_review_decisions(
                failures,
                code_prefix=f"decision_template_{method_id}",
                field_name="allowed_decisions",
                value=row.get("allowed_decisions"),
            )
            expected_card = cards_by_method_id.get(method_id)
            if expected_card:
                for row_field, card_field in (
                    ("method_label", "method_label"),
                    ("objective_family", "objective_family"),
                    ("priority", "priority"),
                    ("priority_label", "priority_label"),
                    ("priority_review_scope", "priority_review_scope"),
                    ("source_evidence", "source_evidence"),
                    ("source_evidence_details", "source_evidence_details"),
                    (
                        "measured_or_audible_problem",
                        "measured_or_audible_problem",
                    ),
                    ("proposed_correction", "proposed_correction"),
                    ("confidence", "confidence"),
                    ("bounded_delta", "bounded_delta"),
                    ("safety_blockers", "safety_blockers"),
                    (
                        "headroom_feedback_true_peak_checks",
                        "headroom_feedback_true_peak_checks",
                    ),
                    (
                        "operator_readable_explanation",
                        "operator_readable_explanation",
                    ),
                    (
                        "operator_visible_explanation_contract",
                        "operator_visible_explanation_contract",
                    ),
                    (
                        "rollback_readback_contract",
                        "rollback_readback_contract",
                    ),
                    (
                        "future_wing_readback_contract",
                        "future_wing_readback_contract",
                    ),
                    ("bounded_delta_unit", "bounded_delta_unit"),
                    ("bounded_delta_policy", "bounded_delta_policy"),
                    ("rollback_expectation", "rollback_expectation"),
                ):
                    _check_equal(
                        failures,
                        code=f"decision_template_{method_id}_{row_field}_mismatch",
                        actual=row.get(row_field),
                        expected=expected_card.get(card_field),
                    )
                _check_equal(
                    failures,
                    code=(
                        f"decision_template_{method_id}_"
                        "expected_test_evidence_references_mismatch"
                    ),
                    actual=_sorted_str_items(
                        row.get("expected_test_evidence_references")
                    ),
                    expected=_sorted_str_items(
                        expected_card.get("test_evidence_references")
                    ),
                )
                for row_field, card_field in (
                    ("allowed_decisions", "operator_review_decisions"),
                    ("required_confirmations", "what_operator_must_confirm"),
                    (
                        "required_confirmation_details",
                        "required_confirmation_details",
                    ),
                    (
                        "operator_acceptance_gate_details",
                        "operator_acceptance_gate_details",
                    ),
                    (
                        "operator_review_decision_option_details",
                        "operator_review_decision_option_details",
                    ),
                    ("required_evidence", "required_evidence"),
                    ("required_safety_checks", "required_safety_checks"),
                    (
                        "test_evidence_references",
                        "test_evidence_references",
                    ),
                    (
                        "operator_payload_required_fields",
                        "operator_payload_required_fields",
                    ),
                    (
                        "operator_evidence_required_fields",
                        "operator_evidence_required_fields",
                    ),
                    (
                        "operator_problem_review_summary",
                        "operator_problem_review_summary",
                    ),
                    (
                        "operator_method_review_focus",
                        "operator_method_review_focus",
                    ),
                    ("why_safe_summary", "why_safe_summary"),
                    ("why_blocked_summary", "why_blocked_summary"),
                    (
                        "goal_requirement_details",
                        "goal_requirement_details",
                    ),
                    (
                        "method_specific_review_details",
                        "method_specific_review_details",
                    ),
                    (
                        "operator_actionable_explanation_details",
                        "operator_actionable_explanation_details",
                    ),
                    (
                        "evidence_problem_correction_chain_details",
                        "evidence_problem_correction_chain_details",
                    ),
                    (
                        "future_wing_readback_targets",
                        "future_wing_readback_targets",
                    ),
                    (
                        "future_wing_readback_contract_summary",
                        "future_wing_readback_contract_summary",
                    ),
                    ("future_live_required_proofs", "future_live_required_proofs"),
                    (
                        "future_live_requirement_details",
                        "future_live_requirement_details",
                    ),
                    ("safety_blocker_details", "safety_blocker_details"),
                    (
                        "headroom_feedback_true_peak_check_details",
                        "headroom_feedback_true_peak_check_details",
                    ),
                    ("blocked_live_actions", "blocked_live_actions"),
                    (
                        "live_authority_boundary_summary",
                        "live_authority_boundary_summary",
                    ),
                    (
                        "operator_review_record_required_fields",
                        "operator_review_record_required_fields",
                    ),
                    (
                        "operator_review_checklist_details",
                        "operator_review_checklist_details",
                    ),
                    (
                        "operator_review_record_field_details",
                        "operator_review_record_field_details",
                    ),
                    (
                        "operator_next_action_details",
                        "operator_next_action_details",
                    ),
                ):
                    _check_equal(
                        failures,
                        code=f"decision_template_{method_id}_{row_field}_mismatch",
                        actual=_sorted_str_items(row.get(row_field)),
                        expected=_sorted_str_items(expected_card.get(card_field)),
                    )
            _check_operator_confirmations(
                failures,
                code_prefix=f"decision_template_{method_id}",
                field_name="required_confirmations",
                value=row.get("required_confirmations"),
            )
            _check_equal(
                failures,
                code=(
                    f"decision_template_{method_id}_"
                    "required_confirmation_details_mismatch"
                ),
                actual=_sorted_str_items(row.get("required_confirmation_details")),
                expected=sorted(
                    _expected_required_confirmation_details(
                        row.get("required_confirmations")
                    )
                ),
            )
            _check_operator_review_record_required_fields(
                failures,
                code_prefix=f"decision_template_{method_id}",
                value=row.get("operator_review_record_required_fields"),
            )
            _check_equal(
                failures,
                code=(
                    f"decision_template_{method_id}_"
                    "operator_review_checklist_details_mismatch"
                ),
                actual=_sorted_str_items(row.get("operator_review_checklist_details")),
                expected=sorted(
                    _expected_operator_review_checklist_details(
                        expected_card.get("operator_review_checklist")
                    )
                ),
            )
            _check_equal(
                failures,
                code=(
                    f"decision_template_{method_id}_"
                    "operator_review_decision_option_details_mismatch"
                ),
                actual=_sorted_str_items(
                    row.get("operator_review_decision_option_details")
                ),
                expected=sorted(
                    _expected_operator_review_decision_option_details(
                        row.get("allowed_decisions")
                    )
                ),
            )
            _check_equal(
                failures,
                code=(
                    f"decision_template_{method_id}_"
                    "operator_review_record_field_details_mismatch"
                ),
                actual=_sorted_str_items(
                    row.get("operator_review_record_field_details")
                ),
                expected=sorted(
                    _expected_operator_review_record_field_details(
                        row.get("operator_review_record_required_fields")
                    )
                ),
            )
            if not row.get("required_evidence"):
                failures.append(
                    _failure(
                        f"decision_template_{method_id}_missing_required_evidence",
                        method_id,
                    )
                )
            if not row.get("required_safety_checks"):
                failures.append(
                    _failure(
                        (
                            f"decision_template_{method_id}_"
                            "missing_required_safety_checks"
                        ),
                        method_id,
                    )
                )
            _check_required_safety_checks(
                failures,
                code_prefix=f"decision_template_{method_id}",
                value=row.get("required_safety_checks"),
            )
            _check_bounded_delta_contract(
                failures,
                code_prefix=f"decision_template_{method_id}",
                bounded_delta_unit=row.get("bounded_delta_unit"),
                bounded_delta_policy=row.get("bounded_delta_policy"),
            )
            _check_rollback_expectation(
                failures,
                code_prefix=f"decision_template_{method_id}",
                value=row.get("rollback_expectation"),
            )
            _check_operator_visible_explanation_contract(
                failures,
                code_prefix=f"decision_template_{method_id}",
                value=row.get("operator_visible_explanation_contract"),
            )
            for field in REQUIRED_METHOD_CARD_CONTRACT_FIELDS:
                contract = row.get(field)
                if not isinstance(contract, Mapping):
                    failures.append(
                        _failure(
                            f"decision_template_{method_id}_{field}_contract_not_object",
                            repr(contract),
                        )
                    )
                    continue
                if contract.get("required") is not True:
                    failures.append(
                        _failure(
                            (
                                f"decision_template_{method_id}_{field}_"
                                "contract_required_not_true"
                            ),
                            repr(contract.get("required")),
                        )
                    )
            for field in (
                "bounded_delta_unit",
                "bounded_delta_policy",
                "rollback_expectation",
            ):
                if not row.get(field):
                    failures.append(
                        _failure(
                            f"decision_template_{method_id}_missing_{field}",
                            method_id,
                        )
                    )
            if not row.get("future_wing_readback_targets"):
                failures.append(
                    _failure(
                        f"decision_template_{method_id}_missing_readback_targets",
                        method_id,
                    )
                )
            _check_future_wing_readback_targets(
                failures,
                code_prefix=f"decision_template_{method_id}",
                value=row.get("future_wing_readback_targets"),
            )
            _check_future_wing_readback_contract_summary(
                failures,
                code_prefix=f"decision_template_{method_id}",
                value=row.get("future_wing_readback_contract_summary"),
            )
            _check_future_live_required_proofs(
                failures,
                code_prefix=f"decision_template_{method_id}",
                value=row.get("future_live_required_proofs"),
            )
            _check_blocked_live_actions(
                failures,
                code_prefix=f"decision_template_{method_id}",
                value=row.get("blocked_live_actions"),
            )
            _check_live_authority_boundary_summary(
                failures,
                code_prefix=f"decision_template_{method_id}",
                value=row.get("live_authority_boundary_summary"),
            )
            _check_equal(
                failures,
                code=f"decision_template_{method_id}_why_safe_summary_keys_mismatch",
                actual=_summary_keys(row.get("why_safe_summary")),
                expected=list(REQUIRED_OPERATOR_WHY_SAFE_SUMMARY_KEYS),
            )
            _check_equal(
                failures,
                code=(
                    f"decision_template_{method_id}_"
                    "why_blocked_summary_keys_mismatch"
                ),
                actual=_summary_keys(row.get("why_blocked_summary")),
                expected=list(REQUIRED_OPERATOR_WHY_BLOCKED_SUMMARY_KEYS),
            )
            if not row.get("blocked_live_actions"):
                failures.append(
                    _failure(
                        f"decision_template_{method_id}_missing_blocked_live_actions",
                        method_id,
                    )
                )
        _check_equal(
            failures,
            code="decision_template_method_ids_mismatch",
            actual=sorted(decision_method_ids),
            expected=sorted(method_ids),
        )

        priority_sections = decision_template.get("priority_decision_sections")
        if not isinstance(priority_sections, list):
            failures.append(
                _failure(
                    "decision_template_priority_sections_not_list",
                    repr(priority_sections),
                )
            )
            priority_sections = []
        _check_equal(
            failures,
            code="decision_template_priority_section_count_mismatch",
            actual=decision_template.get("priority_decision_section_count"),
            expected=len(EXPECTED_PRIORITY_REVIEW_SUMMARIES),
        )
        _check_equal(
            failures,
            code="decision_template_priority_sections_len_mismatch",
            actual=len(priority_sections),
            expected=len(EXPECTED_PRIORITY_REVIEW_SUMMARIES),
        )
        template_sections_by_priority: Dict[int, Mapping[str, Any]] = {}
        for section in priority_sections:
            if not isinstance(section, dict):
                failures.append(
                    _failure(
                        "decision_template_priority_section_not_object",
                        repr(section),
                    )
                )
                continue
            priority = section.get("priority")
            if not isinstance(priority, int):
                failures.append(
                    _failure(
                        "decision_template_priority_section_priority_not_int",
                        repr(priority),
                    )
                )
                continue
            if priority in template_sections_by_priority:
                failures.append(
                    _failure(
                        "decision_template_priority_section_duplicate_priority",
                        str(priority),
                    )
                )
                continue
            template_sections_by_priority[priority] = section

        for (
            _packet_key,
            _summary_prefix,
            _ordinal_label,
            expected_priority,
            expected_label,
        ) in EXPECTED_PRIORITY_REVIEW_SUMMARIES:
            section = template_sections_by_priority.get(expected_priority)
            if section is None:
                failures.append(
                    _failure(
                        f"decision_template_missing_priority_section_{expected_priority}",
                        expected_label,
                    )
                )
                continue
            verified_priority_decision_labels.append(str(section.get("label") or ""))
            _check_equal(
                failures,
                code=f"decision_template_priority_{expected_priority}_label_mismatch",
                actual=section.get("label"),
                expected=expected_label,
            )
            _check_equal(
                failures,
                code=f"decision_template_priority_{expected_priority}_review_scope_mismatch",
                actual=section.get("review_scope"),
                expected="offline_proposal_only",
            )
            _check_equal(
                failures,
                code=f"decision_template_priority_{expected_priority}_review_status_mismatch",
                actual=section.get("review_status"),
                expected="operator_review_pending",
            )
            if section.get("operator_decision") is not None:
                failures.append(
                    _failure(
                        f"decision_template_priority_{expected_priority}_decision_not_pending",
                        repr(section.get("operator_decision")),
                    )
                )
            _check_pending_operator_review_record_fields(
                failures,
                code_prefix=f"decision_template_priority_{expected_priority}",
                payload=section,
            )
            _check_equal(
                failures,
                code=(
                    f"decision_template_priority_{expected_priority}_"
                    "expected_evidence_packet_reference_mismatch"
                ),
                actual=section.get("expected_evidence_packet_reference"),
                expected=f"packet.json#priority_sections/{expected_priority}",
            )
            _check_equal(
                failures,
                code=(
                    f"decision_template_priority_{expected_priority}_"
                    "expected_source_material_reference_mismatch"
                ),
                actual=section.get("expected_source_material_reference"),
                expected=(
                    "packet.json#priority_sections/"
                    f"{expected_priority}/source_evidence"
                ),
            )
            _check_equal(
                failures,
                code=(
                    f"decision_template_priority_{expected_priority}_"
                    "expected_verification_report_reference_mismatch"
                ),
                actual=section.get("expected_verification_report_reference"),
                expected=EXPECTED_VERIFICATION_REPORT_REFERENCE,
            )
            _check_equal(
                failures,
                code=f"decision_template_priority_{expected_priority}_required_priority_decision_mismatch",
                actual=section.get("required_priority_operator_decision"),
                expected="one_offline_decision_per_priority_section",
            )
            _check_equal(
                failures,
                code=f"decision_template_priority_{expected_priority}_required_method_decisions_mismatch",
                actual=section.get("required_operator_decisions"),
                expected="one_offline_decision_per_method",
            )
            _check_true_flags(
                failures,
                payload=section,
                label=f"decision_template_priority_{expected_priority}",
                flags=TRUE_REVIEW_FLAGS + ("read_only",),
            )
            _check_false_flags(
                failures,
                payload=section,
                label=f"decision_template_priority_{expected_priority}",
                flags=FALSE_AUTHORITY_FLAGS + ("can_authorize_live",),
            )
            _check_offline_review_decisions(
                failures,
                code_prefix=f"decision_template_priority_{expected_priority}",
                field_name="allowed_decisions",
                value=section.get("allowed_decisions"),
            )
            section_method_ids = section.get("method_ids")
            if not isinstance(section_method_ids, list) or not section_method_ids:
                failures.append(
                    _failure(
                        f"decision_template_priority_{expected_priority}_method_ids_missing",
                        repr(section_method_ids),
                    )
                )
                section_method_ids = []
            expected_method_ids = priority_method_ids_by_priority.get(
                expected_priority,
                section_method_ids,
            )
            _check_equal(
                failures,
                code=f"decision_template_priority_{expected_priority}_method_ids_mismatch",
                actual=sorted(str(item) for item in section_method_ids),
                expected=sorted(str(item) for item in expected_method_ids),
            )
            _check_equal(
                failures,
                code=f"decision_template_priority_{expected_priority}_method_count_mismatch",
                actual=section.get("method_decision_count"),
                expected=len(section_method_ids),
            )
            expected_priority_cards = [
                cards_by_method_id[str(method_id)]
                for method_id in expected_method_ids
                if str(method_id) in cards_by_method_id
            ]
            expected_packet_section = section_by_priority.get(expected_priority)
            if expected_packet_section:
                for section_field in (
                    "required_evidence",
                    "required_safety_checks",
                    "test_evidence_references",
                    *REQUIRED_METHOD_CARD_CONTRACT_FIELDS,
                    "operator_visible_explanation_contract",
                    "bounded_delta_policies",
                    "rollback_expectations",
                    "required_confirmation_details",
                    "operator_acceptance_gate_details",
                    "operator_review_decision_option_details",
                    "operator_review_checklist_details",
                    "operator_review_record_field_details",
                    "operator_next_action_details",
                    "goal_requirement_details",
                    "method_specific_review_details",
                    "operator_actionable_explanation_details",
                    "evidence_problem_correction_chain_details",
                    "future_live_requirement_details",
                    "safety_blocker_details",
                    "headroom_feedback_true_peak_check_details",
                ):
                    _check_equal(
                        failures,
                        code=(
                            f"decision_template_priority_{expected_priority}_"
                            f"{section_field}_mismatch"
                        ),
                        actual=_sorted_str_items(section.get(section_field)),
                        expected=_sorted_str_items(
                            expected_packet_section.get(section_field)
                        ),
                    )
                _check_equal(
                    failures,
                    code=(
                        f"decision_template_priority_{expected_priority}_"
                        "expected_test_evidence_references_mismatch"
                    ),
                    actual=_sorted_str_items(
                        section.get("expected_test_evidence_references")
                    ),
                    expected=_sorted_str_items(
                        expected_packet_section.get("test_evidence_references")
                    ),
                )
            for section_field, card_field in (
                (
                    "operator_payload_required_fields",
                    "operator_payload_required_fields",
                ),
                (
                    "operator_evidence_required_fields",
                    "operator_evidence_required_fields",
                ),
                (
                    "operator_problem_review_summary",
                    "operator_problem_review_summary",
                ),
                (
                    "operator_method_review_focus",
                    "operator_method_review_focus",
                ),
                ("why_safe_summary", "why_safe_summary"),
                ("why_blocked_summary", "why_blocked_summary"),
                ("goal_requirement_details", "goal_requirement_details"),
                (
                    "method_specific_review_details",
                    "method_specific_review_details",
                ),
                (
                    "operator_actionable_explanation_details",
                    "operator_actionable_explanation_details",
                ),
                (
                    "evidence_problem_correction_chain_details",
                    "evidence_problem_correction_chain_details",
                ),
                (
                    "future_wing_readback_targets",
                    "future_wing_readback_targets",
                ),
                (
                    "future_wing_readback_contract_summary",
                    "future_wing_readback_contract_summary",
                ),
                ("future_live_required_proofs", "future_live_required_proofs"),
                (
                    "future_live_requirement_details",
                    "future_live_requirement_details",
                ),
                ("safety_blocker_details", "safety_blocker_details"),
                (
                    "headroom_feedback_true_peak_check_details",
                    "headroom_feedback_true_peak_check_details",
                ),
                ("blocked_live_actions", "blocked_live_actions"),
                (
                    "live_authority_boundary_summary",
                    "live_authority_boundary_summary",
                ),
                (
                    "operator_review_record_required_fields",
                    "operator_review_record_required_fields",
                ),
                (
                    "required_confirmation_details",
                    "required_confirmation_details",
                ),
                (
                    "operator_acceptance_gate_details",
                    "operator_acceptance_gate_details",
                ),
                (
                    "operator_review_decision_option_details",
                    "operator_review_decision_option_details",
                ),
                (
                    "operator_review_checklist_details",
                    "operator_review_checklist_details",
                ),
                (
                    "operator_next_action_details",
                    "operator_next_action_details",
                ),
                (
                    "operator_review_record_field_details",
                    "operator_review_record_field_details",
                ),
                ):
                expected_items = sorted(
                    {
                        str(item)
                        for card in expected_priority_cards
                        for item in (card.get(card_field) or [])
                    }
                )
                _check_equal(
                    failures,
                    code=(
                        f"decision_template_priority_{expected_priority}_"
                        f"{section_field}_mismatch"
                    ),
                    actual=_sorted_str_items(section.get(section_field)),
                    expected=expected_items,
                )
            expected_explanation_contract_items = sorted(
                {
                    str(item)
                    for card in expected_priority_cards
                    for item in _operator_visible_explanation_contract_items(card)
                }
            )
            _check_equal(
                failures,
                code=(
                    f"decision_template_priority_{expected_priority}_"
                    "operator_visible_explanation_contract_mismatch"
                ),
                actual=_sorted_str_items(
                    section.get("operator_visible_explanation_contract")
                ),
                expected=expected_explanation_contract_items,
            )
            _check_operator_review_record_required_fields(
                failures,
                code_prefix=f"decision_template_priority_{expected_priority}",
                value=section.get("operator_review_record_required_fields"),
            )
            expected_checklist_detail_items = sorted(
                {
                    str(item)
                    for card in expected_priority_cards
                    for item in _expected_operator_review_checklist_details(
                        card.get("operator_review_checklist")
                    )
                }
            )
            _check_equal(
                failures,
                code=(
                    f"decision_template_priority_{expected_priority}_"
                    "operator_review_checklist_details_mismatch"
                ),
                actual=_sorted_str_items(section.get("operator_review_checklist_details")),
                expected=expected_checklist_detail_items,
            )
            _check_equal(
                failures,
                code=(
                    f"decision_template_priority_{expected_priority}_"
                    "operator_review_record_field_details_mismatch"
                ),
                actual=_sorted_str_items(
                    section.get("operator_review_record_field_details")
                ),
                expected=sorted(
                    _expected_operator_review_record_field_details(
                        section.get("operator_review_record_required_fields")
                    )
                ),
            )
            if not section.get("required_evidence"):
                failures.append(
                    _failure(
                        (
                            f"decision_template_priority_{expected_priority}_"
                            "missing_required_evidence"
                        ),
                        expected_label,
                    )
                )
            if not section.get("required_safety_checks"):
                failures.append(
                    _failure(
                        (
                            f"decision_template_priority_{expected_priority}_"
                            "missing_required_safety_checks"
                        ),
                        expected_label,
                    )
                )
            _check_required_safety_checks(
                failures,
                code_prefix=f"decision_template_priority_{expected_priority}",
                value=section.get("required_safety_checks"),
            )
            _check_bounded_delta_policies(
                failures,
                code_prefix=f"decision_template_priority_{expected_priority}",
                value=section.get("bounded_delta_policies"),
            )
            _check_rollback_expectations(
                failures,
                code_prefix=f"decision_template_priority_{expected_priority}",
                value=section.get("rollback_expectations"),
            )
            for field in REQUIRED_METHOD_CARD_CONTRACT_FIELDS:
                if not section.get(field):
                    failures.append(
                        _failure(
                            (
                                f"decision_template_priority_{expected_priority}_"
                                f"missing_{field}"
                            ),
                            expected_label,
                        )
                    )
            if not section.get("operator_visible_explanation_contract"):
                failures.append(
                    _failure(
                        (
                            f"decision_template_priority_{expected_priority}_"
                            "missing_operator_visible_explanation_contract"
                        ),
                        expected_label,
                    )
                )
            if not section.get("bounded_delta_policies"):
                failures.append(
                    _failure(
                        (
                            f"decision_template_priority_{expected_priority}_"
                            "missing_bounded_delta_policies"
                        ),
                        expected_label,
                    )
                )
            if not section.get("rollback_expectations"):
                failures.append(
                    _failure(
                        (
                            f"decision_template_priority_{expected_priority}_"
                            "missing_rollback_expectations"
                        ),
                        expected_label,
                    )
                )
            if not section.get("future_wing_readback_targets"):
                failures.append(
                    _failure(
                        f"decision_template_priority_{expected_priority}_missing_readback_targets",
                        expected_label,
                    )
                )
            _check_future_wing_readback_targets(
                failures,
                code_prefix=f"decision_template_priority_{expected_priority}",
                value=section.get("future_wing_readback_targets"),
            )
            _check_future_wing_readback_contract_summary(
                failures,
                code_prefix=f"decision_template_priority_{expected_priority}",
                value=section.get("future_wing_readback_contract_summary"),
                allow_duplicate_keys=True,
            )
            _check_future_live_required_proofs(
                failures,
                code_prefix=f"decision_template_priority_{expected_priority}",
                value=section.get("future_live_required_proofs"),
            )
            _check_blocked_live_actions(
                failures,
                code_prefix=f"decision_template_priority_{expected_priority}",
                value=section.get("blocked_live_actions"),
            )
            _check_live_authority_boundary_summary(
                failures,
                code_prefix=f"decision_template_priority_{expected_priority}",
                value=section.get("live_authority_boundary_summary"),
            )
            for summary_field, required_keys in (
                ("why_safe_summary", REQUIRED_OPERATOR_WHY_SAFE_SUMMARY_KEYS),
                (
                    "why_blocked_summary",
                    REQUIRED_OPERATOR_WHY_BLOCKED_SUMMARY_KEYS,
                ),
            ):
                actual_keys = set(_summary_keys(section.get(summary_field)))
                missing_keys = [
                    key for key in required_keys if key not in actual_keys
                ]
                if missing_keys:
                    failures.append(
                        _failure(
                            (
                                f"decision_template_priority_{expected_priority}_"
                                f"missing_{summary_field}_keys"
                            ),
                            repr(missing_keys),
                        )
                    )
            if not section.get("blocked_live_actions"):
                failures.append(
                    _failure(
                        f"decision_template_priority_{expected_priority}_missing_blocked_actions",
                        expected_label,
                    )
                )

    if markdown:
        if not markdown.startswith("# Automixer Correction Method Operator Review Packet\n"):
            failures.append(
                _failure(
                    "markdown_header_missing",
                    "operator-review.md must start with the review packet heading",
                )
            )
        if "## Operator Decision Workflow" not in markdown:
            failures.append(
                _failure(
                    "markdown_missing_operator_decision_workflow",
                    "operator-review.md must expose the operator decision workflow",
                )
            )
        packet_intro_end = markdown.find("\n## Operator Next Step")
        packet_intro_block = markdown[
            : packet_intro_end if packet_intro_end != -1 else 0
        ]
        for field, expected_line in _markdown_authority_lines(packet):
            if expected_line not in packet_intro_block:
                failures.append(
                    _failure(
                        f"markdown_packet_{field}_mismatch",
                        expected_line,
                    )
                )
        operator_workflow_block = _markdown_h2_intro_block(
            markdown,
            "## Operator Decision Workflow",
        )
        expected_operator_workflow_lines = (
            (
                "completion_requires",
                (
                    "- completion_requires: "
                    "`review_packet_json`, `operator_review_markdown`, "
                    "`verification_report_passed`, "
                    "`explicit_offline_decision_per_priority_section`, "
                    "`explicit_offline_decision_per_method`, "
                    "`future_real_wing_readback_before_live_promotion`"
                ),
            ),
            ("workflow_scope", "- workflow_scope: `offline_proposal_only`"),
            (
                "required_priority_operator_decision",
                (
                    "- required_priority_operator_decision: "
                    "`one_offline_decision_per_priority_section`"
                ),
            ),
            (
                "required_operator_decisions",
                "- required_operator_decisions: `one_offline_decision_per_method`",
            ),
            *_markdown_authority_lines(packet),
        )
        for field, expected_line in expected_operator_workflow_lines:
            if expected_line not in operator_workflow_block:
                failures.append(
                    _failure(
                        f"markdown_operator_decision_workflow_{field}_mismatch",
                        expected_line,
                    )
                )
        for required_line in (
            "`explicit_offline_decision_per_priority_section`",
            "`explicit_offline_decision_per_method`",
            "`future_real_wing_readback_before_live_promotion`",
            "- workflow_scope: `offline_proposal_only`",
            (
                "- required_priority_operator_decision: "
                "`one_offline_decision_per_priority_section`"
            ),
            "- required_operator_decisions: `one_offline_decision_per_method`",
            "- can_authorize_live: `false`",
            "- can_apply_live: `false`",
            "- can_import_to_queue: `false`",
            "- can_close_readiness: `false`",
            "- blocks_live_promotion: `true`",
            "- blocks_queue_import: `true`",
            "- full_autonomous_ready: `false`",
            "- queue_action: `not_queueable_until_promoted`",
            "- future_wing_readback_required: `true`",
            "- hardware_proof_required_before_live: `true`",
        ):
            if required_line not in markdown:
                failures.append(
                    _failure(
                        "markdown_missing_operator_decision_workflow_line",
                        required_line,
                    )
                )
        expected_denial_reason_line = (
            f"- denial_reason: {_markdown_code(packet.get('denial_reason'))}"
        )
        if expected_denial_reason_line not in markdown:
            failures.append(
                _failure(
                    "markdown_denial_reason_mismatch",
                    expected_denial_reason_line,
                )
            )
        expected_next_step_block = (
            "## Operator Next Step\n"
            f"{packet.get('operator_next_step')}\n"
        )
        if expected_next_step_block not in markdown:
            failures.append(
                _failure(
                    "markdown_operator_next_step_mismatch",
                    expected_next_step_block.strip(),
                )
            )
        markdown_packet_summary = (
            packet.get("summary") if isinstance(packet.get("summary"), dict) else {}
        )
        for key in MARKDOWN_SUMMARY_MIRROR_KEYS:
            expected_summary_line = (
                f"- {key}: {_markdown_code(markdown_packet_summary.get(key))}"
            )
            if expected_summary_line not in markdown:
                failures.append(
                    _failure(
                        f"markdown_summary_{key}_mismatch",
                        expected_summary_line,
                    )
                )
        for forbidden in FORBIDDEN_MARKDOWN_TRUE_FLAGS:
            if forbidden in markdown:
                failures.append(
                    _failure("markdown_forbidden_true_flag", forbidden)
                )
        for proof in REQUIRED_FUTURE_PROOFS:
            if proof not in markdown:
                failures.append(_failure("markdown_missing_future_proof", proof))
        if "## Required Operator Surfaces" not in markdown:
            failures.append(
                _failure(
                    "markdown_missing_required_operator_surfaces_section",
                    "operator-review.md must expose required operator surfaces",
                )
            )
        for surface in REQUIRED_OPERATOR_SURFACES:
            surface_line = f"- {_markdown_code(surface)}"
            if surface_line not in markdown:
                failures.append(
                    _failure(
                        "markdown_missing_required_operator_surface",
                        surface,
                    )
                )
        for (
            packet_key,
            summary_prefix,
            ordinal_label,
            expected_priority,
            expected_label,
        ) in EXPECTED_PRIORITY_REVIEW_SUMMARIES:
            priority_review_heading = f"## {ordinal_label} Priority Review"
            if priority_review_heading not in markdown:
                failures.append(
                    _failure(
                        f"markdown_missing_{summary_prefix}_review",
                        expected_label,
                    )
                )
            priority_review_summary = packet.get(packet_key)
            if isinstance(priority_review_summary, dict):
                priority_review_block = _markdown_h2_block(
                    markdown,
                    priority_review_heading,
                )
                expected_priority_review_lines = (
                    (
                        "label",
                        f"- label: {_markdown_code(priority_review_summary.get('label'))}",
                    ),
                    (
                        "objective_families",
                        (
                            "- objective_families: "
                            f"{_markdown_items(priority_review_summary.get('objective_families'))}"
                        ),
                    ),
                    (
                        "method_ids",
                        (
                            "- method_ids: "
                            f"{_markdown_items(priority_review_summary.get('method_ids'))}"
                        ),
                    ),
                    (
                        "applicable_safety_checks",
                        (
                            "- applicable_safety_checks: "
                            f"{_markdown_items(priority_review_summary.get('applicable_safety_checks'))}"
                        ),
                    ),
                    (
                        "future_wing_readback_targets",
                        (
                            "- future_wing_readback_targets: "
                            f"{_markdown_items(priority_review_summary.get('future_wing_readback_targets'))}"
                        ),
                    ),
                    (
                        "blocked_live_actions",
                        (
                            "- blocked_live_actions: "
                            f"{_markdown_items(priority_review_summary.get('blocked_live_actions'))}"
                        ),
                    ),
                    *_markdown_authority_lines(priority_review_summary),
                    (
                        "operator_next_step",
                        (
                            "- operator_next_step: "
                            f"{_markdown_code(priority_review_summary.get('operator_next_step'))}"
                        ),
                    ),
                )
                for field, expected_line in expected_priority_review_lines:
                    if expected_line not in priority_review_block:
                        failures.append(
                            _failure(
                                f"markdown_{summary_prefix}_{field}_mismatch",
                                expected_line,
                            )
                        )
            if f"### {expected_priority}. {expected_label}" not in markdown:
                failures.append(
                    _failure(
                        f"markdown_missing_{summary_prefix}_priority_section",
                        expected_label,
                    )
                )
            if (
                f"### Decision Section {expected_priority}: {expected_label}"
                not in markdown
            ):
                failures.append(
                    _failure(
                        f"markdown_missing_{summary_prefix}_decision_section",
                        expected_label,
                    )
                )
            decision_section = section_by_priority.get(expected_priority)
            decision_section_heading = (
                f"### Decision Section {expected_priority}: {expected_label}"
            )
            if isinstance(decision_section, Mapping):
                decision_section_block = _markdown_h3_block(
                    markdown,
                    decision_section_heading,
                )
                expected_decision_section_lines = (
                    ("review_scope", "- review_scope: `offline_proposal_only`"),
                    ("review_status", "- review_status: `operator_review_pending`"),
                    (
                        "required_priority_operator_decision",
                        (
                            "- required_priority_operator_decision: "
                            "`one_offline_decision_per_priority_section`"
                        ),
                    ),
                    (
                        "required_operator_decisions",
                        "- required_operator_decisions: `one_offline_decision_per_method`",
                    ),
                    (
                        "method_ids",
                        (
                            "- method_ids: "
                            f"{_markdown_items(_priority_section_method_ids(decision_section))}"
                        ),
                    ),
                    (
                        "allowed_decisions",
                        (
                            "- allowed_decisions: "
                            f"{_markdown_items(_priority_section_method_items(decision_section, 'operator_review_decisions'))}"
                        ),
                    ),
                    (
                        "required_evidence",
                        (
                            "- required_evidence: "
                            f"{_markdown_items(decision_section.get('required_evidence'))}"
                        ),
                    ),
                    (
                        "required_safety_checks",
                        (
                            "- required_safety_checks: "
                            f"{_markdown_items(decision_section.get('required_safety_checks'))}"
                        ),
                    ),
                    (
                        "source_evidence",
                        (
                            "- source_evidence: "
                            f"{_markdown_items(decision_section.get('source_evidence'))}"
                        ),
                    ),
                    (
                        "source_evidence_details",
                        (
                            "- source_evidence_details: "
                            f"{_markdown_items(decision_section.get('source_evidence_details'))}"
                        ),
                    ),
                    (
                        "measured_or_audible_problem",
                        (
                            "- measured_or_audible_problem: "
                            f"{_markdown_items(decision_section.get('measured_or_audible_problem'))}"
                        ),
                    ),
                    (
                        "proposed_correction",
                        (
                            "- proposed_correction: "
                            f"{_markdown_items(decision_section.get('proposed_correction'))}"
                        ),
                    ),
                    (
                        "goal_requirement_details",
                        (
                            "- goal_requirement_details: "
                            f"{_markdown_items(decision_section.get('goal_requirement_details'))}"
                        ),
                    ),
                    (
                        "method_specific_review_details",
                        (
                            "- method_specific_review_details: "
                            f"{_markdown_items(decision_section.get('method_specific_review_details'))}"
                        ),
                    ),
                    (
                        "operator_actionable_explanation_details",
                        (
                            "- operator_actionable_explanation_details: "
                            f"{_markdown_items(decision_section.get('operator_actionable_explanation_details'))}"
                        ),
                    ),
                    (
                        "evidence_problem_correction_chain_details",
                        (
                            "- evidence_problem_correction_chain_details: "
                            f"{_markdown_items(decision_section.get('evidence_problem_correction_chain_details'))}"
                        ),
                    ),
                    (
                        "confidence",
                        (
                            "- confidence: "
                            f"{_markdown_items(decision_section.get('confidence'))}"
                        ),
                    ),
                    (
                        "bounded_delta",
                        (
                            "- bounded_delta_contract: "
                            f"{_markdown_items(decision_section.get('bounded_delta'))}"
                        ),
                    ),
                    (
                        "safety_blockers",
                        (
                            "- safety_blockers: "
                            f"{_markdown_items(decision_section.get('safety_blockers'))}"
                        ),
                    ),
                    (
                        "safety_blocker_details",
                        (
                            "- safety_blocker_details: "
                            f"{_markdown_items(decision_section.get('safety_blocker_details'))}"
                        ),
                    ),
                    (
                        "headroom_feedback_true_peak_checks",
                        (
                            "- headroom_feedback_true_peak_checks: "
                            f"{_markdown_items(decision_section.get('headroom_feedback_true_peak_checks'))}"
                        ),
                    ),
                    (
                        "headroom_feedback_true_peak_check_details",
                        (
                            "- headroom_feedback_true_peak_check_details: "
                            f"{_markdown_items(decision_section.get('headroom_feedback_true_peak_check_details'))}"
                        ),
                    ),
                    (
                        "operator_readable_explanation",
                        (
                            "- operator_readable_explanation: "
                            f"{_markdown_items(decision_section.get('operator_readable_explanation'))}"
                        ),
                    ),
                    (
                        "operator_visible_explanation_contract",
                        (
                            "- operator_visible_explanation_contract: "
                            f"{_markdown_items(decision_section.get('operator_visible_explanation_contract'))}"
                        ),
                    ),
                    (
                        "operator_payload_required_fields",
                        (
                            "- operator_payload_required_fields: "
                            f"{_markdown_items(decision_section.get('operator_payload_required_fields'))}"
                        ),
                    ),
                    (
                        "operator_evidence_required_fields",
                        (
                            "- operator_evidence_required_fields: "
                            f"{_markdown_items(decision_section.get('operator_evidence_required_fields'))}"
                        ),
                    ),
                    (
                        "operator_problem_review_summary",
                        (
                            "- operator_problem_review_summary: "
                            f"{_markdown_items(decision_section.get('operator_problem_review_summary'))}"
                        ),
                    ),
                    (
                        "operator_method_review_focus",
                        (
                            "- operator_method_review_focus: "
                            f"{_markdown_items(decision_section.get('operator_method_review_focus'))}"
                        ),
                    ),
                    (
                        "why_safe_summary",
                        (
                            "- why_safe_summary: "
                            f"{_markdown_items(decision_section.get('why_safe_summary'))}"
                        ),
                    ),
                    (
                        "why_blocked_summary",
                        (
                            "- why_blocked_summary: "
                            f"{_markdown_items(decision_section.get('why_blocked_summary'))}"
                        ),
                    ),
                    (
                        "bounded_delta_policies",
                        (
                            "- bounded_delta_policies: "
                            f"{_markdown_items(decision_section.get('bounded_delta_policies'))}"
                        ),
                    ),
                    (
                        "rollback_expectations",
                        (
                            "- rollback_expectations: "
                            f"{_markdown_items(decision_section.get('rollback_expectations'))}"
                        ),
                    ),
                    (
                        "rollback_readback_contract",
                        (
                            "- rollback_readback_contract: "
                            f"{_markdown_items(decision_section.get('rollback_readback_contract'))}"
                        ),
                    ),
                    (
                        "future_wing_readback_targets",
                        (
                            "- future_wing_readback_targets: "
                            f"{_markdown_items(decision_section.get('future_wing_readback_targets'))}"
                        ),
                    ),
                    (
                        "future_wing_readback_contract",
                        (
                            "- future_wing_readback_contract: "
                            f"{_markdown_items(decision_section.get('future_wing_readback_contract'))}"
                        ),
                    ),
                    (
                        "test_evidence",
                        (
                            "- test_evidence: "
                            f"{_markdown_items(decision_section.get('test_evidence'))}"
                        ),
                    ),
                    (
                        "test_evidence_references",
                        (
                            "- test_evidence_references: "
                            f"{_markdown_items(decision_section.get('test_evidence_references'))}"
                        ),
                    ),
                    (
                        "expected_evidence_packet_reference",
                        (
                            "- expected_evidence_packet_reference: "
                            f"{_markdown_code(f'packet.json#priority_sections/{expected_priority}')}"
                        ),
                    ),
                    (
                        "expected_source_material_reference",
                        (
                            "- expected_source_material_reference: "
                            f"{_markdown_code(f'packet.json#priority_sections/{expected_priority}/source_evidence')}"
                        ),
                    ),
                    (
                        "expected_test_evidence_references",
                        (
                            "- expected_test_evidence_references: "
                            f"{_markdown_items(decision_section.get('test_evidence_references'))}"
                        ),
                    ),
                    (
                        "expected_verification_report_reference",
                        (
                            "- expected_verification_report_reference: "
                            f"{_markdown_code(EXPECTED_VERIFICATION_REPORT_REFERENCE)}"
                        ),
                    ),
                    (
                        "future_wing_readback_contract_summary",
                        (
                            "- future_wing_readback_contract_summary: "
                            f"{_markdown_items(decision_section.get('future_wing_readback_contract_summary'))}"
                        ),
                    ),
                    (
                        "future_live_required_proofs",
                        (
                            "- future_live_required_proofs: "
                            f"{_markdown_items(decision_section.get('future_live_required_proofs'))}"
                        ),
                    ),
                    (
                        "future_live_requirement_details",
                        (
                            "- future_live_requirement_details: "
                            f"{_markdown_items(decision_section.get('future_live_requirement_details'))}"
                        ),
                    ),
                    (
                        "blocked_live_actions",
                        (
                            "- blocked_live_actions: "
                            f"{_markdown_items(decision_section.get('blocked_live_actions'))}"
                        ),
                    ),
                    (
                        "live_authority_boundary_summary",
                        (
                            "- live_authority_boundary_summary: "
                            f"{_markdown_items(decision_section.get('live_authority_boundary_summary'))}"
                        ),
                    ),
                    (
                        "operator_review_record_required_fields",
                        (
                            "- operator_review_record_required_fields: "
                            f"{_markdown_items(decision_section.get('operator_review_record_required_fields'))}"
                        ),
                    ),
                    (
                        "required_confirmation_details",
                        (
                            "- required_confirmation_details: "
                            f"{_markdown_items(decision_section.get('required_confirmation_details'))}"
                        ),
                    ),
                    (
                        "operator_acceptance_gate_details",
                        (
                            "- operator_acceptance_gate_details: "
                            f"{_markdown_items(decision_section.get('operator_acceptance_gate_details'))}"
                        ),
                    ),
                    (
                        "operator_review_decision_option_details",
                        (
                            "- operator_review_decision_option_details: "
                            f"{_markdown_items(decision_section.get('operator_review_decision_option_details'))}"
                        ),
                    ),
                    (
                        "operator_review_checklist_details",
                        (
                            "- operator_review_checklist_details: "
                            f"{_markdown_items(decision_section.get('operator_review_checklist_details'))}"
                        ),
                    ),
                    (
                        "operator_review_record_field_details",
                        (
                            "- operator_review_record_field_details: "
                            f"{_markdown_items(decision_section.get('operator_review_record_field_details'))}"
                        ),
                    ),
                    (
                        "operator_next_action_details",
                        (
                            "- operator_next_action_details: "
                            f"{_markdown_items(decision_section.get('operator_next_action_details'))}"
                        ),
                    ),
                    (
                        "missing_required_operator_surface_method_ids",
                        (
                            "- missing_required_operator_surface_method_ids: "
                            f"{_markdown_items(decision_section.get('missing_required_operator_surface_method_ids'))}"
                        ),
                    ),
                    (
                        "live_authority_method_ids",
                        (
                            "- live_authority_method_ids: "
                            f"{_markdown_items(decision_section.get('live_authority_method_ids'))}"
                        ),
                    ),
                    (
                        "live_apply_method_ids",
                        (
                            "- live_apply_method_ids: "
                            f"{_markdown_items(decision_section.get('live_apply_method_ids'))}"
                        ),
                    ),
                    (
                        "queueable_method_ids",
                        (
                            "- queueable_method_ids: "
                            f"{_markdown_items(decision_section.get('queueable_method_ids'))}"
                        ),
                    ),
                    (
                        "operator_next_step",
                        (
                            "- operator_next_step: "
                            f"{_markdown_code(decision_section.get('operator_next_step'))}"
                        ),
                    ),
                    *_markdown_authority_lines(decision_section),
                )
                for field, expected_line in expected_decision_section_lines:
                    if expected_line not in decision_section_block:
                        failures.append(
                            _failure(
                                (
                                    f"markdown_decision_section_{expected_priority}_"
                                    f"{field}_mismatch"
                                ),
                                expected_line,
                            )
                        )
        for method_id in method_ids:
            card = cards_by_method_id.get(method_id)
            if not card:
                continue
            heading = (
                f"#### {card.get('method_label')} "
                f"({_markdown_code(method_id)})"
            )
            if heading not in markdown:
                failures.append(
                    _failure("markdown_missing_method_card", method_id)
                )
                continue
            card_block = _markdown_card_block(markdown, heading)
            expected_lines = (
                ("review_scope", f"- review_scope: {_markdown_code(card.get('review_scope'))}"),
                ("status", f"- status: {_markdown_code(card.get('status'))}"),
                ("review_status", f"- review_status: {_markdown_code(card.get('review_status'))}"),
                ("domain", f"- domain: {_markdown_code(card.get('domain'))}"),
                (
                    "objective_family",
                    f"- objective_family: {_markdown_code(card.get('objective_family'))}",
                ),
                (
                    "priority",
                    f"- priority: {_markdown_code(card.get('priority'))}",
                ),
                (
                    "priority_label",
                    f"- priority_label: {_markdown_code(card.get('priority_label'))}",
                ),
                (
                    "priority_review_scope",
                    (
                        "- priority_review_scope: "
                        f"{_markdown_code(card.get('priority_review_scope'))}"
                    ),
                ),
                (
                    "operator_question",
                    f"- operator_question: {_markdown_code(card.get('operator_question'))}",
                ),
                (
                    "denial_reason",
                    f"- denial_reason: {_markdown_code(card.get('denial_reason'))}",
                ),
                (
                    "source_evidence",
                    (
                        "- source_evidence: "
                        f"{_markdown_items(_operator_review_contract_items(card, 'source_evidence'))}"
                    ),
                ),
                (
                    "source_evidence_details",
                    (
                        "- source_evidence_details: "
                        f"{_markdown_items(_operator_review_contract_items(card, 'source_evidence_details'))}"
                    ),
                ),
                (
                    "measured_or_audible_problem",
                    (
                        "- measured_or_audible_problem: "
                        f"{_markdown_items(_operator_review_contract_items(card, 'measured_or_audible_problem'))}"
                    ),
                ),
                (
                    "proposed_correction",
                    (
                        "- proposed_correction: "
                        f"{_markdown_items(_operator_review_contract_items(card, 'proposed_correction'))}"
                    ),
                ),
                (
                    "goal_requirement_details",
                    (
                        "- goal_requirement_details: "
                        f"{_markdown_items(card.get('goal_requirement_details'))}"
                    ),
                ),
                (
                    "method_specific_review_details",
                    (
                        "- method_specific_review_details: "
                        f"{_markdown_items(card.get('method_specific_review_details'))}"
                    ),
                ),
                (
                    "operator_actionable_explanation_details",
                    (
                        "- operator_actionable_explanation_details: "
                        f"{_markdown_items(card.get('operator_actionable_explanation_details'))}"
                    ),
                ),
                (
                    "evidence_problem_correction_chain_details",
                    (
                        "- evidence_problem_correction_chain_details: "
                        f"{_markdown_items(card.get('evidence_problem_correction_chain_details'))}"
                    ),
                ),
                (
                    "operator_problem_review_summary",
                    (
                        "- operator_problem_review_summary: "
                        f"{_markdown_items(card.get('operator_problem_review_summary'))}"
                    ),
                ),
                (
                    "operator_method_review_focus",
                    (
                        "- operator_method_review_focus: "
                        f"{_markdown_items(card.get('operator_method_review_focus'))}"
                    ),
                ),
                (
                    "why_safe_summary",
                    (
                        "- why_safe_summary: "
                        f"{_markdown_items(card.get('why_safe_summary'))}"
                    ),
                ),
                (
                    "why_blocked_summary",
                    (
                        "- why_blocked_summary: "
                        f"{_markdown_items(card.get('why_blocked_summary'))}"
                    ),
                ),
                (
                    "confidence",
                    (
                        "- confidence: "
                        f"{_markdown_items(_operator_review_contract_items(card, 'confidence'))}"
                    ),
                ),
                (
                    "bounded_delta",
                    (
                        "- bounded_delta_contract: "
                        f"{_markdown_items(_operator_review_contract_items(card, 'bounded_delta'))}"
                    ),
                ),
                (
                    "safety_blockers",
                    (
                        "- safety_blockers: "
                        f"{_markdown_items(_operator_review_contract_items(card, 'safety_blockers'))}"
                    ),
                ),
                (
                    "safety_blocker_details",
                    (
                        "- safety_blocker_details: "
                        f"{_markdown_items(card.get('safety_blocker_details'))}"
                    ),
                ),
                (
                    "headroom_feedback_true_peak_checks",
                    (
                        "- headroom_feedback_true_peak_checks: "
                        f"{_markdown_items(_operator_review_contract_items(card, 'headroom_feedback_true_peak_checks'))}"
                    ),
                ),
                (
                    "headroom_feedback_true_peak_check_details",
                    (
                        "- headroom_feedback_true_peak_check_details: "
                        f"{_markdown_items(card.get('headroom_feedback_true_peak_check_details'))}"
                    ),
                ),
                (
                    "operator_readable_explanation",
                    (
                        "- operator_readable_explanation: "
                        f"{_markdown_items(_operator_review_contract_items(card, 'operator_readable_explanation'))}"
                    ),
                ),
                (
                    "required_evidence",
                    f"- required_evidence: {_markdown_items(card.get('required_evidence'))}",
                ),
                (
                    "required_safety_checks",
                    (
                        "- required_safety_checks: "
                        f"{_markdown_items(card.get('required_safety_checks'))}"
                    ),
                ),
                (
                    "required_operator_surfaces",
                    (
                        "- required_operator_surfaces: "
                        f"{_markdown_items(card.get('required_operator_surfaces'))}"
                    ),
                ),
                (
                    "operator_payload_required_fields",
                    (
                        "- operator_payload_required_fields: "
                        f"{_markdown_items(card.get('operator_payload_required_fields'))}"
                    ),
                ),
                (
                    "operator_evidence_required_fields",
                    (
                        "- operator_evidence_required_fields: "
                        f"{_markdown_items(card.get('operator_evidence_required_fields'))}"
                    ),
                ),
                (
                    "must_confirm",
                    (
                        "- must_confirm: "
                        f"{_markdown_items(card.get('what_operator_must_confirm'))}"
                    ),
                ),
                (
                    "required_confirmation_details",
                    (
                        "- required_confirmation_details: "
                        f"{_markdown_items(card.get('required_confirmation_details'))}"
                    ),
                ),
                (
                    "operator_acceptance_gate_details",
                    (
                        "- operator_acceptance_gate_details: "
                        f"{_markdown_items(card.get('operator_acceptance_gate_details'))}"
                    ),
                ),
                (
                    "operator_review_decision_option_details",
                    (
                        "- operator_review_decision_option_details: "
                        f"{_markdown_items(card.get('operator_review_decision_option_details'))}"
                    ),
                ),
                (
                    "operator_review_checklist",
                    (
                        "- operator_review_checklist: "
                        f"{_markdown_items(card.get('operator_review_checklist'))}"
                    ),
                ),
                (
                    "operator_review_checklist_details",
                    (
                        "- operator_review_checklist_details: "
                        f"{_markdown_items(card.get('operator_review_checklist_details'))}"
                    ),
                ),
                (
                    "operator_review_record_required_fields",
                    (
                        "- operator_review_record_required_fields: "
                        f"{_markdown_items(card.get('operator_review_record_required_fields'))}"
                    ),
                ),
                (
                    "operator_review_record_field_details",
                    (
                        "- operator_review_record_field_details: "
                        f"{_markdown_items(card.get('operator_review_record_field_details'))}"
                    ),
                ),
                (
                    "operator_next_action_details",
                    (
                        "- operator_next_action_details: "
                        f"{_markdown_items(card.get('operator_next_action_details'))}"
                    ),
                ),
                (
                    "bounded_delta",
                    (
                        f"- bounded_delta: {_markdown_code(card.get('bounded_delta_unit'))}; "
                        f"{_markdown_code(card.get('bounded_delta_policy'))}"
                    ),
                ),
                (
                    "operator_visible_explanation_contract",
                    (
                        "- operator_visible_explanation_contract: "
                        f"{_markdown_items(_operator_visible_explanation_contract_items(card))}"
                    ),
                ),
                (
                    "rollback_expectation",
                    (
                        "- rollback_expectation: "
                        f"{_markdown_code(card.get('rollback_expectation'))}"
                    ),
                ),
                (
                    "rollback_readback_contract",
                    (
                        "- rollback_readback_contract: "
                        f"{_markdown_items(_operator_review_contract_items(card, 'rollback_readback_contract'))}"
                    ),
                ),
                (
                    "future_wing_readback_targets",
                    (
                        "- future_wing_readback_targets: "
                        f"{_markdown_items(card.get('future_wing_readback_targets'))}"
                    ),
                ),
                    (
                        "future_wing_readback_contract",
                        (
                            "- future_wing_readback_contract: "
                            f"{_markdown_items(_operator_review_contract_items(card, 'future_wing_readback_contract'))}"
                        ),
                    ),
                    (
                        "test_evidence",
                        (
                            "- test_evidence: "
                            f"{_markdown_items(_operator_review_contract_items(card, 'test_evidence'))}"
                        ),
                    ),
                    (
                        "test_evidence_references",
                        (
                            "- test_evidence_references: "
                            f"{_markdown_items(card.get('test_evidence_references'))}"
                        ),
                    ),
                    (
                        "expected_evidence_packet_reference",
                        (
                            "- expected_evidence_packet_reference: "
                            f"{_markdown_code(f'packet.json#method_review_cards/{method_id}')}"
                        ),
                    ),
                    (
                        "expected_source_material_reference",
                        (
                            "- expected_source_material_reference: "
                            f"{_markdown_code(f'packet.json#method_review_cards/{method_id}/source_evidence')}"
                        ),
                    ),
                    (
                        "expected_test_evidence_references",
                        (
                            "- expected_test_evidence_references: "
                            f"{_markdown_items(card.get('test_evidence_references'))}"
                        ),
                    ),
                    (
                        "expected_verification_report_reference",
                        (
                            "- expected_verification_report_reference: "
                            f"{_markdown_code(EXPECTED_VERIFICATION_REPORT_REFERENCE)}"
                        ),
                    ),
                    (
                        "future_wing_readback_contract_summary",
                        (
                        "- future_wing_readback_contract_summary: "
                        f"{_markdown_items(card.get('future_wing_readback_contract_summary'))}"
                    ),
                ),
                (
                    "future_live_required_proofs",
                    (
                        "- future_live_required_proofs: "
                        f"{_markdown_items(card.get('future_live_required_proofs'))}"
                    ),
                ),
                (
                    "future_live_requirement_details",
                    (
                        "- future_live_requirement_details: "
                        f"{_markdown_items(card.get('future_live_requirement_details'))}"
                    ),
                ),
                (
                    "blocked_live_actions",
                    (
                        "- blocked_live_actions: "
                        f"{_markdown_items(card.get('blocked_live_actions'))}"
                    ),
                ),
                (
                    "live_authority_boundary_summary",
                    (
                        "- live_authority_boundary_summary: "
                        f"{_markdown_items(card.get('live_authority_boundary_summary'))}"
                    ),
                ),
                (
                    "operator_review_decisions",
                    (
                        "- operator_review_decisions: "
                        f"{_markdown_items(card.get('operator_review_decisions'))}"
                    ),
                ),
                (
                    "operator_next_step",
                    f"- operator_next_step: {_markdown_code(card.get('operator_next_step'))}",
                ),
                *_markdown_authority_lines(card),
                (
                    "authority",
                    (
                        "- authority: "
                        "live_authority_granted="
                        f"{_markdown_bool(card.get('live_authority_granted'))}; "
                        "live_apply_allowed="
                        f"{_markdown_bool(card.get('live_apply_allowed'))}; "
                        "queue_import_allowed="
                        f"{_markdown_bool(card.get('queue_import_allowed'))}; "
                        "contacts_wing="
                        f"{_markdown_bool(card.get('contacts_wing'))}; "
                        "contacts_backend="
                        f"{_markdown_bool(card.get('contacts_backend'))}; "
                        "mutates_mixer="
                        f"{_markdown_bool(card.get('mutates_mixer'))}"
                    ),
                ),
            )
            for field, expected_line in expected_lines:
                if expected_line not in card_block:
                    failures.append(
                        _failure(
                            f"markdown_method_{method_id}_{field}_missing",
                            expected_line,
                        )
                    )

    if goal_audit_markdown:
        if not goal_audit_markdown.startswith(
            "# Automixer Correction Method Goal Audit\n"
        ):
            failures.append(
                _failure(
                    "goal_audit_markdown_header_missing",
                    "goal-audit.md must start with the goal audit heading",
                )
            )
        for expected_line in (
            "- status: `passed`",
            "- read_only: `true`",
            "- contacts_wing: `false`",
            "- live_authority_granted: `false`",
            "- readiness_closure_allowed: `false`",
            "- overall_offline_goal_percent: `100.0`",
            "- `none`",
        ):
            if expected_line not in goal_audit_markdown:
                failures.append(
                    _failure(
                        "goal_audit_markdown_required_line_missing",
                        expected_line,
                    )
                )
        for family in (
            goal_audit.get("required_objective_families")
            if isinstance(goal_audit.get("required_objective_families"), list)
            else []
        ):
            expected_line = f"- `{family}`: `covered`"
            if expected_line not in goal_audit_markdown:
                failures.append(
                    _failure(
                        "goal_audit_markdown_required_family_missing",
                        expected_line,
                    )
                )
        for forbidden in FORBIDDEN_MARKDOWN_TRUE_FLAGS:
            if forbidden in goal_audit_markdown:
                failures.append(
                    _failure("goal_audit_markdown_forbidden_true_flag", forbidden)
                )

    return {
        "schema": "automixer.correction_method_operator_review_export_verification.v1",
        "run_dir": str(run_dir),
        "status": "passed" if not failures else "failed",
        "failed_checks": failures,
        "read_only": True,
        "contacts_wing": False,
        "contacts_backend": False,
        "queries_console": False,
        "mutates_console_state": False,
        "mutates_mixer": False,
        "live_write": False,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "live_apply_allowed": False,
        "queue_import_allowed": False,
        "method_review_card_count": len(method_ids),
        "priority_review_summary_count": len(verified_priority_labels),
        "priority_review_labels": verified_priority_labels,
        "priority_decision_section_count": len(verified_priority_decision_labels),
        "priority_decision_labels": verified_priority_decision_labels,
        "verified_artifacts": sorted(EXPECTED_ARTIFACTS.values()),
    }


def _decision_rows_by_method_id(
    rows: Any,
    *,
    label: str,
    failures: List[Dict[str, str]],
) -> Dict[str, Mapping[str, Any]]:
    if not isinstance(rows, list):
        failures.append(_failure(f"{label}_method_decisions_not_list", repr(rows)))
        return {}

    by_id: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            failures.append(_failure(f"{label}_method_decision_not_object", repr(row)))
            continue
        method_id = str(row.get("method_id") or "")
        if not method_id:
            failures.append(_failure(f"{label}_method_decision_missing_id", repr(row)))
            continue
        if method_id in by_id:
            failures.append(_failure(f"{label}_method_decision_duplicate_id", method_id))
            continue
        by_id[method_id] = row
    return by_id


def validate_correction_method_operator_decision_record(
    run_dir: Path,
    decision_record_path: Path,
) -> Dict[str, Any]:
    """Validate a filled offline operator decision record.

    This is intentionally a local artifact validator. It treats every operator
    decision, including acceptance, as proposal-only and incapable of granting
    live mixer authority or readiness closure.
    """

    run_dir = Path(run_dir)
    decision_record_path = Path(decision_record_path)
    bundle_report = verify_correction_method_operator_review_export(run_dir)
    failures: List[Dict[str, str]] = [
        _failure(f"bundle_{item['code']}", item["detail"])
        for item in bundle_report["failed_checks"]
    ]

    template = _load_json_object(
        run_dir / "operator-decision-template.json",
        label="operator_decision_template",
        failures=failures,
    )
    record = _load_json_object(
        decision_record_path,
        label="operator_decision_record",
        failures=failures,
    )

    accepted_offline_proposal_count = 0
    blocked_live_actions: set[str] = set()
    decision_counts: Dict[str, int] = {"missing_operator_decision": 0}
    future_live_required_proofs: set[str] = set()
    future_wing_readback_targets: set[str] = set()
    method_ids_by_decision: Dict[str, List[str]] = {
        "missing_operator_decision": []
    }
    priority_decision_counts: Dict[str, int] = {"missing_operator_decision": 0}
    priority_ids_by_decision: Dict[str, List[int]] = {
        "missing_operator_decision": []
    }
    record_rows_by_id: Dict[str, Mapping[str, Any]] = {}
    validated_method_decisions: List[Dict[str, Any]] = []
    validated_priority_decision_sections: List[Dict[str, Any]] = []

    if template and record:
        template_allowed_report_decisions = _check_offline_review_decisions(
            failures,
            code_prefix="decision_template",
            field_name="allowed_decisions",
            value=template.get("allowed_decisions"),
        )
        record_allowed_report_decisions = _check_offline_review_decisions(
            failures,
            code_prefix="decision_record",
            field_name="allowed_decisions",
            value=record.get("allowed_decisions"),
        )
        _check_equal(
            failures,
            code="decision_record_allowed_decisions_mismatch",
            actual=_sorted_str_items(record.get("allowed_decisions")),
            expected=_sorted_str_items(template.get("allowed_decisions")),
        )
        allowed_report_decisions = (
            record_allowed_report_decisions or template_allowed_report_decisions
        )
        for decision in allowed_report_decisions:
            decision_key = str(decision)
            decision_counts.setdefault(decision_key, 0)
            method_ids_by_decision.setdefault(decision_key, [])
            priority_decision_counts.setdefault(decision_key, 0)
            priority_ids_by_decision.setdefault(decision_key, [])

        schema = record.get("schema")
        if schema not in (
            EXPECTED_DECISION_TEMPLATE_SCHEMA,
            EXPECTED_DECISION_RECORD_SCHEMA,
        ):
            failures.append(
                _failure(
                    "decision_record_schema_mismatch",
                    (
                        "expected "
                        f"{EXPECTED_DECISION_TEMPLATE_SCHEMA!r} or "
                        f"{EXPECTED_DECISION_RECORD_SCHEMA!r}, got {schema!r}"
                    ),
                )
            )
        if record.get("status") not in ALLOWED_DECISION_RECORD_STATUSES:
            failures.append(
                _failure(
                    "decision_record_status_mismatch",
                    (
                        "expected one of "
                        f"{list(ALLOWED_DECISION_RECORD_STATUSES)!r}, "
                        f"got {record.get('status')!r}"
                    ),
                )
            )

        _check_false_flags(
            failures,
            payload=record,
            label="decision_record",
            flags=FALSE_AUTHORITY_FLAGS + ("can_authorize_live",),
        )
        _check_true_flags(
            failures,
            payload=record,
            label="decision_record",
            flags=TRUE_REVIEW_FLAGS + ("read_only",),
        )

        template_rows_by_id = _decision_rows_by_method_id(
            template.get("method_decisions"),
            label="decision_template",
            failures=failures,
        )
        record_rows_by_id = _decision_rows_by_method_id(
            record.get("method_decisions"),
            label="decision_record",
            failures=failures,
        )
        _check_equal(
            failures,
            code="decision_record_method_ids_mismatch",
            actual=sorted(record_rows_by_id),
            expected=sorted(template_rows_by_id),
        )
        _check_equal(
            failures,
            code="decision_record_method_count_mismatch",
            actual=record.get("method_decision_count"),
            expected=len(template_rows_by_id),
        )

        for method_id, template_row in template_rows_by_id.items():
            record_row = record_rows_by_id.get(method_id)
            if record_row is None:
                continue
            allowed_decisions = _check_offline_review_decisions(
                failures,
                code_prefix=f"decision_template_{method_id}",
                field_name="allowed_decisions",
                value=template_row.get("allowed_decisions"),
            )
            _check_offline_review_decisions(
                failures,
                code_prefix=f"decision_record_{method_id}",
                field_name="allowed_decisions",
                value=record_row.get("allowed_decisions"),
            )
            _check_equal(
                failures,
                code=f"decision_record_{method_id}_allowed_decisions_mismatch",
                actual=_sorted_str_items(record_row.get("allowed_decisions")),
                expected=_sorted_str_items(allowed_decisions),
            )
            for row_field in (
                "method_label",
                "objective_family",
                "priority",
                "priority_label",
                "priority_review_scope",
                "review_scope",
                "review_status",
                "measured_or_audible_problem",
                "proposed_correction",
                "confidence",
                "safety_blockers",
                "headroom_feedback_true_peak_checks",
                "operator_readable_explanation",
                "operator_visible_explanation_contract",
                "bounded_delta_unit",
                "bounded_delta_policy",
                "rollback_expectation",
                "expected_evidence_packet_reference",
                "expected_source_material_reference",
                "expected_verification_report_reference",
            ):
                _check_equal(
                    failures,
                    code=f"decision_record_{method_id}_{row_field}_mismatch",
                    actual=record_row.get(row_field),
                    expected=template_row.get(row_field),
                )
            for row_field in (
                "required_confirmations",
                "required_confirmation_details",
                "operator_acceptance_gate_details",
                "operator_review_decision_option_details",
                "required_evidence",
                "required_safety_checks",
                "operator_payload_required_fields",
                "operator_evidence_required_fields",
                "operator_problem_review_summary",
                "operator_method_review_focus",
                "why_safe_summary",
                "why_blocked_summary",
                "goal_requirement_details",
                "method_specific_review_details",
                "operator_actionable_explanation_details",
                "evidence_problem_correction_chain_details",
                "safety_blocker_details",
                "headroom_feedback_true_peak_check_details",
                "operator_visible_explanation_contract",
                "future_wing_readback_targets",
                "future_wing_readback_contract_summary",
                "future_live_required_proofs",
                "future_live_requirement_details",
                "blocked_live_actions",
                "live_authority_boundary_summary",
                "operator_review_checklist_details",
                "operator_review_record_required_fields",
                "operator_review_record_field_details",
                "operator_next_action_details",
                "expected_test_evidence_references",
            ):
                _check_equal(
                    failures,
                    code=f"decision_record_{method_id}_{row_field}_mismatch",
                    actual=_sorted_str_items(record_row.get(row_field)),
                    expected=_sorted_str_items(template_row.get(row_field)),
                )

            operator_decision = record_row.get("operator_decision")
            if operator_decision is None:
                failures.append(
                    _failure(
                        f"decision_record_{method_id}_operator_decision_missing",
                        "operator_decision must be filled",
                    )
                )
            elif operator_decision not in allowed_decisions:
                failures.append(
                    _failure(
                        f"decision_record_{method_id}_operator_decision_invalid",
                        (
                            f"expected one of {allowed_decisions!r}, "
                            f"got {operator_decision!r}"
                        ),
                    )
                )
            elif operator_decision == "accept_as_offline_proposal":
                accepted_offline_proposal_count += 1
            _check_completed_operator_review_record_fields(
                failures,
                code_prefix=f"decision_record_{method_id}",
                payload=record_row,
            )
            _check_operator_review_reference_bindings(
                failures,
                code_prefix=f"decision_record_{method_id}",
                payload=record_row,
                expected_evidence_packet_reference=template_row.get(
                    "expected_evidence_packet_reference"
                ),
                expected_source_material_reference=template_row.get(
                    "expected_source_material_reference"
                ),
                expected_test_evidence_references=template_row.get(
                    "expected_test_evidence_references"
                ),
                expected_verification_report_reference=template_row.get(
                    "expected_verification_report_reference"
                ),
            )

            decision_key = (
                str(operator_decision)
                if operator_decision is not None
                else "missing_operator_decision"
            )
            decision_counts[decision_key] = decision_counts.get(decision_key, 0) + 1
            method_ids_by_decision.setdefault(decision_key, []).append(method_id)

            readback_targets = record_row.get("future_wing_readback_targets")
            if isinstance(readback_targets, list):
                future_wing_readback_targets.update(str(item) for item in readback_targets)
            payload_required_fields = record_row.get(
                "operator_payload_required_fields"
            )
            evidence_required_fields = record_row.get(
                "operator_evidence_required_fields"
            )
            readback_contract_summary = record_row.get(
                "future_wing_readback_contract_summary"
            )
            required_proofs = record_row.get("future_live_required_proofs")
            if isinstance(required_proofs, list):
                future_live_required_proofs.update(str(item) for item in required_proofs)
            future_live_requirement_details = record_row.get(
                "future_live_requirement_details"
            )
            goal_requirement_details = record_row.get("goal_requirement_details")
            method_specific_review_details = record_row.get(
                "method_specific_review_details"
            )
            operator_actionable_explanation_details = record_row.get(
                "operator_actionable_explanation_details"
            )
            safety_blocker_details = record_row.get("safety_blocker_details")
            evidence_problem_correction_chain_details = record_row.get(
                "evidence_problem_correction_chain_details"
            )
            headroom_feedback_true_peak_check_details = record_row.get(
                "headroom_feedback_true_peak_check_details"
            )
            live_actions = record_row.get("blocked_live_actions")
            if isinstance(live_actions, list):
                blocked_live_actions.update(str(item) for item in live_actions)
            authority_boundary_summary = record_row.get(
                "live_authority_boundary_summary"
            )
            why_safe_summary = record_row.get("why_safe_summary")
            why_blocked_summary = record_row.get("why_blocked_summary")
            review_record_required_fields = record_row.get(
                "operator_review_record_required_fields"
            )
            review_checklist_details = record_row.get(
                "operator_review_checklist_details"
            )
            required_confirmation_details = record_row.get(
                "required_confirmation_details"
            )
            operator_acceptance_gate_details = record_row.get(
                "operator_acceptance_gate_details"
            )
            review_decision_option_details = record_row.get(
                "operator_review_decision_option_details"
            )
            review_record_field_details = record_row.get(
                "operator_review_record_field_details"
            )
            operator_next_action_details = record_row.get(
                "operator_next_action_details"
            )
            required_evidence = record_row.get("required_evidence")
            required_safety_checks = record_row.get("required_safety_checks")
            operator_review_contracts = {
                field: record_row.get(field)
                for field in REQUIRED_METHOD_CARD_CONTRACT_FIELDS
            }
            operator_visible_explanation_contract = record_row.get(
                "operator_visible_explanation_contract"
            )

            validated_method_decisions.append(
                {
                    "method_id": method_id,
                    "operator_decision": operator_decision,
                    "review_scope": record_row.get("review_scope"),
                    "read_only": True,
                    "live_authority_granted": False,
                    "readiness_closure_allowed": False,
                    "live_apply_allowed": False,
                    "queue_import_allowed": False,
                    "can_authorize_live": False,
                    "future_wing_readback_required": True,
                    "future_wing_readback_target_count": (
                        len(readback_targets)
                        if isinstance(readback_targets, list)
                        else 0
                    ),
                    "operator_payload_required_field_count": (
                        len(payload_required_fields)
                        if isinstance(payload_required_fields, list)
                        else 0
                    ),
                    "operator_evidence_required_field_count": (
                        len(evidence_required_fields)
                        if isinstance(evidence_required_fields, list)
                        else 0
                    ),
                    "required_evidence_count": (
                        len(required_evidence)
                        if isinstance(required_evidence, list)
                        else 0
                    ),
                    "required_safety_check_count": (
                        len(required_safety_checks)
                        if isinstance(required_safety_checks, list)
                        else 0
                    ),
                    "operator_review_contract_counts": {
                        field: (
                            len(contract)
                            if isinstance(contract, Mapping)
                            else 0
                        )
                        for field, contract in operator_review_contracts.items()
                    },
                    "operator_visible_explanation_contract_count": (
                        len(operator_visible_explanation_contract)
                        if isinstance(operator_visible_explanation_contract, Mapping)
                        else 0
                    ),
                    "future_wing_readback_contract_summary_count": (
                        len(readback_contract_summary)
                        if isinstance(readback_contract_summary, list)
                        else 0
                    ),
                    "future_live_required_proof_count": (
                        len(required_proofs)
                        if isinstance(required_proofs, list)
                        else 0
                    ),
                    "future_live_requirement_detail_count": (
                        len(future_live_requirement_details)
                        if isinstance(future_live_requirement_details, list)
                        else 0
                    ),
                    "safety_blocker_detail_count": (
                        len(safety_blocker_details)
                        if isinstance(safety_blocker_details, list)
                        else 0
                    ),
                    "evidence_problem_correction_chain_detail_count": (
                        len(evidence_problem_correction_chain_details)
                        if isinstance(
                            evidence_problem_correction_chain_details,
                            list,
                        )
                        else 0
                    ),
                    "headroom_feedback_true_peak_check_detail_count": (
                        len(headroom_feedback_true_peak_check_details)
                        if isinstance(
                            headroom_feedback_true_peak_check_details,
                            list,
                        )
                        else 0
                    ),
                    "blocked_live_action_count": (
                        len(live_actions)
                        if isinstance(live_actions, list)
                        else 0
                    ),
                    "live_authority_boundary_summary_count": (
                        len(authority_boundary_summary)
                        if isinstance(authority_boundary_summary, list)
                        else 0
                    ),
                    "why_safe_summary_count": (
                        len(why_safe_summary)
                        if isinstance(why_safe_summary, list)
                        else 0
                    ),
                    "why_blocked_summary_count": (
                        len(why_blocked_summary)
                        if isinstance(why_blocked_summary, list)
                        else 0
                    ),
                    "goal_requirement_detail_count": (
                        len(goal_requirement_details)
                        if isinstance(goal_requirement_details, list)
                        else 0
                    ),
                    "method_specific_review_detail_count": (
                        len(method_specific_review_details)
                        if isinstance(method_specific_review_details, list)
                        else 0
                    ),
                    "operator_actionable_explanation_detail_count": (
                        len(operator_actionable_explanation_details)
                        if isinstance(
                            operator_actionable_explanation_details,
                            list,
                        )
                        else 0
                    ),
                    "operator_review_record_required_field_count": (
                        len(review_record_required_fields)
                        if isinstance(review_record_required_fields, list)
                        else 0
                    ),
                    "required_confirmation_detail_count": (
                        len(required_confirmation_details)
                        if isinstance(required_confirmation_details, list)
                        else 0
                    ),
                    "operator_acceptance_gate_detail_count": (
                        len(operator_acceptance_gate_details)
                        if isinstance(operator_acceptance_gate_details, list)
                        else 0
                    ),
                    "operator_review_decision_option_detail_count": (
                        len(review_decision_option_details)
                        if isinstance(review_decision_option_details, list)
                        else 0
                    ),
                    "operator_review_checklist_detail_count": (
                        len(review_checklist_details)
                        if isinstance(review_checklist_details, list)
                        else 0
                    ),
                    "operator_review_record_field_detail_count": (
                        len(review_record_field_details)
                        if isinstance(review_record_field_details, list)
                        else 0
                    ),
                    "operator_next_action_detail_count": (
                        len(operator_next_action_details)
                        if isinstance(operator_next_action_details, list)
                        else 0
                    ),
                    "operator_review_record_completed_field_count": (
                        _completed_operator_review_record_field_count(record_row)
                    ),
                }
            )

            if record_row.get("review_scope") != "offline_proposal_only":
                failures.append(
                    _failure(
                        f"decision_record_{method_id}_review_scope_mismatch",
                        repr(record_row.get("review_scope")),
                    )
                )
            _check_false_flags(
                failures,
                payload=record_row,
                label=f"decision_record_{method_id}",
                flags=FALSE_AUTHORITY_FLAGS + ("can_authorize_live",),
            )
            _check_operator_confirmations(
                failures,
                code_prefix=f"decision_record_{method_id}",
                field_name="required_confirmations",
                value=record_row.get("required_confirmations"),
            )
            _check_operator_review_record_required_fields(
                failures,
                code_prefix=f"decision_record_{method_id}",
                value=record_row.get("operator_review_record_required_fields"),
            )
            if not record_row.get("future_wing_readback_targets"):
                failures.append(
                    _failure(
                        f"decision_record_{method_id}_missing_readback_targets",
                        method_id,
                    )
                )
            _check_future_wing_readback_targets(
                failures,
                code_prefix=f"decision_record_{method_id}",
                value=record_row.get("future_wing_readback_targets"),
            )
            _check_future_wing_readback_contract_summary(
                failures,
                code_prefix=f"decision_record_{method_id}",
                value=record_row.get("future_wing_readback_contract_summary"),
            )
            _check_future_live_required_proofs(
                failures,
                code_prefix=f"decision_record_{method_id}",
                value=record_row.get("future_live_required_proofs"),
            )
            _check_blocked_live_actions(
                failures,
                code_prefix=f"decision_record_{method_id}",
                value=record_row.get("blocked_live_actions"),
            )
            _check_live_authority_boundary_summary(
                failures,
                code_prefix=f"decision_record_{method_id}",
                value=record_row.get("live_authority_boundary_summary"),
            )
            _check_equal(
                failures,
                code=f"decision_record_{method_id}_why_safe_summary_keys_mismatch",
                actual=_summary_keys(record_row.get("why_safe_summary")),
                expected=list(REQUIRED_OPERATOR_WHY_SAFE_SUMMARY_KEYS),
            )
            _check_equal(
                failures,
                code=f"decision_record_{method_id}_why_blocked_summary_keys_mismatch",
                actual=_summary_keys(record_row.get("why_blocked_summary")),
                expected=list(REQUIRED_OPERATOR_WHY_BLOCKED_SUMMARY_KEYS),
            )
            if not record_row.get("blocked_live_actions"):
                failures.append(
                    _failure(
                        f"decision_record_{method_id}_missing_blocked_live_actions",
                        method_id,
                    )
                )
            if not record_row.get("required_evidence"):
                failures.append(
                    _failure(
                        f"decision_record_{method_id}_missing_required_evidence",
                        method_id,
                    )
                )
            if not record_row.get("required_safety_checks"):
                failures.append(
                    _failure(
                        (
                            f"decision_record_{method_id}_"
                            "missing_required_safety_checks"
                        ),
                        method_id,
                    )
                )
            _check_required_safety_checks(
                failures,
                code_prefix=f"decision_record_{method_id}",
                value=record_row.get("required_safety_checks"),
            )
            _check_bounded_delta_contract(
                failures,
                code_prefix=f"decision_record_{method_id}",
                bounded_delta_unit=record_row.get("bounded_delta_unit"),
                bounded_delta_policy=record_row.get("bounded_delta_policy"),
                )
            _check_rollback_expectation(
                failures,
                code_prefix=f"decision_record_{method_id}",
                value=record_row.get("rollback_expectation"),
            )
            _check_operator_visible_explanation_contract(
                failures,
                code_prefix=f"decision_record_{method_id}",
                value=record_row.get("operator_visible_explanation_contract"),
            )
            for field in REQUIRED_METHOD_CARD_CONTRACT_FIELDS:
                contract = record_row.get(field)
                if not isinstance(contract, Mapping):
                    failures.append(
                        _failure(
                            f"decision_record_{method_id}_{field}_contract_not_object",
                            repr(contract),
                        )
                    )
                    continue
                if contract.get("required") is not True:
                    failures.append(
                        _failure(
                            (
                                f"decision_record_{method_id}_{field}_"
                                "contract_required_not_true"
                            ),
                            repr(contract.get("required")),
                        )
                    )

        template_priority_sections = template.get("priority_decision_sections")
        record_priority_sections = record.get("priority_decision_sections")
        if not isinstance(template_priority_sections, list):
            failures.append(
                _failure(
                    "decision_template_priority_sections_not_list",
                    repr(template_priority_sections),
                )
            )
            template_priority_sections = []
        if not isinstance(record_priority_sections, list):
            failures.append(
                _failure(
                    "decision_record_priority_sections_not_list",
                    repr(record_priority_sections),
                )
            )
            record_priority_sections = []

        record_priority_sections_by_priority: Dict[int, Mapping[str, Any]] = {}
        for section in record_priority_sections:
            if not isinstance(section, dict):
                failures.append(
                    _failure("decision_record_priority_section_not_object", repr(section))
                )
                continue
            priority = section.get("priority")
            if not isinstance(priority, int):
                failures.append(
                    _failure(
                        "decision_record_priority_section_priority_not_int",
                        repr(priority),
                    )
                )
                continue
            if priority in record_priority_sections_by_priority:
                failures.append(
                    _failure(
                        "decision_record_priority_section_duplicate_priority",
                        str(priority),
                    )
                )
                continue
            record_priority_sections_by_priority[priority] = section

        for template_section in template_priority_sections:
            if not isinstance(template_section, dict):
                failures.append(
                    _failure(
                        "decision_template_priority_section_not_object",
                        repr(template_section),
                    )
                )
                continue
            priority = template_section.get("priority")
            if not isinstance(priority, int):
                failures.append(
                    _failure(
                        "decision_template_priority_section_priority_not_int",
                        repr(priority),
                    )
                )
                continue
            record_section = record_priority_sections_by_priority.get(priority)
            if record_section is None:
                failures.append(
                    _failure(
                        f"decision_record_missing_priority_section_{priority}",
                        str(template_section.get("label")),
                    )
                )
                continue

            template_method_ids = template_section.get("method_ids")
            record_method_ids = record_section.get("method_ids")
            if not isinstance(template_method_ids, list):
                failures.append(
                    _failure(
                        f"decision_template_priority_{priority}_method_ids_not_list",
                        repr(template_method_ids),
                    )
                )
                template_method_ids = []
            if not isinstance(record_method_ids, list):
                failures.append(
                    _failure(
                        f"decision_record_priority_{priority}_method_ids_not_list",
                        repr(record_method_ids),
                    )
                )
                record_method_ids = []
            _check_equal(
                failures,
                code=f"decision_record_priority_{priority}_method_ids_mismatch",
                actual=sorted(str(item) for item in record_method_ids),
                expected=sorted(str(item) for item in template_method_ids),
            )
            _check_equal(
                failures,
                code=f"decision_record_priority_{priority}_label_mismatch",
                actual=record_section.get("label"),
                expected=template_section.get("label"),
            )
            for section_field in (
                "objective_families",
                "allowed_decisions",
                "required_evidence",
                "required_safety_checks",
                "test_evidence_references",
                *REQUIRED_METHOD_CARD_CONTRACT_FIELDS,
                "bounded_delta_policies",
                "rollback_expectations",
                "operator_payload_required_fields",
                "operator_evidence_required_fields",
                "operator_problem_review_summary",
                "operator_method_review_focus",
                "why_safe_summary",
                "why_blocked_summary",
                "goal_requirement_details",
                "method_specific_review_details",
                "operator_actionable_explanation_details",
                "evidence_problem_correction_chain_details",
                "safety_blocker_details",
                "headroom_feedback_true_peak_check_details",
                "operator_visible_explanation_contract",
                "future_wing_readback_targets",
                "future_wing_readback_contract_summary",
                "future_live_required_proofs",
                "future_live_requirement_details",
                "blocked_live_actions",
                "live_authority_boundary_summary",
                "required_confirmation_details",
                "operator_acceptance_gate_details",
                "operator_review_decision_option_details",
                "operator_review_checklist_details",
                "operator_review_record_required_fields",
                "operator_review_record_field_details",
                "operator_next_action_details",
                "expected_test_evidence_references",
            ):
                _check_equal(
                    failures,
                    code=f"decision_record_priority_{priority}_{section_field}_mismatch",
                    actual=_sorted_str_items(record_section.get(section_field)),
                    expected=_sorted_str_items(template_section.get(section_field)),
                )
            for section_field in (
                "expected_evidence_packet_reference",
                "expected_source_material_reference",
                "expected_verification_report_reference",
            ):
                _check_equal(
                    failures,
                    code=f"decision_record_priority_{priority}_{section_field}_mismatch",
                    actual=record_section.get(section_field),
                    expected=template_section.get(section_field),
                )
            _check_operator_review_record_required_fields(
                failures,
                code_prefix=f"decision_record_priority_{priority}",
                value=record_section.get("operator_review_record_required_fields"),
            )
            if record_section.get("review_scope") != "offline_proposal_only":
                failures.append(
                    _failure(
                        f"decision_record_priority_{priority}_review_scope_mismatch",
                        repr(record_section.get("review_scope")),
                    )
                )
            _check_equal(
                failures,
                code=f"decision_record_priority_{priority}_required_priority_decision_mismatch",
                actual=record_section.get("required_priority_operator_decision"),
                expected="one_offline_decision_per_priority_section",
            )
            _check_equal(
                failures,
                code=f"decision_record_priority_{priority}_required_method_decisions_mismatch",
                actual=record_section.get("required_operator_decisions"),
                expected="one_offline_decision_per_method",
            )
            allowed_priority_decisions = _check_offline_review_decisions(
                failures,
                code_prefix=f"decision_template_priority_{priority}",
                field_name="allowed_decisions",
                value=template_section.get("allowed_decisions"),
            )
            _check_offline_review_decisions(
                failures,
                code_prefix=f"decision_record_priority_{priority}",
                field_name="allowed_decisions",
                value=record_section.get("allowed_decisions"),
            )
            priority_operator_decision = record_section.get("operator_decision")
            if priority_operator_decision is None:
                failures.append(
                    _failure(
                        f"decision_record_priority_{priority}_operator_decision_missing",
                        "priority operator_decision must be filled",
                    )
                )
            elif priority_operator_decision not in allowed_priority_decisions:
                failures.append(
                    _failure(
                        f"decision_record_priority_{priority}_operator_decision_invalid",
                        (
                            f"expected one of {allowed_priority_decisions!r}, "
                            f"got {priority_operator_decision!r}"
                        ),
                    )
                )
            _check_completed_operator_review_record_fields(
                failures,
                code_prefix=f"decision_record_priority_{priority}",
                payload=record_section,
            )
            _check_operator_review_reference_bindings(
                failures,
                code_prefix=f"decision_record_priority_{priority}",
                payload=record_section,
                expected_evidence_packet_reference=template_section.get(
                    "expected_evidence_packet_reference"
                ),
                expected_source_material_reference=template_section.get(
                    "expected_source_material_reference"
                ),
                expected_test_evidence_references=template_section.get(
                    "expected_test_evidence_references"
                ),
                expected_verification_report_reference=template_section.get(
                    "expected_verification_report_reference"
                ),
            )
            priority_decision_key = (
                str(priority_operator_decision)
                if priority_operator_decision is not None
                else "missing_operator_decision"
            )
            priority_decision_counts[priority_decision_key] = (
                priority_decision_counts.get(priority_decision_key, 0) + 1
            )
            priority_ids_by_decision.setdefault(priority_decision_key, []).append(
                priority
            )
            _check_false_flags(
                failures,
                payload=record_section,
                label=f"decision_record_priority_{priority}",
                flags=FALSE_AUTHORITY_FLAGS + ("can_authorize_live",),
            )
            _check_true_flags(
                failures,
                payload=record_section,
                label=f"decision_record_priority_{priority}",
                flags=TRUE_REVIEW_FLAGS + ("read_only",),
            )
            if not record_section.get("future_wing_readback_targets"):
                failures.append(
                    _failure(
                        f"decision_record_priority_{priority}_missing_readback_targets",
                        str(priority),
                    )
                )
            _check_future_wing_readback_targets(
                failures,
                code_prefix=f"decision_record_priority_{priority}",
                value=record_section.get("future_wing_readback_targets"),
            )
            _check_future_wing_readback_contract_summary(
                failures,
                code_prefix=f"decision_record_priority_{priority}",
                value=record_section.get("future_wing_readback_contract_summary"),
                allow_duplicate_keys=True,
            )
            _check_future_live_required_proofs(
                failures,
                code_prefix=f"decision_record_priority_{priority}",
                value=record_section.get("future_live_required_proofs"),
            )
            _check_blocked_live_actions(
                failures,
                code_prefix=f"decision_record_priority_{priority}",
                value=record_section.get("blocked_live_actions"),
            )
            _check_live_authority_boundary_summary(
                failures,
                code_prefix=f"decision_record_priority_{priority}",
                value=record_section.get("live_authority_boundary_summary"),
            )
            for summary_field, required_keys in (
                ("why_safe_summary", REQUIRED_OPERATOR_WHY_SAFE_SUMMARY_KEYS),
                (
                    "why_blocked_summary",
                    REQUIRED_OPERATOR_WHY_BLOCKED_SUMMARY_KEYS,
                ),
            ):
                actual_keys = set(_summary_keys(record_section.get(summary_field)))
                missing_keys = [
                    key for key in required_keys if key not in actual_keys
                ]
                if missing_keys:
                    failures.append(
                        _failure(
                            (
                                f"decision_record_priority_{priority}_"
                                f"missing_{summary_field}_keys"
                            ),
                            repr(missing_keys),
                        )
                    )
            if not record_section.get("blocked_live_actions"):
                failures.append(
                    _failure(
                        (
                            f"decision_record_priority_{priority}_"
                            "missing_blocked_live_actions"
                        ),
                        str(priority),
                    )
                )
            if not record_section.get("required_evidence"):
                failures.append(
                    _failure(
                        (
                            f"decision_record_priority_{priority}_"
                            "missing_required_evidence"
                        ),
                        str(priority),
                    )
                )
            if not record_section.get("required_safety_checks"):
                failures.append(
                    _failure(
                        (
                            f"decision_record_priority_{priority}_"
                            "missing_required_safety_checks"
                        ),
                        str(priority),
                    )
                )
            _check_required_safety_checks(
                failures,
                code_prefix=f"decision_record_priority_{priority}",
                value=record_section.get("required_safety_checks"),
            )
            _check_bounded_delta_policies(
                failures,
                code_prefix=f"decision_record_priority_{priority}",
                value=record_section.get("bounded_delta_policies"),
            )
            _check_rollback_expectations(
                failures,
                code_prefix=f"decision_record_priority_{priority}",
                value=record_section.get("rollback_expectations"),
            )
            for field in REQUIRED_METHOD_CARD_CONTRACT_FIELDS:
                if not record_section.get(field):
                    failures.append(
                        _failure(
                            f"decision_record_priority_{priority}_missing_{field}",
                            str(priority),
                        )
                    )
            if not record_section.get("operator_visible_explanation_contract"):
                failures.append(
                    _failure(
                        (
                            f"decision_record_priority_{priority}_"
                            "missing_operator_visible_explanation_contract"
                        ),
                        str(priority),
                    )
                )
            if not record_section.get("bounded_delta_policies"):
                failures.append(
                    _failure(
                        (
                            f"decision_record_priority_{priority}_"
                            "missing_bounded_delta_policies"
                        ),
                        str(priority),
                    )
                )
            if not record_section.get("rollback_expectations"):
                failures.append(
                    _failure(
                        (
                            f"decision_record_priority_{priority}_"
                            "missing_rollback_expectations"
                        ),
                        str(priority),
                    )
                )

            section_payload_required_fields = record_section.get(
                "operator_payload_required_fields"
            )
            section_evidence_required_fields = record_section.get(
                "operator_evidence_required_fields"
            )
            section_readback_contract_summary = record_section.get(
                "future_wing_readback_contract_summary"
            )
            section_authority_boundary_summary = record_section.get(
                "live_authority_boundary_summary"
            )
            section_why_safe_summary = record_section.get("why_safe_summary")
            section_why_blocked_summary = record_section.get(
                "why_blocked_summary"
            )
            section_goal_requirement_details = record_section.get(
                "goal_requirement_details"
            )
            section_method_specific_review_details = record_section.get(
                "method_specific_review_details"
            )
            section_operator_actionable_explanation_details = record_section.get(
                "operator_actionable_explanation_details"
            )
            section_review_record_required_fields = record_section.get(
                "operator_review_record_required_fields"
            )
            section_review_checklist_details = record_section.get(
                "operator_review_checklist_details"
            )
            section_required_confirmation_details = record_section.get(
                "required_confirmation_details"
            )
            section_operator_acceptance_gate_details = record_section.get(
                "operator_acceptance_gate_details"
            )
            section_review_decision_option_details = record_section.get(
                "operator_review_decision_option_details"
            )
            section_review_record_field_details = record_section.get(
                "operator_review_record_field_details"
            )
            section_required_evidence = record_section.get("required_evidence")
            section_required_safety_checks = record_section.get(
                "required_safety_checks"
            )
            section_bounded_delta_policies = record_section.get(
                "bounded_delta_policies"
            )
            section_rollback_expectations = record_section.get(
                "rollback_expectations"
            )
            section_operator_review_contracts = {
                field: record_section.get(field)
                for field in REQUIRED_METHOD_CARD_CONTRACT_FIELDS
            }
            section_operator_visible_explanation_contract = record_section.get(
                "operator_visible_explanation_contract"
            )

            section_decision_counts: Dict[str, int] = {
                "missing_operator_decision": 0
            }
            section_method_ids_by_decision: Dict[str, List[str]] = {
                "missing_operator_decision": []
            }
            section_blocked_live_actions: set[str] = set()
            section_future_live_required_proofs: set[str] = set()
            section_future_wing_readback_targets: set[str] = set()
            section_accepted_count = 0
            for method_id in template_method_ids:
                method_id_key = str(method_id)
                record_row = record_rows_by_id.get(str(method_id))
                operator_decision = (
                    record_row.get("operator_decision")
                    if record_row is not None
                    else None
                )
                decision_key = (
                    str(operator_decision)
                    if operator_decision is not None
                    else "missing_operator_decision"
                )
                section_decision_counts[decision_key] = (
                    section_decision_counts.get(decision_key, 0) + 1
                )
                section_method_ids_by_decision.setdefault(decision_key, []).append(
                    method_id_key
                )
                if operator_decision == "accept_as_offline_proposal":
                    section_accepted_count += 1
                if record_row is None:
                    continue
                readback_targets = record_row.get("future_wing_readback_targets")
                if isinstance(readback_targets, list):
                    section_future_wing_readback_targets.update(
                        str(item) for item in readback_targets
                    )
                required_proofs = record_row.get("future_live_required_proofs")
                if isinstance(required_proofs, list):
                    section_future_live_required_proofs.update(
                        str(item) for item in required_proofs
                    )
                live_actions = record_row.get("blocked_live_actions")
                if isinstance(live_actions, list):
                    section_blocked_live_actions.update(
                        str(item) for item in live_actions
                    )

            section_method_count = len(template_method_ids)
            if (
                priority_operator_decision == "accept_as_offline_proposal"
                and section_accepted_count != section_method_count
            ):
                failures.append(
                    _failure(
                        (
                            f"decision_record_priority_{priority}_"
                            "accept_conflicts_with_method_decisions"
                        ),
                        (
                            "priority section cannot accept as offline proposal "
                            "unless every method in the section is accepted"
                        ),
                    )
                )
            if (
                priority_operator_decision
                in (
                    "request_more_evidence",
                    "reject_proposal",
                    "defer_until_real_wing_proof",
                )
                and section_method_count > 0
                and section_accepted_count == section_method_count
            ):
                failures.append(
                    _failure(
                        (
                            f"decision_record_priority_{priority}_"
                            "non_accept_conflicts_with_method_decisions"
                        ),
                        (
                            "priority section cannot block, defer, or reject "
                            "when every method in the section is accepted"
                        ),
                    )
                )

            validated_priority_decision_sections.append(
                {
                    "priority": priority,
                    "label": record_section.get("label"),
                    "operator_decision": priority_operator_decision,
                    "review_scope": record_section.get("review_scope"),
                    "method_decision_count": section_method_count,
                    "accepted_offline_proposal_count": section_accepted_count,
                    "decision_counts": dict(sorted(section_decision_counts.items())),
                    "method_ids_by_decision": {
                        decision: sorted(method_ids)
                        for decision, method_ids in sorted(
                            section_method_ids_by_decision.items()
                        )
                    },
                    "future_wing_readback_target_count": len(
                        section_future_wing_readback_targets
                    ),
                    "future_live_required_proof_count": len(
                        section_future_live_required_proofs
                    ),
                    "blocked_live_action_count": len(section_blocked_live_actions),
                    "operator_payload_required_field_count": (
                        len(section_payload_required_fields)
                        if isinstance(section_payload_required_fields, list)
                        else 0
                    ),
                    "operator_evidence_required_field_count": (
                        len(section_evidence_required_fields)
                        if isinstance(section_evidence_required_fields, list)
                        else 0
                    ),
                    "required_evidence_count": (
                        len(section_required_evidence)
                        if isinstance(section_required_evidence, list)
                        else 0
                    ),
                    "required_safety_check_count": (
                        len(section_required_safety_checks)
                        if isinstance(section_required_safety_checks, list)
                        else 0
                    ),
                    "bounded_delta_policy_count": (
                        len(section_bounded_delta_policies)
                        if isinstance(section_bounded_delta_policies, list)
                        else 0
                    ),
                    "rollback_expectation_count": (
                        len(section_rollback_expectations)
                        if isinstance(section_rollback_expectations, list)
                        else 0
                    ),
                    "operator_review_contract_counts": {
                        field: (
                            len(values)
                            if isinstance(values, list)
                            else 0
                        )
                        for field, values in section_operator_review_contracts.items()
                    },
                    "operator_visible_explanation_contract_count": (
                        len(section_operator_visible_explanation_contract)
                        if isinstance(
                            section_operator_visible_explanation_contract, list
                        )
                        else 0
                    ),
                    "future_wing_readback_contract_summary_count": (
                        len(section_readback_contract_summary)
                        if isinstance(section_readback_contract_summary, list)
                        else 0
                    ),
                    "live_authority_boundary_summary_count": (
                        len(section_authority_boundary_summary)
                        if isinstance(section_authority_boundary_summary, list)
                        else 0
                    ),
                    "why_safe_summary_count": (
                        len(section_why_safe_summary)
                        if isinstance(section_why_safe_summary, list)
                        else 0
                    ),
                    "why_blocked_summary_count": (
                        len(section_why_blocked_summary)
                        if isinstance(section_why_blocked_summary, list)
                        else 0
                    ),
                    "goal_requirement_detail_count": (
                        len(section_goal_requirement_details)
                        if isinstance(section_goal_requirement_details, list)
                        else 0
                    ),
                    "method_specific_review_detail_count": (
                        len(section_method_specific_review_details)
                        if isinstance(
                            section_method_specific_review_details,
                            list,
                        )
                        else 0
                    ),
                    "operator_actionable_explanation_detail_count": (
                        len(section_operator_actionable_explanation_details)
                        if isinstance(
                            section_operator_actionable_explanation_details,
                            list,
                        )
                        else 0
                    ),
                    "operator_review_record_required_field_count": (
                        len(section_review_record_required_fields)
                        if isinstance(section_review_record_required_fields, list)
                        else 0
                    ),
                    "required_confirmation_detail_count": (
                        len(section_required_confirmation_details)
                        if isinstance(section_required_confirmation_details, list)
                        else 0
                    ),
                    "operator_acceptance_gate_detail_count": (
                        len(section_operator_acceptance_gate_details)
                        if isinstance(
                            section_operator_acceptance_gate_details,
                            list,
                        )
                        else 0
                    ),
                    "operator_review_decision_option_detail_count": (
                        len(section_review_decision_option_details)
                        if isinstance(section_review_decision_option_details, list)
                        else 0
                    ),
                    "operator_review_checklist_detail_count": (
                        len(section_review_checklist_details)
                        if isinstance(section_review_checklist_details, list)
                        else 0
                    ),
                    "operator_review_record_field_detail_count": (
                        len(section_review_record_field_details)
                        if isinstance(section_review_record_field_details, list)
                        else 0
                    ),
                    "operator_review_record_completed_field_count": (
                        _completed_operator_review_record_field_count(
                            record_section
                        )
                    ),
                    "read_only": True,
                    "live_authority_granted": False,
                    "readiness_closure_allowed": False,
                    "live_apply_allowed": False,
                    "queue_import_allowed": False,
                    "can_authorize_live": False,
                    "future_wing_readback_required": True,
                }
            )

    return {
        "schema": "automixer.correction_method_operator_decision_record_validation.v1",
        "run_dir": str(run_dir),
        "decision_record_path": str(decision_record_path),
        "status": "passed" if not failures else "failed",
        "failed_checks": failures,
        "read_only": True,
        "contacts_wing": False,
        "contacts_backend": False,
        "queries_console": False,
        "mutates_console_state": False,
        "mutates_mixer": False,
        "live_write": False,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "live_apply_allowed": False,
        "queue_import_allowed": False,
        "can_authorize_live": False,
        "operator_review_required": True,
        "future_wing_readback_required": True,
        "hardware_proof_required_before_live": True,
        "source_authority_denied": True,
        "denial_reason": (
            "operator_decision_record_cannot_authorize_live_or_close_readiness"
        ),
        "operator_next_step": (
            "use_offline_decisions_for_engineer_review_and_collect_real_wing_"
            "readback_before_live_promotion"
        ),
        "method_decision_count": len(record_rows_by_id),
        "accepted_offline_proposal_count": accepted_offline_proposal_count,
        "decision_counts": dict(sorted(decision_counts.items())),
        "method_ids_by_decision": {
            decision: sorted(method_ids)
            for decision, method_ids in sorted(method_ids_by_decision.items())
        },
        "priority_decision_counts": dict(sorted(priority_decision_counts.items())),
        "priority_ids_by_decision": {
            decision: sorted(priority_ids)
            for decision, priority_ids in sorted(priority_ids_by_decision.items())
        },
        "future_wing_readback_target_count": len(future_wing_readback_targets),
        "future_live_required_proof_count": len(future_live_required_proofs),
        "blocked_live_action_count": len(blocked_live_actions),
        "priority_decision_section_count": len(validated_priority_decision_sections),
        "validated_priority_decision_sections": validated_priority_decision_sections,
        "validated_method_decisions": validated_method_decisions,
    }


def write_correction_method_operator_review_verification_report(
    report: Mapping[str, Any],
    output_path: Path,
) -> Path:
    """Persist a verifier report as an offline/read-only artifact."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_correction_method_operator_decision_validation_report(
    report: Mapping[str, Any],
    output_path: Path,
) -> Path:
    """Persist an operator decision-record validation report."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a correction-method operator review export bundle without "
            "WING/backend contact."
        )
    )
    parser.add_argument("run_dir", type=Path, help="Export run directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for a persisted verification-report.json artifact.",
    )
    parser.add_argument(
        "--decision-record",
        type=Path,
        default=None,
        help=(
            "Optional filled operator-decision-record JSON to validate as "
            "offline/proposal-only."
        ),
    )
    parser.add_argument(
        "--decision-output",
        type=Path,
        default=None,
        help="Optional path for a persisted decision-validation-report.json artifact.",
    )
    args = parser.parse_args(argv)

    export_report = verify_correction_method_operator_review_export(args.run_dir)
    if args.output is not None:
        write_correction_method_operator_review_verification_report(
            export_report,
            args.output,
        )
    report = export_report
    if args.decision_record is not None:
        report = validate_correction_method_operator_decision_record(
            args.run_dir,
            args.decision_record,
        )
        if args.decision_output is not None:
            write_correction_method_operator_decision_validation_report(
                report,
                args.decision_output,
            )
    print(json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
