#!/usr/bin/env python3
"""Generate the latest supervised WING MVP progress report.

This tool is read-only except for the requested report output. It parses the
repo-local blocker table and writes a compact operator-facing status report.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BLOCKERS = PROJECT_ROOT / ".ai" / "project-layer" / "wing_mvp_blockers.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "Docs" / "reports" / "MVP_PROGRESS_LATEST.md"
CLOSED_STATUS = "closed"
CURRENT_FIRST_COMMAND_HEADING = "## Current First Command"
EVIDENCE_PATH_RE = re.compile(r"`([^`]+)`")
CURRENT_FIRST_COMMAND_BASELINE_COVERAGE_MARKERS = (
    "PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick",
    "Docs/runbooks/WING_READONLY_PREFLIGHT.md",
    "operator_sequence_count: 12",
    "operator_sequence_sha256_matches_actual: true",
    "operator_sequence_matches_contract: true",
    "operator_sequence_sha256_matches_contract: true",
    "backend_port_prestart_gate",
    "expected_backend_port_prestart_gate",
    "backend_port_prestart_gate_matches_expected: true",
    "declared_backend_port_prestart_gate_matches_expected: true",
    "backend_port_prestart_gate.grants_live_authority: false",
    "backend_port_prestart_gate.must_pass_before_backend_wing_session: true",
    "backend_port_prestart_check_command",
    "backend_port_prestart_check_command_text",
    "backend_port_prestart_check_required: true",
    "backend_port_prestart_check_must_pass_before_starting_backend: true",
    "backend_port_prestart_gate_execution_allowed: false",
    "backend_port_prestart_gate_readiness_closure_allowed: false",
    "backend_port_prestart_gate_live_authority_granted: false",
    "backend_port_prestart_check_failure_detail_fields",
    "environment_blocked_checks",
    "environment_blocked_check_details",
    "local_dev_server_bind_denied",
    "127.0.0.1:3000",
    "listener_confirmation_template_status: pending_operator_edit",
    "listener_confirmation_template_safe_to_stop_listener_by_gate: false",
    "listener_confirmation_template_live_authority_granted: false",
    "listener_confirmation_template_accepted_by_template: false",
    "listener_confirmation_template_operator_placeholder_fields",
    "listener_confirmation_template_execution_allowed_by_template: false",
    "listener_confirmation_template_readiness_closure_allowed_by_template: false",
    "manual_review_required_before_readiness_update: true",
    "readiness_matrix_update_allowed_by_packet: false",
    "readiness_closure_allowed_by_packet: false",
    "read_only: true",
    "live_write: false",
    "mutates_console_state: false",
    "queries_console: false",
    "executes_tests: false",
    "operator_required: true",
    "expected_plan_required_for_real_run: true",
    "physical_preflight_requires_existing_backend_connection: true",
    "backend_port_prestart_summary_required: true",
    "backend_port_prestart_summary_type: no_wing_release_check",
    "backend_port_prestart_summary_type_matches_plan: true",
    "backend_port_prestart_summary_connection_policy: no_hardware_no_network_no_live_writes",
    "backend_port_prestart_summary_connection_policy_matches_plan: true",
    "backend_port_prestart_summary_status: passed",
    "backend_port_prestart_summary_status_matches_plan: true",
    "backend_port_prestart_summary_read_only: true",
    "backend_port_prestart_summary_read_only_matches_plan: true",
    "backend_port_prestart_summary_live_write: false",
    "backend_port_prestart_summary_live_write_matches_plan: true",
    "backend_port_prestart_summary_contacts_backend: false",
    "backend_port_prestart_summary_contacts_backend_matches_plan: true",
    "backend_port_prestart_summary_contacts_wing: false",
    "backend_port_prestart_summary_contacts_wing_matches_plan: true",
    "backend_port_prestart_summary_mutates_console_state: false",
    "backend_port_prestart_summary_mutates_console_state_matches_plan: true",
    "backend_port_prestart_summary_queries_console: false",
    "backend_port_prestart_summary_queries_console_matches_plan: true",
    "backend_port_prestart_summary_executes_tests: false",
    "backend_port_prestart_summary_executes_tests_matches_plan: true",
    "backend_port_prestart_check_status: passed",
    "plan_safe_to_stop_listener_by_gate: false",
    "plan_live_authority_granted: false",
    "plan_backend_port_prestart_gate_execution_allowed: false",
    "plan_backend_port_prestart_gate_readiness_closure_allowed: false",
    "plan_backend_port_prestart_gate_live_authority_granted: false",
    "backend_port_prestart_check_status_matches_plan: true",
    "safe_to_stop_listener_by_gate_matches_plan: true",
    "live_authority_granted_matches_plan: true",
    "backend_port_prestart_gate_execution_allowed_matches_plan: true",
    "backend_port_prestart_gate_readiness_closure_allowed_matches_plan: true",
    "backend_port_prestart_gate_live_authority_granted_matches_plan: true",
    "declared_safe_to_stop_listener_by_gate_matches_plan: true",
    "declared_live_authority_granted_matches_plan: true",
    "declared_backend_port_prestart_gate_execution_allowed_matches_plan: true",
    "declared_backend_port_prestart_gate_readiness_closure_allowed_matches_plan: true",
    "declared_backend_port_prestart_gate_live_authority_granted_matches_plan: true",
    "execution_allowed_by_packet: false",
    "live_authority_granted: false",
    "full_autonomous_ready: false",
    "next_manual_artifact_required: real wing_readonly_preflight.v1 summary.json",
    "operator_sequence",
    "expected_operator_sequence",
    "wing_readonly_operator_sequence_list_summary_surface_self_check_required: true",
    "wing_readonly_operator_sequence_list_summary_surface_mismatches: []",
    "wing_readonly_operator_sequence_list_summary_surface_match: true",
    "wing_readonly_operator_sequence_list_summary_surface_readiness_closure_allowed: false",
    "wing_readonly_operator_sequence_list_summary_surface_live_authority_granted: false",
    "wing_readonly_operator_sequence_list_markdown_surface_required: true",
    "wing_readonly_operator_sequence_list_markdown_surface_missing_field_lines: []",
    "wing_readonly_operator_sequence_list_markdown_surface_field_lines_rendered: true",
    "wing_readonly_operator_sequence_list_markdown_surface_readiness_closure_allowed: false",
    "wing_readonly_operator_sequence_list_markdown_surface_live_authority_granted: false",
    "expected_artifacts",
    "expected_artifacts_match_plan: true",
    "summary_glob_matches_plan: true",
    "event_log_glob_matches_plan: true",
    "artifact_manifest_verifier_operator_sequence_step_index: 1",
    "expected_artifact_manifest_verifier_operator_sequence_step_index: 1",
    "artifact_manifest_verifier_operator_sequence_step_index_matches_contract: true",
    "artifact_manifest_verifier_operator_sequence_step",
    "expected_artifact_manifest_verifier_operator_sequence_step",
    "artifact_manifest_verifier_operator_sequence_step_matches_contract: true",
    "artifact_manifest_verifier_operator_sequence_before_preview: true",
    "artifact_manifest_verifier_required_markers",
    "backend_connection_confirmation_step_index: 3",
    "expected_backend_connection_confirmation_step_index: 3",
    "operator_sequence_has_backend_connection_confirmation: true",
    "operator_sequence_backend_connection_confirmation_step",
    "operator_sequence_backend_connection_confirmation_step_index: 3",
    "operator_sequence_backend_connection_confirmation_before_capture: true",
    "operator_sequence_validate_readonly_preflight_inputs_step_index: 5",
    "expected_operator_sequence_validate_readonly_preflight_inputs_step_index: 5",
    "operator_sequence_validate_readonly_preflight_inputs_step_index_matches_contract: true",
    "operator_sequence_validate_readonly_preflight_inputs_step: Run validate_readonly_preflight_inputs before capture; it must validate real-run inputs and expected-plan bindings without opening transport.",
    "expected_operator_sequence_validate_readonly_preflight_inputs_step: Run validate_readonly_preflight_inputs before capture; it must validate real-run inputs and expected-plan bindings without opening transport.",
    "operator_sequence_validate_readonly_preflight_inputs_step_matches_contract: true",
    "operator_sequence_capture_readonly_preflight_step_index: 6",
    "expected_operator_sequence_capture_readonly_preflight_step_index: 6",
    "operator_sequence_capture_readonly_preflight_step_index_matches_contract: true",
    "operator_sequence_capture_readonly_preflight_step: Run capture_readonly_preflight.",
    "expected_operator_sequence_capture_readonly_preflight_step: Run capture_readonly_preflight.",
    "operator_sequence_capture_readonly_preflight_step_matches_contract: true",
    "validate_inputs_command_uses_validate_run_inputs_only: true",
    "validate_inputs_command_uses_plan_only: false",
    "validate_inputs_command_prestart_summary",
    "validate_inputs_command_prestart_summary_matches_plan: true",
    "validate_inputs_command_artifact_root_matches_packet: true",
    "validate_inputs_command_expected_plan_matches_packet: true",
    "mvp_blocker_anchor_matches_current_first_open: true",
    "mvp_blocker_anchor_sha256_matches_current_first_open: true",
    "mvp_blocker_anchor.packet_does_not_close_blocker: true",
    "project_director_selected_action_anchor",
    "expected_project_director_selected_action_anchor",
    "project_director_selected_action_anchor_matches_current_directive: true",
    "project_director_selected_action_source: Current First Command",
    "expected_project_director_selected_action_source: Current First Command",
    "project_director_selected_action_source_matches_current_directive: true",
    "project_director_selected_action_sha256",
    "actual_project_director_selected_action_sha256",
    "expected_project_director_selected_action_sha256",
    "project_director_selected_action_sha256_available: true",
    "project_director_selected_action_sha256_available_matches_digest: true",
    "project_director_selected_action_sha256_matches_anchor: true",
    "project_director_selected_action_sha256_matches_text: true",
    "project_director_selected_action_sha256_matches_current_directive: true",
    "project_director_current_first_command_baseline_coverage_required: true",
    "project_director_current_first_command_baseline_coverage_passed: true",
    "project_director_current_first_command_baseline_coverage_marker_count",
    "project_director_current_first_command_baseline_missing_markers: []",
    "project_director_selected_action_value_gate_self_check_required: true",
    "project_director_selected_action_value_gate_self_check_required_markers",
    "project_director_selected_action_value_gate_self_check_missing_markers: []",
    "project_director_selected_action_value_gate_self_check_markers_present: true",
    "project_director_selected_action_value_gate_self_check_readiness_closure_allowed: false",
    "project_director_selected_action_value_gate_self_check_live_authority_granted: false",
    "project_director_selected_action_value_gate_self_check_markdown_surface_required: true",
    "project_director_selected_action_value_gate_self_check_markdown_surface_required_field_lines",
    "project_director_selected_action_value_gate_self_check_markdown_surface_missing_field_lines: []",
    "project_director_selected_action_value_gate_self_check_markdown_surface_field_lines_rendered: true",
    "project_director_selected_action_value_gate_self_check_markdown_surface_readiness_closure_allowed: false",
    "project_director_selected_action_value_gate_self_check_markdown_surface_live_authority_granted: false",
    "operator_sequence_has_supervised_prestart_gate: true",
    "operator_sequence_supervised_prestart_gate_is_final: true",
    "required_supervised_prestart_gate_markers",
    "supervised_prestart_gate",
    "expected_supervised_prestart_gate",
    "supervised_prestart_gate_matches_expected: true",
    "declared_supervised_prestart_gate_matches_expected: true",
    "supervised_prestart_gate_required: true",
    "supervised_prestart_gate_operator_sequence_step_index: 11",
    "supervised_prestart_gate_operator_sequence_step_must_be_final: true",
    "supervised_prestart_gate_operator_sequence_step",
    "supervised_prestart_gate_operator_sequence_step_matches_summary: true",
    "supervised_prestart_gate_required_markers",
    "expected_supervised_prestart_gate_required_markers",
    "expected_required_supervised_prestart_gate_markers",
    "required_supervised_prestart_gate_marker_count: 8",
    "expected_required_supervised_prestart_gate_marker_count: 8",
    "required_supervised_prestart_gate_marker_count_matches_contract: true",
    "required_supervised_prestart_gate_marker_count_matches_supervised_gate: true",
    "required_supervised_prestart_gate_markers_match_contract: true",
    "required_supervised_prestart_gate_markers_match_supervised_gate: true",
    "supervised_prestart_gate_required_marker_count: 8",
    "expected_supervised_prestart_gate_required_marker_count: 8",
    "supervised_prestart_gate_required_marker_count_matches_contract: true",
    "supervised_prestart_gate_required_marker_count_matches_summary: true",
    "supervised_prestart_gate_required_markers_match_contract: true",
    "supervised_prestart_gate_required_markers_match_summary: true",
    "supervised_prestart_gate.grants_live_authority: false",
    "supervised_prestart_gate_grants_live_authority: false",
    "supervised_prestart_gate.requires_accepted_candidate_prestart_trace: true",
    "supervised_prestart_gate.requires_accepted_candidate_verifier_ws_proof: true",
    "supervised_prestart_gate.requires_accepted_candidate_artifact_root_trace: true",
    "supervised_prestart_gate.requires_accepted_candidate_request_sequence_trace: true",
    "supervised_prestart_gate.requires_accepted_candidate_expected_artifact_trace: true",
    "supervised_prestart_gate.requires_accepted_candidate_contact_trace: true",
    "supervised_prestart_gate_requires_operator_decision_status: accepted_decision_ready",
    "supervised_prestart_gate_unlocks_runbook: Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md",
    "supervised_prestart_gate_requires_accepted_candidate_prestart_trace: true",
    "supervised_prestart_gate_requires_accepted_candidate_verifier_ws_proof: true",
    "supervised_prestart_gate_requires_accepted_candidate_artifact_root_trace: true",
    "supervised_prestart_gate_requires_accepted_candidate_request_sequence_trace: true",
    "supervised_prestart_gate_requires_accepted_candidate_expected_artifact_trace: true",
    "supervised_prestart_gate_requires_accepted_candidate_contact_trace: true",
    "valid accepted-candidate artifact-root trace",
    "valid accepted-candidate request-sequence trace",
    "valid accepted-candidate contact trace",
    "required_supervised_proof_plan_summary_fields",
    "expected_required_supervised_proof_plan_summary_fields",
    "required_supervised_proof_plan_summary_fields_match_contract: true",
    "required_supervised_proof_plan_summary_fields_include_invalid_records: true",
    "operator-readable no-WING `summary.md`",
    "readonly_preflight_review_decision_candidate_prestart_contract_valid: true",
    'readonly_preflight_review_decision_candidate_prestart_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]',
    "readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source: {}",
    'readonly_preflight_review_decision_expected_candidate_prestart_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]',
    "readonly_preflight_review_decision_expected_candidate_prestart_contract_violations_by_source: {}",
    "readonly_preflight_review_decision_invalid_records",
    "accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof",
    "accepted_readonly_preflight_candidate_verifier_ws_proof_required_before_supervised_proof",
    "readonly_preflight_review_decision_candidate_artifact_root_contract_valid: true",
    'readonly_preflight_review_decision_candidate_artifact_roots: ["artifacts/wing_readonly_preflight"]',
    "readonly_preflight_review_decision_candidate_artifact_root_contract_violations_by_source: {}",
    'readonly_preflight_review_decision_expected_candidate_artifact_roots: ["artifacts/wing_readonly_preflight"]',
    "readonly_preflight_review_decision_expected_candidate_artifact_root_contract_violations_by_source: {}",
    "next_wing_proof_plan_steps_with_expected_candidate_artifact_root_violations",
    "accepted_readonly_preflight_candidate_artifact_root_required_before_supervised_proof",
    "accepted_readonly_preflight_candidate_expected_artifact_trace_required_before_supervised_proof",
    "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_source_ids",
    "readonly_preflight_review_decision_accepted_candidate_expected_artifacts_by_source",
    "readonly_preflight_review_decision_accepted_candidate_expected_summary_artifact_by_source",
    "readonly_preflight_review_decision_accepted_candidate_expected_events_artifact_by_source",
    "readonly_preflight_review_decision_accepted_candidate_summary_path_matches_expected_artifact_by_source",
    "readonly_preflight_review_decision_accepted_candidate_events_path_matches_expected_artifact_by_source",
    "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_missing_source_ids: []",
    "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_complete: true",
    "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_required: true",
    "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_violations: []",
    "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_passed: true",
    "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_readiness_closure_allowed: false",
    "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_live_authority_granted: false",
    "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_self_check_required: true",
    "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_required_fields",
    "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_mismatches: []",
    "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_match: true",
    "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_readiness_closure_allowed: false",
    "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_live_authority_granted: false",
    "readonly_preflight_review_decision_candidate_request_sequence_contract_valid: true",
    "readonly_preflight_review_decision_candidate_request_sequence_source_ids",
    "readonly_preflight_review_decision_candidate_request_sequence_contract_violations_by_source: {}",
    "readonly_preflight_review_decision_expected_candidate_request_sequence_source_ids",
    "readonly_preflight_review_decision_expected_candidate_request_sequence_contract_violations_by_source: {}",
    "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_source_ids",
    "readonly_preflight_review_decision_accepted_candidate_request_sequence_by_source",
    "readonly_preflight_review_decision_accepted_candidate_plan_request_sequence_by_source",
    "readonly_preflight_review_decision_accepted_candidate_request_sequence_matches_plan_by_source",
    "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_missing_source_ids: []",
    "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_complete: true",
    "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_required: true",
    "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_violations: []",
    "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_passed: true",
    "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_readiness_closure_allowed: false",
    "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_live_authority_granted: false",
    "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_summary_surface_self_check_required: true",
    "readonly_preflight_review_decision_accepted_candidate_contact_trace_source_ids",
    "readonly_preflight_review_decision_accepted_candidate_contacts_backend_by_source",
    "readonly_preflight_review_decision_accepted_candidate_queries_console_by_source",
    "readonly_preflight_review_decision_accepted_candidate_contact_trace_missing_source_ids: []",
    "readonly_preflight_review_decision_accepted_candidate_contact_trace_complete: true",
    "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_required: true",
    "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_violations: []",
    "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_passed: true",
    "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_readiness_closure_allowed: false",
    "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_live_authority_granted: false",
    "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_summary_surface_required_fields",
    "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_summary_surface_mismatches: []",
    "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_summary_surface_match: true",
    "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_summary_surface_readiness_closure_allowed: false",
    "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_summary_surface_live_authority_granted: false",
    "operator_markdown_summary_required: true",
    "operator_markdown_summary_required_field_lines",
    "readonly_preflight_review_decision_candidate_verifier_ws_proof_source_ids",
    "readonly_preflight_review_decision_candidate_verifier_ws_proof_violations_by_source: {}",
    "readonly_preflight_review_decision_candidate_verifier_ws_proof_complete: true",
    "operator_markdown_summary_required_field_lines_missing_members: []",
    "operator_markdown_summary_required_field_lines_match_contract: true",
    "operator_markdown_summary_field_lines_rendered: true",
    "operator_markdown_summary_missing_field_lines: []",
    "operator_markdown_summary_artifact: summary.md",
    "operator_markdown_summary_sha256",
    "operator_markdown_summary_sha256_available: true",
    "operator_markdown_summary_actual_sha256",
    "operator_markdown_summary_sha256_matches_actual: true",
    "artifact_manifest_summary_surface_required: true",
    "artifact_manifest_summary_surface_missing_fields: []",
    "artifact_manifest_summary_surface_required_field_lines",
    "artifact_manifest_summary_surface_missing_field_lines: []",
    "artifact_manifest_summary_surface_field_lines_rendered: true",
    "artifact_manifest_contract_id",
    "artifact_manifest_contract_required: true",
    "artifact_manifest_contract_fields",
    "artifact_manifest_contract_sha256",
    "artifact_manifest_contract_sha256_available: true",
    "artifact_manifest_contract_sha256_matches_expected: true",
    "artifact_manifest_contract_cross_surface_required: true",
    "artifact_manifest_contract_cross_surface_required_surfaces",
    "artifact_manifest_contract_cross_surface_required_fields",
    "artifact_manifest_contract_cross_surface_expected_match: true",
    "artifact_manifest_contract_cross_surface_match: true",
    "artifact_manifest_summary_json_contract_sha256_matches_manifest: true",
    "artifact_manifest_summary_md_contract_sha256_matches_manifest: true",
    "artifact_manifest_contract_cross_surface_mismatches: []",
    "artifact_manifest_contract_cross_surface_readiness_closure_allowed: false",
    "artifact_manifest_contract_cross_surface_live_authority_granted: false",
    "artifact_manifest_path: artifact-manifest.json",
    "artifact_manifest_artifact_count: 2",
    "artifact_manifest_required_artifact_count: 2",
    "artifact_manifest_artifact_count_matches_required: true",
    "artifact_manifest_required_artifact_ids",
    "artifact_manifest_artifact_ids",
    "artifact_manifest_required_artifacts_present: true",
    "artifact_manifest_required_artifact_paths",
    "artifact_manifest_artifact_paths",
    "artifact_manifest_required_artifact_paths_present: true",
    "artifact_manifest_required_artifact_sha256_available: true",
    "artifact_manifest_required_artifact_sha256_fields",
    "artifact_manifest_summary_json_sha256",
    "artifact_manifest_summary_json_sha256_available: true",
    "artifact_manifest_summary_md_sha256",
    "artifact_manifest_summary_md_sha256_available: true",
    "artifact_manifest_readiness_closure_allowed: false",
    "artifact_manifest_live_authority_granted: false",
    "PYTHONPATH=backend python3 tools/verify_no_wing_artifact_manifest.py <no_wing_artifact_dir>",
    "status: passed",
    "failed_checks: []",
    "artifact_manifest_self_consistency_match: true",
    "artifact_manifest_self_consistency_summary_surface_match: true",
    "read_only: true",
    "live_write: false",
    "contacts_wing: false",
    "contacts_backend: false",
    "mutates_console_state: false",
    "queries_console: false",
    "executes_tests: false",
    "artifact_manifest_verifier_required: true",
    "artifact_manifest_verifier_command",
    "artifact_manifest_verifier_status_required: passed",
    "artifact_manifest_verifier_failed_checks_required: []",
    "artifact_manifest_verifier_report_required_fields",
    "artifact_manifest_verifier_read_only: true",
    "artifact_manifest_verifier_live_write: false",
    "artifact_manifest_verifier_contacts_wing: false",
    "artifact_manifest_verifier_contacts_backend: false",
    "artifact_manifest_verifier_mutates_console_state: false",
    "artifact_manifest_verifier_queries_console: false",
    "artifact_manifest_verifier_executes_tests: false",
    "artifact_manifest_verifier_readiness_matrix_update_allowed: false",
    "artifact_manifest_verifier_readiness_closure_allowed: false",
    "artifact_manifest_verifier_live_authority_granted: false",
    "artifact-manifest.json",
    "summary_json_exists: true",
    "summary_md_exists: true",
    "artifact_manifest_exists: true",
    "artifact_bundle_loadable: true",
    "manifest_type: true",
    "manifest_required_artifact_ids_match_contract: true",
    "manifest_required_artifact_paths_match_contract: true",
    "manifest_records_required_artifact_ids: true",
    "summary_json_path_matches_contract: true",
    "summary_json_sha256_matches_actual: true",
    "summary_json_sha256_matches_manifest: true",
    "summary_md_path_matches_contract: true",
    "summary_md_sha256_matches_actual: true",
    "summary_md_sha256_matches_manifest: true",
    "manifest_denies_readiness_matrix_update: true",
    "manifest_denies_readiness_closure: true",
    "manifest_denies_live_authority: true",
    "summary_denies_readiness_matrix_update: true",
    "summary_denies_readiness_closure: true",
    "summary_denies_live_authority: true",
    "summary_json_artifact_manifest_verifier_fields_present: true",
    "manifest_artifact_manifest_verifier_fields_present: true",
    "summary_md_artifact_manifest_verifier_fields_rendered: true",
    "manifest_artifact_manifest_verifier_command_matches_expected: true",
    "summary_md_readiness_closure_allowed_by_packet_denies_authority: true",
    "summary_md_live_authority_granted_denies_authority: true",
    "summary_md_artifact_manifest_contract_cross_surface_readiness_closure_allowed_denies_authority: true",
    "summary_md_artifact_manifest_contract_cross_surface_readiness_matrix_update_allowed_denies_authority: true",
    "summary_md_artifact_manifest_contract_cross_surface_live_authority_granted_denies_authority: true",
    "summary_md_artifact_manifest_readiness_matrix_update_allowed_denies_authority: true",
    "summary_md_artifact_manifest_readiness_closure_allowed_denies_authority: true",
    "summary_md_artifact_manifest_live_authority_granted_denies_authority: true",
    "summary_md_artifact_manifest_verifier_readiness_matrix_update_allowed_denies_authority: true",
    "summary_md_artifact_manifest_verifier_readiness_closure_allowed_denies_authority: true",
    "summary_md_artifact_manifest_verifier_live_authority_granted_denies_authority: true",
    "summary_md_wing_readonly_handoff_plan_verifier_summary_surface_readiness_matrix_update_allowed_denies_authority: true",
    "summary_md_wing_readonly_handoff_plan_verifier_summary_surface_readiness_closure_allowed_denies_authority: true",
    "summary_md_wing_readonly_handoff_plan_verifier_summary_surface_live_authority_granted_denies_authority: true",
    "artifact_manifest_verifier_check_summary_md_readiness_closure_allowed_by_packet_denies_authority: true",
    "artifact_manifest_verifier_check_summary_md_live_authority_granted_denies_authority: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_contract_cross_surface_readiness_closure_allowed_denies_authority: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_contract_cross_surface_readiness_matrix_update_allowed_denies_authority: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_contract_cross_surface_live_authority_granted_denies_authority: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_readiness_matrix_update_allowed_denies_authority: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_readiness_closure_allowed_denies_authority: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_live_authority_granted_denies_authority: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_readiness_matrix_update_allowed_denies_authority: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_readiness_closure_allowed_denies_authority: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_live_authority_granted_denies_authority: true",
    "artifact_manifest_verifier_check_summary_md_wing_readonly_handoff_plan_verifier_summary_surface_readiness_matrix_update_allowed_denies_authority: true",
    "artifact_manifest_verifier_check_summary_md_wing_readonly_handoff_plan_verifier_summary_surface_readiness_closure_allowed_denies_authority: true",
    "artifact_manifest_verifier_check_summary_md_wing_readonly_handoff_plan_verifier_summary_surface_live_authority_granted_denies_authority: true",
    "summary_json_artifact_manifest_verifier_report_required_fields_matches_expected: true",
    "summary_md_artifact_manifest_verifier_report_required_fields_matches_expected: true",
    "manifest_artifact_manifest_verifier_report_required_fields_matches_expected: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_read_only_matches_expected: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_read_only_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_read_only_matches_expected: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_live_write_matches_expected: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_live_write_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_live_write_matches_expected: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_contacts_backend_matches_expected: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_contacts_backend_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_contacts_backend_matches_expected: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_executes_tests_matches_expected: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_executes_tests_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_executes_tests_matches_expected: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_contacts_wing_matches_expected: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_contacts_wing_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_contacts_wing_matches_expected: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_mutates_console_state_matches_expected: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_mutates_console_state_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_mutates_console_state_matches_expected: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_queries_console_matches_expected: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_queries_console_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_queries_console_matches_expected: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_readiness_matrix_update_allowed_matches_expected: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_readiness_matrix_update_allowed_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_readiness_matrix_update_allowed_matches_expected: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_readiness_closure_allowed_matches_expected: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_readiness_closure_allowed_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_readiness_closure_allowed_matches_expected: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_live_authority_granted_matches_expected: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_live_authority_granted_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_live_authority_granted_matches_expected: true",
    "verifier_report_required_fields_present: true",
    "verifier_report_readiness_closure_allowed_matches_expected: true",
    "verifier_report_live_authority_granted_matches_expected: true",
    "artifact_manifest_verifier_check_verifier_report_required_fields_missing_empty: true",
    "artifact_manifest_verifier_check_verifier_report_required_fields_match_contract: true",
    "artifact_manifest_verifier_check_verifier_report_artifact_manifest_self_consistency_match_matches_expected: true",
    "artifact_manifest_verifier_check_verifier_report_artifact_manifest_contract_cross_surface_match_matches_expected: true",
    "artifact_manifest_verifier_check_verifier_report_artifact_manifest_self_consistency_summary_surface_match_matches_expected: true",
    "artifact_manifest_verifier_check_verifier_report_read_only_matches_expected: true",
    "artifact_manifest_verifier_check_verifier_report_live_write_matches_expected: true",
    "artifact_manifest_verifier_check_verifier_report_contacts_wing_matches_expected: true",
    "artifact_manifest_verifier_check_verifier_report_contacts_backend_matches_expected: true",
    "artifact_manifest_verifier_check_verifier_report_mutates_console_state_matches_expected: true",
    "artifact_manifest_verifier_check_verifier_report_queries_console_matches_expected: true",
    "artifact_manifest_verifier_check_verifier_report_executes_tests_matches_expected: true",
    "artifact_manifest_verifier_check_verifier_report_readiness_matrix_update_allowed_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_wing_readonly_handoff_plan_verifier_summary_surface_fields_present: true",
    "artifact_manifest_verifier_check_manifest_wing_readonly_handoff_plan_verifier_summary_surface_required_matches_summary: true",
    "artifact_manifest_verifier_check_manifest_wing_readonly_handoff_plan_verifier_summary_surface_mismatches_matches_summary: true",
    "artifact_manifest_verifier_check_manifest_wing_readonly_handoff_plan_verifier_summary_surface_match_matches_summary: true",
    "artifact_manifest_verifier_check_manifest_wing_readonly_handoff_plan_verifier_summary_surface_readiness_matrix_update_allowed_matches_summary: true",
    "artifact_manifest_verifier_check_manifest_wing_readonly_handoff_plan_verifier_summary_surface_readiness_closure_allowed_matches_summary: true",
    "artifact_manifest_verifier_check_manifest_wing_readonly_handoff_plan_verifier_summary_surface_live_authority_granted_matches_summary: true",
    "artifact_manifest_verifier_check_manifest_wing_readonly_handoff_plan_verifier_summary_surface_required_is_boolean: true",
    "artifact_manifest_verifier_check_manifest_wing_readonly_handoff_plan_verifier_summary_surface_mismatches_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_wing_readonly_handoff_plan_verifier_summary_surface_match_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_wing_readonly_handoff_plan_verifier_summary_surface_readiness_matrix_update_allowed_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_wing_readonly_handoff_plan_verifier_summary_surface_readiness_closure_allowed_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_wing_readonly_handoff_plan_verifier_summary_surface_live_authority_granted_matches_expected: true",
    "artifact_manifest_verifier_check_verifier_report_wing_readonly_handoff_plan_verifier_summary_surface_required_is_boolean: true",
    "artifact_manifest_verifier_check_verifier_report_wing_readonly_handoff_plan_verifier_summary_surface_mismatches_matches_expected: true",
    "artifact_manifest_verifier_check_verifier_report_wing_readonly_handoff_plan_verifier_summary_surface_match_matches_expected: true",
    "artifact_manifest_verifier_check_verifier_report_wing_readonly_handoff_plan_verifier_summary_surface_readiness_matrix_update_allowed_matches_expected: true",
    "artifact_manifest_verifier_check_verifier_report_wing_readonly_handoff_plan_verifier_summary_surface_readiness_closure_allowed_matches_expected: true",
    "artifact_manifest_verifier_check_verifier_report_wing_readonly_handoff_plan_verifier_summary_surface_live_authority_granted_matches_expected: true",
    "verifier_report_required_fields_missing: []",
    "verifier_report_required_fields_match_contract: true",
    "failed_checks_have_check_entries: true",
    "artifact_manifest_verifier_check_failed_checks_missing_check_entries_empty: true",
    "failed_checks_missing_check_entries: []",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def unique_ordered(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def extract_evidence_files(evidence_text: str) -> list[str]:
    return unique_ordered(
        match.group(1)
        for match in EVIDENCE_PATH_RE.finditer(evidence_text)
        if "/" in match.group(1)
    )


def split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_blocker_table(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    in_table = False
    headers: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            if in_table and rows:
                break
            continue
        cells = split_markdown_row(line)
        if not headers:
            headers = [cell.lower() for cell in cells]
            in_table = True
            continue
        if set(cells) <= {"---", ":---", "---:", ":---:"}:
            continue
        if len(cells) != len(headers):
            continue
        row = dict(zip(headers, cells))
        if {"blocker", "status", "evidence", "next action"} <= set(row):
            rows.append(row)
    return rows


def first_open_blocker(rows: Sequence[dict[str, str]]) -> dict[str, str] | None:
    for row in rows:
        if row.get("status") != CLOSED_STATUS:
            return dict(row)
    return None


def markdown_section_text(path: Path, heading: str) -> str | None:
    section_lines: list[str] = []
    in_section = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == heading:
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if in_section and line:
            section_lines.append(line)
    if not section_lines:
        return None
    return " ".join(section_lines)


def current_first_command(path: Path) -> str | None:
    return markdown_section_text(path, CURRENT_FIRST_COMMAND_HEADING)


def current_first_command_missing_baseline_markers(
    current_first_command: str | None,
) -> list[str]:
    if not current_first_command:
        return list(CURRENT_FIRST_COMMAND_BASELINE_COVERAGE_MARKERS)
    return [
        marker
        for marker in CURRENT_FIRST_COMMAND_BASELINE_COVERAGE_MARKERS
        if marker not in current_first_command
    ]


def current_first_command_covers_baseline(current_first_command: str | None) -> bool:
    return not current_first_command_missing_baseline_markers(current_first_command)


def selected_action_text(
    *,
    blocker_next_action: str | None,
    current_first_command: str | None,
) -> tuple[str, str]:
    if current_first_command and blocker_next_action:
        if current_first_command == blocker_next_action:
            return current_first_command, "Current First Command"
        if current_first_command_covers_baseline(current_first_command):
            return current_first_command, "Current First Command"
        return (
            f"{current_first_command} Baseline blocker next action remains: "
            f"{blocker_next_action}",
            "Current First Command + blocker table baseline",
        )
    if current_first_command:
        return current_first_command, "Current First Command"
    if blocker_next_action:
        return blocker_next_action, "blocker table"
    return "All MVP blockers are closed.", "all blockers closed"


def build_progress_report(
    rows: Sequence[dict[str, str]],
    *,
    generated_at: str | None = None,
    current_first_command: str | None = None,
) -> str:
    generated = generated_at or utc_now()
    first_open = first_open_blocker(rows)
    total = len(rows)
    closed = sum(1 for row in rows if row.get("status") == CLOSED_STATUS)
    first_name = first_open["blocker"] if first_open else "none"
    next_action, next_action_source = selected_action_text(
        blocker_next_action=first_open["next action"] if first_open else None,
        current_first_command=current_first_command if first_open else None,
    )
    missing_markers = current_first_command_missing_baseline_markers(
        current_first_command if first_open else None
    )
    baseline_coverage_required = bool(
        first_open
        and current_first_command
        and current_first_command != first_open["next action"]
    )
    baseline_coverage_passed = bool(
        current_first_command and not missing_markers
    )
    selected_evidence_files = (
        extract_evidence_files(first_open["evidence"]) if first_open else []
    )

    lines = [
        "# MVP Progress Latest",
        "",
        f"Generated: `{generated}`",
        "",
        "## Current MVP",
        "",
        "Supervised WING Rack proof.",
        "",
        "## Summary",
        "",
        f"- Total blockers: {total}",
        f"- Closed blockers: {closed}",
        f"- Current first open blocker: {first_name}",
        f"- Next required action source: {next_action_source}",
        f"- Next required action: {next_action}",
        f"- Current First Command baseline coverage required: {str(baseline_coverage_required).lower()}",
        f"- Current First Command baseline coverage passed: {str(baseline_coverage_passed).lower()}",
        "- Current First Command baseline coverage marker count: "
        f"{len(CURRENT_FIRST_COMMAND_BASELINE_COVERAGE_MARKERS)}",
        "- Current First Command baseline missing markers: "
        f"{json.dumps(missing_markers)}",
        f"- Selected evidence file count: {len(selected_evidence_files)}",
        "",
        "## Blockers",
        "",
        "| Blocker | Status | Next action |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['blocker']} | {row['status']} | {row['next action']} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This report does not grant live authority. Offline tooling, shadow",
            "evidence, and readiness candidates are not live readiness. A blocker",
            "closes only after real WING artifacts are manually reviewed.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Update supervised WING MVP progress report.")
    parser.add_argument("--blockers", default=str(DEFAULT_BLOCKERS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--print", action="store_true", help="Print report instead of writing only.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    blockers_path = Path(args.blockers)
    rows = parse_blocker_table(blockers_path)
    if not rows:
        print(f"No blocker rows found in {blockers_path}", file=sys.stderr)
        return 2
    command_text = current_first_command(blockers_path)
    report = build_progress_report(rows, current_first_command=command_text)
    write_report(Path(args.output), report)
    if args.print:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
