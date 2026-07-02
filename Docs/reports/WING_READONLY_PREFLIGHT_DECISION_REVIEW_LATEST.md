# WING Read-Only Preflight Decision Review

Generated: `2026-07-02T06:56:50Z`
Status: `blocked`

## Summary

- ledger_status: `operator_review_required`
- preflight_evidence_status: `readonly_preflight_candidate_available`
- review_candidate_count: `2`
- decision_record_count: `0`
- accepted_record_count: `0`
- rejected_record_count: `0`
- invalid_record_count: `0`
- unreviewed_candidate_count: `2`
- manual_readiness_update_candidate_available: `false`
- accepted_candidate_prestart_contract_valid: `false`
- accepted_candidate_prestart_source_ids: `[]`
- accepted_candidate_prestart_contract_violations_by_source: `{}`
- accepted_candidate_verifier_ws_proof_valid: `false`
- accepted_candidate_verifier_ws_proof_source_ids: `[]`
- accepted_candidate_verifier_ws_proof_violations_by_source: `{}`
- accepted_candidate_artifact_root_contract_valid: `false`
- accepted_candidate_artifact_roots: `[]`
- accepted_candidate_artifact_root_contract_violations_by_source: `{}`
- accepted_candidate_request_sequence_contract_valid: `false`
- accepted_candidate_request_sequence_source_ids: `[]`
- accepted_candidate_request_sequence_contract_violations_by_source: `{}`
- accepted_candidate_contact_trace_complete: `false`
- accepted_candidate_contact_trace_source_ids: `[]`
- accepted_candidate_contacts_backend_by_source: `{}`
- accepted_candidate_queries_console_by_source: `{}`
- accepted_candidate_contact_trace_missing_source_ids: `[]`
- accepted_candidate_expected_artifact_trace_complete: `false`
- accepted_candidate_expected_artifact_trace_source_ids: `[]`
- accepted_candidate_expected_artifacts_by_source: `{}`
- accepted_candidate_expected_summary_artifact_by_source: `{}`
- accepted_candidate_expected_events_artifact_by_source: `{}`
- accepted_candidate_summary_path_matches_expected_artifact_by_source: `{}`
- accepted_candidate_events_path_matches_expected_artifact_by_source: `{}`
- accepted_candidate_expected_artifact_trace_missing_source_ids: `[]`

## Safety

- read_only: `true`
- live_write: `false`
- mutates_console_state: `false`
- queries_console: `false`
- executes_tests: `false`
- operator_required: `true`
- execution_allowed_by_decision: `false`
- readiness_matrix_update_allowed_by_decision: `false`
- readiness_closure_allowed_by_decision: `false`
- live_authority_granted: `false`
- full_autonomous_ready: `false`
- source_authority_denied: `true`
- source_authority_flags: `{"full_autonomous_ready": false, "live_authority_granted": false, "readiness_closure_allowed_by_decision": false, "readiness_matrix_update_allowed_by_decision": false}`

## Decisions

- accepted_source_ids: `[]`
- rejected_source_ids: `[]`
- invalid_record_ids: `[]`
- duplicate_source_ids: `[]`
- conflicting_source_ids: `[]`
- unreviewed_candidate_source_ids: `["artifacts/wing_readonly_preflight/20260701T220146Z_wing_readonly_preflight/summary.json", "artifacts/wing_readonly_preflight/20260702T061111Z_wing_readonly_preflight/summary.json"]`
- required_checklist_ids: `["confirm_artifact_paths", "confirm_operator_channel_scope", "confirm_operator_session_ready", "confirm_physical_readbacks", "confirm_event_log_readonly_sequence", "confirm_event_log_run_id_binding", "confirm_no_live_authority"]`

## Invalid Decision Records

- no invalid decision records

## Candidate Prestart Contract

- source: `artifacts/wing_readonly_preflight/20260701T220146Z_wing_readonly_preflight/summary.json`
  - status: `valid_readonly_preflight_artifact`
  - run_id: `20260701T220146Z_wing_readonly_preflight`
  - artifact_root: `artifacts/wing_readonly_preflight`
  - summary_path: `artifacts/wing_readonly_preflight/20260701T220146Z_wing_readonly_preflight/summary.json`
  - expected_artifacts: `["artifacts/wing_readonly_preflight/<run_id>/summary.json", "artifacts/wing_readonly_preflight/<run_id>/events.jsonl"]`
  - expected_summary_artifact: `artifacts/wing_readonly_preflight/20260701T220146Z_wing_readonly_preflight/summary.json`
  - expected_events_artifact: `artifacts/wing_readonly_preflight/20260701T220146Z_wing_readonly_preflight/events.jsonl`
  - summary_path_matches_expected_artifact: `true`
  - events_path_matches_expected_artifact: `true`
  - contacts_backend: `true`
  - queries_console: `true`
  - backend_port_prestart_check_required: `true`
  - backend_port_prestart_check_must_pass_before_starting_backend: `true`
  - backend_port_prestart_check_command: `["PYTHONPATH=backend", "python3", "tools/no_wing_release_check.py", "--quick"]`
  - backend_port_prestart_check_command_text: `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick`
  - backend_port_prestart_check_failure_detail_fields: `["listener_processes", "listener_process_error", "listener_confirmation_checklist", "preflight_blocked_until_listener_confirmed", "intended_backend_confirmation_required", "safe_to_stop_listener_by_gate", "listener_confirmation_template_status", "listener_confirmation_template_path", "listener_confirmation_template_acceptance_status", "listener_confirmation_template_safe_to_stop_listener_by_gate", "listener_confirmation_template_live_authority_granted", "listener_confirmation_template_accepted_by_template", "listener_confirmation_template_operator_placeholder_fields", "listener_confirmation_template_execution_allowed_by_template", "listener_confirmation_template_readiness_closure_allowed_by_template", "listener_confirmation_template_required_edits", "backend_port_prestart_gate_execution_allowed", "backend_port_prestart_gate_readiness_closure_allowed", "backend_port_prestart_gate_live_authority_granted", "fail_closed_reason", "operator_next_step"]`
  - backend_port_prestart_summary_required: `true`
  - backend_port_prestart_summary_path: `artifacts/no_wing_release_check/20260701T215031Z_no_wing_release_check/backend_port_prestart/summary.json`
  - backend_port_prestart_summary_sha256: `80506f5f377d923598e2d9292492fbb6a959705ed83893f6cdea62de38335148`
  - backend_port_prestart_summary_type: `no_wing_release_check`
  - backend_port_prestart_summary_status: `passed`
  - backend_port_prestart_summary_connection_policy: `no_hardware_no_network_no_live_writes`
  - backend_port_prestart_summary_read_only: `true`
  - backend_port_prestart_summary_live_write: `false`
  - backend_port_prestart_summary_contacts_backend: `false`
  - backend_port_prestart_summary_contacts_wing: `false`
  - backend_port_prestart_summary_mutates_console_state: `false`
  - backend_port_prestart_summary_queries_console: `false`
  - backend_port_prestart_summary_executes_tests: `false`
  - backend_port_prestart_check_status: `passed`
  - artifact_manifest_verifier_must_pass_before_websocket: `true`
  - artifact_manifest_verifier_passed_before_websocket: `true`
  - artifact_manifest_verifier_websocket_opened_after_pass: `true`
  - safe_to_stop_listener_by_gate: `false`
  - live_authority_granted: `false`
  - backend_port_prestart_gate_execution_allowed: `false`
  - backend_port_prestart_gate_readiness_closure_allowed: `false`
  - backend_port_prestart_gate_live_authority_granted: `false`
  - plan_safe_to_stop_listener_by_gate: `false`
  - plan_live_authority_granted: `false`
  - plan_backend_port_prestart_gate_execution_allowed: `false`
  - plan_backend_port_prestart_gate_readiness_closure_allowed: `false`
  - plan_backend_port_prestart_gate_live_authority_granted: `false`
  - safe_to_stop_listener_by_gate_matches_plan: `true`
  - live_authority_granted_matches_plan: `true`
  - backend_port_prestart_gate_execution_allowed_matches_plan: `true`
  - backend_port_prestart_gate_readiness_closure_allowed_matches_plan: `true`
  - backend_port_prestart_gate_live_authority_granted_matches_plan: `true`
- source: `artifacts/wing_readonly_preflight/20260702T061111Z_wing_readonly_preflight/summary.json`
  - status: `valid_readonly_preflight_artifact`
  - run_id: `20260702T061111Z_wing_readonly_preflight`
  - artifact_root: `artifacts/wing_readonly_preflight`
  - summary_path: `artifacts/wing_readonly_preflight/20260702T061111Z_wing_readonly_preflight/summary.json`
  - expected_artifacts: `["artifacts/wing_readonly_preflight/<run_id>/summary.json", "artifacts/wing_readonly_preflight/<run_id>/events.jsonl"]`
  - expected_summary_artifact: `artifacts/wing_readonly_preflight/20260702T061111Z_wing_readonly_preflight/summary.json`
  - expected_events_artifact: `artifacts/wing_readonly_preflight/20260702T061111Z_wing_readonly_preflight/events.jsonl`
  - summary_path_matches_expected_artifact: `true`
  - events_path_matches_expected_artifact: `true`
  - contacts_backend: `true`
  - queries_console: `true`
  - backend_port_prestart_check_required: `true`
  - backend_port_prestart_check_must_pass_before_starting_backend: `true`
  - backend_port_prestart_check_command: `["PYTHONPATH=backend", "python3", "tools/no_wing_release_check.py", "--quick"]`
  - backend_port_prestart_check_command_text: `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick`
  - backend_port_prestart_check_failure_detail_fields: `["listener_processes", "listener_process_error", "listener_confirmation_checklist", "preflight_blocked_until_listener_confirmed", "intended_backend_confirmation_required", "safe_to_stop_listener_by_gate", "listener_confirmation_template_status", "listener_confirmation_template_path", "listener_confirmation_template_acceptance_status", "listener_confirmation_template_safe_to_stop_listener_by_gate", "listener_confirmation_template_live_authority_granted", "listener_confirmation_template_accepted_by_template", "listener_confirmation_template_operator_placeholder_fields", "listener_confirmation_template_execution_allowed_by_template", "listener_confirmation_template_readiness_closure_allowed_by_template", "listener_confirmation_template_required_edits", "backend_port_prestart_gate_execution_allowed", "backend_port_prestart_gate_readiness_closure_allowed", "backend_port_prestart_gate_live_authority_granted", "fail_closed_reason", "operator_next_step"]`
  - backend_port_prestart_summary_required: `true`
  - backend_port_prestart_summary_path: `artifacts/no_wing_release_check/20260702T060916Z_no_wing_release_check/backend_port_prestart/summary.json`
  - backend_port_prestart_summary_sha256: `5df49ae579811b469b0c01f08325f66af6f08cc46c5c109d47079c11bc716e08`
  - backend_port_prestart_summary_type: `no_wing_release_check`
  - backend_port_prestart_summary_status: `passed`
  - backend_port_prestart_summary_connection_policy: `no_hardware_no_network_no_live_writes`
  - backend_port_prestart_summary_read_only: `true`
  - backend_port_prestart_summary_live_write: `false`
  - backend_port_prestart_summary_contacts_backend: `false`
  - backend_port_prestart_summary_contacts_wing: `false`
  - backend_port_prestart_summary_mutates_console_state: `false`
  - backend_port_prestart_summary_queries_console: `false`
  - backend_port_prestart_summary_executes_tests: `false`
  - backend_port_prestart_check_status: `passed`
  - artifact_manifest_verifier_must_pass_before_websocket: `true`
  - artifact_manifest_verifier_passed_before_websocket: `true`
  - artifact_manifest_verifier_websocket_opened_after_pass: `true`
  - safe_to_stop_listener_by_gate: `false`
  - live_authority_granted: `false`
  - backend_port_prestart_gate_execution_allowed: `false`
  - backend_port_prestart_gate_readiness_closure_allowed: `false`
  - backend_port_prestart_gate_live_authority_granted: `false`
  - plan_safe_to_stop_listener_by_gate: `false`
  - plan_live_authority_granted: `false`
  - plan_backend_port_prestart_gate_execution_allowed: `false`
  - plan_backend_port_prestart_gate_readiness_closure_allowed: `false`
  - plan_backend_port_prestart_gate_live_authority_granted: `false`
  - safe_to_stop_listener_by_gate_matches_plan: `true`
  - live_authority_granted_matches_plan: `true`
  - backend_port_prestart_gate_execution_allowed_matches_plan: `true`
  - backend_port_prestart_gate_readiness_closure_allowed_matches_plan: `true`
  - backend_port_prestart_gate_live_authority_granted_matches_plan: `true`

## Blockers

- `operator_must_record_readonly_preflight_review_decision`

## Next Required Action

Operator records a review decision for each clean preflight candidate.

## Boundary

This review is offline and read-only. It does not contact WING,
does not grant live authority, and does not close readiness rows.
