# WING Read-Only Preflight Artifact Review

Generated: `2026-07-02T06:22:13Z`
Status: `candidate_ready_for_manual_review`

## Summary

- evidence_status: `readonly_preflight_candidate_available`
- artifact_count: `2`
- valid_candidate_count: `2`
- invalid_candidate_count: `0`
- readiness_update_candidate_available: `true`
- latest_valid_candidate_run_id: `20260702T061111Z_wing_readonly_preflight`
- artifact_operator_next_steps: `[]`

## Safety

- read_only: `true`
- live_write: `false`
- queries_console: `false`
- execution_allowed_by_review: `false`
- live_authority_granted: `false`
- full_autonomous_ready: `false`
- source_authority_denied: `true`
- source_authority_flags: `{"full_autonomous_ready": false, "live_authority_granted": false, "readiness_closure_allowed_by_evidence": false}`
- manual_review_required_before_readiness_update: `true`
- manual_review_decision_required: `true`
- request_id_policy: `required_for_every_query_mixer_parameter_response`
- request_id_correlation_required: `true`
- review_decision_template_available: `true`
- review_decision_template_status: `operator_must_replace_placeholders_before_ledger_acceptance`
- review_decision_template_required_edits: `["set decision to accepted or rejected after manual review", "set reviewed_by to the operator name", "set reviewed_at to a UTC timestamp at or after artifact_captured_at", "remove template_status or replace the generated pending marker before ledger acceptance"]`

## Manual Review Gate

- `confirm_artifact_paths`: Confirm summary_path and events_path point to the intended real WING run.
- `confirm_operator_channel_scope`: Confirm operator_confirmed_channels exactly match the queried channels.
- `confirm_operator_session_ready`: Confirm operator_confirmed_session_ready is true for the intended backend/WING session.
- `confirm_physical_readbacks`: Confirm every readback is verified_from_physical_readback for the declared value types.
- `confirm_event_log_readonly_sequence`: Confirm events.jsonl contains only get_operator_mode, get_state, and query_mixer_parameter sends.
- `confirm_event_log_run_id_binding`: Confirm every events.jsonl row has the same run_id as summary.json.
- `confirm_no_live_authority`: Confirm the review does not grant live authority or close readiness rows.

## Review Decision Template

- available: `true`
- required_fields: `["review_id", "artifact_source_id", "run_id", "artifact_captured_at", "artifact_root", "summary_path", "summary_sha256", "events_path", "events_sha256", "channels", "value_types", "operator_confirmed_channels", "operator_confirmed_session_ready", "readback_count", "physical_readback_count", "request_id_policy", "request_id_correlation_required", "event_log_run_id_matches_summary", "decision", "reviewed_by", "reviewed_at", "checklist_ids", "backend_port_prestart_check_required", "backend_port_prestart_check_must_pass_before_starting_backend", "backend_port_prestart_check_command", "backend_port_prestart_check_command_text", "backend_port_prestart_check_failure_detail_fields", "backend_port_prestart_summary_required", "backend_port_prestart_summary_path", "backend_port_prestart_summary_sha256", "backend_port_prestart_summary_type", "backend_port_prestart_summary_status", "backend_port_prestart_summary_connection_policy", "backend_port_prestart_summary_read_only", "backend_port_prestart_summary_live_write", "backend_port_prestart_summary_contacts_backend", "backend_port_prestart_summary_contacts_wing", "backend_port_prestart_summary_mutates_console_state", "backend_port_prestart_summary_queries_console", "backend_port_prestart_summary_executes_tests", "backend_port_prestart_check_status", "artifact_manifest_verifier_must_pass_before_websocket", "artifact_manifest_verifier_passed_before_websocket", "artifact_manifest_verifier_websocket_opened_after_pass", "safe_to_stop_listener_by_gate", "live_authority_granted", "backend_port_prestart_gate_execution_allowed", "backend_port_prestart_gate_readiness_closure_allowed", "backend_port_prestart_gate_live_authority_granted", "plan_safe_to_stop_listener_by_gate", "plan_live_authority_granted", "plan_backend_port_prestart_gate_execution_allowed", "plan_backend_port_prestart_gate_readiness_closure_allowed", "plan_backend_port_prestart_gate_live_authority_granted", "safe_to_stop_listener_by_gate_matches_plan", "live_authority_granted_matches_plan", "backend_port_prestart_gate_execution_allowed_matches_plan", "backend_port_prestart_gate_readiness_closure_allowed_matches_plan", "backend_port_prestart_gate_live_authority_granted_matches_plan"]`
- required_edits: `["set decision to accepted or rejected after manual review", "set reviewed_by to the operator name", "set reviewed_at to a UTC timestamp at or after artifact_captured_at", "remove template_status or replace the generated pending marker before ledger acceptance"]`
- output: `artifacts/wing_readonly_preflight_review_decisions/pending_review_decision_template.json`

## Blockers

- `operator_must_review_readonly_preflight_candidate_before_readiness_update`

## Violations

- none

## Artifact Sources

- `artifacts/wing_readonly_preflight/20260701T220146Z_wing_readonly_preflight/summary.json`
- `artifacts/wing_readonly_preflight/20260702T061111Z_wing_readonly_preflight/summary.json`

## Artifact Details

- source: `artifacts/wing_readonly_preflight/20260701T220146Z_wing_readonly_preflight/summary.json`
  - status: `valid_readonly_preflight_artifact`
  - run_id: `20260701T220146Z_wing_readonly_preflight`
  - captured_at: `2026-07-01T22:01:46.878Z`
  - artifact_root: `artifacts/wing_readonly_preflight`
  - summary_path: `artifacts/wing_readonly_preflight/20260701T220146Z_wing_readonly_preflight/summary.json`
  - summary_sha256: `0d434e94b667b28ecfbb0defa924393ed3a794b6746ce7a92f4709c24ff6acf4`
  - events_path: `artifacts/wing_readonly_preflight/20260701T220146Z_wing_readonly_preflight/events.jsonl`
  - events_sha256: `d833ae782144539a86ccd129ba87797872a3204e9710e82e37a0a6e6d34da2b0`
  - event_log_run_id_matches_summary: `true`
  - expected_artifacts: `["artifacts/wing_readonly_preflight/<run_id>/summary.json", "artifacts/wing_readonly_preflight/<run_id>/events.jsonl"]`
  - expected_summary_artifact: `artifacts/wing_readonly_preflight/20260701T220146Z_wing_readonly_preflight/summary.json`
  - expected_events_artifact: `artifacts/wing_readonly_preflight/20260701T220146Z_wing_readonly_preflight/events.jsonl`
  - summary_path_matches_expected_artifact: `true`
  - events_path_matches_expected_artifact: `true`
  - contacts_backend: `true`
  - queries_console: `true`
  - write_request_sent: `false`
  - proposal_request_sent: `false`
  - sent_request_types: `["get_operator_mode", "get_state", "query_mixer_parameter", "query_mixer_parameter", "query_mixer_parameter"]`
  - forbidden_request_types_sent: `[]`
  - channels: `[3]`
  - operator_confirmed_channels: `[3]`
  - operator_session_confirmation_required: `true`
  - operator_confirmed_session_ready: `true`
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
  - operator_next_step: `None`
  - value_types: `["fader", "gain", "pan"]`
  - readback_count: `3`
  - physical_readback_count: `3`
  - violations: `[]`
- source: `artifacts/wing_readonly_preflight/20260702T061111Z_wing_readonly_preflight/summary.json`
  - status: `valid_readonly_preflight_artifact`
  - run_id: `20260702T061111Z_wing_readonly_preflight`
  - captured_at: `2026-07-02T06:11:11.626Z`
  - artifact_root: `artifacts/wing_readonly_preflight`
  - summary_path: `artifacts/wing_readonly_preflight/20260702T061111Z_wing_readonly_preflight/summary.json`
  - summary_sha256: `102a70155ee24c86ee7dee242eb4fab7aef6cb17eed9048d1cb7112304d475bc`
  - events_path: `artifacts/wing_readonly_preflight/20260702T061111Z_wing_readonly_preflight/events.jsonl`
  - events_sha256: `62b841b6b1267dae8c3448d195005f7bbaf97cbb0169ed225c8fa3788b6d21cb`
  - event_log_run_id_matches_summary: `true`
  - expected_artifacts: `["artifacts/wing_readonly_preflight/<run_id>/summary.json", "artifacts/wing_readonly_preflight/<run_id>/events.jsonl"]`
  - expected_summary_artifact: `artifacts/wing_readonly_preflight/20260702T061111Z_wing_readonly_preflight/summary.json`
  - expected_events_artifact: `artifacts/wing_readonly_preflight/20260702T061111Z_wing_readonly_preflight/events.jsonl`
  - summary_path_matches_expected_artifact: `true`
  - events_path_matches_expected_artifact: `true`
  - contacts_backend: `true`
  - queries_console: `true`
  - write_request_sent: `false`
  - proposal_request_sent: `false`
  - sent_request_types: `["get_operator_mode", "get_state", "query_mixer_parameter", "query_mixer_parameter", "query_mixer_parameter", "query_mixer_parameter", "query_mixer_parameter", "query_mixer_parameter", "query_mixer_parameter", "query_mixer_parameter"]`
  - forbidden_request_types_sent: `[]`
  - channels: `[3]`
  - operator_confirmed_channels: `[3]`
  - operator_session_confirmation_required: `true`
  - operator_confirmed_session_ready: `true`
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
  - operator_next_step: `None`
  - value_types: `["eq_on", "eq_mix", "eq_low_gain", "eq_high_gain", "dyn_on", "dyn_threshold", "gate_on", "gate_threshold"]`
  - readback_count: `8`
  - physical_readback_count: `8`
  - violations: `[]`

## Next Required Action

Operator manually reviews summary.json and events.jsonl before readiness update.

## Boundary

This review is offline and read-only. It does not contact WING,
does not grant live authority, and does not close readiness rows
without operator review of real artifacts.
