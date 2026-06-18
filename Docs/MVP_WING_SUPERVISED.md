# MVP: Supervised WING Rack Proof

Status: current execution contract

## Purpose

This is the current MVP rail for Automixer. The long-term goal is a capable AI
mixing engineer, but the current project milestone is narrower: prove one safe
supervised hardware loop on WING Rack before broad autonomy work continues.

Goal-mode Codex automation must also follow
`.ai/project-layer/autonomous_mvp_and_model_routing.md` and
`.ai/agent-core/model_budget_policy.md`. Those files preserve the full project
goal while preventing daily/weekly/monthly model budget from being consumed by
broad, high-reasoning work before the supervised WING proof is ready.

## In Scope

The MVP includes only:

1. WING read-only connection proof.
2. Physical console readback proof.
3. Supervised fader write proof with explicit operator approval.
4. Supervised gain-down / trim proof with explicit operator approval.
5. Rollback proof.
6. Emergency-stop proof.
7. Connection-loss / timeout fail-closed proof.
8. Operator-visible evidence report.
9. Readiness matrix update based only on real proof.

## Out Of Scope

These are not part of the current MVP:

- Full autonomous live mixing.
- Live EQ, compressor, gate, routing, mute, FX, scene, matrix, or DCA writes.
- New broad proposal-only surfaces.
- New shadow-only methods that do not close a blocker below.
- New memory-trust, queue-import, result-consumer, report-only, or read-model
  chains that do not directly close a blocker below.
- Measurement-microphone / room-tuning workflows. They are useful later, but
  not required for the current console proof gate.

Correction-method development for Gain/Trim, Pan, EQ, Comp, FX, Balance, and
related families continues through `Docs/CORRECTION_METHODS_ROADMAP.md`, but
before this MVP is proven those methods must remain offline, dry-run,
proposal-only, read-only, and tested. They must not widen live writes.

## Source Of Truth

Use these files in this order:

1. `AGENTS.md`
2. `CLAUDE.md`
3. `Docs/MVP_WING_SUPERVISED.md`
4. `.ai/project-layer/wing_mvp_blockers.md`
5. `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
6. `tools/wing_readonly_preflight_runner.py`
7. `tools/review_wing_readonly_preflight_artifacts.py`
8. `backend/wing_readonly_preflight_review_decision.py`
9. `backend/supervised_wing_proof_plan.py`
10. `tools/no_wing_release_check.py`
11. `Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md`
12. `Docs/reports/MVP_PROGRESS_LATEST.md`
13. `Docs/CORRECTION_METHODS_ROADMAP.md`
14. `.ai/project-layer/correction_methods_backlog.md`

## Definition Of Done

The MVP is done only when every row in
`.ai/project-layer/wing_mvp_blockers.md` is `closed` with real proof artifacts.

Offline tests, dry-run artifacts, shadow-only evidence, valid report shape, or
readiness candidates are not enough to close a blocker by themselves.

## Correct Work Order

1. Run read-only preflight against the intended WING backend.
2. Verify physical readback from the artifact and event log.
3. Review the clean preflight candidate manually.
4. Record an accepted operator decision JSON for that candidate.
5. Run supervised fader proof only with explicit operator approval.
6. Run supervised gain-down / trim proof only with explicit operator approval.
7. Prove rollback.
8. Prove emergency stop.
9. Prove connection-loss / timeout fail-closed behavior.
10. Update the readiness matrix from real evidence.

## Stop-The-Line Rules

Stop the current Codex work direction and return to the first open blocker if:

- `tools/codex_autoloop/codex_direction_guard.py` reports `drifting`.
- The work adds memory/queue/result-consumer/report-only layers without closing
  a blocker.
- The work treats candidate, shadow, offline, or dry-run evidence as live
  readiness.
- A supervised proof is planned before clean read-only preflight evidence is
  available for manual review.
- A supervised proof is planned before the clean read-only preflight candidate
  has an accepted review decision record.
- A supervised proof is planned while accepted review source ids lack a valid
  accepted-candidate prestart trace, or while the proof plan still carries
  `accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof`.
- A supervised proof is planned while accepted review source ids lack a valid
  accepted-candidate verifier WebSocket proof, or while the proof plan still carries
  `accepted_readonly_preflight_candidate_verifier_ws_proof_required_before_supervised_proof`.
- A supervised proof is planned while accepted review source ids lack a valid
  accepted-candidate artifact-root trace, or while the proof plan still carries
  `accepted_readonly_preflight_candidate_artifact_root_required_before_supervised_proof`.
- A supervised proof is planned while accepted review source ids lack a valid
  accepted-candidate request-sequence trace, or while the proof plan still carries
  `accepted_readonly_preflight_candidate_request_sequence_required_before_supervised_proof`.
- A supervised proof is planned while accepted review source ids lack a valid
  accepted-candidate contact trace, or while the proof plan still carries
  `accepted_readonly_preflight_candidate_contact_trace_required_before_supervised_proof`.
- A supervised proof is planned while
  `accepted_candidate_prestart_contract_violations_by_source` or
  `readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source`
  is non-empty.
- A supervised proof is planned while the accepted-candidate prestart trace does
  not bind to a passed no-WING prestart `summary.json` by path and SHA256, with
	  `backend_port_prestart_summary_required: true`,
	  `backend_port_prestart_summary_type: no_wing_release_check`,
	  `backend_port_prestart_summary_status: passed`,
	  `backend_port_prestart_summary_connection_policy: no_hardware_no_network_no_live_writes`,
	  `backend_port_prestart_summary_read_only: true`,
	  `backend_port_prestart_summary_live_write: false`,
	  `backend_port_prestart_summary_contacts_backend: false`,
	  `backend_port_prestart_summary_contacts_wing: false`,
	  `backend_port_prestart_summary_mutates_console_state: false`,
	  `backend_port_prestart_summary_queries_console: false`,
	  `backend_port_prestart_summary_executes_tests: false`, and
	  `backend_port_prestart_check_status: passed`.
- A supervised proof is planned while
  `accepted_candidate_artifact_root_contract_violations_by_source` or
  `readonly_preflight_review_decision_candidate_artifact_root_contract_violations_by_source`
  is non-empty.
- A supervised proof is planned while the no-WING handoff summary does not show
  command bindings to the packet roots:
  `preview_command_artifact_root_matches_packet: true`,
  `preview_command_plan_output_matches_packet: true`,
  `validate_inputs_command_uses_validate_run_inputs_only: true`,
  `validate_inputs_command_uses_plan_only: false`,
  `validate_inputs_command_artifact_root_matches_packet: true`,
  `validate_inputs_command_expected_plan_matches_packet: true`,
  `capture_command_artifact_root_matches_packet: true`,
  `capture_command_expected_plan_matches_packet: true`,
  `review_command_artifact_root_matches_packet: true`,
  `decision_review_command_artifact_root_matches_packet: true`, and
  `decision_review_command_decision_root_matches_packet: true`.
- A supervised proof is planned while the no-WING handoff summary does not show
  `backend_port_prestart_summary_sha256_matches_plan: true`.
- A supervised proof is planned while the no-WING handoff summary does not show
  `backend_port_prestart_summary_sha256_available: true` and
  `backend_port_prestart_summary_sha256_available_matches_digest: true`.
- A supervised proof is planned while the no-WING handoff summary does not show
  `required_supervised_proof_plan_summary_fields_include_invalid_records: true`.
- A supervised proof is planned while the no-WING handoff summary does not show
  `project_director_selected_action_value_gate_self_check_required: true`,
  `project_director_selected_action_value_gate_self_check_missing_markers: []`,
  `project_director_selected_action_value_gate_self_check_markers_present: true`,
  `project_director_selected_action_value_gate_self_check_readiness_closure_allowed: false`,
  and
  `project_director_selected_action_value_gate_self_check_live_authority_granted: false`.
- A supervised proof is planned while the no-WING operator Markdown does not
  show
  `operator_markdown_summary_required_field_lines` containing
  `readonly_preflight_review_decision_candidate_verifier_ws_proof_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]`,
  `readonly_preflight_review_decision_candidate_verifier_ws_proof_violations_by_source: {}`,
  `readonly_preflight_review_decision_candidate_verifier_ws_proof_complete: true`,
  `readonly_preflight_review_decision_candidate_request_sequence_contract_valid: true`,
  `readonly_preflight_review_decision_candidate_request_sequence_contract_violations_by_source: {}`,
  `readonly_preflight_review_decision_expected_candidate_request_sequence_contract_violations_by_source: {}`,
  `readonly_preflight_review_decision_accepted_candidate_request_sequence_by_source`,
  `readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_complete`,
  `next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_summary_surface_readiness_closure_allowed`,
  `next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_summary_surface_live_authority_granted`,
  `next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_readiness_closure_allowed`
  and
  `next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_live_authority_granted`,
  `operator_markdown_summary_required_field_lines_missing_members: []`, and
  `operator_markdown_summary_required_field_lines_match_contract: true`,
  `project_director_selected_action_value_gate_self_check_markdown_surface_required: true`,
  `project_director_selected_action_value_gate_self_check_markdown_surface_missing_field_lines: []`,
  `project_director_selected_action_value_gate_self_check_markdown_surface_field_lines_rendered: true`,
  `project_director_selected_action_value_gate_self_check_markdown_surface_readiness_closure_allowed: false`,
  and
  `project_director_selected_action_value_gate_self_check_markdown_surface_live_authority_granted: false`.
- A supervised proof is planned while the no-WING summary surface does not show
  `artifact_manifest_summary_json_contract_sha256_matches_manifest: true`,
  `artifact_manifest_summary_md_contract_sha256_matches_manifest: true`,
  `artifact_manifest_contract_cross_surface_mismatches: []`, and
  `artifact_manifest_contract_cross_surface_match: true`.
- A supervised proof is planned while the no-WING handoff summary does not show
  `backend_port_prestart_summary_type_matches_plan: true`,
  `backend_port_prestart_summary_status_matches_plan: true`,
  `backend_port_prestart_summary_connection_policy_matches_plan: true`,
  `backend_port_prestart_summary_read_only_matches_plan: true`,
  `backend_port_prestart_summary_live_write_matches_plan: true`,
  `backend_port_prestart_summary_contacts_backend_matches_plan: true`,
  `backend_port_prestart_summary_contacts_wing_matches_plan: true`,
  `backend_port_prestart_summary_mutates_console_state_matches_plan: true`,
  `backend_port_prestart_summary_queries_console_matches_plan: true`,
  `backend_port_prestart_summary_executes_tests_matches_plan: true`,
  `plan_safe_to_stop_listener_by_gate: false`,
  `plan_live_authority_granted: false`,
  `plan_backend_port_prestart_gate_execution_allowed: false`,
  `plan_backend_port_prestart_gate_readiness_closure_allowed: false`,
  `plan_backend_port_prestart_gate_live_authority_granted: false`,
  `backend_port_prestart_check_status_matches_plan: true`,
  `safe_to_stop_listener_by_gate_matches_plan: true`,
  `live_authority_granted_matches_plan: true`,
  `backend_port_prestart_gate_execution_allowed_matches_plan: true`,
  `backend_port_prestart_gate_readiness_closure_allowed_matches_plan: true`, and
  `backend_port_prestart_gate_live_authority_granted_matches_plan: true`;
  it must also show
  `declared_safe_to_stop_listener_by_gate_matches_plan: true`,
  `declared_live_authority_granted_matches_plan: true`,
  `declared_backend_port_prestart_gate_execution_allowed_matches_plan: true`,
  `declared_backend_port_prestart_gate_readiness_closure_allowed_matches_plan: true`,
  and
  `declared_backend_port_prestart_gate_live_authority_granted_matches_plan: true`.
- A supervised proof is planned while the no-WING handoff summary does not show
  `operator_sequence`,
  `expected_operator_sequence`,
  `expected_operator_sequence_count: 12`,
  `operator_sequence_count_matches_contract: true`,
  `operator_sequence_sha256`,
  `actual_operator_sequence_sha256`,
  `operator_sequence_sha256_matches_actual: true`,
  `expected_operator_sequence_sha256`,
  `actual_expected_operator_sequence_sha256`,
  `operator_sequence_matches_contract: true`,
  `operator_sequence_sha256_matches_contract: true`,
  `declared_expected_operator_sequence_sha256_matches_contract: true`,
  `declared_operator_sequence_matches_contract_flag: true`,
  `declared_operator_sequence_matches_contract_flag_matches_actual: true`,
  `declared_operator_sequence_sha256_matches_contract_flag: true`,
  `declared_operator_sequence_sha256_matches_contract_flag_matches_actual: true`,
  `declared_operator_sequence_count: 12`,
  `declared_expected_operator_sequence_count: 12`,
  `declared_operator_sequence_count_contract_flag: true`,
  `declared_operator_sequence_count_matches_actual: true`,
  `declared_operator_sequence_count_matches_contract: true`,
  `declared_expected_operator_sequence_count_matches_contract: true`,
  `declared_operator_sequence_count_contract_flag_matches_declared_count: true`,
  `expected_mvp_blocker_anchor_digest_review_step_index: 10`,
  `mvp_blocker_anchor_digest_review_step_index_matches_contract: true`,
  `mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true`,
  `expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10`,
  `operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true`,
  `expected_operator_sequence_supervised_prestart_gate_step_index: 11`, and
  `operator_sequence_supervised_prestart_gate_step_index_matches_contract: true`.
- A supervised proof is planned before the no-WING handoff summary shows
  `backend_port_prestart_gate` exactly matching
  `expected_backend_port_prestart_gate`.
- A supervised proof is planned while the no-WING handoff summary does not show
  `backend_port_prestart_gate_matches_expected: true` and
  `declared_backend_port_prestart_gate_matches_expected: true`.
- A supervised proof is planned while `backend_port_prestart_gate` claims
  `grants_live_authority: true`, or while
  `backend_port_prestart_gate.must_pass_before_backend_wing_session` is not
  `true`.
- A supervised proof is planned while `backend_port_prestart_gate` does not show
  `operator_sequence_step_index: 0` and
  `operator_sequence_step_must_be_first: true`.
- A supervised proof is planned while `backend_port_prestart_gate` does not show
  `runs_command: prestart_backend_port_check`.
- A supervised proof is planned while `backend_port_prestart_gate` does not
  exactly list `required_markers` for
  `Before starting the intended backend/WING session`,
  `prestart_backend_port_check`, and `unexpected listeners`.
- A supervised proof is planned while the failed busy-port no-WING summary or
  no-WING handoff summary does not show
  `backend_port_prestart_check_command` exactly matching
  `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick`.
- A supervised proof is planned while the failed busy-port no-WING summary or
  no-WING handoff summary does not show
  `backend_port_prestart_check_command_text` exactly matching
  `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick`.
- A supervised proof is planned while the failed busy-port no-WING summary or
  no-WING handoff summary does not show
  `backend_port_prestart_check_required: true` and
  `backend_port_prestart_check_must_pass_before_starting_backend: true`.
- A supervised proof is planned while the failed busy-port no-WING summary or
  no-WING handoff summary does not show
  `backend_port_prestart_gate_execution_allowed: false`,
  `backend_port_prestart_gate_readiness_closure_allowed: false`, and
  `backend_port_prestart_gate_live_authority_granted: false`.
- A supervised proof is planned while the failed busy-port no-WING summary or
  no-WING handoff summary does not show
  `backend_port_prestart_check_failure_detail_fields` exactly listing
  `listener_processes`, `listener_process_error`, `listener_confirmation_checklist`,
  `preflight_blocked_until_listener_confirmed`,
  `intended_backend_confirmation_required`, `safe_to_stop_listener_by_gate`,
  `listener_confirmation_template_status`,
  `listener_confirmation_template_path`,
  `listener_confirmation_template_acceptance_status`,
  `listener_confirmation_template_safe_to_stop_listener_by_gate`,
  `listener_confirmation_template_live_authority_granted`,
  `listener_confirmation_template_accepted_by_template`,
  `listener_confirmation_template_operator_placeholder_fields`,
  `listener_confirmation_template_execution_allowed_by_template`,
  `listener_confirmation_template_readiness_closure_allowed_by_template`,
  `listener_confirmation_template_required_edits`,
  `backend_port_prestart_gate_execution_allowed`,
  `backend_port_prestart_gate_readiness_closure_allowed`,
  `backend_port_prestart_gate_live_authority_granted`,
  `fail_closed_reason`, and `operator_next_step`.
- A supervised proof is planned while no-WING quick reports
  `environment_blocked_checks` without
  `environment_blocked_check_details`, `local_dev_server_bind_denied`,
  `127.0.0.1:3000`, log paths, and `operator_next_step`. Such a no-WING result
  remains failed until rerun in an environment that can bind the frontend dev
  server.
- A WING read-only preflight is planned from an existing backend listener while
  the failed no-WING prestart summary lacks a pending
  `listener_confirmation_template_path`, or while the generated template does
  not show `listener_confirmation_template_status: pending_operator_edit`,
  `listener_confirmation_template_safe_to_stop_listener_by_gate: false`, and
  `listener_confirmation_template_live_authority_granted: false`, or while the
  failed summary does not show
  `listener_confirmation_template_accepted_by_template: false` and
  `listener_confirmation_template_operator_placeholder_fields`, or while it
  does not show
  `listener_confirmation_template_execution_allowed_by_template: false` and
  `listener_confirmation_template_readiness_closure_allowed_by_template: false`,
  or while
  `listener_confirmation_template_required_edits` does not include
  `set template_status to operator_reviewed_and_edited if using the generated
  template` and `set accepted_by_template to true if using the generated
  template`.
- A WING read-only preflight is planned from a template-derived accepted
  backend-listener confirmation while `REPLACE_WITH_*` operator placeholders
  remain, `template_status: operator_reviewed_and_edited` is missing, or
  `accepted_by_template: true` is missing.
- A readiness update is planned while the no-WING handoff summary does not show
  `manual_review_required_before_readiness_update: true`.
- A readiness update is planned while the no-WING handoff summary does not show
  `readiness_matrix_update_allowed_by_packet: false`.
- A readiness closure is planned while the no-WING handoff summary does not
  show `readiness_closure_allowed_by_packet: false`.
- A failed no-WING handoff summary hides unsafe packet values by rendering
  `execution_allowed_by_packet`, `readiness_matrix_update_allowed_by_packet`,
  `readiness_closure_allowed_by_packet`, `live_authority_granted`, or
  `full_autonomous_ready` as `false` when the failed packet set them to `true`.
- A failed no-WING handoff summary hides unsafe safety flags by rendering
  `live_write`, `mutates_console_state`, `queries_console`, or `executes_tests`
  as `false` when the failed packet set them to `true`.
- A supervised proof is planned while the no-WING handoff summary does not show
  `read_only: true`.
- A supervised proof is planned while the no-WING handoff summary does not show
  `live_write: false`.
- A supervised proof is planned while the no-WING handoff summary does not show
  `mutates_console_state: false`.
- A supervised proof is planned while the no-WING handoff summary does not show
  `queries_console: false`.
- A supervised proof is planned while the no-WING handoff summary does not show
  `executes_tests: false`.
- A supervised proof is planned while the no-WING handoff summary does not show
  `operator_required: true`.
- A supervised proof is planned while the no-WING handoff summary does not show
  `expected_plan_required_for_real_run: true`.
- A supervised proof is planned while the no-WING handoff summary does not show
  `physical_preflight_requires_existing_backend_connection: true`.
- A supervised proof is planned while the no-WING handoff summary does not show
  `backend_port_prestart_summary_required: true`.
- A supervised proof is planned while the no-WING handoff summary does not show
  `backend_port_prestart_summary_path_placeholder` pointing to the passed
  no-WING prestart `summary.json`.
- A supervised proof is planned while the no-WING handoff summary does not show
  `preview_command_prestart_summary`, `validate_inputs_command_prestart_summary`,
  and `capture_command_prestart_summary` matching that same passed no-WING
  prestart `summary.json`.
- A supervised proof is planned while the no-WING handoff summary does not show
  `preview_command_prestart_summary_matches_plan: true`,
  `validate_inputs_command_prestart_summary_matches_plan: true`, and
  `capture_command_prestart_summary_matches_plan: true`.
- A supervised proof is planned while the no-WING handoff summary does not show
  `execution_allowed_by_packet: false`.
- A supervised proof is planned while the no-WING handoff summary does not show
  `live_authority_granted: false`.
- A supervised proof is planned while the no-WING handoff summary does not show
  `full_autonomous_ready: false`.
- A supervised proof is planned while the no-WING handoff summary does not show
  `next_manual_artifact_required: real wing_readonly_preflight.v1 summary.json`.
- A supervised proof is planned while the no-WING handoff summary does not show
  `expected_artifacts` exactly matching `expected_artifact_globs` and
  `plan_expected_artifacts`.
- A supervised proof is planned while the no-WING handoff summary does not show
  `expected_artifacts_match_plan: true`, `summary_glob_matches_plan: true`,
  and `event_log_glob_matches_plan: true`.
- A supervised proof is planned before the no-WING handoff summary shows
  `supervised_prestart_gate` exactly matching
  `expected_supervised_prestart_gate`.
- A supervised proof is planned while the no-WING handoff summary does not show
  `supervised_prestart_gate_matches_expected: true` and
  `declared_supervised_prestart_gate_matches_expected: true`.
- A supervised proof is planned while `supervised_prestart_gate` claims
  `grants_live_authority: true`.
- A supervised proof is planned while `supervised_prestart_gate` does not show
  `operator_sequence_step_index: 9` and
  `operator_sequence_step_must_be_final: true`.
- A supervised proof is planned while the no-WING handoff summary does not show
  `mvp_blocker_anchor_digest_review_step_index: 10`,
  `expected_mvp_blocker_anchor_digest_review_step_index: 10`,
  `mvp_blocker_anchor_digest_review_step_index_matches_contract: true`,
  `mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true`,
  `operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10`,
  `expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10`,
  `operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true`,
  and
  `operator_sequence_supervised_prestart_gate_step_index: 11`.
- A supervised proof is planned while the no-WING handoff summary does not show
  `operator_sequence_count: 12`.
- A supervised proof is planned while the no-WING handoff summary does not show
  `declared_operator_sequence_count: 12`,
  `declared_expected_operator_sequence_count: 12`,
  `declared_operator_sequence_count_contract_flag: true`,
  `declared_operator_sequence_count_matches_actual: true`,
  `declared_operator_sequence_count_matches_contract: true`,
  `declared_expected_operator_sequence_count_matches_contract: true`, and
  `declared_operator_sequence_count_contract_flag_matches_declared_count: true`.
- A supervised proof is planned while
  `backend_port_prestart_check_operator_sequence_step` does not include
  `prestart_backend_port_check` and `unexpected listeners`.

This is not a stop to development. Codex must continue uninterrupted Goal-mode
work, but only on the first non-closed MVP blocker.

## Current Next Action

Run the read-only preflight runbook when WING Rack and backend are ready:

`Docs/runbooks/WING_READONLY_PREFLIGHT.md`

Before any supervised proof, accepted review source ids must have a valid
accepted-candidate prestart trace from read-only preflight review. If that trace
is missing or invalid, the proof plan must keep
`accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof`
as an operator blocker.
The same accepted source ids must also have a valid accepted-candidate verifier
WebSocket proof before supervised proof planning; otherwise the proof plan must
keep
`accepted_readonly_preflight_candidate_verifier_ws_proof_required_before_supervised_proof`
as an operator blocker.
That trace must include `expected_artifacts`, `expected_summary_artifact`,
`expected_events_artifact`, `summary_path_matches_expected_artifact: true`,
`events_path_matches_expected_artifact: true`, and a path plus matching SHA256
for the passed no-WING prestart `summary.json`; a candidate that only repeats
the prestart field names without the bound artifacts remains blocked.
The decision-review CLI must not report `accepted_decision_ready` until the
same accepted source ids have that valid accepted-candidate prestart trace.
The decision-review ledger/report must expose
`accepted_candidate_prestart_contract_violations_by_source`, and the supervised
proof plan summary must expose
`readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source`.
Both maps must be `{}` before supervised proof steps are planned; no-WING
validation must fail with
`next_wing_proof_plan_steps_with_candidate_prestart_violations` if the proof
plan carries a non-empty map while proof steps are present.
The recomputed
`readonly_preflight_review_decision_expected_candidate_prestart_contract_violations_by_source: {}`
value must also stay true; no-WING validation must fail with
`next_wing_proof_plan_steps_with_expected_candidate_prestart_violations` if
that expected map is non-empty while proof steps are present.
No-WING validation must also fail with
`next_wing_proof_plan_steps_with_candidate_artifact_root_violations` if the
proof plan carries a non-empty accepted-candidate artifact-root violations map
while proof steps are present.
The recomputed
`readonly_preflight_review_decision_expected_candidate_artifact_root_contract_violations_by_source: {}`
value must also stay true; no-WING validation must fail with
`next_wing_proof_plan_steps_with_expected_candidate_artifact_root_violations`
if that expected map is non-empty while proof steps are present.

The no-WING handoff summary must also show `backend_port_prestart_gate` exactly
matching `expected_backend_port_prestart_gate`, with
`backend_port_prestart_gate.grants_live_authority: false` and
`backend_port_prestart_gate.must_pass_before_backend_wing_session: true`, before
any supervised fader/gain proof runbook is used.

The same `backend_port_prestart_gate` must show
`backend_port_prestart_gate.operator_sequence_step_index: 0` and
`backend_port_prestart_gate.operator_sequence_step_must_be_first: true`.

The same `backend_port_prestart_gate` must also show
`backend_port_prestart_gate.runs_command: prestart_backend_port_check`.

The same `backend_port_prestart_gate` must also show
`backend_port_prestart_gate.required_markers` exactly listing
`Before starting the intended backend/WING session`,
`prestart_backend_port_check`, and `unexpected listeners`.

The failed busy-port no-WING summary and no-WING handoff summary must both show
`backend_port_prestart_check_command` exactly matching
`PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick`.

The failed busy-port no-WING summary and no-WING handoff summary must both show
`backend_port_prestart_check_command_text` exactly matching
`PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick`.

The same summaries must also show
`backend_port_prestart_check_required: true` and
`backend_port_prestart_check_must_pass_before_starting_backend: true`.

The same summaries must also show
`backend_port_prestart_gate_execution_allowed: false`,
`backend_port_prestart_gate_readiness_closure_allowed: false`, and
`backend_port_prestart_gate_live_authority_granted: false`.

The same summaries must also show
`backend_port_prestart_check_failure_detail_fields` exactly listing
`listener_processes`, `listener_process_error`, `listener_confirmation_checklist`,
`preflight_blocked_until_listener_confirmed`,
`intended_backend_confirmation_required`, `safe_to_stop_listener_by_gate`,
`listener_confirmation_template_status`,
`listener_confirmation_template_path`,
`listener_confirmation_template_acceptance_status`,
`listener_confirmation_template_safe_to_stop_listener_by_gate`,
`listener_confirmation_template_live_authority_granted`,
`listener_confirmation_template_accepted_by_template`,
`listener_confirmation_template_operator_placeholder_fields`,
`listener_confirmation_template_execution_allowed_by_template`,
`listener_confirmation_template_readiness_closure_allowed_by_template`,
`listener_confirmation_template_required_edits`,
`backend_port_prestart_gate_execution_allowed`,
`backend_port_prestart_gate_readiness_closure_allowed`,
`backend_port_prestart_gate_live_authority_granted`, `fail_closed_reason`, and
`operator_next_step`.

If no-WING quick reports `environment_blocked_checks`, the no-WING summary must
also expose `environment_blocked_check_details` with
`local_dev_server_bind_denied`, `127.0.0.1:3000`, log paths, and
`operator_next_step`; that result remains failed until rerun in an environment
that can bind the frontend dev server.

When the prestart gate fails closed on an existing listener, the no-WING summary
must expose the generated `listener_confirmation_template_path` and show
`listener_confirmation_template_status: pending_operator_edit`,
`listener_confirmation_template_safe_to_stop_listener_by_gate: false`, and
`listener_confirmation_template_live_authority_granted: false`.
It must also show `listener_confirmation_template_accepted_by_template: false`
and `listener_confirmation_template_operator_placeholder_fields` listing
`operator_reviewed_at` and `operator_name`.
It must also show
`listener_confirmation_template_execution_allowed_by_template: false` and
`listener_confirmation_template_readiness_closure_allowed_by_template: false`.
`listener_confirmation_template_required_edits` must include
`set template_status to operator_reviewed_and_edited if using the generated
template` and `set accepted_by_template to true if using the generated
template`, and any template-derived accepted confirmation must replace
`REPLACE_WITH_*` operator placeholders, set
`template_status: operator_reviewed_and_edited`, and set
`accepted_by_template: true`. That pending template is not an accepted
confirmation and cannot close readiness.

The no-WING handoff summary must also show
`manual_review_required_before_readiness_update: true`.

The no-WING handoff summary must also show
`readiness_matrix_update_allowed_by_packet: false`.

The no-WING handoff summary must also show
`readiness_closure_allowed_by_packet: false`.

The no-WING handoff summary must also show `read_only: true`.

The no-WING handoff summary must also show `live_write: false`.

The no-WING handoff summary must also show `mutates_console_state: false`.

The no-WING handoff summary must also show `queries_console: false`.

The no-WING handoff summary must also show `executes_tests: false`.

The no-WING handoff summary must also show `operator_required: true`.

The no-WING handoff summary must also show
`expected_plan_required_for_real_run: true`.

The no-WING handoff summary must also show
`physical_preflight_requires_existing_backend_connection: true`.

The no-WING handoff summary must also show the same
`backend_port_prestart_check_required: true` and
`backend_port_prestart_check_must_pass_before_starting_backend: true`.

The no-WING handoff summary must also show
`backend_port_prestart_gate_execution_allowed: false`,
`backend_port_prestart_gate_readiness_closure_allowed: false`, and
`backend_port_prestart_gate_live_authority_granted: false`.

The no-WING handoff summary must also show
`plan_safe_to_stop_listener_by_gate: false`,
`plan_live_authority_granted: false`,
`plan_backend_port_prestart_gate_execution_allowed: false`,
`plan_backend_port_prestart_gate_readiness_closure_allowed: false`, and
`plan_backend_port_prestart_gate_live_authority_granted: false`.

The no-WING handoff summary must also show
`safe_to_stop_listener_by_gate_matches_plan: true`,
`live_authority_granted_matches_plan: true`,
`backend_port_prestart_gate_execution_allowed_matches_plan: true`,
`backend_port_prestart_gate_readiness_closure_allowed_matches_plan: true`, and
`backend_port_prestart_gate_live_authority_granted_matches_plan: true`.

The no-WING handoff summary must also show
`declared_safe_to_stop_listener_by_gate_matches_plan: true`,
`declared_live_authority_granted_matches_plan: true`,
`declared_backend_port_prestart_gate_execution_allowed_matches_plan: true`,
`declared_backend_port_prestart_gate_readiness_closure_allowed_matches_plan: true`,
and
`declared_backend_port_prestart_gate_live_authority_granted_matches_plan: true`.

The no-WING handoff summary must also show
`execution_allowed_by_packet: false`.

The no-WING handoff summary must also show
`live_authority_granted: false`.

The no-WING handoff summary must also show
`full_autonomous_ready: false`.

The no-WING handoff summary must also show
`next_manual_artifact_required: real wing_readonly_preflight.v1 summary.json`.

The no-WING handoff summary must also show `expected_artifacts` exactly
matching `expected_artifact_globs` and `plan_expected_artifacts`.

The no-WING handoff summary must also show
`expected_artifacts_match_plan: true`, `summary_glob_matches_plan: true`, and
`event_log_glob_matches_plan: true`.

The no-WING handoff summary must also show command binding to the same packet
artifacts and roots before a real WING run:
`preview_command_artifact_root_matches_packet: true`,
`preview_command_plan_output_matches_packet: true`,
`validate_inputs_command_uses_validate_run_inputs_only: true`,
`validate_inputs_command_uses_plan_only: false`,
`validate_inputs_command_artifact_root_matches_packet: true`,
`validate_inputs_command_expected_plan_matches_packet: true`,
`capture_command_artifact_root_matches_packet: true`,
`capture_command_expected_plan_matches_packet: true`,
`review_command_artifact_root_matches_packet: true`,
`decision_review_command_artifact_root_matches_packet: true`, and
`decision_review_command_decision_root_matches_packet: true`.

It must also show `supervised_prestart_gate` exactly matching
`expected_supervised_prestart_gate`, with
`supervised_prestart_gate.grants_live_authority: false`.

The same `supervised_prestart_gate` must show
`supervised_prestart_gate.operator_sequence_step_index: 11` and
`supervised_prestart_gate.operator_sequence_step_must_be_final: true`.
It must also expose top-level
`supervised_prestart_gate_required: true`,
`supervised_prestart_gate_operator_sequence_step_index: 11`, and
`supervised_prestart_gate_operator_sequence_step_must_be_final: true`.
It must also expose top-level `supervised_prestart_gate_operator_sequence_step`
and `supervised_prestart_gate_operator_sequence_step_matches_summary: true`.
It must also expose top-level
`supervised_prestart_gate_required_markers`,
`expected_supervised_prestart_gate_required_markers`,
`expected_required_supervised_prestart_gate_markers`,
`required_supervised_prestart_gate_marker_count: 8`,
`expected_required_supervised_prestart_gate_marker_count: 8`,
`required_supervised_prestart_gate_marker_count_matches_contract: true`,
`required_supervised_prestart_gate_marker_count_matches_supervised_gate: true`,
`required_supervised_prestart_gate_markers_match_contract: true`,
`required_supervised_prestart_gate_markers_match_supervised_gate: true`,
`supervised_prestart_gate_required_marker_count: 8`,
`expected_supervised_prestart_gate_required_marker_count: 8`,
`supervised_prestart_gate_required_marker_count_matches_contract: true`,
`supervised_prestart_gate_required_marker_count_matches_summary: true`,
`supervised_prestart_gate_required_markers_match_contract: true`, and
`supervised_prestart_gate_required_markers_match_summary: true`.

It must also show
`supervised_prestart_gate.requires_operator_decision_status: accepted_decision_ready`
and
`supervised_prestart_gate.unlocks_runbook: Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md`.
It must also expose top-level
`supervised_prestart_gate_grants_live_authority: false`,
`supervised_prestart_gate_requires_operator_decision_status: accepted_decision_ready`,
and
`supervised_prestart_gate_unlocks_runbook: Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md`.

It must also show
`supervised_prestart_gate.requires_accepted_candidate_prestart_trace: true`,
`supervised_prestart_gate.requires_accepted_candidate_artifact_root_trace: true`,
`supervised_prestart_gate.requires_accepted_candidate_request_sequence_trace: true`,
`supervised_prestart_gate.requires_accepted_candidate_contact_trace: true`,
and
`supervised_prestart_gate.requires_accepted_candidate_expected_artifact_trace: true`.
It must also expose top-level
`supervised_prestart_gate_requires_accepted_candidate_prestart_trace: true`,
`supervised_prestart_gate_requires_accepted_candidate_artifact_root_trace: true`,
`supervised_prestart_gate_requires_accepted_candidate_request_sequence_trace: true`,
`supervised_prestart_gate_requires_accepted_candidate_contact_trace: true`,
and
`supervised_prestart_gate_requires_accepted_candidate_expected_artifact_trace: true`
so release checks can grep the supervised prestart gates without parsing the
nested object.

The handoff summary must also show `required_supervised_prestart_gate_markers`
exactly listing `accepted_decision_ready`,
`valid accepted-candidate prestart trace`,
  `valid accepted-candidate verifier WebSocket proof`,
  `valid accepted-candidate artifact-root trace`,
`valid accepted-candidate request-sequence trace`,
`valid accepted-candidate contact trace`,
`expected-artifact-bound accepted source id`, and
`WING_SUPERVISED_FADER_GAIN_PROOF`.

The operator sequence indexes must also remain explicit:
`mvp_blocker_anchor_digest_review_step_index: 10`,
`expected_mvp_blocker_anchor_digest_review_step_index: 10`,
`mvp_blocker_anchor_digest_review_step_index_matches_contract: true`,
`mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true`,
`operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10`,
`expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10`,
`operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true`,
and
`operator_sequence_supervised_prestart_gate_step_index: 11`.

The handoff summary must also show `operator_sequence_count: 12` so reviewers
can reject packets with missing or extra operator steps.
It must also show `operator_sequence` and `expected_operator_sequence` so
reviewers can inspect the ordered checklist directly from the flat no-WING
summary before opening a backend/WING WebSocket.

It must also show
`wing_readonly_operator_sequence_list_summary_surface_self_check_required: true`,
`wing_readonly_operator_sequence_list_summary_surface_mismatches: []`,
`wing_readonly_operator_sequence_list_summary_surface_match: true`,
`wing_readonly_operator_sequence_list_summary_surface_readiness_closure_allowed: false`,
and
`wing_readonly_operator_sequence_list_summary_surface_live_authority_granted: false`
so the promoted ordered lists fail closed if they drift from the handoff artifact
or contract.

It must also show
`wing_readonly_operator_sequence_list_markdown_surface_required: true`,
`wing_readonly_operator_sequence_list_markdown_surface_required_field_lines`,
`wing_readonly_operator_sequence_list_markdown_surface_required_field_lines_expected_value`,
`wing_readonly_operator_sequence_list_markdown_surface_required_field_lines_actual_value`,
`wing_readonly_operator_sequence_list_markdown_surface_required_field_lines_match_contract: true`,
`wing_readonly_operator_sequence_list_markdown_surface_missing_field_lines: []`,
`wing_readonly_operator_sequence_list_markdown_surface_field_lines_rendered: true`,
`wing_readonly_operator_sequence_list_markdown_surface_readiness_closure_allowed: false`,
and
`wing_readonly_operator_sequence_list_markdown_surface_live_authority_granted: false`
so the operator-readable `summary.md` fails closed if it stops rendering the
ordered list review fields or the Markdown-surface verdict fields themselves.
It also fails closed if the rendered `required_field_lines` value drifts from
the runtime contract.
It must also show `operator_sequence_sha256`,
`actual_operator_sequence_sha256`, and
`operator_sequence_sha256_matches_actual: true` so reviewers can reject
packets whose ordered checklist text drifted from the declared digest.
It must also show `expected_operator_sequence_sha256`,
`actual_expected_operator_sequence_sha256`,
`operator_sequence_matches_contract: true`,
`operator_sequence_sha256_matches_contract: true`,
`declared_expected_operator_sequence_sha256_matches_contract: true`,
`declared_operator_sequence_matches_contract_flag: true`,
`declared_operator_sequence_matches_contract_flag_matches_actual: true`,
`declared_operator_sequence_sha256_matches_contract_flag: true`, and
`declared_operator_sequence_sha256_matches_contract_flag_matches_actual: true`
so reviewers can reject packets whose checklist no longer matches the shared
contract.
It must also show `declared_operator_sequence_count: 12`,
`declared_expected_operator_sequence_count: 12`,
`declared_operator_sequence_count_contract_flag: true`,
`declared_operator_sequence_count_matches_actual: true`,
`declared_operator_sequence_count_matches_contract: true`,
`declared_expected_operator_sequence_count_matches_contract: true`, and
`declared_operator_sequence_count_contract_flag_matches_declared_count: true`.

The first operator step text must also be explicit:
`backend_port_prestart_check_operator_sequence_step` must include
`prestart_backend_port_check` and `unexpected listeners`.

The handoff summary must also prove the backend/WING connection confirmation
step is explicit and occurs before capture:
`backend_connection_confirmation_step_index: 3`,
`expected_backend_connection_confirmation_step_index: 3`,
`operator_sequence_has_backend_connection_confirmation: true`,
`operator_sequence_backend_connection_confirmation_step_index: 3`,
`operator_sequence_backend_connection_confirmation_before_capture: true`,
`operator_sequence_backend_connection_confirmation_step` includes
`existing backend connection`, `intended WING Rack`, and
`read-only inspection`, and
`operator_sequence_validate_readonly_preflight_inputs_step_index: 5`,
`expected_operator_sequence_validate_readonly_preflight_inputs_step_index: 5`,
`operator_sequence_validate_readonly_preflight_inputs_step_index_matches_contract: true`,
`operator_sequence_validate_readonly_preflight_inputs_step: Run validate_readonly_preflight_inputs before capture; it must validate real-run inputs and expected-plan bindings without opening transport.`,
`expected_operator_sequence_validate_readonly_preflight_inputs_step: Run validate_readonly_preflight_inputs before capture; it must validate real-run inputs and expected-plan bindings without opening transport.`,
`operator_sequence_validate_readonly_preflight_inputs_step_matches_contract: true`,
`operator_sequence_capture_readonly_preflight_step_index: 6`,
`expected_operator_sequence_capture_readonly_preflight_step_index: 6`,
`operator_sequence_capture_readonly_preflight_step_index_matches_contract: true`,
`operator_sequence_capture_readonly_preflight_step: Run capture_readonly_preflight.`,
`expected_operator_sequence_capture_readonly_preflight_step: Run capture_readonly_preflight.`,
and
`operator_sequence_capture_readonly_preflight_step_matches_contract: true`.

## Autonomy Budget Rule

Automated cycles may continue without asking "continue?", but they must be
bounded by `tools/codex_autoloop/model_budget_profile.json`. If the expensive
profile is exhausted, Codex must keep work on the MVP rail with a cheaper model
profile or skip the cycle cleanly instead of spending GPT-5.5 high reasoning on
routine reports.
