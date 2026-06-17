# Codex Autoloop LaunchAgent

This folder prepares a local macOS LaunchAgent that can run safe Codex
development cycles for Automixer every 6 hours, with a local model-budget gate
before each cycle.

It is not installed automatically. Review the files first, then install only if
you want the background automation active.

## Files

- `run_codex_cycle.sh` - one-shot wrapper that runs one Codex cycle and exits.
- `codex_cycle_prompt.md` - prompt used by each Codex cycle.
- `model_budget_profile.json` - local daily/weekly/monthly model profile caps.
- `check_model_budget.py` - selects the cheapest safe model profile for one
  cycle and records reservations in `logs/codex_autoloop/model_budget_state.json`.
- `validate_cycle_report.py` - local checker for required cycle-report evidence.
- `../no_wing_release_check.py` - timeout-bounded offline release-check runner
  for backend/frontend gates without WING access.
- `com.tubeslave.automixer.codex-autoloop.plist` - LaunchAgent definition.

## Behavior

- The LaunchAgent attempts one Codex cycle every `21600` seconds.
- Before Codex starts, `check_model_budget.py` selects a model profile from
  `model_budget_profile.json` and reserves one local budget event.
- The budget gate cannot read provider-side tariff usage. It protects the plan
  conservatively by limiting automated requests for expensive profiles:
  `budget_light`, `budget_balanced`, `budget_strict`, and rare `premium_high`.
- If all allowed local profiles are exhausted, the wrapper exits successfully
  without starting Codex so quota is preserved for the next day/week/month.
- Codex limit checks are disabled. The current runner does not use the older
  `CODEX_LIMITS_*` provider-limit files and does not skip runs based on limits;
  the active model-budget profile gate is a separate local cycle-routing guard.
  Current policy supersedes older local reports that described a limit gate.
- The default automated path avoids GPT-5.5 high reasoning. High reasoning is
  reserved for rare architecture/safety review windows and is disabled for
  normal daily automation by the profile.
- The wrapper has a hard project-root guard and refuses to run unless both the
  script path and git root resolve to
  `/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main`.
- The prompt explicitly forbids work in neighboring projects such as external
  Paperclip/PiperClip folders. Repo-local `.paperclip/reports/` is used only as
  an Automixer report directory.
- The prompt keeps TODO/FIXME scans focused on owned Automixer code, tests,
  tooling, and current docs. Vendored, generated, artifact, and reference
  corpora such as `external/`, `models/training_state.json`,
  `backend/ai/knowledge/`, `frontend/node_modules/`, nested
  `**/node_modules/`, `backend/venv/`, `**/.venv/`, `artifacts/`, and
  generated/log/report-history paths such as `frontend/dist/`,
  `logs/codex_autoloop/`, `.paperclip/reports/`, plus archived/manual docs are
  treated as scan noise unless the selected task explicitly targets them. The
  TODO/FIXME scan should use `rg --hidden --glob ...` directly rather than
  `find ... | xargs rg`, because generated app bundle paths can contain spaces.
- Does not run immediately at login; `launchd` starts it on the 6-hour
  interval.
- Uses a lock directory at `.codex_autoloop.lock` to avoid parallel runs.
- Writes lock owner metadata to `.codex_autoloop.lock/owner.env` after acquiring
  the lock, including pid, parent pid, UTC start time, host, project root, and
  wrapper log path. If another run finds the lock, it prints that metadata when
  available instead of guessing whether the lock is active or stale.
- Writes wrapper logs to `logs/codex_autoloop/`.
- After a successful Codex run, finds the new repo-local cycle report and
  validates the required safety/report shape, including a valid, not-future UTC
  report title timestamp that matches the filename, a short plan section with
  one to three numbered/bulleted items for the selected cycle tasks,
  discovery evidence for the branch/worktree, latest/recent repo-local report
  directories, latest/recent test artifacts, TODO/FIXME scan, local `.ai`
  tasks, and non-empty evidence sections for
  findings, changes, tests, remaining work, risks, and each plain-language
  explanation subsection. Worktree discovery must name the exact
  `git status --short` command and the current branch. Core report sections
  must remain in the expected
  order: scope, findings, short plan, changes, changed files, tests, memory
  lint result, remaining work, and risks. The TODO/FIXME evidence must state
  the owned-code focus and the excluded vendored/generated/reference scan-noise
  boundary. The tests section must state an outcome such as passed, failed,
  skipped, blocked, or not run, not only list commands, and must include a
  memory lint result.
  Reports must also include a dedicated non-empty `## Memory Lint Result`
  section with the `memory_lint.py` command/result and an explicit outcome.
  The `Changed Files` section must be
  non-empty, include repo-local markdown links, and link back to the cycle
  report itself so the audit artifact is visible in the file list. Reports must
  explicitly state the exact Scope lines
  `Codex CLI was the only agent workflow used.` and
  `External Paperclip/PiperClip was not used as an agent runner.`, and must
  explicitly state that no neighboring repositories or external local projects
  were analyzed, modified, or used. Scope must include `Repo-local report path:`
  with `/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main/.paperclip/reports` so
  repo-local reports cannot be confused with external Paperclip/PiperClip
  output. The worktree evidence must mention both modified/changed files and
  new/untracked files, even when one side is empty.
  The plain-language project-state subsection must explicitly cover what
  already works, what works partially, what is not implemented, and the current
  risks. The end-of-report order is fixed: `## Три обязательных вопроса`
  appears before the final top-level `## Объяснение простым языком` section,
  and no top-level report section may follow that plain-language section.
  Reports must not contradict the repo-local boundary later by claiming
  that external Paperclip/PiperClip ran the cycle or that neighboring project
  paths such as `/Users/dmitrijvolkov/paperclip` were analyzed, modified, or
  used.
  Forbidden live command tokens, including input-trim/preamp aliases, must be
  listed in an explicitly negative audit sentence, such as a statement that no
  such command was run. Non-destructive
  action terms such as commit, push, reset, destructive cleanup, secret access,
  and user-file deletion must also appear in an explicitly negative audit
  sentence. The validator also rejects reports that include a valid no-action
  audit in `## Scope` and then contradict it in later report sections with an
  explicit executed/run/sent/applied/performed action sentence.
- Direction guard activation is inside each Codex cycle: the prompt requires
  `tools/codex_autoloop/codex_direction_guard.py` before task selection and
  again before the final cycle report. The guard reads git status, recent
  commit subjects, and repo-local reports, then scores whether the cycle is
  moving toward supervised WING Rack MVP blockers or drifting into
  memory/queue/result-consumer/report-only work. The wall-clock cadence remains
  the existing LaunchAgent interval of `21600` seconds when installed; no new
  background service is installed by the guard.
- When WING Rack is connected, the guarded next action starts with
  `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick` before
  starting the intended backend/WING session, then continues to the read-only
  WING preflight or manual review of a real preflight artifact. This prestart
  gate is diagnostic only and does not stop processes, contact WING, or grant
  live authority.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `backend_port_prestart_gate` exactly matching
  `expected_backend_port_prestart_gate`,
  `backend_port_prestart_gate.grants_live_authority: false`, and
  `backend_port_prestart_gate.must_pass_before_backend_wing_session: true`.
  This first gate proves only that the backend-port prestart check is still
  before the intended backend/WING session; it does not contact WING or grant
  live authority.
- Before any supervised fader/gain proof, the structured
  `backend_port_prestart_gate` must also show
  `backend_port_prestart_gate.operator_sequence_step_index: 0` and
  `backend_port_prestart_gate.operator_sequence_step_must_be_first: true`.
  These fields keep the structured gate aligned with the first operator step.
- Before any supervised fader/gain proof, the structured
  `backend_port_prestart_gate` must also show
  `backend_port_prestart_gate.runs_command: prestart_backend_port_check`.
  This binds the structured gate to the no-WING backend-port prestart command.
- Before any supervised fader/gain proof, the structured
  `backend_port_prestart_gate.required_markers` must exactly list
  `Before starting the intended backend/WING session`,
  `prestart_backend_port_check`, and `unexpected listeners`.
  These markers bind the first gate to the operator-facing stop condition.
- Before any supervised fader/gain proof, the failed busy-port no-WING summary
  and no-WING handoff summary must both show
  `backend_port_prestart_check_command` exactly matching
  `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick`.
  This keeps the structured prestart command aligned with the required
  no-WING release check.
- The same summaries must also show `backend_port_prestart_check_command_text`
  exactly matching
  `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick`.
  This gives operators and automation a stable exact-string mirror of the
  structured prestart command.
- The same summaries must also show
  `backend_port_prestart_check_required: true` and
  `backend_port_prestart_check_must_pass_before_starting_backend: true`.
  This keeps the backend listener prestart check non-optional and before the
  intended session.
- Before any supervised fader/gain proof, accepted review source ids must show
  `accepted_candidate_prestart_contract_violations_by_source: {}` in the
  decision-review ledger/report and
  `readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source: {}`
  in the supervised proof plan summary. A non-empty map keeps
  `accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof`
  in force, must trigger
  `next_wing_proof_plan_steps_with_candidate_prestart_violations` if proof
  steps are still present, and grants no live authority.
- The recomputed
  `readonly_preflight_review_decision_expected_candidate_prestart_contract_violations_by_source`
  map must also stay `{}`. A non-empty expected map must trigger
  `next_wing_proof_plan_steps_with_expected_candidate_prestart_violations` if
  proof steps are still present.
- Before any supervised fader/gain proof, accepted review source ids must show
  `readonly_preflight_review_decision_candidate_verifier_ws_proof_complete: true`
  in the supervised proof plan summary. Missing verifier proof keeps
  `accepted_readonly_preflight_candidate_verifier_ws_proof_required_before_supervised_proof`
  in force and grants no live authority.
- Accepted review source ids must also show
  `readonly_preflight_review_decision_candidate_artifact_root_contract_violations_by_source: {}`
  in the supervised proof plan summary. A non-empty map keeps
  `accepted_readonly_preflight_candidate_artifact_root_required_before_supervised_proof`
  in force, must trigger
  `next_wing_proof_plan_steps_with_candidate_artifact_root_violations` if proof
  steps are still present, and grants no live authority.
- The recomputed
  `readonly_preflight_review_decision_expected_candidate_artifact_root_contract_violations_by_source`
  map must also stay `{}`. A non-empty expected map must trigger
  `next_wing_proof_plan_steps_with_expected_candidate_artifact_root_violations`
  if proof steps are still present.
- Accepted review source ids must also expose an expected-artifact trace in the
  supervised proof plan summary. Missing or mismatched expected-artifact trace
  evidence keeps
  `accepted_readonly_preflight_candidate_expected_artifact_trace_required_before_supervised_proof`
  in force, must trigger
  `next_wing_proof_plan_steps_with_expected_artifact_trace_violations` if proof
  steps are still present, and grants no live authority.
- The same summaries must also show
  `backend_port_prestart_gate_execution_allowed: false`,
  `backend_port_prestart_gate_readiness_closure_allowed: false`, and
  `backend_port_prestart_gate_live_authority_granted: false`.
  This keeps the failed prestart gate from being treated as execution,
  readiness, or live-console authority.
- The same summaries must also show
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
  This preserves the fail-closed review evidence when the prestart check
  detects unexpected listeners.
- If no-WING quick reports `environment_blocked_checks`, the no-WING summary
  must also expose `environment_blocked_check_details` with
  `local_dev_server_bind_denied`, `127.0.0.1:3000`, log paths, and
  `operator_next_step`. That result remains failed until rerun in an
  environment that can bind the frontend dev server.
- When the backend-port prestart check detects an existing listener, the failed
  no-WING summary must expose `listener_confirmation_template_path` and keep the
  generated template pending with
  `listener_confirmation_template_status: pending_operator_edit`,
  `listener_confirmation_template_safe_to_stop_listener_by_gate: false`, and
  `listener_confirmation_template_live_authority_granted: false`. The template
  is only an operator-edit starting point, not an accepted confirmation or live
  authority.
- The same failed no-WING summary must expose
  `listener_confirmation_template_accepted_by_template: false` and
  `listener_confirmation_template_operator_placeholder_fields` while the
  generated template remains pending.
- The same failed no-WING summary must expose
  `listener_confirmation_template_execution_allowed_by_template: false` and
  `listener_confirmation_template_readiness_closure_allowed_by_template: false`
  while the generated template remains pending.
- The generated backend-listener template must also expose
  `listener_confirmation_template_required_edits` with
  `set template_status to operator_reviewed_and_edited if using the generated
  template` and `set accepted_by_template to true if using the generated
  template`; any template-derived accepted confirmation must replace
  `REPLACE_WITH_*` operator placeholders, set
  `template_status: operator_reviewed_and_edited`, and set
  `accepted_by_template: true`.
- Before any readiness update, the no-WING handoff summary must show
  `manual_review_required_before_readiness_update: true`.
  This keeps offline handoff packets from being treated as accepted real WING
  proof.
- Before any readiness update, the no-WING handoff summary must show
  `readiness_matrix_update_allowed_by_packet: false`.
  This keeps packet handoff review from being treated as permission to update
  the readiness matrix.
- Before closing any readiness item, the no-WING handoff summary must show
  `readiness_closure_allowed_by_packet: false`.
  This keeps packet handoff review from being treated as blocker-closure
  authority.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `read_only: true`.
  This keeps the handoff packet tied to read-only proof preparation.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `live_write: false`.
  This keeps the handoff packet from being treated as a live-write artifact.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `mutates_console_state: false`.
  This keeps the handoff packet from being treated as console mutation
  evidence.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `queries_console: false`.
  This keeps the no-WING packet from being treated as proof that a console was
  queried.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `executes_tests: false`.
  This keeps the no-WING packet from being treated as validation evidence.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `operator_required: true`.
  This keeps the no-WING packet tied to explicit operator action.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `expected_plan_required_for_real_run: true`.
  This keeps the real read-only preflight tied to the offline expected plan.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
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
  This keeps the packet's no-WING prestart summary contract bound to the plan.
- Before any supervised fader/gain proof, the accepted-candidate prestart trace
	  must show `backend_port_prestart_summary_required: true`,
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
  This keeps the read-only candidate tied to the actual passed no-WING
  prestart `summary.json`, not only to prestart field names.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `backend_port_prestart_summary_path_placeholder` pointing to the passed
  no-WING prestart `summary.json`, plus matching
  `preview_command_prestart_summary`,
  `preview_command_prestart_summary_matches_plan: true`, matching
  `validate_inputs_command_prestart_summary`,
  `validate_inputs_command_prestart_summary_matches_plan: true`, matching
  `capture_command_prestart_summary`, and
  `capture_command_prestart_summary_matches_plan: true`.
  This keeps the operator handoff commands aligned with the runner prestart
  summary gate.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `physical_preflight_requires_existing_backend_connection: true`.
  This keeps the physical read-only preflight tied to the intended existing
  backend/WING session.
- Before any supervised fader/gain proof, the no-WING handoff summary must also
  show the same `backend_port_prestart_check_required: true` and
  `backend_port_prestart_check_must_pass_before_starting_backend: true`.
  This keeps the backend listener prestart check non-optional and before the
  intended session.
- Before any supervised fader/gain proof, the no-WING handoff summary must also
  show `backend_port_prestart_gate_execution_allowed: false`,
  `backend_port_prestart_gate_readiness_closure_allowed: false`, and
  `backend_port_prestart_gate_live_authority_granted: false`.
  This keeps the handoff prestart gate non-authorizing even when rendered as an
  operator checklist.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `execution_allowed_by_packet: false`.
  This keeps the handoff packet from being treated as authority to execute.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `live_authority_granted: false`.
  This keeps the handoff packet from granting live console authority.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `full_autonomous_ready: false`.
  This keeps the handoff packet from being treated as autonomous readiness.
- If handoff validation fails because execution, readiness, live-authority, or
  full-autonomous packet flags are `true`, the no-WING summary must preserve
  those unsafe `true` values so the failed handoff is auditable.
- If handoff validation fails because packet safety flags such as `live_write`,
  `mutates_console_state`, `queries_console`, or `executes_tests` are `true`,
  the no-WING summary must preserve those unsafe `true` values.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `next_manual_artifact_required: real wing_readonly_preflight.v1 summary.json`.
  This keeps the next blocking artifact tied to a real WING read-only summary,
  not to the offline handoff packet.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `expected_artifacts` exactly matching `expected_artifact_globs` and
  `plan_expected_artifacts`, with `expected_artifacts_match_plan: true`,
  `summary_glob_matches_plan: true`, and `event_log_glob_matches_plan: true`.
  This keeps the packet artifact list, named summary/events globs, and expected
  plan artifact list tied to the same real-run outputs.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
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
  `decision_review_command_decision_root_matches_packet: true`. This keeps
  capture, review, and decision-review commands bound to the packet's expected
  plan, artifact root, and decision root.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `backend_port_prestart_check_operator_sequence_step_index: 0`,
  `backend_port_prestart_check_operator_sequence_step` including
  `prestart_backend_port_check` and `unexpected listeners`,
  `backend_port_prestart_check_must_be_first_operator_step: true`, and
  `backend_port_prestart_check_operator_sequence_step_is_first: true`.
  This proves the backend-port prestart check remains the first operator step
  before any intended backend/WING session.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `backend_connection_confirmation_step_index: 3`,
  `expected_backend_connection_confirmation_step_index: 3`,
  `operator_sequence_has_backend_connection_confirmation: true`,
  `operator_sequence_backend_connection_confirmation_step_index: 3`,
  `operator_sequence_backend_connection_confirmation_before_capture: true`,
  `operator_sequence_backend_connection_confirmation_step` including
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
  and `operator_sequence_capture_readonly_preflight_step_matches_contract: true`.
  This proves the intended backend/WING confirmation is explicit before capture
  starts and the capture step still matches the shared handoff contract.
- Before any supervised fader/gain proof, the no-WING handoff summary must also
  show
  `supervised_prestart_gate` exactly matching
  `expected_supervised_prestart_gate` and
  `supervised_prestart_gate.grants_live_authority: false`. This structured gate
  is evidence for the proof boundary only and does not grant live authority.
- Before any supervised fader/gain proof, the structured
  `supervised_prestart_gate` must also show
  `supervised_prestart_gate.operator_sequence_step_index: 11` and
  `supervised_prestart_gate.operator_sequence_step_must_be_final: true`.
  These fields keep the supervised gate aligned with the final operator step.
- Before any supervised fader/gain proof, the structured
  `supervised_prestart_gate` must also show
  `supervised_prestart_gate.requires_operator_decision_status:
  accepted_decision_ready` and `supervised_prestart_gate.unlocks_runbook:
  Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md`.
  These fields keep the gate tied to an accepted decision and the supervised
  fader/gain proof runbook only.
- Before any supervised fader/gain proof, the no-WING handoff summary must also
  show `mvp_blocker_anchor_matches_current_first_open: true`,
  `mvp_blocker_anchor_sha256_matches_current_first_open: true`,
  `mvp_blocker_anchor.packet_does_not_close_blocker: true`,
  `operator_sequence_has_mvp_blocker_anchor_digest_review: true`,
  `operator_sequence_has_supervised_prestart_gate: true`,
  `operator_sequence_supervised_prestart_gate_is_final: true`, and
  `supervised_prestart_gate.requires_accepted_candidate_prestart_trace: true`,
  plus `supervised_prestart_gate.requires_accepted_candidate_artifact_root_trace: true`,
  `supervised_prestart_gate.requires_accepted_candidate_request_sequence_trace: true`,
  and
  `supervised_prestart_gate.requires_accepted_candidate_expected_artifact_trace: true`.
  It must also expose top-level
  `supervised_prestart_gate_required: true`,
  `supervised_prestart_gate_operator_sequence_step_index: 11`,
  `supervised_prestart_gate_operator_sequence_step_must_be_final: true`,
  `supervised_prestart_gate_operator_sequence_step`,
  `supervised_prestart_gate_operator_sequence_step_matches_summary: true`,
  `supervised_prestart_gate_required_markers`,
  `expected_supervised_prestart_gate_required_markers`,
  `expected_required_supervised_prestart_gate_markers`,
  `required_supervised_prestart_gate_marker_count: 6`,
  `expected_required_supervised_prestart_gate_marker_count: 6`,
  `required_supervised_prestart_gate_marker_count_matches_contract: true`,
  `required_supervised_prestart_gate_marker_count_matches_supervised_gate: true`,
  `required_supervised_prestart_gate_markers_match_contract: true`,
  `required_supervised_prestart_gate_markers_match_supervised_gate: true`,
  `supervised_prestart_gate_required_marker_count: 6`,
  `expected_supervised_prestart_gate_required_marker_count: 6`,
  `supervised_prestart_gate_required_marker_count_matches_contract: true`,
  `supervised_prestart_gate_required_marker_count_matches_summary: true`,
  `supervised_prestart_gate_required_markers_match_contract: true`,
  `supervised_prestart_gate_required_markers_match_summary: true`,
  `supervised_prestart_gate_grants_live_authority: false`,
  `supervised_prestart_gate_requires_operator_decision_status: accepted_decision_ready`,
  `supervised_prestart_gate_unlocks_runbook: Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md`,
  `supervised_prestart_gate_requires_accepted_candidate_prestart_trace: true`,
  `supervised_prestart_gate_requires_accepted_candidate_artifact_root_trace: true`,
  `supervised_prestart_gate_requires_accepted_candidate_request_sequence_trace: true`,
  and
  `supervised_prestart_gate_requires_accepted_candidate_expected_artifact_trace: true`.
  These fields prove only that the offline handoff packet still points at the
  current first WING MVP blocker and keeps supervised proof planning behind a
  valid accepted-candidate prestart trace and artifact-root trace; they do not
  grant live authority.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `operator_sequence_count: 12`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  top-level `operator_sequence` and top-level `expected_operator_sequence`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `wing_readonly_operator_sequence_list_summary_surface_match: true` and
  `wing_readonly_operator_sequence_list_summary_surface_mismatches: []`.
- Before any supervised fader/gain proof, the no-WING `summary.md` must show
  `wing_readonly_operator_sequence_list_markdown_surface_field_lines_rendered: true`
  and `wing_readonly_operator_sequence_list_markdown_surface_missing_field_lines: []`.
- Before any supervised fader/gain proof, the no-WING `summary.md` must show
  `wing_readonly_operator_sequence_list_markdown_surface_required_field_lines`.
- Before any supervised fader/gain proof, the no-WING `summary.md` must show
  `wing_readonly_operator_sequence_list_markdown_surface_required_field_lines_expected_value`
  and `wing_readonly_operator_sequence_list_markdown_surface_required_field_lines_actual_value`.
- Before any supervised fader/gain proof, the no-WING `summary.md` must show
  `wing_readonly_operator_sequence_list_markdown_surface_required_field_lines_match_contract: true`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `expected_operator_sequence_count: 12` and
  `operator_sequence_count_matches_contract: true`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `operator_sequence_sha256`, `actual_operator_sequence_sha256`, and
  `operator_sequence_sha256_matches_actual: true`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `expected_operator_sequence_sha256`,
  `actual_expected_operator_sequence_sha256`,
  `operator_sequence_matches_contract: true`,
  `operator_sequence_sha256_matches_contract: true`,
  `declared_expected_operator_sequence_sha256_matches_contract: true`,
  `declared_operator_sequence_matches_contract_flag: true`,
  `declared_operator_sequence_matches_contract_flag_matches_actual: true`,
  `declared_operator_sequence_sha256_matches_contract_flag: true`, and
  `declared_operator_sequence_sha256_matches_contract_flag_matches_actual: true`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `declared_operator_sequence_count: 12`,
  `declared_expected_operator_sequence_count: 12`,
  `declared_operator_sequence_count_contract_flag: true`,
  `declared_operator_sequence_count_matches_actual: true`,
  `declared_operator_sequence_count_matches_contract: true`,
  `declared_expected_operator_sequence_count_matches_contract: true`, and
  `declared_operator_sequence_count_contract_flag_matches_declared_count: true`.
- Before any supervised fader/gain proof, the
  `mvp_blocker_anchor_digest_review_step_index` must be `9`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `expected_mvp_blocker_anchor_digest_review_step_index: 10`,
  `mvp_blocker_anchor_digest_review_step_index_matches_contract: true`, and
  `mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true`.
- Before any supervised fader/gain proof, the
  `operator_sequence_mvp_blocker_anchor_digest_review_step_index` must be `9`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10`
  and
  `operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true`.
- Before any supervised fader/gain proof, the
  `operator_sequence_mvp_blocker_anchor_digest_review_step` must include
  `mvp_blocker_anchor`, `mvp_blocker_anchor_sha256`, and
  `current first open blocker`.
- Before any supervised fader/gain proof, the
  `operator_sequence_supervised_prestart_gate_step_index` must be `10`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `expected_operator_sequence_supervised_prestart_gate_step_index: 11` and
  `operator_sequence_supervised_prestart_gate_step_index_matches_contract: true`.
- Before any supervised fader/gain proof, the final
  `operator_sequence_supervised_prestart_gate_step` must include
  `accepted_decision_ready`, `valid accepted-candidate prestart trace`,
  `valid accepted-candidate artifact-root trace`,
  `expected-artifact-bound accepted source id`, and
  `WING_SUPERVISED_FADER_GAIN_PROOF`.
- Before any supervised fader/gain proof,
  `required_supervised_prestart_gate_markers` must exactly list
  `accepted_decision_ready`, `valid accepted-candidate prestart trace`,
  `valid accepted-candidate artifact-root trace`,
  `expected-artifact-bound accepted source id`, and
  `WING_SUPERVISED_FADER_GAIN_PROOF`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `backend_port_prestart_gate_matches_expected: true`,
  `declared_backend_port_prestart_gate_matches_expected: true`,
  `supervised_prestart_gate_matches_expected: true`, and
  `declared_supervised_prestart_gate_matches_expected: true`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `required_supervised_proof_plan_summary_fields`,
  `expected_required_supervised_proof_plan_summary_fields`, and
  `required_supervised_proof_plan_summary_fields_match_contract: true`.
- Before any supervised fader/gain proof, the no-WING handoff summary must show
  `required_supervised_proof_plan_summary_fields_include_invalid_records: true`.
- Before any supervised fader/gain proof,
  `required_supervised_proof_plan_summary_fields` must include
  `readonly_preflight_review_decision_expected_candidate_prestart_source_ids`,
  `readonly_preflight_review_decision_expected_candidate_prestart_contract_violations_by_source`,
  `readonly_preflight_review_decision_candidate_artifact_root_contract_valid`,
  `readonly_preflight_review_decision_candidate_artifact_roots`,
  `readonly_preflight_review_decision_candidate_artifact_root_contract_violations_by_source`,
  `readonly_preflight_review_decision_expected_candidate_artifact_roots`, and
  `readonly_preflight_review_decision_expected_candidate_artifact_root_contract_violations_by_source`,
  and `readonly_preflight_review_decision_invalid_records`.
- Before any supervised fader/gain proof, the operator-readable no-WING
  `summary.md` proof plan section must render those canonical helper fields.
- Before any supervised fader/gain proof, the no-WING `next_wing_proof_plan`
  summary must show `operator_markdown_summary_required: true`,
  `operator_markdown_summary_required_field_lines`,
  `readonly_preflight_review_decision_candidate_verifier_ws_proof_source_ids`,
  `readonly_preflight_review_decision_candidate_verifier_ws_proof_violations_by_source`,
  `readonly_preflight_review_decision_candidate_verifier_ws_proof_complete`,
  `operator_markdown_summary_required_field_lines_missing_members: []`,
  `operator_markdown_summary_required_field_lines_match_contract: true`,
  `operator_markdown_summary_field_lines_rendered: true`, and
  `operator_markdown_summary_missing_field_lines: []`.
- Before any supervised fader/gain proof, the no-WING `next_wing_proof_plan`
  summary must bind the operator markdown artifact with
  `operator_markdown_summary_artifact: summary.md`,
  `operator_markdown_summary_sha256`, and
  `operator_markdown_summary_sha256_available: true`.
- Before any supervised fader/gain proof, the no-WING `next_wing_proof_plan`
  summary must verify that binding with
  `operator_markdown_summary_actual_sha256` and
  `operator_markdown_summary_sha256_matches_actual: true`.
- Before any supervised fader/gain proof, the no-WING `summary.json` and
  operator-readable `summary.md` must expose the manifest summary surface with
  `artifact_manifest_summary_surface_required: true`,
  `artifact_manifest_summary_surface_missing_fields: []`,
  `artifact_manifest_summary_surface_required_field_lines`,
  `artifact_manifest_summary_surface_missing_field_lines: []`,
  `artifact_manifest_summary_surface_field_lines_rendered: true`, and
  `artifact_manifest_contract_id`, `artifact_manifest_contract_required: true`,
  `artifact_manifest_contract_fields`, `artifact_manifest_contract_sha256`,
  `artifact_manifest_contract_sha256_available: true`,
  `artifact_manifest_contract_sha256_matches_expected: true`,
  `artifact_manifest_contract_cross_surface_required: true`,
  `artifact_manifest_contract_cross_surface_required_surfaces`,
  `artifact_manifest_contract_cross_surface_required_fields`,
  `artifact_manifest_contract_cross_surface_expected_match: true`,
  `artifact_manifest_contract_cross_surface_readiness_closure_allowed: false`,
  `artifact_manifest_contract_cross_surface_live_authority_granted: false`, and
  `artifact_manifest_required_artifact_sha256_fields`.
- Before any supervised fader/gain proof, the no-WING run artifact bundle must
  write `artifact-manifest.json` with
  `artifact_manifest_contract_id`, `artifact_manifest_contract_required: true`,
  `artifact_manifest_contract_fields`, `artifact_manifest_contract_sha256`,
  `artifact_manifest_contract_sha256_available: true`,
  `artifact_manifest_contract_sha256_matches_expected: true`,
  `artifact_manifest_contract_cross_surface_required: true`,
  `artifact_manifest_contract_cross_surface_required_surfaces`,
  `artifact_manifest_contract_cross_surface_required_fields`,
  `artifact_manifest_contract_cross_surface_expected_match: true`,
  `artifact_manifest_contract_cross_surface_match: true`,
  `artifact_manifest_summary_json_contract_sha256_matches_manifest: true`,
  `artifact_manifest_summary_md_contract_sha256_matches_manifest: true`,
  `artifact_manifest_contract_cross_surface_mismatches: []`,
  `artifact_manifest_contract_cross_surface_readiness_closure_allowed: false`,
  `artifact_manifest_contract_cross_surface_live_authority_granted: false`,
  `artifact_manifest_path: artifact-manifest.json`,
  `artifact_manifest_artifact_count: 2`,
  `artifact_manifest_required_artifact_count: 2`,
  `artifact_manifest_artifact_count_matches_required: true`,
  `artifact_manifest_required_artifact_ids`, `artifact_manifest_artifact_ids`,
  `artifact_manifest_required_artifacts_present: true`,
  `artifact_manifest_required_artifact_paths`, `artifact_manifest_artifact_paths`,
  `artifact_manifest_required_artifact_paths_present: true`,
  `artifact_manifest_required_artifact_sha256_available: true`,
  `artifact_manifest_summary_json_sha256`,
  `artifact_manifest_summary_json_sha256_available: true`,
  `artifact_manifest_summary_md_sha256`,
  `artifact_manifest_summary_md_sha256_available: true`,
  `artifact_manifest_readiness_closure_allowed: false`, and
  `artifact_manifest_live_authority_granted: false`.
- Before opening the backend/WING session, the operator must run
  `PYTHONPATH=backend python3 tools/verify_no_wing_artifact_manifest.py <no_wing_artifact_dir>`
  against the same passed no-WING artifact directory and require
  `status: passed`, `failed_checks: []`,
  `artifact_manifest_self_consistency_match: true`,
  `artifact_manifest_contract_cross_surface_match: true`,
  `artifact_manifest_self_consistency_summary_surface_match: true`,
  `readiness_closure_allowed: false`, and `live_authority_granted: false`.
  The verifier is read-only and grants no live authority.
  The no-WING summary must also expose `artifact_manifest_verifier_required:
  true`, `artifact_manifest_verifier_command`,
  `artifact_manifest_verifier_status_required: passed`,
  `artifact_manifest_verifier_failed_checks_required: []`,
  `artifact_manifest_verifier_contacts_wing: false`,
  `artifact_manifest_verifier_mutates_console_state: false`,
  `artifact_manifest_verifier_queries_console: false`,
  `artifact_manifest_verifier_readiness_closure_allowed: false`, and
  `artifact_manifest_verifier_live_authority_granted: false`.
- Before any supervised fader/gain proof, the no-WING `next_wing_proof_plan`
  summary must expose `required_supervised_proof_plan_summary_fields` and
  `missing_required_supervised_proof_plan_summary_fields`.
- Before any supervised fader/gain proof, `next_wing_proof_plan_schema` must
  fail closed with `next_wing_proof_plan_required_summary_fields_missing` when
  any required summary field is absent.
- Project Director Agent activation is inside each Codex cycle after the
  direction guard. `tools/codex_autoloop/project_director_agent.py` reads
  `Docs/MVP_WING_SUPERVISED.md`, `.ai/project-layer/wing_mvp_blockers.md`, and
  the guard result, then selects the first non-closed MVP blocker as the only
  acceptable task direction. It writes `.paperclip/reports/project_director_latest.md`
  and never writes code, contacts WING, or grants live authority.
  When the selected action source is `Current First Command`, the directive also
  reports `current_first_command_baseline_coverage_passed`,
  `current_first_command_baseline_missing_markers`, and
  `selected_action_sha256` so the autonomous cycle can prove that the shorter
  action text still covers the WING safety/proof baseline.
  For WING handoff work, the no-WING handoff summary must also show
  `project_director_selected_action_anchor`,
  `expected_project_director_selected_action_anchor`,
  `project_director_selected_action_anchor_matches_current_directive: true`,
  `project_director_selected_action_source: Current First Command`,
  `expected_project_director_selected_action_source: Current First Command`,
  `project_director_selected_action_source_matches_current_directive: true`,
  `project_director_selected_action_sha256`,
  `actual_project_director_selected_action_sha256`,
  `expected_project_director_selected_action_sha256`,
  `project_director_selected_action_sha256_available: true`,
  `project_director_selected_action_sha256_available_matches_digest: true`,
  `project_director_selected_action_sha256_matches_anchor: true`,
  `project_director_selected_action_sha256_matches_text: true`,
  `project_director_selected_action_sha256_matches_current_directive: true`,
  `project_director_current_first_command_baseline_coverage_required: true`,
  `project_director_current_first_command_baseline_coverage_passed: true`,
  `project_director_current_first_command_baseline_coverage_marker_count`, and
  `project_director_current_first_command_baseline_missing_markers: []`.
  If it detects broad-work drift, development still continues in Goal mode; the
  next task is redirected to the first non-closed MVP blocker instead of
  stopping the autonomous cycle.
- `tools/codex_autoloop/update_mvp_progress.py` refreshes
  `Docs/reports/MVP_PROGRESS_LATEST.md` from the blocker table before planning
  and before the final report.
- Focused offline release gates can use `tools/no_wing_release_check.py`; each
  subprocess check has a default timeout and records a `timeout` result instead
  of leaving an autonomous run silent indefinitely.
- Requires `codex` CLI to be available.
- Uses `codex exec -C <project> --sandbox workspace-write --model <selected>`.
- Sets Codex approval policy to `never` for non-interactive execution.
- Sets `model_reasoning_effort` from the selected profile.
- Does not enable live WING/OSC writes. The prompt explicitly requires
  dry-run/audit-only behavior for risky actions.
- Does not commit or push.

## Dry Run

From the project root:

```bash
chmod +x tools/codex_autoloop/run_codex_cycle.sh
CODEX_AUTOLOOP_DRY_RUN=1 tools/codex_autoloop/run_codex_cycle.sh
plutil -lint tools/codex_autoloop/com.tubeslave.automixer.codex-autoloop.plist
```

Dry run still checks the project-root, prompt, validator, and lock contract, but
it does not require a working Codex CLI executable because it never starts
`codex exec`.

The dry run also prints the selected local model profile without reserving a
budget event.

## Model Budget Overrides

The default profile is conservative. For one manual cycle you can override it:

```bash
CODEX_AUTOLOOP_MODEL_PROFILE=budget_light tools/codex_autoloop/run_codex_cycle.sh
CODEX_AUTOLOOP_MODEL_PROFILE=budget_strict tools/codex_autoloop/run_codex_cycle.sh
```

Use `premium_high` only for explicit architecture/safety review. It is capped
by `model_budget_profile.json` and is not part of normal daily automation.

You may override the exact model string if the local Codex installation exposes
a different model name:

```bash
CODEX_AUTOLOOP_MODEL=gpt-5.5 CODEX_AUTOLOOP_REASONING_EFFORT=low \
  tools/codex_autoloop/run_codex_cycle.sh
```

If `shellcheck` is installed:

```bash
shellcheck tools/codex_autoloop/run_codex_cycle.sh
```

## One Manual Real Run

This starts one Codex cycle in the foreground and then exits:

```bash
tools/codex_autoloop/run_codex_cycle.sh
```

## Install LaunchAgent

Only run this when you intentionally want the 6-hour background automation:

```bash
mkdir -p "$HOME/Library/LaunchAgents" logs/codex_autoloop .paperclip/reports
cp tools/codex_autoloop/com.tubeslave.automixer.codex-autoloop.plist \
  "$HOME/Library/LaunchAgents/com.tubeslave.automixer.codex-autoloop.plist"
launchctl bootstrap "gui/$(id -u)" \
  "$HOME/Library/LaunchAgents/com.tubeslave.automixer.codex-autoloop.plist"
launchctl enable "gui/$(id -u)/com.tubeslave.automixer.codex-autoloop"
```

## Inspect Status

```bash
launchctl print "gui/$(id -u)/com.tubeslave.automixer.codex-autoloop"
tail -f logs/codex_autoloop/launchd.out.log
tail -f logs/codex_autoloop/launchd.err.log
```

## Uninstall LaunchAgent

```bash
launchctl bootout "gui/$(id -u)" \
  "$HOME/Library/LaunchAgents/com.tubeslave.automixer.codex-autoloop.plist"
rm -f "$HOME/Library/LaunchAgents/com.tubeslave.automixer.codex-autoloop.plist"
```

## Notes

- Do not install multiple Codex autoloops for the same repo.
- If Codex.app starts opening unexpectedly, inspect active launchd jobs:

```bash
launchctl list | rg -i "codex|openai|chatgpt|paperclip|telegram|claw"
```

- This LaunchAgent is interval-based. The shell script must not contain its own
  infinite loop.
