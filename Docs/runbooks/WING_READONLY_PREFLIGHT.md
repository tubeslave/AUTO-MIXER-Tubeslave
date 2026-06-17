# Runbook: WING Read-Only Preflight

Status: required first real hardware step for the supervised WING Rack MVP

## Goal

Capture a real read-only WING preflight artifact before any supervised fader or
gain write is attempted.

This runbook proves only:

- backend is in WING mode;
- current WING state can be read;
- selected fader/gain values are verified from physical readback;
- the runner sent no write, proposal, mode-change, rollback, emergency-stop,
  connect, or disconnect messages.

It does not approve live writes.

## Preconditions

- Before starting the intended backend/WING session, run
  `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick` and confirm
  it does not fail `backend_port_free` against an unexpected local listener. If
  it does, inspect the generated summary fields `listener_processes`,
  `listener_process_error`, `listener_confirmation_checklist`,
  `backend_port_prestart_check_required`,
  `backend_port_prestart_check_must_pass_before_starting_backend`,
  `backend_port_prestart_check_command`,
  `backend_port_prestart_check_failure_detail_fields`,
  `backend_port_prestart_gate_execution_allowed`,
  `backend_port_prestart_gate_readiness_closure_allowed`,
  `backend_port_prestart_gate_live_authority_granted`,
  `preflight_blocked_until_listener_confirmed`,
  `intended_backend_confirmation_required`, `safe_to_stop_listener_by_gate`,
  `fail_closed_reason`, and `operator_next_step` before changing local
  services. Optional `command_line` diagnostics may identify the listener, but
  the real preflight remains blocked until the operator follows the checklist
  to confirm the listener is the intended backend or frees/reconfigures the
  unexpected listener. The gate is diagnostic only and does not stop processes.
  If the existing listener is the intended backend, use the generated
  `listener_confirmation_template_path` from the failed no-WING summary as the
  starting point for a separate operator confirmation JSON, then rerun the gate
  with `--backend-listener-confirmation`. The generated template must remain
  pending with `listener_confirmation_template_status: pending_operator_edit`
  until an operator edits it; it must keep
  `listener_confirmation_template_safe_to_stop_listener_by_gate: false` and
  `listener_confirmation_template_live_authority_granted: false`. It must also
  expose `listener_confirmation_template_accepted_by_template: false` plus
  `listener_confirmation_template_operator_placeholder_fields` for the
  generated `REPLACE_WITH_*` operator fields. It must also expose
  `listener_confirmation_template_execution_allowed_by_template: false` and
  `listener_confirmation_template_readiness_closure_allowed_by_template: false`.
  The accepted JSON must match the
  current listener PID and command line, keep
  `safe_to_stop_listener_by_gate: false`, and keep
  `live_authority_granted: false`. If it was edited from the generated
  template, it must also replace placeholder operator fields, set
  `template_status: operator_reviewed_and_edited`, and set
  `accepted_by_template: true`:

  ```json
  {
    "type": "wing_backend_port_listener_confirmation.v1",
    "status": "operator_confirmed_intended_backend",
    "host": "127.0.0.1",
    "port": 8765,
    "confirms_intended_backend": true,
    "reviewed_listener_pid": 87210,
    "reviewed_listener_command_line": "python3 backend/server.py --mode wing",
    "operator_reviewed_at": "2026-06-13T09:15:00Z",
    "operator_name": "operator",
    "safe_to_stop_listener_by_gate": false,
    "live_authority_granted": false,
    "template_status": "operator_reviewed_and_edited",
    "accepted_by_template": true
  }
  ```

  ```bash
  PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick \
    --backend-listener-confirmation path/to/backend-listener-confirmation.json
  ```

  A valid confirmation only clears the local backend-port prestart blocker; it
  does not contact WING, does not approve writes, and does not close readiness.
  Keep the passed no-WING `summary.json` path. The read-only preflight plan and
  real run must both use it with `--prestart-summary`; the runner records its
  SHA256 and fails closed before WebSocket if the file is missing, stale, or not
  a passed `backend_port_free` prestart result.
- Before opening a backend/WING WebSocket, verify the detached
  `artifact-manifest.json` in the same no-WING artifact directory as the passed
  prestart `summary.json`:

  ```bash
  PYTHONPATH=backend python3 tools/verify_no_wing_artifact_manifest.py \
    path/to/no_wing_artifact_dir
  ```

  The verifier must return `status: passed`, `failed_checks: []`,
  `failed_checks_missing_check_entries: []`, and
  `failed_checks_have_check_entries: true` in `checks`. Its report must also
  show `verifier_report_required_fields_present: true`,
  `verifier_report_required_fields_missing: []`, and
  `verifier_report_required_fields_match_contract: true`. Its `checks` must
  also show `summary_json_exists: true`,
  `summary_md_exists: true`,
  `artifact_manifest_exists: true`,
  `artifact_bundle_loadable: true`,
  `manifest_type: true`,
  `manifest_required_artifact_ids_match_contract: true`,
  `manifest_required_artifact_paths_match_contract: true`,
  `manifest_records_required_artifact_ids: true`,
  `summary_json_path_matches_contract: true`,
  `summary_json_sha256_matches_actual: true`,
  `summary_md_path_matches_contract: true`,
  `summary_md_sha256_matches_actual: true`,
  `summary_json_sha256_matches_manifest: true`,
  `summary_md_sha256_matches_manifest: true`,
  `manifest_denies_readiness_closure: true`,
  `manifest_denies_live_authority: true`,
  `summary_denies_readiness_closure: true`,
  `summary_denies_live_authority: true`,
  `summary_json_artifact_manifest_verifier_fields_present: true`,
  `manifest_artifact_manifest_verifier_fields_present: true`,
  `summary_md_artifact_manifest_verifier_fields_rendered: true`,
  `summary_json_artifact_manifest_verifier_command_matches_expected: true`,
  `summary_md_artifact_manifest_verifier_command_matches_expected: true`,
  `manifest_artifact_manifest_verifier_command_matches_expected: true`,
  `summary_json_artifact_manifest_verifier_report_required_fields_matches_expected: true`,
  `summary_md_artifact_manifest_verifier_report_required_fields_matches_expected: true`,
  and
  `manifest_artifact_manifest_verifier_report_required_fields_matches_expected: true`.
  The same `checks` must show verifier report safety values match the contract:
  `verifier_report_read_only_matches_expected: true`,
  `verifier_report_live_write_matches_expected: true`,
  `verifier_report_contacts_wing_matches_expected: true`,
  `verifier_report_contacts_backend_matches_expected: true`,
  `verifier_report_mutates_console_state_matches_expected: true`,
  `verifier_report_queries_console_matches_expected: true`, and
  `verifier_report_executes_tests_matches_expected: true`,
  `verifier_report_readiness_closure_allowed_matches_expected: true`, and
  `verifier_report_live_authority_granted_matches_expected: true`.
  The published
  no-WING summary and detached sidecar must show
  `artifact_manifest_verifier_required: true`,
  `artifact_manifest_verifier_command`,
  `artifact_manifest_verifier_target_artifact_dir`,
  `artifact_manifest_verifier_status_required: passed`,
  `artifact_manifest_verifier_failed_checks_required: []`,
  `artifact_manifest_verifier_required_fields_missing: []`,
  `artifact_manifest_verifier_contacts_wing: false`,
  `artifact_manifest_verifier_mutates_console_state: false`,
  `artifact_manifest_verifier_queries_console: false`,
  `artifact_manifest_verifier_readiness_closure_allowed: false`, and
  `artifact_manifest_verifier_live_authority_granted: false`. The published
  `artifact_manifest_verifier_report_required_fields` list must include
  `checks`, `verifier_report_required_fields`,
  `verifier_report_required_fields_missing`, and
  `verifier_report_required_fields_match_contract`. The sidecar must preserve
  the same summary-surface value-gate verdict as `summary.json` and
  `summary.md`:
  `artifact_manifest_self_consistency_summary_surface_match: true`,
  `artifact_manifest_self_consistency_summary_surface_json_mismatches: []`,
  `artifact_manifest_self_consistency_summary_surface_markdown_mismatches: []`,
  `artifact_manifest_self_consistency_summary_surface_mismatches: []`,
  `artifact_manifest_self_consistency_summary_surface_readiness_closure_allowed: false`,
  `artifact_manifest_self_consistency_summary_surface_live_authority_granted: false`,
  `artifact_manifest_self_consistency_match: true`,
  `artifact_manifest_self_consistency_mismatches: []`,
  `artifact_manifest_contract_cross_surface_match: true`, and
  `artifact_manifest_contract_cross_surface_mismatches: []`. Missing or failing
  verifier/report/sidecar markers mean rerun the no-WING quick check and do not
  open the WING session. The verifier is diagnostic only and does not contact
  WING; its results do not close readiness, approve live writes, or grant live
  authority. The read-only preflight runner repeats this verifier against the
  artifact directory that contains `--prestart-summary` and fails closed before
  opening a backend/WING WebSocket when the detached manifest verifier fails.
  Its summary/expected plan must expose
  `artifact_manifest_verifier_check_mismatches: []` and mirror every required
  verifier check as `artifact_manifest_verifier_check_<check_name>: true`,
  including `artifact_manifest_verifier_check_summary_json_exists: true`,
  `artifact_manifest_verifier_check_summary_md_exists: true`,
  `artifact_manifest_verifier_check_artifact_manifest_exists: true`,
  `artifact_manifest_verifier_check_artifact_bundle_loadable: true`,
  `artifact_manifest_verifier_check_manifest_type: true`,
  `artifact_manifest_verifier_check_manifest_required_artifact_ids_match_contract: true`,
  `artifact_manifest_verifier_check_manifest_required_artifact_paths_match_contract: true`,
  `artifact_manifest_verifier_check_manifest_records_required_artifact_ids: true`,
  `artifact_manifest_verifier_check_summary_json_path_matches_contract: true`,
  `artifact_manifest_verifier_check_summary_json_sha256_matches_actual: true`,
  `artifact_manifest_verifier_check_summary_md_path_matches_contract: true`,
  `artifact_manifest_verifier_check_summary_md_sha256_matches_actual: true`,
  `artifact_manifest_verifier_check_summary_json_sha256_matches_manifest: true`,
  `artifact_manifest_verifier_check_summary_md_sha256_matches_manifest: true`,
  `artifact_manifest_verifier_check_manifest_denies_readiness_closure: true`,
  `artifact_manifest_verifier_check_manifest_denies_live_authority: true`,
  `artifact_manifest_verifier_check_summary_denies_readiness_closure: true`,
  `artifact_manifest_verifier_check_summary_denies_live_authority: true`,
  `artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_fields_present: true`,
  `artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_fields_present: true`,
  `artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_fields_rendered: true`,
  `artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_command_matches_expected: true`,
  `artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_report_required_fields_matches_expected: true`,
  `artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_report_required_fields_matches_expected: true`,
  `artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_report_required_fields_matches_expected: true`,
  `artifact_manifest_verifier_check_verifier_report_required_fields_present: true`,
  `artifact_manifest_verifier_check_verifier_report_readiness_closure_allowed_matches_expected: true`,
  `artifact_manifest_verifier_check_verifier_report_live_authority_granted_matches_expected: true`,
  and `artifact_manifest_verifier_check_failed_checks_have_check_entries: true`.
  A real captured artifact must later prove the ordering with
  `artifact_manifest_verifier_must_pass_before_websocket: true`,
  `artifact_manifest_verifier_passed_before_websocket: true`, and
  `artifact_manifest_verifier_websocket_opened_after_pass: true`.
- If the no-WING release summary includes
  `wing_readonly_preflight_evidence.artifact_operator_next_steps`, inspect and
  complete those repair instructions before opening a backend/WING WebSocket.
  They come from blocked read-only preflight artifacts and are diagnostic only;
  they do not approve live writes or close readiness rows.
- Then connect the intended WING Rack to the backend and start the backend
  WebSocket, usually at `ws://localhost:8765`.
- The operator knows which channel is safe to inspect.
- No live performance is in progress unless the operator explicitly allows
  read-only inspection.
- Do not run supervised proof yet.
- Do not mix latency, OSC RTT, audio E2E, tone-trigger, DSP-host, or plugin-host
  benchmarks into this read-only proof. Those measurements may be captured
  later as separate lab-only artifacts, but this preflight may send only the
  read-only request types listed below and must not trigger audio or console
  mutation behavior.

## Command

Optional: before the hardware session, generate the offline operator handoff
packet. This does not contact WING and does not approve live writes:

```bash
python3 tools/build_wing_readonly_preflight_handoff_packet.py \
  --url ws://localhost:8765 \
  --channels 3 \
  --value-types fader,gain,pan \
  --artifact-root artifacts/wing_readonly_preflight \
  --plan-output artifacts/wing_readonly_preflight_plan/latest_plan.json \
  --output Docs/reports/WING_READONLY_PREFLIGHT_HANDOFF_PACKET_LATEST.md
```

The packet must also render `mvp_blocker_anchor` for the current first open row
in `.ai/project-layer/wing_mvp_blockers.md`. The no-WING validator must report
`mvp_blocker_anchor_matches_current_first_open: true`; otherwise the handoff is
stale and cannot gate supervised proof planning. The packet must also include a
matching `mvp_blocker_anchor_sha256` so reviewers can compare the displayed
anchor with the exact blocker-row digest. The packet operator sequence must
include a dedicated review step that names `mvp_blocker_anchor`,
`mvp_blocker_anchor_sha256`, and the current first open blocker before
supervised proof planning. The no-WING handoff summary must expose that review
step as `operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10` so
reviewers can confirm the digest check occurs immediately before the final
supervised prestart gate. The same summary must show
`mvp_blocker_anchor_digest_review_step_index: 10`,
`expected_mvp_blocker_anchor_digest_review_step_index: 10`,
`mvp_blocker_anchor_digest_review_step_index_matches_contract: true`,
`mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true`,
`expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10` and
`operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true`.
Before any supervised proof, the handoff summary must also show
`required_supervised_proof_plan_summary_fields_include_invalid_records: true`
so invalid review-decision record diagnostics cannot be hidden behind a stale
proof-plan field list.
The same summary must show the detached no-WING manifest verifier as the
operator-sequence step before preview:
`artifact_manifest_verifier_operator_sequence_step_index: 1`,
`expected_artifact_manifest_verifier_operator_sequence_step_index: 1`,
`artifact_manifest_verifier_operator_sequence_step_index_matches_contract: true`,
`artifact_manifest_verifier_operator_sequence_step`,
`expected_artifact_manifest_verifier_operator_sequence_step`,
`artifact_manifest_verifier_operator_sequence_step_matches_contract: true`,
`artifact_manifest_verifier_required_markers`, and
`artifact_manifest_verifier_operator_sequence_before_preview: true`.
The same summary must also show the Project Director selected-action value-gate
self-check fields before supervised proof planning:
`project_director_selected_action_value_gate_self_check_required: true`,
`project_director_selected_action_value_gate_self_check_required_markers`,
`project_director_selected_action_value_gate_self_check_missing_markers: []`,
`project_director_selected_action_value_gate_self_check_markers_present: true`,
`project_director_selected_action_value_gate_self_check_readiness_closure_allowed: false`,
and
`project_director_selected_action_value_gate_self_check_live_authority_granted: false`.
The operator Markdown surface must also prove those Project Director value-gate
self-check lines were rendered:
`operator_markdown_summary_required_field_lines` must include
`readonly_preflight_review_decision_candidate_verifier_ws_proof_source_ids`,
`readonly_preflight_review_decision_candidate_verifier_ws_proof_violations_by_source`,
`readonly_preflight_review_decision_candidate_verifier_ws_proof_complete`,
`readonly_preflight_review_decision_candidate_request_sequence_contract_valid`,
`readonly_preflight_review_decision_accepted_candidate_request_sequence_by_source`,
`readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_complete`,
`next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_summary_surface_readiness_closure_allowed`,
`next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_summary_surface_live_authority_granted`,
`next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_readiness_closure_allowed`
and
`next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_live_authority_granted`,
and the same summary must show
`operator_markdown_summary_required_field_lines_missing_members: []` and
`operator_markdown_summary_required_field_lines_match_contract: true`,
`project_director_selected_action_value_gate_self_check_markdown_surface_required: true`,
`project_director_selected_action_value_gate_self_check_markdown_surface_required_field_lines`,
`project_director_selected_action_value_gate_self_check_markdown_surface_missing_field_lines: []`,
`project_director_selected_action_value_gate_self_check_markdown_surface_field_lines_rendered: true`,
`project_director_selected_action_value_gate_self_check_markdown_surface_readiness_closure_allowed: false`,
and
`project_director_selected_action_value_gate_self_check_markdown_surface_live_authority_granted: false`.
Missing markers mean rerun the no-WING quick check and keep Project Director
audit/planning-only.

Replace channel `3` with the operator-approved channel:

Before opening the backend/WING session, preview the exact read-only request
plan. This validates the channel confirmation and writes only a local JSON plan;
it does not open WebSocket, contact WING, query the console, or grant live
authority. The plan also records the current MVP blocker, fail-closed
conditions, and the offline review commands to run after the real artifact is
captured. It must also show `expected_plan_required_for_real_run: true` and
`physical_preflight_requires_existing_backend_connection: true`. Replace
`$PRESTART_SUMMARY` with the passed no-WING prestart `summary.json`:

```bash
python3 tools/wing_readonly_preflight_runner.py \
  --url ws://localhost:8765 \
  --channels 3 \
  --operator-confirm-channels 3 \
  --operator-confirm-session-ready \
  --value-types fader,gain,pan \
  --require-physical-readback \
  --prestart-summary "$PRESTART_SUMMARY" \
  --artifact-root artifacts/wing_readonly_preflight \
  --plan-only \
  --plan-output artifacts/wing_readonly_preflight_plan/latest_plan.json
```

Then run the real read-only preflight only when the intended backend/WING
session is ready. Before opening WING transport, validate the exact real-run
inputs against the offline plan with `--validate-run-inputs-only`. This step
must use the same URL, channels, value types, artifact root, no-WING prestart
summary, and expected plan as the real capture command:

```bash
python3 tools/wing_readonly_preflight_runner.py \
  --url ws://localhost:8765 \
  --channels 3 \
  --operator-confirm-channels 3 \
  --operator-confirm-session-ready \
  --value-types fader,gain,pan \
  --require-physical-readback \
  --prestart-summary "$PRESTART_SUMMARY" \
  --artifact-root artifacts/wing_readonly_preflight \
  --expected-plan artifacts/wing_readonly_preflight_plan/latest_plan.json \
  --validate-run-inputs-only
```

After the validation-only command passes, run the real read-only preflight. The
real run must point at the offline plan generated above with `--expected-plan`;
if the flag is omitted, or if the runtime arguments or request sequence do not
match that plan, the runner fails closed before opening WebSocket and writes
`operator_next_step` in `summary.json`. It must also point at the same no-WING
prestart summary used for the plan and validation-only command:

```bash
python3 tools/wing_readonly_preflight_runner.py \
  --url ws://localhost:8765 \
  --channels 3 \
  --operator-confirm-channels 3 \
  --operator-confirm-session-ready \
  --value-types fader,gain,pan \
  --require-physical-readback \
  --prestart-summary "$PRESTART_SUMMARY" \
  --artifact-root artifacts/wing_readonly_preflight \
  --expected-plan artifacts/wing_readonly_preflight_plan/latest_plan.json
```

## Expected Result

The command should produce a `summary.json` and `events.jsonl` under:

`artifacts/wing_readonly_preflight/<run_id>/`

The summary must show:

- `status: passed`
- `proof_artifact_schema: wing_readonly_preflight.v1`
- `proof_runner: wing_readonly_preflight_runner`
- `connection_mode: wing`
- `operator_channel_confirmation_required: true`
- `operator_confirmed_channels == channels`
- `operator_session_confirmation_required: true`
- `operator_confirmed_session_ready: true`
- `read_only: true`
- `live_write: false`
- `write_request_sent: false`
- `proposal_request_sent: false`
- `forbidden_request_types_sent: []`
- `require_physical_readback: true`
- `physical_preflight_requires_existing_backend_connection: true`
- `request_id_policy: required_for_every_query_mixer_parameter_response`
- `expected_plan_path: artifacts/wing_readonly_preflight_plan/latest_plan.json`
- `artifact_root` matches the runtime `--artifact-root` and the referenced
  offline plan `artifact_root`
- `expected_plan_match: true`
- `expected_plan_violations: []`
- `expected_plan_sha256` is non-empty and matches the referenced offline plan
  file during artifact review
- `backend_port_prestart_summary_required: true`
- `backend_port_prestart_summary_path` points to the passed no-WING prestart
  `summary.json`
- `backend_port_prestart_summary_sha256` is non-empty and matches that
  `summary.json` during artifact review
- `backend_port_prestart_summary_type: no_wing_release_check`
- `backend_port_prestart_summary_status: passed`
- `backend_port_prestart_summary_connection_policy:
  no_hardware_no_network_no_live_writes`
- `backend_port_prestart_summary_read_only: true`
- `backend_port_prestart_summary_live_write: false`
- `backend_port_prestart_summary_contacts_backend: false`
- `backend_port_prestart_summary_contacts_wing: false`
- `backend_port_prestart_summary_mutates_console_state: false`
- `backend_port_prestart_summary_queries_console: false`
- `backend_port_prestart_summary_executes_tests: false`
- `backend_port_prestart_check_status: passed`
- `artifact_manifest_verifier_must_pass_before_websocket: true`
- `artifact_manifest_verifier_passed_before_websocket: true`
- `artifact_manifest_verifier_websocket_opened_after_pass: true`
- referenced offline plan fields and `request_sequence` match the captured
  summary, operator-confirmed channels, value types, and readback identities
- `physical_readback_count == readback_count`
- every readback has the expected WING address for its channel/value type and a
  finite numeric `value`

The event log must show only these request types:

- `get_operator_mode`
- `get_state`
- `query_mixer_parameter`

The event log must also be strict `send`/`recv` pairs. The request order must
be `get_operator_mode`, `get_state`, then `query_mixer_parameter` rows, and each
query must match its readback by request id, channel, value type, and address.
The `operator_mode_status.connection_mode` and `state_update.mode` event
payloads must prove `wing` mode.
Event timestamps must be valid UTC values, monotonic in event order, and not
later than `summary.json` `captured_at`. Event payloads must not include
mutation evidence such as apply/rollback/write results, requested changes, or
truthy live-authority flags.
The review must also show `event_log_run_id_matches_summary: true`; event rows
with a missing or different `run_id` keep the artifact invalid.
The offline plan and handoff `fail_closed_conditions` must explicitly include
`event_log_run_id_mismatch`.

The offline review report must show:

- `manual_review_required_before_readiness_update: true`
- `manual_review_decision_required: true` only when a clean candidate exists
- `artifact_operator_next_steps` when blocked artifacts include a specific
  repair instruction
- checklist rows for artifact paths, operator channel scope, operator session
  readiness, physical readbacks, read-only event-log sequence, and no live
  authority

After a clean candidate is reviewed, record the operator decision as a JSON file
under:

`artifacts/wing_readonly_preflight_review_decisions/`

Minimum accepted-decision shape:

```json
{
  "review_id": "review-wing-readonly-<run_id>",
  "artifact_source_id": "artifacts/wing_readonly_preflight/<run_id>/summary.json",
  "run_id": "<run_id>",
  "artifact_captured_at": "2026-06-07T00:00:00Z",
  "artifact_root": "artifacts/wing_readonly_preflight",
  "summary_path": "artifacts/wing_readonly_preflight/<run_id>/summary.json",
  "summary_sha256": "<sha256 of summary.json>",
  "events_path": "artifacts/wing_readonly_preflight/<run_id>/events.jsonl",
  "events_sha256": "<sha256 of events.jsonl>",
  "channels": [3],
  "value_types": ["fader", "gain", "pan"],
  "operator_confirmed_channels": [3],
  "operator_confirmed_session_ready": true,
  "readback_count": 3,
  "physical_readback_count": 3,
  "request_id_policy": "required_for_every_query_mixer_parameter_response",
  "request_id_correlation_required": true,
  "event_log_run_id_matches_summary": true,
  "decision": "accepted",
  "reviewed_by": "<operator>",
  "reviewed_at": "2026-06-07T00:00:00Z",
  "checklist_ids": [
    "confirm_artifact_paths",
    "confirm_operator_channel_scope",
    "confirm_operator_session_ready",
    "confirm_physical_readbacks",
    "confirm_event_log_readonly_sequence",
    "confirm_event_log_run_id_binding",
    "confirm_no_live_authority"
  ]
}
```

Use `decision: "rejected"` if the operator does not accept the candidate. The
`review_id` field must remain present in the saved decision record. A missing
`review_id` keeps the record invalid even if the artifact hashes still match.
The `artifact_captured_at`, `artifact_root`, `summary_sha256`, `events_sha256`,
`channels`, `value_types`, `operator_confirmed_channels`,
`operator_confirmed_session_ready`, `readback_count`,
`physical_readback_count`, `request_id_policy`,
`request_id_correlation_required`, and `event_log_run_id_matches_summary` fields
must match the current captured files; stale decisions for replaced artifacts
fail closed.
`reviewed_at` must be the same time as or later than `artifact_captured_at`, so a
pre-filled or pre-capture operator decision cannot accept a later artifact. The
decision ledger still does not grant live authority or close readiness rows by
itself. The next supervised fader/gain proof plan remains blocked until a clean
candidate has exactly one accepted decision record. Duplicate decision records
for the same candidate, including accepted/rejected conflicts, are invalid and
must be replaced before the next step.

## Review Command

Run the offline artifact review after the artifact is captured:

```bash
python3 tools/review_wing_readonly_preflight_artifacts.py \
  --artifact-root artifacts/wing_readonly_preflight \
  --output Docs/reports/WING_READONLY_PREFLIGHT_REVIEW_LATEST.md \
  --decision-template-output artifacts/wing_readonly_preflight_review_decisions/pending_review_decision_template.json
```

Exit codes:

- `0`: clean candidate is ready for manual review;
- `2`: no artifact was found;
- `3`: invalid, corrupt, incomplete, or unsafe artifact evidence was found.

The decision template is intentionally not a valid accepted decision yet. The
operator must edit `decision`, `reviewed_by`, and `reviewed_at`, then save the
file as a durable review decision. Do not leave the generated
`template_status: operator_must_replace_placeholders_before_ledger_acceptance`
marker in an accepted decision; the decision ledger treats that marker as an
unedited pending template and fails closed. Keep `artifact_source_id`, `run_id`,
`artifact_captured_at`, `artifact_root`, `summary_path`, `events_path`, and
`checklist_ids` aligned with the template.

After editing the decision JSON, run the focused decision review:

```bash
python3 tools/review_wing_readonly_preflight_decisions.py \
  --artifact-root artifacts/wing_readonly_preflight \
  --decision-root artifacts/wing_readonly_preflight_review_decisions \
  --output Docs/reports/WING_READONLY_PREFLIGHT_DECISION_REVIEW_LATEST.md
```

Exit codes:

- `0`: an accepted decision is ready for the next manual WING MVP step only
  when the accepted source ids also have a valid accepted-candidate prestart
  trace, a valid accepted-candidate artifact-root trace, a valid
  accepted-candidate request-sequence trace, and the generated handoff keeps
  the supervised-proof transition behind a final accepted-candidate
  prestart/artifact-root/request-sequence gate;
- `2`: no clean preflight candidate exists yet;
- `3`: the decision record is missing, incomplete, invalid, or unsafe;
- `4`: the operator rejected the candidate, so supervised proof remains blocked.

If the decision review exits `3`, inspect the `Invalid Decision Records`
section in the generated markdown. It must show the blocked `review_id`,
`artifact_source_id`, `template_status`, and `violations`; stale generated
decision templates must expose `review_decision_template_status_unedited` until
the pending `template_status` marker is removed or replaced.
The no-WING `summary.json` and operator-readable `summary.md` mirror the same
decision-ledger `invalid_records` field, so the quick release gate must also
show the stale-template violation instead of only an invalid record id.

Then run the focused validation checks:

```bash
PYTHONPATH=backend python3 -m pytest -q \
  tests/test_wing_readonly_preflight_evidence.py \
  tests/test_wing_readonly_preflight_review_decision.py \
  tests/test_wing_readonly_preflight_review_tool.py \
  tests/test_no_wing_release_check.py
```

Then update:

- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`

## Fail Closed If

- backend is not in WING mode;
- any readback comes from cache instead of physical readback;
- any forbidden request type is sent;
- `--operator-confirm-session-ready` is missing for a physical read-only
  preflight;
- backend returns an unexpected response type for the request in flight;
- the backend WebSocket listener rejects the handshake, including HTTP 403
  before any read-only request is sent. In this case verify that `--url`
  points to the intended Automixer backend WebSocket and not another local
  service on the same port;
- backend transport fails (including send failure), the request times out, or the
  WebSocket connection is lost;
- `events.jsonl` is missing or does not match `summary.json`;
- `events.jsonl` is not strict `send`/`recv` pairs;
- `events.jsonl` does not prove WING mode from both `operator_mode_status` and
  `state_update` payloads;
- `events.jsonl` timestamps are invalid, non-monotonic, or later than
  `captured_at`;
- `events.jsonl` payloads include apply/write/rollback evidence or grant live
  authority;
- any readback omits its expected WING address or finite numeric value;
- a `query_mixer_parameter` response does not match its request identity;
- a `query_mixer_parameter` response omits or mismatches the request id;
- the real run omits the expected offline plan or does not match it;
- the real run omits `--prestart-summary`, the referenced no-WING summary is
  missing, stale, not `status: passed`, not `type: no_wing_release_check`, not
  `connection_policy: no_hardware_no_network_no_live_writes`, has
  `live_write: true`, lacks `backend_port_free`, or has
  `backend_port_free.status` other than `passed`;
- the referenced no-WING summary's `backend_port_free.detail` omits or changes
  `backend_port_prestart_check_failure_detail_fields`, including missing
  `listener_confirmation_checklist`, or any backend-port required-false detail
  field is missing or not `false`;
- the captured read-only summary's `backend_port_prestart_summary_sha256` does
  not match the referenced no-WING prestart summary file;
- the referenced no-WING prestart artifact's detached `artifact-manifest.json`
  is missing, absent from the same artifact directory as `summary.json`, or does
  not pass
  `PYTHONPATH=backend python3 tools/verify_no_wing_artifact_manifest.py
  <no_wing_artifact_dir>` with `status: passed`, `failed_checks: []`,
  `failed_checks_missing_check_entries: []`,
  `failed_checks_have_check_entries: true`,
  `verifier_report_required_fields_present: true`,
  `verifier_report_required_fields_missing: []`, and
  `verifier_report_required_fields_match_contract: true`,
  `summary_json_exists: true`,
  `summary_md_exists: true`,
  `artifact_manifest_exists: true`,
  `artifact_bundle_loadable: true`,
  `manifest_type: true`,
  `manifest_required_artifact_ids_match_contract: true`,
  `manifest_required_artifact_paths_match_contract: true`,
  `manifest_records_required_artifact_ids: true`,
  `summary_json_path_matches_contract: true`,
  `summary_json_sha256_matches_actual: true`,
  `summary_md_path_matches_contract: true`,
  `summary_md_sha256_matches_actual: true`,
  `summary_json_sha256_matches_manifest: true`,
  `summary_md_sha256_matches_manifest: true`,
  `manifest_denies_readiness_closure: true`,
  `manifest_denies_live_authority: true`,
  `summary_denies_readiness_closure: true`,
  `summary_denies_live_authority: true`,
  `summary_json_artifact_manifest_verifier_fields_present: true`,
  `manifest_artifact_manifest_verifier_fields_present: true`,
  `summary_md_artifact_manifest_verifier_fields_rendered: true`,
  `summary_json_artifact_manifest_verifier_command_matches_expected: true`,
  `summary_md_artifact_manifest_verifier_command_matches_expected: true`,
  `manifest_artifact_manifest_verifier_command_matches_expected: true`,
  `summary_json_artifact_manifest_verifier_report_required_fields_matches_expected: true`,
  `summary_md_artifact_manifest_verifier_report_required_fields_matches_expected: true`,
  and
  `manifest_artifact_manifest_verifier_report_required_fields_matches_expected: true`,
  `verifier_report_read_only_matches_expected: true`,
  `verifier_report_live_write_matches_expected: true`,
  `verifier_report_contacts_wing_matches_expected: true`,
  `verifier_report_contacts_backend_matches_expected: true`,
  `verifier_report_mutates_console_state_matches_expected: true`,
  `verifier_report_queries_console_matches_expected: true`, and
  `verifier_report_executes_tests_matches_expected: true`,
  `verifier_report_readiness_closure_allowed_matches_expected: true`, and
  `verifier_report_live_authority_granted_matches_expected: true`,
  or does
  not show `artifact_manifest_self_consistency_summary_surface_match: true`,
  `artifact_manifest_self_consistency_summary_surface_json_mismatches: []`,
  `artifact_manifest_self_consistency_summary_surface_markdown_mismatches: []`,
  `artifact_manifest_self_consistency_summary_surface_mismatches: []`,
  `artifact_manifest_self_consistency_summary_surface_readiness_closure_allowed: false`,
  `artifact_manifest_self_consistency_summary_surface_live_authority_granted: false`,
  `artifact_manifest_self_consistency_match: true`,
  `artifact_manifest_self_consistency_mismatches: []`,
  `artifact_manifest_contract_cross_surface_match: true`, and
  `artifact_manifest_contract_cross_surface_mismatches: []`, and the referenced
  no-WING summary or detached sidecar does not show
  `artifact_manifest_verifier_required: true`,
  `artifact_manifest_verifier_command`,
  `artifact_manifest_verifier_target_artifact_dir`,
  `artifact_manifest_verifier_status_required: passed`,
  `artifact_manifest_verifier_failed_checks_required: []`,
  `artifact_manifest_verifier_required_fields_missing: []`,
  `artifact_manifest_verifier_contacts_wing: false`,
  `artifact_manifest_verifier_mutates_console_state: false`,
  `artifact_manifest_verifier_queries_console: false`,
  `artifact_manifest_verifier_readiness_closure_allowed: false`, and
  `artifact_manifest_verifier_live_authority_granted: false`, and the referenced
  `artifact_manifest_verifier_report_required_fields` list does not include
  `checks`, `verifier_report_required_fields`,
  `verifier_report_required_fields_missing`, and
  `verifier_report_required_fields_match_contract`, and the referenced
  no-WING operator-readable `summary.md` does not show
  `wing_readonly_operator_sequence_list_markdown_surface_required_field_lines`,
  `wing_readonly_operator_sequence_list_markdown_surface_missing_field_lines: []`
  and
  `wing_readonly_operator_sequence_list_markdown_surface_field_lines_rendered: true`;
- the captured read-only summary does not show
  `artifact_manifest_verifier_must_pass_before_websocket: true`,
  `artifact_manifest_verifier_passed_before_websocket: true`, and
  `artifact_manifest_verifier_websocket_opened_after_pass: true`;
- the expected offline plan uses a stale backend-port prestart failure-detail
  contract, including missing `listener_confirmation_checklist`;
- the referenced offline plan file is missing or its SHA256 does not match
  `expected_plan_sha256`;
- the referenced offline plan content does not match the captured summary or
  read-only request sequence;
- the captured summary omits `artifact_root` or it does not match the
  referenced offline plan `artifact_root`;
- the summary does not prove the run required an already-connected
  backend/WING session;
- the run contacts the wrong mixer;
- the operator cannot confirm the selected channel;
- `operator_confirmed_channels` is missing or does not exactly match
  `channels`;
- the review report omits the manual review checklist;
- the review decision JSON is missing, malformed, not UTC stamped, linked to
  the wrong artifact/run, duplicated for the same candidate, conflicts with
  another decision for the same candidate, lacks a valid accepted-candidate
  prestart trace, lacks a valid accepted-candidate artifact-root trace, lacks a
  valid accepted-candidate request-sequence trace, or claims live/readiness
  authority.

## Next Step

Only after a clean artifact is manually reviewed, an accepted decision JSON is
recorded, and the no-WING release handoff proves the supervised prestart gate,
continue to:

`Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md`

The handoff validation must report:

- `operator_sequence_count: 12`
- `operator_sequence_sha256`
- `actual_operator_sequence_sha256`
- `operator_sequence_sha256_matches_actual: true`
- `expected_operator_sequence_sha256`
- `actual_expected_operator_sequence_sha256`
- `operator_sequence_matches_contract: true`
- `operator_sequence_sha256_matches_contract: true`
- `declared_expected_operator_sequence_sha256_matches_contract: true`
- `declared_operator_sequence_matches_contract_flag: true`
- `declared_operator_sequence_matches_contract_flag_matches_actual: true`
- `declared_operator_sequence_sha256_matches_contract_flag: true`
- `declared_operator_sequence_sha256_matches_contract_flag_matches_actual: true`
- `declared_operator_sequence_count: 12`
- `declared_expected_operator_sequence_count: 12`
- `declared_operator_sequence_count_contract_flag: true`
- `declared_operator_sequence_count_matches_actual: true`
- `declared_operator_sequence_count_matches_contract: true`
- `declared_expected_operator_sequence_count_matches_contract: true`
- `declared_operator_sequence_count_contract_flag_matches_declared_count: true`
- `backend_port_prestart_check_operator_sequence_step_index: 0`
- `backend_port_prestart_check_operator_sequence_step` includes
  `prestart_backend_port_check` and `unexpected listeners`
- `backend_port_prestart_check_must_be_first_operator_step: true`
- `backend_port_prestart_check_operator_sequence_step_is_first: true`
- `backend_port_prestart_gate` exactly matches
  `expected_backend_port_prestart_gate`
- `backend_port_prestart_gate.grants_live_authority: false`
- `backend_port_prestart_gate.must_pass_before_backend_wing_session: true`
- `backend_port_prestart_gate.operator_sequence_step_index: 0`
- `backend_port_prestart_gate.operator_sequence_step_must_be_first: true`
- `backend_port_prestart_gate.runs_command: prestart_backend_port_check`
- `backend_port_prestart_gate.required_markers` exactly lists
  `Before starting the intended backend/WING session`,
  `prestart_backend_port_check`, and `unexpected listeners`
- the failed busy-port no-WING summary and no-WING handoff summary both show
  `backend_port_prestart_check_command` exactly matching
  `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick`
- the same summaries also show `backend_port_prestart_check_command_text`
  exactly matching
  `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick`
- the same summaries also show `backend_port_prestart_check_required: true`
  and `backend_port_prestart_check_must_pass_before_starting_backend: true`
- the same summaries also show
  `backend_port_prestart_gate_execution_allowed: false`,
  `backend_port_prestart_gate_readiness_closure_allowed: false`, and
  `backend_port_prestart_gate_live_authority_granted: false`
- the no-WING handoff summary also shows
  `plan_safe_to_stop_listener_by_gate: false`,
  `plan_live_authority_granted: false`,
  `plan_backend_port_prestart_gate_execution_allowed: false`,
  `plan_backend_port_prestart_gate_readiness_closure_allowed: false`,
  and `plan_backend_port_prestart_gate_live_authority_granted: false`
- the no-WING handoff summary also shows
  `safe_to_stop_listener_by_gate_matches_plan: true`,
  `live_authority_granted_matches_plan: true`,
  `backend_port_prestart_gate_execution_allowed_matches_plan: true`,
  `backend_port_prestart_gate_readiness_closure_allowed_matches_plan: true`,
  and `backend_port_prestart_gate_live_authority_granted_matches_plan: true`
- the no-WING handoff summary shows
  `artifact_manifest_verifier_operator_sequence_step_index: 1`,
  `expected_artifact_manifest_verifier_operator_sequence_step_index: 1`,
  `artifact_manifest_verifier_operator_sequence_step_index_matches_contract: true`,
  `artifact_manifest_verifier_operator_sequence_step`,
  `expected_artifact_manifest_verifier_operator_sequence_step`,
  `artifact_manifest_verifier_operator_sequence_step_matches_contract: true`,
  `artifact_manifest_verifier_required_markers`, and
  `artifact_manifest_verifier_operator_sequence_before_preview: true`
- the same summaries also show `backend_port_prestart_check_failure_detail_fields`
  exactly listing
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
  `fail_closed_reason`, and `operator_next_step`
- If no-WING quick reports `environment_blocked_checks`, the no-WING summary
  must also expose `environment_blocked_check_details` with
  `local_dev_server_bind_denied`, `127.0.0.1:3000`, log paths, and
  `operator_next_step`. That result remains failed until rerun in an
  environment that can bind the frontend dev server.
- `manual_review_required_before_readiness_update: true`
- `readiness_matrix_update_allowed_by_packet: false`
- `readiness_closure_allowed_by_packet: false`
- If validation fails because a handoff packet set execution, readiness,
  live-authority, or full-autonomous flags to `true`, the no-WING summary must
  preserve those unsafe `true` values so the failed handoff is auditable.
- If validation fails because a handoff packet set `live_write`,
  `mutates_console_state`, `queries_console`, or `executes_tests` to `true`, the
  no-WING summary must preserve those unsafe `true` values so the failed handoff
  is auditable.
- `read_only: true`
- `live_write: false`
- `mutates_console_state: false`
- `queries_console: false`
- `executes_tests: false`
- `operator_required: true`
- `expected_plan_required_for_real_run: true`
- `physical_preflight_requires_existing_backend_connection: true`
- the accepted read-only preflight artifact shows
  `artifact_manifest_verifier_must_pass_before_websocket: true`,
  `artifact_manifest_verifier_passed_before_websocket: true`, and
  `artifact_manifest_verifier_websocket_opened_after_pass: true`
- the no-WING handoff summary also shows the same
  `backend_port_prestart_check_required: true` and
  `backend_port_prestart_check_must_pass_before_starting_backend: true`
- the no-WING handoff summary also shows
  `backend_port_prestart_gate_execution_allowed: false`,
  `backend_port_prestart_gate_readiness_closure_allowed: false`, and
  `backend_port_prestart_gate_live_authority_granted: false`
- the no-WING handoff summary also shows
  `plan_safe_to_stop_listener_by_gate: false`,
  `plan_live_authority_granted: false`,
  `plan_backend_port_prestart_gate_execution_allowed: false`,
  `plan_backend_port_prestart_gate_readiness_closure_allowed: false`,
  and `plan_backend_port_prestart_gate_live_authority_granted: false`
- the no-WING handoff summary also shows
  `safe_to_stop_listener_by_gate_matches_plan: true`,
  `live_authority_granted_matches_plan: true`,
  `backend_port_prestart_gate_execution_allowed_matches_plan: true`,
  `backend_port_prestart_gate_readiness_closure_allowed_matches_plan: true`,
  and `backend_port_prestart_gate_live_authority_granted_matches_plan: true`
- the no-WING handoff summary also shows
  `backend_port_prestart_summary_path_placeholder` pointing to the passed
  no-WING prestart `summary.json`,
  `preview_command_prestart_summary` matching that path,
  `preview_command_prestart_summary_matches_plan: true`,
  `validate_inputs_command_prestart_summary` matching that path,
  `validate_inputs_command_prestart_summary_matches_plan: true`,
  `capture_command_prestart_summary` matching that path, and
  `capture_command_prestart_summary_matches_plan: true`
- `execution_allowed_by_packet: false`
- `live_authority_granted: false`
- `full_autonomous_ready: false`
- `next_manual_artifact_required: real wing_readonly_preflight.v1 summary.json`
- `operator_sequence` lists the actual ordered handoff checklist
- `expected_operator_sequence` lists the expected ordered handoff checklist
- `wing_readonly_operator_sequence_list_summary_surface_self_check_required: true`
- `wing_readonly_operator_sequence_list_summary_surface_mismatches: []`
- `wing_readonly_operator_sequence_list_summary_surface_match: true`
- `wing_readonly_operator_sequence_list_summary_surface_readiness_closure_allowed: false`
- `wing_readonly_operator_sequence_list_summary_surface_live_authority_granted: false`
- `wing_readonly_operator_sequence_list_markdown_surface_required: true`
- `wing_readonly_operator_sequence_list_markdown_surface_required_field_lines`
- `wing_readonly_operator_sequence_list_markdown_surface_required_field_lines_expected_value`
- `wing_readonly_operator_sequence_list_markdown_surface_required_field_lines_actual_value`
- `wing_readonly_operator_sequence_list_markdown_surface_required_field_lines_match_contract: true`
- `wing_readonly_operator_sequence_list_markdown_surface_missing_field_lines: []`
- `wing_readonly_operator_sequence_list_markdown_surface_field_lines_rendered: true`
- `wing_readonly_operator_sequence_list_markdown_surface_readiness_closure_allowed: false`
- `wing_readonly_operator_sequence_list_markdown_surface_live_authority_granted: false`
These Markdown-surface verdict lines are part of the no-WING fail-closed
contract; if any are missing, do not open the WING session.
- `expected_artifacts` exactly matches `expected_artifact_globs` and
  `plan_expected_artifacts`
- `expected_artifacts_match_plan: true`
- `summary_glob_matches_plan: true`
- `event_log_glob_matches_plan: true`
- `preview_command_artifact_root_matches_packet: true`
- `preview_command_plan_output_matches_packet: true`
- `validate_inputs_command_uses_validate_run_inputs_only: true`
- `validate_inputs_command_uses_plan_only: false`
- `validate_inputs_command_artifact_root_matches_packet: true`
- `validate_inputs_command_expected_plan_matches_packet: true`
- `capture_command_artifact_root_matches_packet: true`
- `capture_command_expected_plan_matches_packet: true`
- `review_command_artifact_root_matches_packet: true`
- `decision_review_command_artifact_root_matches_packet: true`
- `decision_review_command_decision_root_matches_packet: true`
- `mvp_blocker_anchor` exactly matches the current first open blocker row
- `mvp_blocker_anchor_matches_current_first_open: true`
- `mvp_blocker_anchor_sha256_matches_current_first_open: true`
- `mvp_blocker_anchor.packet_does_not_close_blocker: true`
- `operator_sequence_has_mvp_blocker_anchor_digest_review: true`
- `mvp_blocker_anchor_digest_review_step_index: 10`
- `expected_mvp_blocker_anchor_digest_review_step_index: 10`
- `mvp_blocker_anchor_digest_review_step_index_matches_contract: true`
- `mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true`
- `operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10`
- `expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10`
- `operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true`
- `operator_sequence_mvp_blocker_anchor_digest_review_step` includes
  `mvp_blocker_anchor`, `mvp_blocker_anchor_sha256`, and
  `current first open blocker`
- `backend_connection_confirmation_step_index: 3`
- `expected_backend_connection_confirmation_step_index: 3`
- `operator_sequence_has_backend_connection_confirmation: true`
- `operator_sequence_backend_connection_confirmation_step_index: 3`
- `operator_sequence_backend_connection_confirmation_before_capture: true`
- `operator_sequence_backend_connection_confirmation_step` includes
  `existing backend connection`, `intended WING Rack`, and
  `read-only inspection`
- `operator_sequence_validate_readonly_preflight_inputs_step_index: 5`
- `expected_operator_sequence_validate_readonly_preflight_inputs_step_index: 5`
- `operator_sequence_validate_readonly_preflight_inputs_step_index_matches_contract: true`
- `operator_sequence_validate_readonly_preflight_inputs_step: Run validate_readonly_preflight_inputs before capture; it must validate real-run inputs and expected-plan bindings without opening transport.`
- `expected_operator_sequence_validate_readonly_preflight_inputs_step: Run validate_readonly_preflight_inputs before capture; it must validate real-run inputs and expected-plan bindings without opening transport.`
- `operator_sequence_validate_readonly_preflight_inputs_step_matches_contract: true`
- `operator_sequence_capture_readonly_preflight_step_index: 6`
- `expected_operator_sequence_capture_readonly_preflight_step_index: 6`
- `operator_sequence_capture_readonly_preflight_step_index_matches_contract: true`
- `operator_sequence_capture_readonly_preflight_step: Run capture_readonly_preflight.`
- `expected_operator_sequence_capture_readonly_preflight_step: Run capture_readonly_preflight.`
- `operator_sequence_capture_readonly_preflight_step_matches_contract: true`
- the final handoff operator-sequence step contains the supervised prestart gate
- `operator_sequence_has_supervised_prestart_gate: true`
- `operator_sequence_supervised_prestart_gate_is_final: true`
- `operator_sequence_supervised_prestart_gate_step_index: 11`
- `operator_sequence_supervised_prestart_gate_step` includes
  `accepted_decision_ready`, `valid accepted-candidate prestart trace`,
  `valid accepted-candidate artifact-root trace`,
  `valid accepted-candidate request-sequence trace`,
  `expected-artifact-bound accepted source id`, and
  `WING_SUPERVISED_FADER_GAIN_PROOF`
- `required_supervised_prestart_gate_markers` exactly lists
  `accepted_decision_ready`, `valid accepted-candidate prestart trace`,
  `valid accepted-candidate artifact-root trace`,
  `valid accepted-candidate request-sequence trace`,
  `expected-artifact-bound accepted source id`, and
  `WING_SUPERVISED_FADER_GAIN_PROOF`
- `supervised_prestart_gate` exactly matches
  `expected_supervised_prestart_gate`
- `supervised_prestart_gate.grants_live_authority: false`
- `supervised_prestart_gate.operator_sequence_step_index: 11`
- `supervised_prestart_gate.operator_sequence_step_must_be_final: true`
- `supervised_prestart_gate.requires_operator_decision_status:
  accepted_decision_ready`
- `supervised_prestart_gate.unlocks_runbook:
  Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md`
- `supervised_prestart_gate.requires_accepted_candidate_prestart_trace: true`
- `supervised_prestart_gate.requires_accepted_candidate_artifact_root_trace: true`
- `supervised_prestart_gate.requires_accepted_candidate_request_sequence_trace: true`
- `accepted_candidate_prestart_contract_violations_by_source: {}`
- `readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source: {}`
- `readonly_preflight_review_decision_expected_candidate_prestart_contract_violations_by_source: {}`
- `readonly_preflight_review_decision_candidate_artifact_root_contract_violations_by_source: {}`
- `readonly_preflight_review_decision_expected_candidate_artifact_root_contract_violations_by_source: {}`

Until those markers are present in the final operator step, the structured gate
matches the expected gate, and no-WING validation avoids
`next_wing_proof_plan_steps_with_candidate_prestart_violations`,
`next_wing_proof_plan_steps_with_expected_candidate_prestart_violations`, and
`next_wing_proof_plan_steps_with_candidate_artifact_root_violations`, and
`next_wing_proof_plan_steps_with_expected_candidate_artifact_root_violations`,
the result remains a read-only proof artifact and grants no supervised write
authority.
