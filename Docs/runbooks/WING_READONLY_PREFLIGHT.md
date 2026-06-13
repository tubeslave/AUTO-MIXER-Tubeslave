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
step as `operator_sequence_mvp_blocker_anchor_digest_review_step_index: 8` so
reviewers can confirm the digest check occurs immediately before the final
supervised prestart gate.

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
session is ready. The real run must point at the offline plan generated above
with `--expected-plan`; if the flag is omitted, or if the runtime arguments or
request sequence do not match that plan, the runner fails closed before opening
WebSocket and writes `operator_next_step` in `summary.json`. It must also point
at the same no-WING prestart summary used for the plan:

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
- `backend_port_prestart_summary_live_write: false`
- `backend_port_prestart_summary_contacts_backend: false`
- `backend_port_prestart_summary_queries_console: false`
- `backend_port_prestart_summary_executes_tests: false`
- `backend_port_prestart_check_status: passed`
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
  "summary_path": "artifacts/wing_readonly_preflight/<run_id>/summary.json",
  "summary_sha256": "<sha256 of summary.json>",
  "events_path": "artifacts/wing_readonly_preflight/<run_id>/events.jsonl",
  "events_sha256": "<sha256 of events.jsonl>",
  "decision": "accepted",
  "reviewed_by": "<operator>",
  "reviewed_at": "2026-06-07T00:00:00Z",
  "checklist_ids": [
    "confirm_artifact_paths",
    "confirm_operator_channel_scope",
    "confirm_operator_session_ready",
    "confirm_physical_readbacks",
    "confirm_event_log_readonly_sequence",
    "confirm_no_live_authority"
  ]
}
```

Use `decision: "rejected"` if the operator does not accept the candidate. The
`artifact_captured_at`, `summary_sha256`, and `events_sha256` fields must match
the current captured files; stale decisions for replaced artifacts fail closed.
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
file as a durable review decision. Keep `artifact_source_id`, `run_id`,
`artifact_captured_at`, `summary_path`, `events_path`, and `checklist_ids`
aligned with the template.

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
  trace and the generated handoff keeps the supervised-proof transition behind
  a final accepted-candidate prestart gate;
- `2`: no clean preflight candidate exists yet;
- `3`: the decision record is missing, incomplete, invalid, or unsafe;
- `4`: the operator rejected the candidate, so supervised proof remains blocked.

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
- the captured read-only summary's `backend_port_prestart_summary_sha256` does
  not match the referenced no-WING prestart summary file;
- the expected offline plan uses a stale backend-port prestart failure-detail
  contract, including missing `listener_confirmation_checklist`;
- the referenced offline plan file is missing or its SHA256 does not match
  `expected_plan_sha256`;
- the referenced offline plan content does not match the captured summary or
  read-only request sequence;
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
  prestart trace, or claims live/readiness authority.

## Next Step

Only after a clean artifact is manually reviewed, an accepted decision JSON is
recorded, and the no-WING release handoff proves the supervised prestart gate,
continue to:

`Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md`

The handoff validation must report:

- `operator_sequence_count: 10`
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
- the same summaries also show `backend_port_prestart_check_required: true`
  and `backend_port_prestart_check_must_pass_before_starting_backend: true`
- the same summaries also show
  `backend_port_prestart_gate_execution_allowed: false`,
  `backend_port_prestart_gate_readiness_closure_allowed: false`, and
  `backend_port_prestart_gate_live_authority_granted: false`
- the same summaries also show `backend_port_prestart_check_failure_detail_fields`
  exactly listing
  `listener_processes`, `listener_process_error`, `listener_confirmation_checklist`,
  `preflight_blocked_until_listener_confirmed`,
  `intended_backend_confirmation_required`, `safe_to_stop_listener_by_gate`,
  `listener_confirmation_template_status`,
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
- `manual_review_required_before_readiness_update: true`
- `readiness_matrix_update_allowed_by_packet: false`
- `readiness_closure_allowed_by_packet: false`
- `read_only: true`
- `live_write: false`
- `mutates_console_state: false`
- `queries_console: false`
- `executes_tests: false`
- `operator_required: true`
- `expected_plan_required_for_real_run: true`
- `physical_preflight_requires_existing_backend_connection: true`
- the no-WING handoff summary also shows the same
  `backend_port_prestart_check_required: true` and
  `backend_port_prestart_check_must_pass_before_starting_backend: true`
- the no-WING handoff summary also shows
  `backend_port_prestart_gate_execution_allowed: false`,
  `backend_port_prestart_gate_readiness_closure_allowed: false`, and
  `backend_port_prestart_gate_live_authority_granted: false`
- `execution_allowed_by_packet: false`
- `live_authority_granted: false`
- `full_autonomous_ready: false`
- `next_manual_artifact_required: real wing_readonly_preflight.v1 summary.json`
- `mvp_blocker_anchor` exactly matches the current first open blocker row
- `mvp_blocker_anchor_matches_current_first_open: true`
- `mvp_blocker_anchor_sha256_matches_current_first_open: true`
- `mvp_blocker_anchor.packet_does_not_close_blocker: true`
- `operator_sequence_has_mvp_blocker_anchor_digest_review: true`
- `operator_sequence_mvp_blocker_anchor_digest_review_step_index: 8`
- `operator_sequence_mvp_blocker_anchor_digest_review_step` includes
  `mvp_blocker_anchor`, `mvp_blocker_anchor_sha256`, and
  `current first open blocker`
- the final handoff operator-sequence step contains the supervised prestart gate
- `operator_sequence_has_supervised_prestart_gate: true`
- `operator_sequence_supervised_prestart_gate_is_final: true`
- `operator_sequence_supervised_prestart_gate_step_index: 9`
- `operator_sequence_supervised_prestart_gate_step` includes
  `accepted_decision_ready`, `valid accepted-candidate prestart trace`, and
  `WING_SUPERVISED_FADER_GAIN_PROOF`
- `required_supervised_prestart_gate_markers` exactly lists
  `accepted_decision_ready`, `valid accepted-candidate prestart trace`, and
  `WING_SUPERVISED_FADER_GAIN_PROOF`
- `supervised_prestart_gate` exactly matches
  `expected_supervised_prestart_gate`
- `supervised_prestart_gate.grants_live_authority: false`
- `supervised_prestart_gate.operator_sequence_step_index: 9`
- `supervised_prestart_gate.operator_sequence_step_must_be_final: true`
- `supervised_prestart_gate.requires_operator_decision_status:
  accepted_decision_ready`
- `supervised_prestart_gate.unlocks_runbook:
  Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md`
- `supervised_prestart_gate.requires_accepted_candidate_prestart_trace: true`

Until those markers are present in the final operator step and the structured
gate matches the expected gate, the result remains a read-only proof artifact
and grants no supervised write authority.
