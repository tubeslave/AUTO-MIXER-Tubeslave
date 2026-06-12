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
  `preflight_blocked_until_listener_confirmed`,
  `intended_backend_confirmation_required`, `safe_to_stop_listener_by_gate`,
  `fail_closed_reason`, and `operator_next_step` before changing local
  services. Optional `command_line` diagnostics may identify the listener, but
  the real preflight remains blocked until the operator confirms the listener
  is the intended backend or frees/reconfigures the unexpected listener. The
  gate is diagnostic only and does not stop processes.
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
  --channels 3 \
  --output Docs/reports/WING_READONLY_PREFLIGHT_HANDOFF_PACKET_LATEST.md
```

Replace channel `3` with the operator-approved channel:

Before opening the backend/WING session, preview the exact read-only request
plan. This validates the channel confirmation and writes only a local JSON plan;
it does not open WebSocket, contact WING, query the console, or grant live
authority. The plan also records the current MVP blocker, fail-closed
conditions, and the offline review commands to run after the real artifact is
captured. It must also show `expected_plan_required_for_real_run: true` and
`physical_preflight_requires_existing_backend_connection: true`:

```bash
python3 tools/wing_readonly_preflight_runner.py \
  --url ws://localhost:8765 \
  --channels 3 \
  --operator-confirm-channels 3 \
  --operator-confirm-session-ready \
  --value-types fader,gain \
  --require-physical-readback \
  --artifact-root artifacts/wing_readonly_preflight \
  --plan-only \
  --plan-output artifacts/wing_readonly_preflight_plan/latest_plan.json
```

Then run the real read-only preflight only when the intended backend/WING
session is ready. The real run must point at the offline plan generated above
with `--expected-plan`; if the flag is omitted, or if the runtime arguments or
request sequence do not match that plan, the runner fails closed before opening
WebSocket:

```bash
python3 tools/wing_readonly_preflight_runner.py \
  --url ws://localhost:8765 \
  --channels 3 \
  --operator-confirm-channels 3 \
  --operator-confirm-session-ready \
  --value-types fader,gain \
  --require-physical-readback \
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

- `0`: an accepted decision is ready for the next manual WING MVP step;
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
  another decision for the same candidate, or claims live/readiness authority.

## Next Step

Only after a clean artifact is manually reviewed and an accepted decision JSON
is recorded, continue to:

`Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md`
