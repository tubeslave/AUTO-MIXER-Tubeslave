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

- The intended WING Rack is connected to the backend.
- The backend WebSocket is running, usually at `ws://localhost:8765`.
- The operator knows which channel is safe to inspect.
- No live performance is in progress unless the operator explicitly allows
  read-only inspection.
- Do not run supervised proof yet.

## Command

Replace channel `3` with the operator-approved channel:

```bash
python3 tools/wing_readonly_preflight_runner.py \
  --url ws://localhost:8765 \
  --channels 3 \
  --operator-confirm-channels 3 \
  --value-types fader,gain \
  --require-physical-readback \
  --artifact-root artifacts/wing_readonly_preflight
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
- `read_only: true`
- `live_write: false`
- `write_request_sent: false`
- `proposal_request_sent: false`
- `forbidden_request_types_sent: []`
- `require_physical_readback: true`
- `physical_readback_count == readback_count`

The event log must show only these request types:

- `get_operator_mode`
- `get_state`
- `query_mixer_parameter`

The event log must also be strict `send`/`recv` pairs. The request order must
be `get_operator_mode`, `get_state`, then `query_mixer_parameter` rows, and each
query must match its readback by channel, value type, and address.

The offline review report must show:

- `manual_review_required_before_readiness_update: true`
- `manual_review_decision_required: true` only when a clean candidate exists
- checklist rows for artifact paths, operator channel scope, physical
  readbacks, read-only event-log sequence, and no live authority

After a clean candidate is reviewed, record the operator decision as a JSON file
under:

`artifacts/wing_readonly_preflight_review_decisions/`

Minimum accepted-decision shape:

```json
{
  "review_id": "review-wing-readonly-<run_id>",
  "artifact_source_id": "artifacts/wing_readonly_preflight/<run_id>/summary.json",
  "run_id": "<run_id>",
  "summary_path": "artifacts/wing_readonly_preflight/<run_id>/summary.json",
  "events_path": "artifacts/wing_readonly_preflight/<run_id>/events.jsonl",
  "decision": "accepted",
  "reviewed_by": "<operator>",
  "reviewed_at": "2026-06-07T00:00:00Z",
  "checklist_ids": [
    "confirm_artifact_paths",
    "confirm_operator_channel_scope",
    "confirm_physical_readbacks",
    "confirm_event_log_readonly_sequence",
    "confirm_no_live_authority"
  ]
}
```

Use `decision: "rejected"` if the operator does not accept the candidate. The
decision ledger still does not grant live authority or close readiness rows by
itself. The next supervised fader/gain proof plan remains blocked until a clean
candidate has an accepted decision record.

## Review Command

Run the offline artifact review after the artifact is captured:

```bash
python3 tools/review_wing_readonly_preflight_artifacts.py \
  --artifact-root artifacts/wing_readonly_preflight \
  --output Docs/reports/WING_READONLY_PREFLIGHT_REVIEW_LATEST.md
```

Exit codes:

- `0`: clean candidate is ready for manual review;
- `2`: no artifact was found;
- `3`: invalid, corrupt, incomplete, or unsafe artifact evidence was found.

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
- `events.jsonl` is missing or does not match `summary.json`;
- `events.jsonl` is not strict `send`/`recv` pairs;
- a `query_mixer_parameter` response does not match its request identity;
- the run contacts the wrong mixer;
- the operator cannot confirm the selected channel;
- `operator_confirmed_channels` is missing or does not exactly match
  `channels`;
- the review report omits the manual review checklist;
- the review decision JSON is missing, malformed, not UTC stamped, linked to
  the wrong artifact/run, or claims live/readiness authority.

## Next Step

Only after a clean artifact is manually reviewed and an accepted decision JSON
is recorded, continue to:

`Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md`
