# Offline No-WING Validation Harness

## Context

Automixer needs a way to validate proposal generation, safety gating, rollback
logic, and cockpit-facing evidence when the real WING Rack is not connected.
Existing real-WING proof is valuable but must not be required for every
frontend/backend iteration.

## Options Considered

1. Reuse the real WING validation runner in dry-run mode.
2. Start the backend and connect it to a virtual mixer service.
3. Add a small in-process offline harness that uses Product Layer analysis and
   simulated readback.

## Decision

Use an in-process offline harness:

- `backend/offline_validation_harness.py`
- `tools/offline_operator_validation_harness.py`
- `tests/test_offline_validation_harness.py`

## Why This Won

The harness is deterministic, fast, testable, and impossible to confuse with a
real mixer session. It exercises proposal shape and safety review without
network transport or live write dependencies.

## Rejected Alternatives

- Real-runner dry-run only: useful, but it still assumes a WebSocket backend and
  does not simulate full proposal/readback/rollback proof in one local unit.
- Virtual mixer service: closer to integration testing, but heavier and easier
  to mistake for deployment readiness.

## Implementation Plan

1. Build deterministic offline fixture channels.
2. Use existing operator analysis to produce proposal previews.
3. Review requested changes fail-closed.
4. Simulate supervised apply/readback/rollback only for fader/gain down changes.
5. Prove Analyze/Assist/Auto do not apply.
6. Emit summary and event artifacts.

## Test Plan

- Focused pytest for the harness.
- CLI run that writes `summary.json` and `events.jsonl`.
- Existing backend/frontend suites before merge.

## Risks And Rollback

Risk: offline evidence could be overstated as live readiness.

Mitigation: every output marks `connection_policy` as
`no_hardware_no_network_no_live_writes`, and the report explicitly says real
WING validation is still required.

Rollback: remove the harness files and tests; no production live path depends
on them.
