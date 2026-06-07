# Brief: WING Read-Only Preflight Existing Backend Connection Evidence Gate

## Task ID

wing-readonly-preflight-existing-backend-connection-evidence-gate-20260607

## Context

The supervised WING Rack MVP is still on the first open blocker: real WING
read-only connection proof. The runner already writes
`physical_preflight_requires_existing_backend_connection: true` for real
preflight summaries, showing that it does not connect to or disconnect from the
mixer directly.

Evidence intake should require that explicit boundary before accepting a
summary as a clean read-only preflight candidate.

## Scope

- Keep the change read-only and offline-testable.
- Do not contact WING Rack, WebSocket, OSC, or MIDI hardware.
- Do not widen live writes, proposal apply paths, rollback, emergency stop, or
  readiness closure.
- Reject preflight summaries that omit or falsify the existing backend
  connection boundary.

## Deliverables

- Evidence validation gate.
- Updated valid fixtures and focused regression test.
- Runbook and documentation update.
- ADR/report with validation evidence.

## Validation

- Focused WING preflight evidence/review/decision tests.
- Runner test for the offline plan field.
- No-WING release quick check.
- Full pytest suite before commit.
