# Brief: WING Read-Only Preflight Expected-Plan Gate - 2026-06-07

## Task

Bind the real WING read-only preflight capture run to the previously generated
offline request plan.

## Context

The first open MVP blocker remains `WING read-only connection proof`. The
runbook already asks the operator to generate a plan-only JSON preview before
opening the backend/WING session, then run the real read-only capture. The real
capture did not yet prove it used the same URL, channel scope, value types,
artifact root, request ids, and request sequence as the reviewed offline plan.

## Safety Boundary

- No WING, OSC, or WebSocket transport is contacted by this slice.
- No live write path is widened.
- No readiness row or MVP blocker is closed.
- The real preflight still requires operator channel confirmation and physical
  readback.

## Expected Output

- Real runner supports `--expected-plan`.
- Runner validates runtime args and request sequence against the expected plan
  before opening WebSocket.
- Summary records expected plan path, SHA256, match flag, and violations.
- Evidence intake rejects clean-looking artifacts that are not linked to a
  matching expected plan.
- Handoff/no-WING/runbook commands use the expected plan.
