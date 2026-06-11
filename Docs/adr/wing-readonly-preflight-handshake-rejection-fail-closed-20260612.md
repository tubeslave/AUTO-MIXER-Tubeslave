# wing-readonly-preflight-handshake-rejection-fail-closed-20260612

## Context

A real local probe found a backend listener on `ws://localhost:8765`, but the
read-only WING preflight WebSocket handshake was rejected with HTTP 403 before
any request was sent. The runner previously grouped this with generic transport
errors, which made the first WING MVP blocker harder to diagnose.

## Options considered

1. Leave HTTP 403 as a generic transport failure.
2. Treat handshake rejection as a successful backend-contact proof.
3. Keep it fail-closed, but classify it as a distinct handshake blocker.

## Decision

Use option 3. The preflight runner now reports
`read_only_preflight_ws_handshake_rejected` with blocker
`ws_handshake_rejected` when the WebSocket handshake is rejected before
read-only requests are sent. The summary also records transport context that
tells the operator to verify the `--url` points at the intended Automixer
backend rather than another local service.

## Why this won

The backend listener is not enough to prove WING read-only readiness. A rejected
handshake still blocks the proof, but the operator now sees the exact boundary:
the backend port is open, yet the preflight WebSocket session was not accepted.

## Rejected alternatives

Generic transport failure was rejected because it hides a common setup problem.
Treating the listener as progress was rejected because no operator mode, state,
or physical readback evidence was captured.

## Implementation plan

1. Add a small transport-failure classifier to the runner.
2. Add a regression test for HTTP 403 handshake rejection before any send.
3. Record an operator next-step hint in the fail-closed summary.
4. Document the fail-closed condition in the WING read-only runbook.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_runner.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_no_wing_release_check.py`

## Risks and rollback

The change only affects diagnostic reason strings for preflight transport
failures. If a downstream report expects the old generic reason, update it to
accept the specific fail-closed blocker or revert this ADR and the runner/test
changes.
