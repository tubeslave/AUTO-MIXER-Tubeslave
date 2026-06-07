# ADR: Fail Closed On Unexpected WING Read-Only Preflight Responses

## Context

The read-only preflight artifact is the first real WING proof. Its event log
must be strict `send`/`recv` pairs with the expected response type for each
request.

## Options Considered

1. Continue waiting through unrelated backend events until the expected response
   arrives.
2. Buffer and classify unrelated events separately.
3. Fail closed on the first unexpected response type.

## Decision

Use option 3. For proof mode, the next payload after a request must be the
expected response type.

## Why This Won

It aligns runtime behavior with the evidence-review contract and prevents a run
from continuing after the event log is already invalid. It also avoids
introducing buffering semantics into the proof path.

## Rejected Alternatives

- Waiting through extra events weakens the proof artifact.
- Buffering is more complex than needed for a first hardware gate and could hide
  backend noise that the operator should see.

## Implementation Plan

- Add a typed unexpected-response error to the runner.
- Make `send_and_wait` consume exactly one response.
- On mismatch, write a blocked summary with request type, expected types,
  actual type, and payload.
- Add a regression test proving later readback queries are not sent.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_runner.py`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_no_wing_unexpected_response_gate_20260607`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks And Rollback

If the backend emits benign unsolicited events, operators may need a quiet
backend session for proof capture. Rollback would restore permissive waiting,
but that should require a deliberate safety decision.
