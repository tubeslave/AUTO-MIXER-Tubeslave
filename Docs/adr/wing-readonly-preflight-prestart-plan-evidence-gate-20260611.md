# ADR: WING Read-Only Preflight Prestart Plan Evidence Gate

## Context

The supervised WING Rack MVP is still waiting on real WING read-only connection proof. The runbook requires `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick` before starting the intended backend/WING session. The offline plan already records this boundary, but the evidence validator must also reject a linked expected plan that omits or changes it.

## Options considered

1. Trust only the runner's pre-WebSocket expected-plan validation.
2. Validate the prestart boundary again during file-backed evidence intake.
3. Require a completed no-WING artifact hash in every WING preflight plan.

## Decision

Use option 2. The evidence intake now validates the prestart required flags, command, and diagnostic failure detail fields against the shared WING preflight contract constants.

## Why this won

Artifact review may happen after capture, from files, and without rerunning the original command. The review path therefore needs its own fail-closed check for the required prestart boundary.

## Rejected alternatives

Option 1 leaves archived or hand-edited plan files too trusted. Option 3 may be valuable later, but it expands the runbook contract and is not needed for this narrow blocker hardening.

## Implementation plan

- Import the shared prestart contract constants in `backend/wing_readonly_preflight_evidence.py`.
- Reject expected plans with missing, false, or mismatched prestart fields.
- Extend the expected-plan policy metadata regression.

## Test plan

- `python3 -m py_compile backend/wing_readonly_preflight_evidence.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py`
- Adjacent WING preflight/no-WING tests before commit.

## Risks and rollback

Risk: older local sample artifacts may need regenerated expected-plan files. Rollback: remove the new prestart field checks and test assertions. No live write authority is changed.
