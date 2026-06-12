# Offline no-WING Runbook Proof Plan Candidate Prestart Gate

## Context

The no-WING release gate now requires a specific blocker when accepted WING read-only preflight source ids are not backed by candidate review rows with the backend-port prestart contract. The offline no-WING validation runbook should name the same gate so an operator does not rely on a generic `next_wing_proof_plan_schema` pass.

## Task

Update `Docs/OFFLINE_NO_WING_VALIDATION_RUNBOOK.md` success criteria to include the candidate-prestart fields and blocker required before physical WING proof steps are acceptable.

## Constraints

- Documentation/test-only change.
- No backend startup.
- No WING transport usage.
- No OSC/WebSocket writes.

## Test Plan

- Focused no-WING runbook test.
- Affected no-WING tests.
- Full backend pytest suite.
- `git diff --check`.
- `tools/memory/memory_lint.py`.
