# ADR: WING read-only runbook artifact next-step gate

## Context

Blocked WING read-only artifacts can now surface repair guidance through the
no-WING summary as `artifact_operator_next_steps`. The runbook previously did
not name that field.

## Options Considered

1. Leave the runbook unchanged.
2. Mention only `operator_next_step` from individual runner summaries.
3. Add `artifact_operator_next_steps` to the prestart runbook gate.

## Decision

Add `artifact_operator_next_steps` to the prestart runbook gate.

## Why This Won

The operator runs the no-WING gate before opening the backend/WING WebSocket.
If that summary includes artifact repair guidance, the runbook should require
handling it before proceeding.

## Rejected Alternatives

Leaving the runbook unchanged was rejected because it hides a newly surfaced
prestart field. Runner-only guidance was rejected because the no-WING summary
can aggregate blocked artifact guidance.

## Implementation Plan

- Update `Docs/runbooks/WING_READONLY_PREFLIGHT.md`.
- Add a rails test requiring the field and safety boundary language.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_mvp_rails.py -k runbooks`
- Adjacent WING/no-WING tests.
- Full repository pytest.

## Risks And Rollback

The field must remain diagnostic. Rollback is to remove the precondition and
test; runtime behavior is unchanged.
