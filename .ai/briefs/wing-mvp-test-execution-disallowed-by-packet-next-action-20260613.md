# Brief: WING MVP Test Execution Disallowed By Packet Next Action

## Context

The current first non-closed MVP blocker is still `WING read-only connection
proof`. The no-WING handoff packet already states `executes_tests: false`, but
the operator-facing selected action did not require reviewers to check that
field before supervised proof planning.

## Task

Promote `executes_tests: false` into the WING MVP rails that drive Project
Director, autoloop, runbook review, generated progress, and tests.

## Constraints

- Do not contact WING.
- Do not execute live WING proof commands.
- Do not treat the offline handoff packet as validation evidence.
- Do not close readiness from offline packet evidence.

## Validation

- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Run focused WING/autoloop tests.
- Run Project Director and direction guard.
- Run standard repository validation before commit.
