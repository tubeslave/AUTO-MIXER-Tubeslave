# Brief: WING MVP Operator Required By Packet Next Action

## Context

The first non-closed MVP blocker remains `WING read-only connection proof`. The
no-WING handoff packet already states `operator_required: true`, but the
operator-facing selected action did not require reviewers to check that field
before supervised proof planning.

## Task

Promote `operator_required: true` into the WING MVP rails that drive Project
Director, autoloop, runbook review, generated progress, and tests.

## Constraints

- Do not contact WING.
- Do not grant live authority.
- Do not let the offline handoff become autonomous execution.
- Do not close readiness from offline packet evidence.

## Validation

- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Run focused WING/autoloop tests.
- Run Project Director and direction guard.
- Run standard repository validation before commit.
