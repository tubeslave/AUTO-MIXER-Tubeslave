# Brief: WING MVP Expected Plan Required By Packet Next Action

## Context

The first non-closed MVP blocker remains `WING read-only connection proof`. The
no-WING handoff packet already states `expected_plan_required_for_real_run:
true`, but the operator-facing selected action did not require reviewers to
check that packet field before supervised proof planning.

## Task

Promote `expected_plan_required_for_real_run: true` into the WING MVP rails that
drive Project Director, autoloop, runbook review, generated progress, and tests.

## Constraints

- Do not contact WING.
- Do not grant live authority.
- Do not allow ad hoc real WING read-only preflight without an offline expected
  plan.
- Do not close readiness from offline packet evidence.

## Validation

- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Run focused WING/autoloop tests.
- Run Project Director and direction guard.
- Run standard repository validation before commit.
