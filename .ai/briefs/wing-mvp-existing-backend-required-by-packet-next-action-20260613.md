# Brief: WING MVP Existing Backend Required By Packet Next Action

## Context

The first non-closed MVP blocker remains `WING read-only connection proof`. The
no-WING handoff packet already states
`physical_preflight_requires_existing_backend_connection: true`, but the
operator-facing selected action did not require reviewers to check that packet
field before supervised proof planning.

## Task

Promote `physical_preflight_requires_existing_backend_connection: true` into the
WING MVP rails that drive Project Director, autoloop, runbook review, generated
progress, and tests.

## Constraints

- Do not contact WING.
- Do not grant live authority.
- Do not let the handoff packet imply direct backend/WING connect or disconnect
  authority.
- Do not close readiness from offline packet evidence.

## Validation

- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Run focused WING/autoloop tests.
- Run Project Director and direction guard.
- Run standard repository validation before commit.
