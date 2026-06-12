# Brief: WING MVP Console Query Disallowed By Packet Next Action

## Context

The first non-closed WING MVP blocker is still `WING read-only connection proof`.
The offline WING read-only handoff packet already carries `queries_console:
false`, but the operator-facing next action did not require reviewers to check
that field before supervised proof planning.

## Task

Promote the packet-level `queries_console: false` boundary into the WING MVP
rails that drive Project Director, autoloop, runbook review, generated progress,
and tests.

## Constraints

- Do not contact WING.
- Do not add live write authority.
- Do not close readiness from offline packet evidence.
- Keep the first blocker as the selected work until a real accepted WING
  read-only preflight artifact exists.

## Validation

- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Run focused WING/autoloop tests.
- Run direction guard and Project Director.
- Run standard repository validation before commit.
