# WING MVP Supervised Rows Final Prestart Gate

## Context

The handoff validator requires the supervised prestart gate to be the final
operator sequence step. The MVP blocker table is the Project Director work
queue, so downstream supervised proof rows must reflect that prerequisite.

## Options considered

1. Keep fader/gain rows as accepted-decision-only.
2. Add final handoff prestart gate wording to fader/gain rows.
3. Add a separate blocker row.

## Decision

Use option 2. The supervised fader and gain-down rows now require the final
handoff operator-sequence step with valid accepted-candidate prestart trace
before running the supervised proof runbook.

## Why this won

It keeps the blocker structure stable while preventing future Project Director
actions from omitting the current handoff gate.

## Rejected alternatives

- Accepted-decision-only wording was rejected because it is now stale.
- A new blocker row was rejected because this is a prerequisite inside the
  existing supervised fader/gain proof path.

## Implementation plan

- Update `.ai/project-layer/wing_mvp_blockers.md`.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Add focused assertions.

## Test plan

- Focused MVP rails test.
- Affected Project Director/MVP tests.
- Full pytest suite.
- `git diff --check`.

## Risks and rollback

Risk is status wording only. Rollback is safe by restoring the old next-action
text and regenerating the report, but that would reintroduce stale guidance.
