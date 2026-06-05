# ADR: frontend-read-surface-remediation-target-20260605

## Context

Backend processing snapshots include `read_surface_remediation_plan` with the
next physical readback target and remediation action ID. The cockpit adapter did
not render that plan as a Module Status row.

## Options considered

1. Add a read-only Module Status row for read-surface remediation.
2. Keep the data only in registry JSON.
3. Fold remediation details into supervised proof evidence.

## Decision

Use option 1. Render one `read_surface_remediation_plan` row with read-only
profiles and next-step text.

## Why this won

Module Status already hosts proof evidence, mode gates, promotion readiness, and
gap rows. Adding the remediation row there makes the next readback task visible
without creating a new control path.

## Rejected alternatives

Registry-only data is too hidden for operator planning. Folding remediation into
proof evidence mixes artifact validation with follow-up planning.

## Implementation plan

1. Adapt `read_surface_remediation_plan` into a read-only Module Status row.
2. Add adapter contract assertions for the row and safety flags.
3. Add e2e fixture coverage on Settings.
4. Update project memory.

## Test plan

- `node frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend test`
- `npm --prefix frontend run build`
- `npm --prefix frontend run test:e2e -- automixer-prototype.spec.ts`

## Risks and rollback

Risk: operators could mistake remediation actions for live commands. The row
has `canWriteLive=false`, empty live write kinds, and read-only text.

Rollback is removing the adapter row, frontend tests, and this ADR.
