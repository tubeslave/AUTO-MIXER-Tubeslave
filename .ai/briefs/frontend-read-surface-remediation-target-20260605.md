# frontend-read-surface-remediation-target-20260605

## Context

`read_surface_remediation_plan` now exposes physical readback gap actions and
the next action ID. The frontend adapter currently renders proof evidence, mode
gates, promotion gaps, and autonomy overlays, but not the read-surface
remediation plan.

## Goal

Show the read-surface remediation plan as a read-only Module Status row so the
operator can see the next physical readback target and action ID.

## Non-goals

- Do not execute remediation actions.
- Do not query WING or dLive.
- Do not widen live write kinds or Full Autonomous readiness.

## Safety constraints

- `canWriteLive` must remain false.
- `liveWriteKinds` must remain empty.
- The row must describe actions as evidence/remediation, not runtime commands.

## Expected evidence

- Frontend adapter produces a `read_surface_remediation_plan` row.
- The row carries action count, physical gap count, next target, and next action
  as pipeline profiles.
- Settings e2e shows the row without unsafe writes.
