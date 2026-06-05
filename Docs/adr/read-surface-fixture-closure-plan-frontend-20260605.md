# ADR: read-surface-fixture-closure-plan-frontend-20260605

## Context

The backend fixture closure plan surfaces accepted readback records as manual
fixture-baseline update candidates. The cockpit already exposes capture plans,
contracts, per-surface artifact intake, and the aggregate artifact review queue.
Without a dedicated frontend row, operators cannot see the closure-plan gate in
the same module-status view.

## Options Considered

1. Keep fixture closure plan visibility only in no-WING release summaries.
2. Add closure-plan tags only to supervised proof evidence.
3. Add closure-plan tags plus a dedicated read-only module-status row.

## Decision

Use option 3. The frontend adapter exposes the closure plan as
`read_surface_physical_readback_fixture_closure_plan` and adds
`fixture_closure_plan_*` tags to supervised proof evidence.

## Why This Won

A dedicated row keeps the lifecycle visible: artifact intake, aggregate review,
acceptance records, manual fixture closure planning, and later baseline update.
It also makes safety flags filterable and reviewable in the cockpit.

## Rejected Alternatives

- Release-summary-only visibility was rejected because normal cockpit review
  would hide the gate.
- Tags-only visibility was rejected because closure planning is first-class
  operator work.
- Any fixture update or closure action from the frontend was rejected because
  this row is display-only evidence.

## Implementation Plan

1. Parse `read_surface_physical_readback_fixture_closure_plan` from proof
   quality summaries.
2. Add closure-plan status/count/safety tags to supervised proof evidence.
3. Append a dedicated read-only module-status row after the artifact review
   queue.
4. Preserve missing-plan fallback behavior.
5. Cover present and missing payloads in frontend contract tests.

## Test Plan

- `node --test frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend test`
- `npm --prefix frontend run build`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks And Rollback

Risk: operators may treat closure candidates as completed fixture proof.
Mitigation: the row and tags explicitly keep
`fixture_update_allowed=false`, `fixture_closure_allowed=false`,
`physical_gap_remains=true`, `live_authority=false`, and
`full_autonomous_ready=false`.

Rollback: remove the closure-plan row and `fixture_closure_plan_*` tags from
`operatorAdapters.ts`, then remove the related contract expectations. Backend
release summaries can remain unchanged.
