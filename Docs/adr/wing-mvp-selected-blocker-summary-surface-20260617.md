# ADR: WING MVP Selected Blocker Summary Surface

## Context

The active Project Director guard still reports drift and selects `WING
read-only connection proof` as the first open MVP blocker. The WING read-only
handoff packet already includes selected blocker fields, but the top-level
no-WING summary does not expose them alongside the blocker anchor surface.

## Options considered

1. Keep selected blocker fields only inside the nested handoff packet artifact.
2. Mirror the selected blocker fields into the existing MVP blocker anchor
   summary surface.
3. Add a separate selected-blocker summary surface.

## Decision

Mirror `selected_mvp_blocker` and `selected_mvp_blocker_status` into the
existing MVP blocker anchor summary surface.

## Why this won

This is the smallest change that makes the top-level no-WING artifact easier to
review during the WING read-only proof handoff. It keeps the packet visibly bound
to the current blocker without creating a new concept or implying readiness
closure.

## Rejected alternatives

Keeping the fields nested preserves data but weakens the operator-facing proof
surface. Adding a separate surface would add structure without new behavior.

## Implementation plan

- Add the selected blocker fields to
  `build_wing_readonly_mvp_blocker_anchor_summary_surface()`.
- Assert the top-level values match the handoff packet artifact in the existing
  no-WING summary test.
- Validate with focused no-WING/WING tests and full pytest.

## Test plan

- Compile touched Python files.
- Run focused no-WING/WING guard tests.
- Run no-WING quick and inspect the generated summary fields.
- Run full pytest before commit.

## Risks and rollback

The change is summary-only and does not contact WING hardware or grant live
authority. Rollback is removing the two mirrored fields and their assertions.
