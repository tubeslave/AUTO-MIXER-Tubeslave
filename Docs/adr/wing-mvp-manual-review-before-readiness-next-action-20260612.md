# ADR: WING MVP Manual Review Before Readiness Next Action

## Context

The supervised WING Rack MVP remains blocked on real WING read-only proof.
No-WING handoff packets can prepare operator review, but they must not update
the readiness matrix or close the blocker without manual review of real WING
artifacts.

## Options Considered

1. Keep only the blocker status and readiness prose.
2. Add prose saying manual review is required.
3. Require `manual_review_required_before_readiness_update: true` in the
   selected action and operator-facing rails.

## Decision

Use option 3.

## Why This Won

The structured field is already emitted by the handoff packet builder and can
be asserted across the blocker table, runbook, generated progress, Project
Director, and autoloop surfaces. It prevents no-WING evidence from being
misread as permission to update readiness.

## Rejected Alternatives

Prose-only wording was rejected because it is harder to assert and easier to
drift from generated evidence.

## Implementation Plan

1. Add the manual-review-before-readiness field to the first WING blocker
   selected action and current command.
2. Mirror the requirement in MVP/runbook/autoloop docs.
3. Update tests and regenerate progress.
4. Record validation in a report.

## Test Plan

- Focused WING rail tests.
- Direction guard and Project Director checks.
- Diff check, memory lint, and full pytest.

## Risks And Rollback

Risk is limited to docs/test drift. Rollback removes the field requirement from
docs and tests. No live WING behavior changes are included.
