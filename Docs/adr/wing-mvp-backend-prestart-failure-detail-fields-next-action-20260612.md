# ADR: WING MVP Backend Prestart Failure Detail Fields Next Action

## Context

The supervised WING Rack MVP remains blocked on real WING read-only proof.
Before any backend/WING session or supervised fader/gain proof, the no-WING
handoff must show the failure-detail fields operators need when the
backend-port prestart check detects unexpected listeners.

## Options Considered

1. Keep only the exact prestart command requirement.
2. Add prose saying listener failures should be reviewed.
3. Require `backend_port_prestart_check_failure_detail_fields` to exactly list
   the listener, confirmation, fail-closed, and operator-next-step fields.

## Decision

Use option 3.

## Why This Won

The structured field list is already emitted by the handoff packet builder and
can be asserted across the blocker table, runbook, generated progress, Project
Director, and autoloop surfaces. It prevents a handoff packet from appearing
proof-ready while the prestart failure path lacks the evidence needed for safe
operator review.

## Rejected Alternatives

Prose-only wording was rejected because it is harder to assert and easier to
drift from generated evidence.

## Implementation Plan

1. Add the `backend_port_prestart_check_failure_detail_fields` requirement to
   the first WING blocker selected action and current command.
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
