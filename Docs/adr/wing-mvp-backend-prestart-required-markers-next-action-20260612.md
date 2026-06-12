# ADR: WING MVP Backend Prestart Required Markers Next Action

## Context

The supervised WING Rack MVP remains blocked on real WING read-only proof.
Before any backend/WING session or supervised fader/gain proof, the no-WING
handoff must show that the backend-port prestart gate is first, grants no live
authority, runs the intended diagnostic command, and carries the markers that
make the stop condition operator-reviewable.

## Options Considered

1. Keep only the structured gate equality and command checks.
2. Add prose saying the first gate marker list should be reviewed.
3. Require `backend_port_prestart_gate.required_markers` to exactly list the
   session-boundary, command, and unexpected-listener markers.

## Decision

Use option 3.

## Why This Won

The structured marker list is already emitted by the handoff packet builder and
can be asserted across the blocker table, runbook, generated progress, Project
Director, and autoloop surfaces. It prevents a handoff packet from appearing
proof-ready while the first gate lacks the operator-facing stop-condition
markers.

## Rejected Alternatives

Prose-only wording was rejected because it is harder to assert and easier to
drift from generated evidence.

## Implementation Plan

1. Add the `required_markers` field requirement to the first WING blocker
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
