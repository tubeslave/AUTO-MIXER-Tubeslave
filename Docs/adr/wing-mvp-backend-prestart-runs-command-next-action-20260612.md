# ADR: WING MVP Backend Prestart Runs Command Next Action

## Context

The supervised WING Rack MVP remains blocked on real WING read-only proof.
Before any backend/WING session or supervised fader/gain proof, the no-WING
handoff must show that the backend-port prestart gate is first, grants no live
authority, and is tied to the intended diagnostic step.

## Options Considered

1. Keep only first-step index and text requirements.
2. Add prose saying the structured gate uses the backend-port check.
3. Require `backend_port_prestart_gate.runs_command:
   prestart_backend_port_check`.

## Decision

Use option 3.

## Why This Won

The structured field is direct and assertable across the blocker table,
runbook, generated progress, Project Director, and autoloop surfaces. It
prevents a handoff packet from appearing proof-ready while the structured
backend prestart gate is not explicitly bound to `prestart_backend_port_check`.

## Rejected Alternatives

Prose-only wording was rejected because it is harder to assert and easier to
drift from generated evidence.

## Implementation Plan

1. Add the `runs_command` field requirement to the first WING blocker selected
   action and current command.
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
