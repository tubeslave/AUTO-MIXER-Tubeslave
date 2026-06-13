# ADR: WING MVP Backend Listener Pending Summary Fields Canon

Date: 2026-06-13
Status: accepted

## Context

The no-WING busy-port summary now exposes pending listener-template markers,
including `listener_confirmation_template_accepted_by_template: false` and
`listener_confirmation_template_operator_placeholder_fields`. The WING MVP
blocker rail must require those fields so operator planning and Project
Director output match runtime evidence.

## Options considered

1. Keep the requirement only in the runbook and no-WING tests.
2. Add the pending summary fields to the current blocker next action.
3. Replace the long blocker action with a new structured schema.

## Decision

Use option 2. The first WING blocker next action and generated progress report
now require the pending accepted-by-template marker and placeholder-field list
when an existing backend listener causes the no-WING prestart gate to fail
closed.

## Why this won

Goal-mode work is directed by `.ai/project-layer/wing_mvp_blockers.md`. Keeping
the requirement there ensures future cycles do not ignore the new runtime
summary fields.

## Rejected alternatives

Runbook-only documentation is too easy for automation to miss. A structured
schema may be cleaner later, but would be a broader refactor.

## Implementation plan

Update the blocker table, regenerate `MVP_PROGRESS_LATEST.md`, update MVP and
autoloop docs, and extend tests around Project Director/rail output.

## Test plan

Run focused WING rail and Project Director tests, memory lint, direction guard,
and the standard full pytest command.

## Risks and rollback

If the blocker action becomes unwieldy, move the summary-field contract into a
schema file and reference it from the blocker row.
