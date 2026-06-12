# ADR: WING MVP Backend Prestart Step Text Next Action

## Context

The supervised WING Rack MVP remains blocked on real WING read-only proof.
Before any backend/WING session or supervised fader/gain proof, the no-WING
handoff must show that the backend-port prestart check is the first operator
step and that it grants no live authority.

The handoff summary also exposes the raw first step text as
`backend_port_prestart_check_operator_sequence_step`.

## Options Considered

1. Keep only the structured gate equality check.
2. Add prose saying the first step must be correct.
3. Require `backend_port_prestart_check_operator_sequence_step` to include
   `prestart_backend_port_check` and `unexpected listeners`.

## Decision

Use option 3.

## Why This Won

The structured field is already available in no-WING evidence and is direct for
operator review. Naming the expected markers prevents a correctly indexed but
semantically wrong first operator step from being accepted before supervised
proof planning.

## Rejected Alternatives

Prose-only wording was rejected because it is harder to assert across Project
Director, generated progress, runbook, and autoloop surfaces.

## Implementation Plan

1. Add the raw first-step field and required markers to the first WING blocker
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
