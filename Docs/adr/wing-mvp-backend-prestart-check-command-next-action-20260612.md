# ADR: WING MVP Backend Prestart Check Command Next Action

## Context

The supervised WING Rack MVP remains blocked on real WING read-only proof.
Before any backend/WING session or supervised fader/gain proof, the no-WING
handoff must show the exact command used for the backend-port prestart check.

## Options Considered

1. Keep only the selected-action prose command.
2. Add prose saying the command should be reviewed.
3. Require `backend_port_prestart_check_command` to exactly match
   `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick`.

## Decision

Use option 3.

## Why This Won

The structured command field is already emitted by the handoff packet builder
and can be asserted across the blocker table, runbook, generated progress,
Project Director, and autoloop surfaces. It prevents a handoff packet from
appearing proof-ready while the structured prestart command is stale or points
to a different command.

## Rejected Alternatives

Prose-only wording was rejected because it is harder to assert and easier to
drift from generated evidence.

## Implementation Plan

1. Add the `backend_port_prestart_check_command` requirement to the first WING
   blocker selected action and current command.
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
