# ADR: WING Read-Only Preflight Runbook Prestart Detail Fields

## Context

The no-WING prestart gate emits stable busy-port detail fields, and the shared
WING preflight contract now requires them in plans, handoff packets, and summary
validation. The operator runbook still instructed the operator to inspect only
`listener_processes`, leaving manual procedure behind the generated contract.

## Options considered

1. Leave the runbook unchanged.
2. Update the runbook to list the shared required fields and test the sync.
3. Add optional `command_line` diagnostics to the shared required contract.

## Decision

Use option 2. The runbook now names every shared busy-port failure detail field
and keeps `command_line` as optional diagnostics.

## Why this won

The first hardware step depends on a human operator following the runbook before
starting the intended backend/WING session. The runbook should expose the same
fail-closed evidence that generated artifacts already require, without making
optional OS process diagnostics mandatory.

## Rejected alternatives

Option 1 preserves operator-facing drift. Option 3 over-requires diagnostics
that may be unavailable on some systems.

## Implementation plan

- Update `Docs/runbooks/WING_READONLY_PREFLIGHT.md`.
- Add a static test in `tests/test_no_wing_release_check.py` that reads the
  runbook and checks the shared prestart failure detail fields.
- Validate the WING/no-WING stack and full backend suite.

## Test plan

- Focused runbook sync test.
- Adjacent WING/no-WING tests.
- Full backend suite.

## Risks and rollback

Risk: the runbook becomes verbose. Rollback: remove the added runbook language
and the static sync test. No hardware access or live write behavior is changed.
