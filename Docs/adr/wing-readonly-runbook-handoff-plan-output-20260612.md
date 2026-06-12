# ADR: WING read-only runbook handoff plan output

## Context

The WING read-only handoff packet default `plan_output` matches the expected
offline plan. The runbook did not show that flag in the optional handoff
command.

## Options Considered

1. Keep relying on the handoff tool default.
2. Add the explicit plan-output flag to the runbook.

## Decision

Add the explicit `--plan-output` flag to the runbook handoff example.

## Why This Won

The read-only preflight path intentionally fails closed on expected-plan drift.
The operator-facing handoff command should name the same plan artifact as the
preview and capture commands.

## Rejected Alternatives

Relying on defaults was rejected because defaults are less visible during
manual hardware prep.

## Implementation Plan

- Update `Docs/runbooks/WING_READONLY_PREFLIGHT.md`.
- Add a rails test assertion.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_mvp_rails.py -k runbooks`
- Adjacent WING/no-WING tests.
- Full repository pytest.

## Risks And Rollback

The command remains offline/read-only. Rollback is to remove the explicit flag
if the handoff default path changes.
