# ADR: WING read-only runbook value-types match

## Context

`tools/wing_readonly_preflight_runner.py` defaults to `fader,gain,pan`, and the
no-WING release gate validates plan/capture commands with the same value types.
The read-only runbook examples still showed `fader,gain`.

## Options Considered

1. Keep the runbook examples narrower.
2. Change the no-WING gate to `fader,gain`.
3. Align the runbook examples to `fader,gain,pan`.

## Decision

Align the runbook examples to `fader,gain,pan`.

## Why This Won

The expected-plan gate is strict. Operator-facing commands should match the
checked no-WING prestart contract to avoid false expected-plan mismatches.

## Rejected Alternatives

Changing the no-WING gate was rejected because the current read-only proof
contract already covers pan as a read-only value type. Keeping the mismatch was
rejected because it makes the runbook less reliable.

## Implementation Plan

- Update plan-only and real-run examples.
- Add a rails test preventing stale value-type examples.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_mvp_rails.py -k runbooks`
- Adjacent WING/no-WING tests.
- Full pytest before commit.

## Risks And Rollback

The change is documentation/test-only and read-only. Rollback is to restore the
old examples and test if the release gate contract changes back.
