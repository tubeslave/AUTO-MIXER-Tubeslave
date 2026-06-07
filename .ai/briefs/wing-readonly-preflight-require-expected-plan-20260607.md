# Brief: WING Read-Only Preflight Requires Expected Plan

## Context

The supervised WING MVP is blocked on the first real hardware step: WING
read-only connection proof. The runbook and no-WING plan already require a real
capture to match an offline `--plan-only` JSON before WebSocket opens.

## Problem

`tools/wing_readonly_preflight_runner.py` accepted `--expected-plan`, but a real
run could still omit it and open WebSocket. Downstream evidence would reject the
artifact later, but the runner did not enforce the stated fail-closed boundary at
the first contact point.

## Scope

- Make `--expected-plan` mandatory for non-`--plan-only` runs.
- Fail closed before WebSocket if it is missing.
- Keep all behavior read-only and do not contact WING in tests.
- Update the operator runbook and tests.

## Out Of Scope

- Closing any MVP blocker.
- Running a real WING session.
- Adding supervised writes or widening live authority.
