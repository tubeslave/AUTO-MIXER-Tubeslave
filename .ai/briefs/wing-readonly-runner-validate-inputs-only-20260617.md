# Brief: WING Read-Only Runner Validate Inputs Only

## Task ID

wing-readonly-runner-validate-inputs-only-20260617

## Goal

Add a runner mode that validates the real WING read-only preflight command
inputs without opening the backend WebSocket.

## Context

The next allowed Project Director action is a real `wing_readonly_preflight.v1`
capture after a passed no-WING prestart check. The runner already has
`--plan-only` and fail-closed gates before WebSocket, but operators need a final
dry validation of the exact real-run command before they start the intended
backend/WING session.

## Scope

- Add `--validate-run-inputs-only` to `tools/wing_readonly_preflight_runner.py`.
- Reuse the same argument, prestart summary, and expected-plan checks as the
  real run.
- Write a summary artifact with `ready_for_websocket` only when inputs are valid.
- Do not open WebSocket, contact backend/WING, query console, or grant live
  authority.

## Validation

- Compile touched files.
- Run focused runner/no-WING/WING guard tests.
- Run a dry chain: no-WING quick, plan-only, validate-run-inputs-only.
- Run full pytest before commit.
