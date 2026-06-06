# Brief: WING Operator Proposal Timeout Fail-Closed

Task id: `wing-operator-proposal-timeout-fail-closed-20260606`
Date: 2026-06-06

## Goal

Reduce the supervised WING Rack MVP blocker for connection-loss / timeout
fail-closed proof without touching live hardware.

## Scope

- Harden `tools/wing_operator_proposal_validation_runner.py` evidence when the
  backend WebSocket connection fails or times out.
- Keep timeout artifacts blocked through the supervised proof evaluator,
  report tool, and promotion-evidence bridge.
- Preserve the existing Product Layer proposal-only write path.
- Add offline fake-WebSocket tests only.
- Keep proof-plan commands runnable in this repo by using `python3`.

## Safety Boundary

No live WING/OSC write is performed by this task. The runner still requires
`--execute` before any physical proof attempt, and physical mutation still goes
only through `apply_operator_proposal`.

## Definition of Done

- Connection failure before any apply request records `fail_closed=true`.
- Timeout after an apply request records `manual_review_required`, not a false
  safe pass.
- Timeout artifacts cannot become promotion evidence or safety-row review
  candidates downstream.
- Proof-plan command uses `python3`.
- Focused runner/plan tests pass.
