# Codex Proposal: WING Read-Only Runner Validate Inputs Only

## Short thesis

Add a no-WebSocket validation mode for the exact real WING read-only preflight
command so operators can verify expected plan and prestart gates before opening
the backend/WING session.

## What I understood

The project is blocked on real `wing_readonly_preflight.v1` evidence. We should
not open a WebSocket without an intended WING session, but we can reduce risk by
making the real-run command independently checkable.

## Proposed solution

Add `--validate-run-inputs-only`. The mode runs normal argument validation,
prestart summary validation, expected-plan loading, and expected-plan matching.
If every gate passes, it writes a normal runner summary with
`status=ready_for_websocket`, `websocket_opened=false`, and empty sent request
types, then exits before `websockets.connect`.

## Likely files to touch

- `tools/wing_readonly_preflight_runner.py`
- `tests/test_wing_readonly_preflight_runner.py`
- `.ai/briefs/wing-readonly-runner-validate-inputs-only-20260617.md`
- `Docs/adr/wing-readonly-runner-validate-inputs-only-20260617.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_RUNNER_VALIDATE_INPUTS_ONLY_20260617.md`

## Alternatives considered

- Use `--plan-only` only. This validates plan construction but not the exact
  real-run command with `--expected-plan`.
- Run the real command and rely on fail-closed behavior. That risks opening a
  WebSocket before the operator has intentionally started the backend/WING
  session.

## Risks

The mode must not be mistaken for a real preflight artifact. It marks
`validation_only=true`, `websocket_opened=false`, and does not grant live
authority or readiness closure.

## Test plan

- Unit tests for parser conflict and no-WebSocket validation behavior.
- Focused runner/no-WING/WING guard tests.
- Fresh no-WING quick plus plan-only plus validate-run-inputs-only dry chain.
- Full pytest.

## Where the other agent may disagree

Kimi may argue that only hardware evidence matters now. That is true for
readiness, but this mode improves the final operator handoff immediately before
capturing that hardware evidence.
