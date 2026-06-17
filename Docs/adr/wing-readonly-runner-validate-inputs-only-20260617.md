# ADR: WING Read-Only Runner Validate Inputs Only

## Context

The active WING MVP blocker is a real read-only preflight artifact. The runner
already supports offline plan generation and fail-closed real execution, but it
does not expose a command that validates the exact real-run inputs without
opening WebSocket.

## Options considered

1. Keep `--plan-only` as the only offline preparation command.
2. Add `--validate-run-inputs-only` before the WebSocket boundary.
3. Attempt a real preflight capture from automation without operator session
   confirmation.

## Decision

Add `--validate-run-inputs-only`.

## Why this won

It reuses the same gates that protect the real run and produces an artifact that
says the command is ready for WebSocket, while proving no WebSocket was opened
and no backend/WING contact occurred.

## Rejected alternatives

`--plan-only` does not validate the exact `--expected-plan` real-run command.
Attempting a real capture without the operator-confirmed backend/WING session
would violate the current safety boundary.

## Implementation plan

- Add the CLI flag.
- Reject combining it with `--plan-only`.
- After expected-plan validation passes, write a summary with
  `ready_for_websocket`, `validation_only=true`, `websocket_opened=false`, and
  no sent requests.
- Add focused tests.

## Test plan

- `py_compile` touched files.
- Focused runner/no-WING/WING guard pytest.
- no-WING quick, `--plan-only`, and `--validate-run-inputs-only` dry chain.
- Full pytest before commit.

## Risks and rollback

The main risk is an operator confusing validation with capture. The summary uses
`validation_only=true` and `websocket_opened=false`. Rollback removes the flag,
summary branch, and tests.
