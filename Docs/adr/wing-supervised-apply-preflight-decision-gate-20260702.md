# WING Supervised Apply Preflight Decision Gate

## Context

The WING supervised proof runner can perform a real `fader` or negative `gain`
apply through `apply_operator_proposal` when `--execute` is set. The project
contract requires an accepted read-only preflight decision before any supervised
apply proof, but the runner itself also needs to fail closed if it is invoked
directly.

## Options considered

1. Keep the gate only in the runbook and proof-plan documentation.
2. Require the runner to re-check the read-only decision ledger before opening
   the backend WebSocket.
3. Allow a bypass flag for operator convenience.

## Decision

Use option 2. Any `--execute` run must provide `--accepted-readonly-source-id`.
The runner rebuilds the read-only decision review from the configured artifact
and decision roots before contacting the backend. If the decision review is not
`accepted_decision_ready`, or if the source id is not accepted, the runner
exits before any WebSocket connection or apply request.

## Why this won

This keeps the safety gate close to the live executor. A stale runbook command
or manual CLI mistake cannot skip the accepted read-only decision boundary.

## Rejected alternatives

- Documentation-only gating was rejected because it still allows direct runner
  execution.
- A bypass flag was rejected because this runner is intended to prove the MVP
  safety loop, not override it.

## Implementation plan

- Add `--accepted-readonly-source-id`, `--preflight-artifact-root`, and
  `--preflight-decision-root` to the runner.
- Block `--execute` before backend connection unless the decision review is
  accepted and source authority still denies live authority by itself.
- Emit the gate result in `summary.json`.
- Update generated proof-plan commands and the supervised proof runbook.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_wing_operator_proposal_validation_runner.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_supervised_wing_proof_plan.py -q`
- Targeted no-WING release-check policy tests.
- Targeted WING MVP rail test for the runbook wording.

## Risks and rollback

Risk: operator commands must include one more explicit source-id flag. Rollback
is to remove the runner gate and command flag, but that would reopen a direct
execution path and should only happen with a replacement safety gate.
