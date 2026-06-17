# WING Read-Only Validate Inputs Handoff Step

## Context

The WING read-only runner supports `--validate-run-inputs-only`, which validates real-run inputs and the expected plan without opening transport. The official handoff packet and no-WING command binding surface did not yet include that dry-run step before `capture_readonly_preflight`.

## Options considered

1. Keep validation-only mode as a runner-only affordance.
2. Add it to the operator handoff sequence and no-WING command binding surface.

## Decision

Use option 2. Add `validate_readonly_preflight_inputs` between request-plan preview and real capture.

## Why this won

The operator gets a real command that validates the exact capture inputs before touching WING transport. This closes a practical handoff gap without changing write authority or claiming readiness.

## Rejected alternatives

Runner-only documentation was rejected because it is easy to skip during real hardware capture preparation.

## Implementation plan

- Add the validation-only step to the shared WING handoff operator sequence.
- Emit the validation command in the handoff packet.
- Require and summarize the command in no-WING checks.
- Update WING/no-WING tests to use shared sequence constants where possible.

## Test plan

- `python3 -m py_compile` for edited Python files.
- Focused WING handoff/no-WING tests.
- no-WING quick dry run.
- Full pytest.

## Risks and rollback

The main risk is stale documentation or fixtures that still expect the old 10-step handoff. Rollback is to remove the validation step and command binding, restoring the previous operator sequence count.
