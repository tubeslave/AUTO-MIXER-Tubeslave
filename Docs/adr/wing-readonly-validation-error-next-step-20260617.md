# ADR: WING Read-Only Validation Error Next Step

## Context

The current MVP remains blocked on real WING read-only connection proof. The
read-only preflight runner already fails closed before WebSocket when command
arguments are invalid, and the plan-only branch records stable
`validation_violations`. Unlike prestart and expected-plan failures, validation
failures did not include a concrete operator recovery step.

## Options considered

1. Keep validation failures as machine-readable violations only.
2. Add a shared `operator_next_step` for validation failures.
3. Let invalid arguments fall through to argparse or later gates.

## Decision

Use option 2. Both plan-only and real-run validation failures now tell the
operator to fix the read-only preflight arguments, regenerate the offline
expected plan with `--plan-only`, and rerun `--validate-run-inputs-only` before
opening the backend/WING WebSocket.

## Why this won

Invalid inputs are recoverable before any hardware contact. The recovery path
keeps the operator inside the expected-plan evidence chain and makes the next
safe action visible in the same blocked JSON surface as the violation.

## Rejected alternatives

- Option 1 preserves fail-closed behavior but leaves the operator without the
  exact recovery sequence.
- Option 3 weakens the pre-WebSocket contract and risks less structured failure
  evidence.

## Implementation plan

- Add a shared validation-error `operator_next_step` helper.
- Include it in plan-only blocked validation output.
- Include it in real-run blocked validation summaries.
- Extend regression tests for both surfaces.

## Test plan

- Targeted validation-error runner regressions.
- no-WING quick release check.
- Focused WING preflight, no-WING, handoff, and Project Director tests.
- `git diff --check`.
- Full pytest.

## Risks and rollback

The risk is limited to blocked validation surfaces. Rollback is to remove the
new `operator_next_step` field and its assertions; the existing fail-closed
validation behavior would remain.
