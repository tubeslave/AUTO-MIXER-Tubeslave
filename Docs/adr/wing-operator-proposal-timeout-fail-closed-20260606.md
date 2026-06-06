# ADR: WING Operator Proposal Timeout Fail-Closed

## Context

The active project priority is a supervised WING Rack MVP. One named blocker is
connection-loss / timeout fail-closed proof. The existing proposal validation
runner already routes physical proof through Product Layer proposals, but a
transport exception was reported as a generic error. The downstream proof
report and promotion-evidence bridge also need to reject timeout artifacts
explicitly, even if a future artifact contains readback-shaped fields.

## Options considered

1. Leave generic errors unchanged.
2. Mark every timeout as fail-closed.
3. Distinguish pre-apply transport failure from post-apply transport failure.

## Decision

Use option 3.

## Why this won

Before `apply_operator_proposal`, a connection or timeout failure proves that no
live apply request was sent by the runner. After `apply_operator_proposal`, the
physical console state is unknown until readback or operator inspection, so the
artifact must require manual review instead of becoming promotion evidence.

## Rejected alternatives

- Generic error artifacts are too weak for a supervised hardware proof blocker.
- Treating post-apply timeout as fail-closed would overstate safety.

## Implementation plan

- Track sent request types in the runner recorder.
- Emit `fail_closed=true` only when no live apply request was sent.
- Emit `manual_review_required` after a live apply request timeout.
- Keep timeout/manual-review summaries blocked in the proof evaluator and
  promotion-evidence bridge.
- Use `python3` in generated proof-plan commands.

## Test plan

- Fake-WebSocket connection failure before apply.
- Fake-WebSocket timeout after apply request.
- Evaluator/report regression showing timeout artifacts do not become
  promotion evidence.
- Proof-plan command assertion.
- Focused pytest and memory lint.

## Risks and rollback

The change may classify some inconclusive proof attempts as manual-review
artifacts. Rollback is a normal git revert; no runtime data migration is
required.
