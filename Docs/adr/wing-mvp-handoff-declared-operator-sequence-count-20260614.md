# ADR: WING MVP Handoff Declared Operator Sequence Count

## Context

The supervised WING MVP still waits on real WING read-only connection proof.
The no-WING handoff validator computes and checks the operator sequence count,
but the handoff packet itself did not declare that count as top-level metadata.

## Options considered

1. Keep count only as a computed no-WING summary field.
2. Add declared count fields without validation.
3. Add declared count fields and validate them against the actual sequence and
   shared contract.

## Decision

Use option 3.

## Why this won

The handoff packet is reviewed before the hardware session. If it declares the
operator checklist size, that declaration should prove it matches both the
actual sequence and the contract. This keeps packet metadata auditable without
granting live authority.

## Rejected alternatives

Computed-only count hides the packet metadata from the operator artifact.
Unvalidated declared count fields can become stale or misleading.

## Implementation plan

Emit count, expected count, and count-match flag from the handoff builder.
Validate declared count, expected count, and flag consistency in no-WING release
checks. Expose declared match fields in JSON and Markdown summaries. Update WING
MVP/autoloop docs and rails tests.

## Test plan

Run focused validator and rails tests, no-WING quick summary generation,
Project Director, direction guard, memory lint, `git diff --check`, and the
standard full test command.

## Risks and rollback

Risk is limited to stricter offline handoff validation. Rollback is to remove
the declared count fields and mismatch violations if the packet schema changes.
