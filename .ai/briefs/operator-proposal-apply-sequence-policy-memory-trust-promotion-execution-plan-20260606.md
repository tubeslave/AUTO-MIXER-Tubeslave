# Brief: operator-proposal-apply-sequence-policy-memory-trust-promotion-execution-plan-20260606

## Context

Sequence-memory trust promotion acceptance records can now record explicit
operator accepted/rejected decisions on promotion contract records. The next
safe learning step is to expose accepted records as future executor candidates
without granting execution or policy authority.

## Objective

Add a read-only execution plan for accepted sequence-memory trust promotion
records. The plan may surface candidates for a later executor contract only. It
must not promote sequence policy, mutate queues, policy stores, models,
runtime memory, append-only memory, or live mixer transport.

## Constraints

- Consume only the current promotion acceptance ledger.
- Require valid accepted records and no unreviewed contract records before
  surfacing candidates.
- Reject forged ledger envelopes, summary drift, unsafe record flags, and
  ambiguous contract reviews.
- Keep `sequence_policy_promotion_allowed_by_plan=false` and
  `execution_allowed_by_plan=false`.
- Register as `shadow_only`; never add proposal or executor authority.

## Expected Proof

- Focused execution-plan tests for not-ready, unreviewed, accepted candidate,
  forged-ledger, record-level authority, registry, and goal matrix behavior.
- Product Layer preview visibility with one non-executing candidate.
- Registry and capability matrix tests proving no live widening.
- Compile, memory lint, diff check, and backend pytest.
