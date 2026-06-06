# Brief: operator-proposal-apply-sequence-policy-memory-trust-promotion-executor-dry-run-20260606

## Context

Accepted sequence-memory trust promotion execution candidates can now be
surfaced as non-executing future executor candidates. The next safe autonomy
step is to rehearse what a future executor would write to the sequence-policy
trust gate without mutating policy or live mixer state.

## Objective

Add a read-only executor dry-run for sequence-memory trust promotion execution
candidates. The dry-run may emit deterministic apply specifications for
`operator_sequence_policy_trust_gate` only. It must not apply promotion,
execute proposals, mutate queues, policy stores, models, runtime memory,
append-only memory, or live mixer transport.

## Constraints

- Consume only the current promotion execution plan.
- Reject forged execution-plan envelopes, summary drift, unsafe candidate
  flags, duplicate policy specs, and non-UTC review timestamps.
- Block `apply=true` even when operator acknowledgement is present.
- Keep `sequence_policy_promotion_applied=false` and
  `execution_allowed_by_dry_run=false`.
- Register as `shadow_only`; never add proposal or executor authority.

## Expected Proof

- Focused dry-run tests for not-ready, apply spec creation, apply blocking,
  forged-plan rejection, duplicate spec rejection, registry, and goal matrix
  behavior.
- Product Layer preview visibility with one dry-run record.
- Registry and capability matrix tests proving no live widening.
- Compile, memory lint, diff check, and backend pytest.
