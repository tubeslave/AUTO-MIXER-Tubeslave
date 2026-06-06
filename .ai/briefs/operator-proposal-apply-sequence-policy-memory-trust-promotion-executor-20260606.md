# Brief: operator-proposal-apply-sequence-policy-memory-trust-promotion-executor-20260606

## Context

The sequence-policy promotion executor dry-run can emit deterministic specs
for future updates to `operator_sequence_policy_trust_gate`. The next safe
autonomy step is an operator-gated executor that can apply those specs only to
an explicit caller-provided policy store, never to live mixer state.

## Objective

Add an operator-gated sequence-policy promotion executor. It may mutate only a
caller-provided `operator_sequence_policy_trust_gate` mapping when
`apply=true`, `operator_ack=true`, and dry-run specs validate. Product Layer
preview must call it with `apply=false`.

## Constraints

- Consume only the current executor dry-run payload.
- Reject forged dry-run envelopes, summary drift, unsafe dry-run records,
  duplicate policy keys, and duplicate idempotency keys.
- Require explicit operator acknowledgement before mutation.
- Require an explicit target policy store before mutation.
- Never mutate live mixer state, operator queues, proposal state, models, or
  append-only project memory.
- Register as `shadow_only`; do not widen live write kinds.

## Expected Proof

- Focused executor tests for not-ready, preview-ready, missing ack, missing
  store, explicit-store apply, forged dry-run rejection, registry, and goal
  matrix behavior.
- Product Layer preview visibility with zero apply results.
- Registry and capability matrix tests proving no live widening.
- Compile, memory lint, diff check, and backend pytest.
