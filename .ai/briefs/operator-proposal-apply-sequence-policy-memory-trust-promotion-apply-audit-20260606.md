# Brief: operator-proposal-apply-sequence-policy-memory-trust-promotion-apply-audit-20260606

## Context

The sequence-policy promotion executor can apply trust-state changes to an
explicit caller-provided `operator_sequence_policy_trust_gate` mapping. Before
any downstream policy consumer can trust those applied results, the system
needs a read-only rollback audit.

## Objective

Add a read-only apply audit for sequence-policy memory trust promotion executor
results. The audit may convert valid applied results into rollback evidence
records only. It must not promote sequence policy, execute proposals, mutate
queues, mutate models, mutate runtime memory, or touch live mixer transport.

## Constraints

- Consume only the current promotion executor output.
- Accept audit records only when the executor status is applied and apply
  results are internally consistent.
- Reject forged executor envelopes, summary drift, unsafe apply-result flags,
  duplicate policy keys, and duplicate idempotency keys.
- Keep rollback evidence unverified until a separate rollback verification
  gate exists.
- Register as `shadow_only`; do not widen live write kinds.

## Expected Proof

- Focused audit tests for not-ready, rollback target records, forged executor
  authority, forged apply-result safety, registry, and goal matrix behavior.
- Product Layer preview visibility with zero audit records during dry preview.
- Registry and capability matrix tests proving no live widening.
- Compile, memory lint, diff check, and backend pytest.
