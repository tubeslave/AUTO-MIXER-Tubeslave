# Brief: operator-proposal-apply-sequence-policy-memory-trust-promotion-contract-20260606

## Context

Accepted sequence-memory trust records can now become manual promotion-plan
candidates. The next safe learning step is to require fresh sequence-policy
context and rollback evidence before any candidate is review-ready for a later
promotion executor.

## Objective

Add a read-only sequence-memory trust promotion contract. The contract may
surface promotion-plan candidates as operator-review-ready records only when
fresh context and rollback evidence are valid. It must not mutate policy,
queues, models, runtime memory, append-only memory, or execute mixer changes.

## Constraints

- Consume only the current manual sequence-memory trust promotion plan.
- Require fresh read-only sequence-policy context.
- Require rollback evidence with no live write, no apply attempt, no queue or
  policy mutation, and a supported rollback strategy.
- Reject forged plan envelopes, summary count drift, unsafe candidate flags,
  stale context, unsafe rollback evidence, and non-UTC review timestamps.
- Register as `shadow_only`; never add proposal or executor authority.

## Expected Proof

- Focused contract tests for missing candidates, ready records, stale context,
  missing rollback evidence, unsafe evidence rejection, and registry/matrix
  safety.
- Product Layer preview visibility with one contract record.
- Registry and capability matrix tests proving no live widening.
- Compile, memory lint, diff check, and backend pytest.
