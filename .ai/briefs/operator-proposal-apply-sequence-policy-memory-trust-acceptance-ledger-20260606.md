# Brief: operator-proposal-apply-sequence-policy-memory-trust-acceptance-ledger-20260606

## Context

Sequence policy memory can be indexed and surfaced as read-only operator-review
candidates. The next safe learning step is to record whether the operator
accepted or rejected those candidates.

## Objective

Add a read-only acceptance ledger for sequence-memory trust-review candidates.
The ledger records operator decisions as evidence only and must not promote
policy, mutate queues/models/runtime memory, or execute mixer changes.

## Constraints

- Accept only records linked to current trust-review candidates.
- Reject mismatched candidate fields, non-UTC review timestamps, unknown
  decisions, unsafe authority flags, and forged review envelopes.
- Keep `sequence_policy_promotion_allowed_by_acceptance=false` and
  `execution_allowed_by_acceptance=false`.
- Register as `shadow_only`; never add proposal or executor authority.

## Expected Proof

- Focused ledger tests for missing candidates, unreviewed candidates, accepted
  evidence, unsafe record rejection, and forged review rejection.
- Product Layer preview visibility with one accepted record.
- Registry and capability matrix tests proving no live widening.
- Compile, memory lint, diff check, and backend pytest.
