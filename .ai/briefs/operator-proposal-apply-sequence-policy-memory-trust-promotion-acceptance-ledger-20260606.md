# Brief: operator-proposal-apply-sequence-policy-memory-trust-promotion-acceptance-ledger-20260606

## Context

Sequence-memory trust promotion contract records can now be surfaced for
operator review after fresh context and rollback evidence are valid. The next
safe learning step is to record whether the operator accepted or rejected those
contract records.

## Objective

Add a read-only acceptance ledger for sequence-memory trust promotion contract
records. The ledger may record operator decisions as evidence only. It must not
promote sequence policy, mutate queues/models/runtime memory, execute proposals,
or grant live authority.

## Constraints

- Accept only records linked to current promotion contract records.
- Reject mismatched contract fields, unknown decisions, non-UTC review
  timestamps, unsafe authority flags, and forged contract envelopes.
- Keep `sequence_policy_promotion_allowed_by_acceptance=false` and
  `execution_allowed_by_acceptance=false`.
- Register as `shadow_only`; never add proposal or executor authority.

## Expected Proof

- Focused ledger tests for missing contract records, unreviewed records,
  accepted evidence, unsafe record rejection, and forged contract rejection.
- Product Layer preview visibility with one accepted promotion record.
- Registry and capability matrix tests proving no live widening.
- Compile, memory lint, diff check, and backend pytest.
