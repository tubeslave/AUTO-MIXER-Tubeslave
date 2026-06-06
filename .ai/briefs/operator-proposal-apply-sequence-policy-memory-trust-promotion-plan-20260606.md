# Brief: operator-proposal-apply-sequence-policy-memory-trust-promotion-plan-20260606

## Context

Sequence policy memory can now be indexed, reviewed with fresh context, and
accepted or rejected by the operator as ledger evidence. The next safe learning
step is to surface accepted records as manual promotion candidates.

## Objective

Add a read-only sequence-memory trust promotion plan. The plan may expose
accepted trust records as candidates for a later promotion contract, but it must
not promote policy, mutate queues/models/runtime memory, execute mixer changes,
or grant live authority.

## Constraints

- Consume only the current sequence-memory trust acceptance ledger.
- Reject malformed ledger envelopes, forged summary counts, unsafe safety
  claims, non-UTC timestamps, duplicate candidate reviews, and unsafe accepted
  record flags.
- Keep `sequence_policy_promotion_allowed_by_plan=false` and
  `execution_allowed_by_plan=false`.
- Register as `shadow_only`; never add proposal or executor authority.

## Expected Proof

- Focused promotion-plan tests for missing records, manual candidates,
  unreviewed candidates, unsafe ledger rejection, and registry/matrix safety.
- Product Layer preview visibility with one manual candidate.
- Registry and capability matrix tests proving no live widening.
- Compile, memory lint, diff check, and backend pytest.
