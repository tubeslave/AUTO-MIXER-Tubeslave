# ADR: Operator Proposal Apply Sequence Policy Memory Trust Promotion Sequence Policy Gate

## Context

Rollback verification records confirm that applied sequence-policy memory trust
promotion has operator-reviewed rollback evidence. The next boundary is
eligibility: a future sequence-policy consumer should only see records that have
passed a dedicated gate.

## Options considered

1. Feed rollback verification records directly to future policy consumers.
2. Add a read-only sequence-policy gate.
3. Add queue ordering or direct execution behavior now.

## Decision

Add a read-only sequence-policy gate.

## Why this won

It creates an explicit evidence-to-policy boundary without changing runtime
policy stores, queue order, proposal state, or live execution. It also gives the
Product Layer a visible status for records that are verified but not eligible.

## Rejected alternatives

Direct consumer wiring was rejected because it would blur verified rollback
evidence with policy eligibility. Queue or execution changes were rejected as too
broad before a separate sequence-policy consumer contract exists.

## Implementation plan

- Add a gate module that consumes rollback verification.
- Mark only `trusted_for_sequence_policy` verified records as eligible.
- Reject forged live, queue, model, runtime-memory, policy-store, proposal-state,
  and sequence-execution authority.
- Register the method as `shadow_only`.
- Add preview payload, adapter, and summary exposure.
- Track the method in the long-term memory capability matrix.

## Test plan

- Unit-test not-ready, eligible records, forged authority, blocked policy state,
  registry, and capability matrix behavior.
- Extend Product Layer preview tests.
- Run focused pytest and memory lint.

## Risks and rollback

The main risk is accidentally implying execution authority. Rollback is to remove
the gate from preview, registry, and capability matrix while keeping rollback
verification as the last visible evidence layer.
