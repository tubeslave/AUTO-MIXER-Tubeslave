# ADR: Operator Proposal Apply Sequence Policy Memory Trust Promotion Rollback Verification

## Context

Sequence-policy memory trust promotion can now be planned, contracted, accepted,
dry-run, operator-applied to an explicit store, and audited after apply. The next
required trust boundary is rollback verification: an applied trust-state change
must not influence sequence-policy consumers until rollback evidence is reviewed.

## Options considered

1. Keep apply-audit as the final evidence layer.
2. Add a separate read-only rollback verification gate.
3. Let rollback verification directly update sequence policy consumers.

## Decision

Implement a separate read-only rollback verification gate.

## Why this won

It preserves a clean boundary between applied store mutation, audit evidence, and
future policy consumption. It also gives the Product Layer a visible status for
missing or invalid rollback proof without widening live writes or queue mutation.

## Rejected alternatives

Keeping apply-audit as final evidence is too weak because it records rollback
targets but does not prove operator-reviewed rollback evidence. Direct policy
updates are too broad for this stage and would blur evidence with authority.

## Implementation plan

- Add a verifier module that consumes apply-audit and verification records.
- Reject forged live, queue, model, runtime-memory, policy-store, proposal-state,
  and direct sequence-execution authority.
- Register the method as `shadow_only`.
- Add Product Layer preview payload, adapter, and summary exposure.
- Track the method in the long-term memory capability matrix.

## Test plan

- Unit-test not-ready, missing verification, valid verified evidence, forged
  authority, registry, and capability matrix behavior.
- Extend Product Layer preview tests for payload/adapter/summary exposure.
- Run focused pytest and memory lint.

## Risks and rollback

The main risk is implying policy authority before a policy gate exists. Rollback
is to remove the verifier from preview/registry/capability matrix and keep
apply-audit as the last visible sequence-policy memory promotion layer.
