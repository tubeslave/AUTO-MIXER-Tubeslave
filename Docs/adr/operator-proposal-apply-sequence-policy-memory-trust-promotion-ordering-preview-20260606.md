# ADR: Operator Proposal Apply Sequence Policy Memory Trust Promotion Ordering Preview

## Context

Sequence-policy consumer records can identify preview proposals influenced by
trusted sequence-memory evidence. The system needs a visible ordered view before
any queue-ordering or sequence-execution consumer can safely use that evidence.

## Options considered

1. Feed policy-consumer records directly to queue import or sequence execution.
2. Add a read-only ordering preview.
3. Mutate Product Layer proposal payload order immediately.

## Decision

Add a read-only ordering preview.

## Why this won

It creates a separate evidence layer for recommended order while keeping the
current proposal list, operator queue, policy stores, runtime memory, models,
proposal state, sequence execution, and live mixer state unchanged.

## Rejected alternatives

Direct queue or execution behavior was rejected because it combines policy
evidence with mutation. Mutating Product Layer preview order was rejected because
the preview should remain an observational payload.

## Implementation plan

- Add an ordering-preview module that validates sequence-policy consumer
  records.
- Sort ranked proposals by adjusted policy score and leave unranked proposals in
  original order.
- Emit read-only ordering rows with no queue, policy-store, runtime-memory,
  model, proposal-state, sequence-execution, or live authority.
- Register the method as `shadow_only`.
- Add preview payload, adapter, and summary exposure.
- Track the method in decision-basis and long-term-memory capability rows.

## Test plan

- Unit-test not-ready, recommended order, forged authority, duplicate records,
  registry, and capability matrix behavior.
- Extend Product Layer preview tests.
- Run focused pytest and memory lint.

## Risks and rollback

The main risk is implying preview rows can import queues or execute sequences.
Rollback is to remove the ordering preview from preview, registry, and
capability matrix while leaving the policy consumer as the last visible evidence
layer.
