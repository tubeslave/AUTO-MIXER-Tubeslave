# ADR: Operator Proposal Apply Sequence Policy Memory Trust Promotion Queue Ordering Plan

## Context

Sequence-policy ordering preview rows can rank preview proposals from trusted
sequence-memory evidence. The system needs a queue-ordering plan before any
future importer or sequence consumer mutates the operator queue or execution
behavior.

## Options considered

1. Feed ordering-preview rows directly to queue import or sequence execution.
2. Add a read-only queue-ordering plan.
3. Mutate the Product Layer queue immediately.

## Decision

Add a read-only queue-ordering plan.

## Why this won

It creates a separate reviewable queue-planning layer while keeping the current
proposal payloads, operator queue, policy stores, runtime memory, models,
proposal state, sequence execution, and live mixer state unchanged.

## Rejected alternatives

Direct queue import was rejected because it needs explicit operator
acknowledgement, idempotency, target queue validation, and rollback thinking.
Immediate queue mutation was rejected because ordering evidence is not import or
execution authority.

## Implementation plan

- Add a queue-ordering plan module that validates sequence-policy ordering
  preview rows.
- Emit read-only queue-ordering records in recommended order.
- Keep queue import, queue reorder, policy-store mutation, runtime-memory
  mutation, model mutation, proposal-state mutation, sequence execution, and
  live authority false.
- Register the method as `shadow_only`.
- Add preview payload, adapter, and summary exposure.
- Track the method in decision-basis and long-term-memory capability rows.

## Test plan

- Unit-test not-ready, queue-ordering records, forged authority, duplicate rows,
  registry, and capability matrix behavior.
- Extend Product Layer preview tests.
- Run focused pytest and memory lint.

## Risks and rollback

The main risk is implying plan records can import or reorder the queue. Rollback
is to remove the queue-ordering plan from preview, registry, and capability
matrix while leaving ordering preview as the last visible evidence layer.
