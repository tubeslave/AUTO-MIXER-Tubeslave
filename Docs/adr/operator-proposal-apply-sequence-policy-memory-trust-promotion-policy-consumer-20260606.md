# ADR: Operator Proposal Apply Sequence Policy Memory Trust Promotion Policy Consumer

## Context

Sequence-policy gate records can expose eligible promoted sequence-memory trust
as read-only policy evidence. Before any future ordering or execution consumer
uses that evidence, the Product Layer needs a projection that matches eligible
gate records to current preview proposals and rejects forged authority.

## Options considered

1. Feed sequence-policy gate records directly to future ordering behavior.
2. Add a read-only policy consumer projection.
3. Reorder proposals or import queue items now.

## Decision

Add a read-only policy consumer projection.

## Why this won

It creates a narrow boundary between policy eligibility and proposal handling.
The consumer can expose matched evidence to operators and tests without changing
proposal order, queues, policy stores, runtime memory, models, proposal state,
sequence execution, or live mixer state.

## Rejected alternatives

Direct ordering was rejected because it would combine evidence projection with
behavior change. Queue import or execution was rejected as too broad before a
separate ordering and sequence-consumer contract exists.

## Implementation plan

- Add a consumer module that validates sequence-policy gate records.
- Match eligible gate records to preview proposals by stable proposal and policy
  tokens.
- Emit read-only consumer records with no ordering, queue, policy-store,
  runtime-memory, model, proposal-state, sequence-execution, or live authority.
- Register the method as `shadow_only`.
- Add preview payload, adapter, and summary exposure.
- Track the method in decision-basis and long-term-memory capability rows.

## Test plan

- Unit-test not-ready, matched records, forged authority, duplicate policy keys,
  registry, and capability matrix behavior.
- Extend Product Layer preview tests.
- Run focused pytest and memory lint.

## Risks and rollback

The main risk is implying consumer records can execute or reorder proposals.
Rollback is to remove the consumer from preview, registry, and capability matrix
while leaving the sequence-policy gate as the last visible evidence layer.
