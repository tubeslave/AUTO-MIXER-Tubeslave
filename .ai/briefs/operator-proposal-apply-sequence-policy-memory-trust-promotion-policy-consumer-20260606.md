# Brief: Operator Proposal Apply Sequence Policy Memory Trust Promotion Policy Consumer

## Context

Sequence-policy gate records can mark verified sequence-memory trust promotion
as eligible policy evidence. A separate consumer is needed to project that gate
evidence onto current preview proposals without changing proposal order, queues,
policy stores, runtime memory, models, or live mixer state.

## Goal

Add a read-only policy consumer that consumes sequence-policy gate records and
preview proposals, emits matched consumer records, and leaves all runtime and
live authority false.

## Constraints

- No live OSC/WING writes.
- No proposal ordering mutation, queue import, direct sequence execution,
  policy-store mutation, proposal-state mutation, model mutation, or
  runtime-memory mutation.
- Reject forged gate records that claim live, queue, sequence, policy, or model
  authority.
- Product Layer preview must expose the consumer as evidence only.
- Register the method as `shadow_only`.
- Update tests, capability matrix, registry, preview, ADR, and project memory.

## Validation

- Focused pytest for the consumer, upstream gates, Product Layer preview,
  registry, and capability matrix.
- `python3 tools/memory/memory_lint.py`.
