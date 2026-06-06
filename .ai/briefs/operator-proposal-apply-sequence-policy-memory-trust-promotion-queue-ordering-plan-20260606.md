# Brief: Operator Proposal Apply Sequence Policy Memory Trust Promotion Queue Ordering Plan

## Context

Sequence-policy ordering preview rows can recommend a read-only proposal order.
The next boundary is a queue-ordering plan that future queue or sequence
consumers can review without mutating the operator queue, proposal payloads,
policy stores, runtime memory, models, proposal state, or live mixer transport.

## Goal

Add a read-only queue-ordering plan that consumes sequence-policy ordering
preview rows and emits queue-ordering candidates for later acknowledged import
or sequence consumers.

## Constraints

- No live OSC/WING writes.
- No proposal payload mutation, queue import, queue reorder, direct sequence
  execution, policy-store mutation, proposal-state mutation, model mutation, or
  runtime-memory mutation.
- Reject forged ordering-preview rows that claim queue, sequence, policy, model,
  or live authority.
- Product Layer preview must expose the plan as evidence only.
- Register the method as `shadow_only`.
- Update tests, capability matrix, registry, preview, ADR, and project memory.

## Validation

- Focused pytest for the queue-ordering plan, upstream ordering preview,
  Product Layer preview, registry, and capability matrix.
- `python3 tools/memory/memory_lint.py`.
