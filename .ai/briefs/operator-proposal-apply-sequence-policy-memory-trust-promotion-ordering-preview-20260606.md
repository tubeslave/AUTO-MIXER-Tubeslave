# Brief: Operator Proposal Apply Sequence Policy Memory Trust Promotion Ordering Preview

## Context

Sequence-policy consumer records can project trusted sequence-memory evidence
onto current preview proposals. The next boundary is a read-only recommended
order view that operators and future consumers can inspect without mutating the
proposal payload, operator queue, policy stores, runtime memory, models, or live
console state.

## Goal

Add a read-only ordering preview that consumes sequence-policy consumer records
and emits an ordered proposal view for future queue or sequence consumers.

## Constraints

- No live OSC/WING writes.
- No proposal payload mutation, queue import, direct sequence execution,
  policy-store mutation, proposal-state mutation, model mutation, or
  runtime-memory mutation.
- Reject forged consumer records that claim ordering, queue, sequence, policy,
  model, or live authority.
- Product Layer preview must expose the ordering preview as evidence only.
- Register the method as `shadow_only`.
- Update tests, capability matrix, registry, preview, ADR, and project memory.

## Validation

- Focused pytest for the ordering preview, upstream consumer/gates, Product
  Layer preview, registry, and capability matrix.
- `python3 tools/memory/memory_lint.py`.
