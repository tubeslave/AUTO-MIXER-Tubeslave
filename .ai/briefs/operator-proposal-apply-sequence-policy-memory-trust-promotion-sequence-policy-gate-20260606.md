# Brief: Operator Proposal Apply Sequence Policy Memory Trust Promotion Sequence Policy Gate

## Context

Sequence-policy memory trust promotion now has operator-reviewed rollback
verification records. Those records are still only evidence; a separate gate is
needed before any future sequence-policy consumer can treat promoted memory as
eligible policy input.

## Goal

Add a read-only sequence-policy gate that consumes verified rollback records and
emits eligibility records for a later sequence-policy consumer. The gate must not
mutate queues, proposal order, proposal state, runtime memory, policy stores,
models, or live mixer state.

## Constraints

- No live OSC/WING writes.
- No queue import, direct sequence execution, policy-store mutation, proposal-state
  mutation, model mutation, or runtime-memory mutation.
- Product Layer preview must expose the gate but remain not-ready unless upstream
  verified rollback records exist.
- Register the method as `shadow_only`.
- Update tests, capability matrix, registry, preview, ADR, and project memory.

## Validation

- Focused pytest for the new gate, rollback verifier, Product Layer preview,
  registry, and capability matrix.
- `python3 tools/memory/memory_lint.py`.
