# Brief: Operator Proposal Apply Sequence Policy Memory Trust Promotion Rollback Verification

## Context

Sequence-policy memory trust promotion now has a plan, contract, acceptance ledger,
execution plan, dry-run executor, operator-gated executor, and apply audit. The
remaining safety gap is that applied policy-store changes produce rollback audit
records, but no separate operator-reviewed verification gate confirms those
records before a future sequence-policy consumer may trust them.

## Goal

Add a read-only rollback verification layer that consumes apply-audit records and
operator-provided verification evidence. The layer must be usable only as
evidence for a future sequence-policy gate, not as queue import, direct sequence
execution, runtime-memory mutation, policy-store mutation, or live authority.

## Constraints

- No live OSC/WING writes.
- No queue, model, proposal-state, runtime-memory, or policy-store mutation.
- Product Layer preview must remain dry-run/not-ready unless applied audit records
  and explicit operator verification records exist.
- Register the method as `shadow_only`.
- Update tests, capability matrix, registry, preview, ADR, and project memory.

## Validation

- Focused pytest for the new verifier, Product Layer preview slice, registry, and
  capability matrix.
- `python3 tools/memory/memory_lint.py`.
