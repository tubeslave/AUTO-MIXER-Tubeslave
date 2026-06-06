# Brief: source-identity-promotion-acceptance-record-safety-20260606

## Context

Source identity memory-trust promotion acceptance records are now consumed by a
separate non-executing promotion execution plan. The acceptance ledger emits
safe records, but the execution plan must also distrust externally supplied or
forged ledger rows that claim live authority directly.

## Objective

Make promotion acceptance records self-contained safety evidence and make the
execution plan reject record-level live, console-query, queue, model, runtime
memory, downstream-priority, or UI-queue authority claims.

## Safety Constraints

- Do not execute memory trust promotion.
- Do not change downstream proposal priority.
- Do not mutate queues, models, runtime memory, append-only memory, or console
  state.
- Do not create mixer clients, sockets, OSC, MIDI, TCP, or WING writes.
- Preserve the promotion chain as read-only evidence until a separate reviewed
  executor and rollback path exist.

## Expected Evidence

- Acceptance ledger records carry explicit no-live/no-query/no-mutation flags.
- The promotion execution plan rejects forged record-level authority claims even
  when `unsafe_authority_flags` is empty.
- Focused source-identity and Product Layer preview tests pass.
