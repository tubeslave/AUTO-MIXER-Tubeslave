# Brief: source-identity-apply-result-safety-envelope-20260606

## Context

The source identity memory-trust promotion apply audit validates applied
executor output before it becomes rollback evidence. Executor-level safety
already denies console queries, downstream-priority authority, and live
widening, but each apply-result row should carry and preserve the same
record-level safety envelope.

## Objective

Make promotion executor apply results self-contained safety records, and make
the apply audit reject forged row-level authority even when the executor
summary is otherwise internally consistent.

## Safety Constraints

- Do not touch WING, dLive, OSC, MIDI, TCP, sockets, or mixer clients.
- Do not import audit records into the Product Layer task queue.
- Do not grant downstream-priority, queue, model, runtime-memory, live, or
  full-autonomous authority.
- Product Layer preview must continue to call the executor with `apply=false`.

## Expected Evidence

- Apply-result rows explicitly carry `queries_console=false`,
  `downstream_priority_change_allowed_by_executor=false`, and
  `live_widening_allowed=false`.
- The apply audit fails closed if a supplied apply result forges any of those
  fields.
- Focused executor/apply-audit tests pass.
