# Brief: source-identity-memory-trust-promotion-priority-gate-20260606

## Context

Source identity memory-trust promotion can now apply to an explicit trust store,
produce apply-audit records, and verify rollback evidence through explicit
operator review. Verified rollback evidence is still not a downstream-priority
policy decision.

## Objective

Add a read-only downstream-priority eligibility gate that consumes verified
rollback records and emits policy-gate records for a future proposal-priority
consumer.

## Safety Constraints

- Do not touch WING, dLive, OSC, MIDI, TCP, sockets, or mixer clients.
- Do not import priority-gate records into the Product Layer task queue.
- Do not mutate proposal order, runtime memory, classifier models, trust stores,
  or downstream proposal priority.
- Product Layer preview must stay non-mutating and show zero priority-gate
  records unless verified rollback evidence exists.
- Eligible records are policy evidence only; live authority and full autonomous
  readiness remain false.

## Expected Evidence

- A backend gate validates rollback-verification envelope, records, summary
  counts, safety flags, and duplicate IDs.
- Valid verified rollback records are evaluated through the shared source
  identity downstream gate.
- Trusted source identity states produce eligible priority-gate records while
  keeping direct priority mutation false.
- Held, blocked, unknown, or stale downstream-gate decisions stay not-ready.
- Product Layer, registry, and capability matrix expose the gate as non-live
  memory-learning infrastructure.
