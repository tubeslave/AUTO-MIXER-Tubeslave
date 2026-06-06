# Brief: source-identity-memory-trust-promotion-rollback-verification-20260606

## Context

The source identity memory-trust promotion chain can now apply trust state to an
explicit store and produce rollback audit records. Those records still have
`rollback_verified=false`, so downstream proposal-priority policy must not
consume them until rollback evidence is independently reviewed.

## Objective

Add a read-only rollback verification gate that consumes apply-audit records and
operator verification records, then emits verified rollback evidence for a
future downstream-priority gate.

## Safety Constraints

- Do not touch WING, dLive, OSC, MIDI, TCP, sockets, or mixer clients.
- Do not import verification records into the Product Layer task queue.
- Do not mutate runtime memory, classifier models, trust stores, or downstream
  proposal priority.
- Product Layer preview must stay non-mutating and show zero verified records
  unless operator verification records are supplied.
- Verified rollback records are evidence only; live authority and full
  autonomous readiness remain false.

## Expected Evidence

- A backend verifier validates apply-audit envelope, records, summary counts,
  safety flags, and operator verification records.
- Missing verification keeps the gate not-ready and names unverified audit IDs.
- Valid operator verification records produce `rollback_verified=true` records.
- Forged live, queue, model, runtime-memory, or direct priority authority is
  rejected.
- Product Layer, registry, and capability matrix expose the verifier as
  non-live memory-learning infrastructure.
