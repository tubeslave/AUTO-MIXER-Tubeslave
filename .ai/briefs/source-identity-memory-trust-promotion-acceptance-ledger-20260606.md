# Brief: source-identity-memory-trust-promotion-acceptance-ledger-20260606

## Context

Source identity memory-trust promotion contracts can now surface
operator-review-ready records after promotion planning, fresh replay context,
and rollback evidence validate. The next safe step is recording operator
accepted/rejected decisions on those records without changing downstream
priority.

## Objective

Add a transport-free promotion acceptance ledger that consumes promotion
contract records and operator decisions, then emits evidence-only accepted or
rejected records.

## Safety Constraints

- Do not promote source identity trust.
- Do not change downstream proposal priority.
- Do not mutate queues, models, runtime memory, or console state.
- Do not create mixer clients, sockets, OSC, MIDI, TCP, or WING writes.
- Treat forged contract records or unsafe acceptance records as
  replacement-required.
- Preserve `execution_allowed_by_acceptance=false`,
  `memory_trust_promotion_allowed_by_acceptance=false`,
  `downstream_priority_change_allowed_by_acceptance=false`, and
  `live_authority_granted=false`.

## Expected Evidence

- A backend ledger validates promotion contract records and operator decisions.
- Product Layer preview exposes promotion acceptance counts without queue
  mutation.
- The processing registry and project goal matrix register the ledger as
  `shadow_only` memory-learning evidence, not a proposal or live executor.
- Focused ledger, Product Layer, registry, and capability tests pass.
