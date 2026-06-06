# Brief: source-identity-memory-trust-promotion-executor-dry-run-20260606

## Context

Source identity memory-trust promotion execution plans can now surface accepted
promotion records as non-executing candidates. The next safe step is describing
the exact future apply operation without applying it.

## Objective

Add a transport-free executor dry-run that consumes promotion execution
candidates and emits deterministic apply specifications for
`source_identity_downstream_gate`.

## Safety Constraints

- Do not apply memory trust promotion.
- Do not change downstream proposal priority.
- Do not mutate queues, models, runtime memory, append-only memory, or console
  state.
- Do not create mixer clients, sockets, OSC, MIDI, TCP, or WING writes.
- Treat forged execution plans, unsafe candidates, duplicate source specs, and
  summary drift as replacement-required.
- Preserve `execution_allowed_by_dry_run=false`,
  `memory_trust_promotion_applied=false`,
  `memory_trust_promotion_allowed_by_dry_run=false`,
  `downstream_priority_change_allowed_by_dry_run=false`,
  `queue_mutation_allowed_by_dry_run=false`,
  `model_mutation_allowed_by_dry_run=false`,
  `runtime_memory_mutation_allowed_by_dry_run=false`,
  `live_authority_granted=false`, and `full_autonomous_ready=false`.

## Expected Evidence

- A backend dry-run builder validates promotion execution plans before
  surfacing apply specs.
- Dry-run records include target store, target path, previous/target trust
  state, rollback strategy, and idempotency key.
- Product Layer preview exposes dry-run counts without queue mutation.
- The processing registry and project goal matrix register the dry run as
  `shadow_only` memory-learning evidence, not a proposal or live executor.
- Focused dry-run, Product Layer, registry, and capability tests pass.
