# Brief: source-identity-calibration-memory-trust-promotion-contract-20260606

## Context

Source identity memory-trust promotion planning can now surface accepted
memory-trust records as manual candidates. The next safe step is an explicit
promotion contract that checks replay freshness and rollback evidence before any
future downstream priority promotion is considered.

## Objective

Add a transport-free source identity memory trust promotion contract that
validates manual promotion candidates, fresh replay context, and rollback
evidence while keeping all runtime trust and priority mutation disabled.

## Safety Constraints

- Do not mutate source identity trust, models, memory, queues, or downstream
  proposal priority.
- Do not create mixer writes, queue entries, or live apply paths.
- Require fresh replay context before contract-ready records are exposed.
- Require rollback evidence before contract-ready records are exposed.
- Treat forged promotion-plan payloads, unsafe freshness evidence, or unsafe
  rollback evidence as replacement-required.
- Preserve `queue_mutation=false`, `model_mutation=false`,
  `runtime_memory_mutation=false`, `downstream_priority_mutation=false`, and
  `live_write=false`.

## Expected Evidence

- A backend contract validates promotion plans and rollback evidence.
- Product Layer preview exposes contract-ready counts without queue mutation.
- The processing registry and project goal matrix register the contract as
  `shadow_only` memory-learning evidence, not a proposal or live executor.
- Focused contract, Product Layer, registry, and capability tests pass.
