# Brief: source-identity-memory-trust-promotion-execution-plan-20260606

## Context

Source identity memory-trust promotion acceptance can now record operator
accepted/rejected decisions on promotion contract records. The next safe step
is surfacing accepted decisions as non-executing execution candidates while
keeping downstream priority changes behind a separate future executor.

## Objective

Add a transport-free promotion execution plan that consumes promotion acceptance
ledger evidence and emits review-only execution candidates.

## Safety Constraints

- Do not execute memory trust promotion.
- Do not change downstream proposal priority.
- Do not mutate queues, models, runtime memory, append-only memory, or console
  state.
- Do not create mixer clients, sockets, OSC, MIDI, TCP, or WING writes.
- Treat forged acceptance ledgers or unsafe records as replacement-required.
- Preserve `execution_allowed_by_plan=false`,
  `memory_trust_promotion_allowed_by_plan=false`,
  `downstream_priority_change_allowed_by_plan=false`,
  `queue_mutation_allowed_by_plan=false`,
  `model_mutation_allowed_by_plan=false`,
  `runtime_memory_mutation_allowed_by_plan=false`,
  `live_authority_granted=false`, and `full_autonomous_ready=false`.

## Expected Evidence

- A backend plan validates the promotion acceptance ledger before surfacing
  execution candidates.
- Product Layer preview exposes execution candidate counts without queue
  mutation.
- The processing registry and project goal matrix register the plan as
  `shadow_only` memory-learning evidence, not a proposal or live executor.
- Focused execution-plan, Product Layer, registry, and capability tests pass.
