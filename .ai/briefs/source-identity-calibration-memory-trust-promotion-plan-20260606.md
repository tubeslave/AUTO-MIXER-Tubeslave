# Brief: source-identity-calibration-memory-trust-promotion-plan-20260606

## Context

Source identity memory-trust acceptance records can now capture operator
accepted/rejected decisions. The next safe step is a manual promotion plan that
surfaces accepted records as candidates for a future promotion contract without
changing runtime trust or downstream proposal priority.

## Objective

Add a transport-free source identity memory trust promotion plan that consumes
accepted memory-trust ledger records and emits manual promotion candidates only.

## Safety Constraints

- Do not mutate source identity models or runtime trust state.
- Do not change downstream proposal priority.
- Do not import promotion candidates into the decision queue.
- Do not create mixer write requests, `requested_change`, or live apply paths.
- Treat unsafe acceptance ledgers, summary drift, ambiguous records, or unsafe
  accepted-record flags as replacement-required.
- Preserve `queue_mutation=false`, `model_mutation=false`,
  `runtime_memory_mutation=false`, `downstream_priority_mutation=false`, and
  `live_write=false`.

## Expected Evidence

- A backend plan validates memory-trust acceptance ledgers and accepted records.
- Product Layer preview exposes manual promotion candidate counts without queue
  mutation.
- The processing registry and project goal matrix register the plan as
  `shadow_only` memory-learning evidence, not a proposal or live executor.
- Focused plan, Product Layer, registry, and capability tests pass.
