# Brief: source-identity-calibration-persistence-plan-20260606

## Context

Source identity replay calibration can surface preview-only trust candidates, and
the acceptance ledger can validate operator accept/reject decisions. The next
gap is persistence planning: accepted records should become auditable manual
persistence candidates, but they still must not write classifier models,
long-term memory, downstream proposal priority, queues, or live console state.

## Objective

Add a transport-free source identity calibration persistence plan that validates
the acceptance ledger again and surfaces accepted calibration records as manual
persistence candidates only.

## Safety Constraints

- Do not mutate source identity models or long-term memory.
- Do not change downstream proposal priority automatically.
- Do not import persistence candidates into the decision queue.
- Do not create mixer write requests, `requested_change`, or live apply paths.
- Treat unsafe ledger envelope, summary, or accepted-record claims as invalid.
- Preserve `queue_mutation=false`, `model_mutation=false`,
  `memory_mutation=false`, `downstream_priority_mutation=false`, and
  `live_write=false`.

## Expected Evidence

- A backend plan validates ledger identity, UTC timestamps, status, safety flags,
  summary counts, accepted/rejected candidate-id summaries, and accepted-record
  structure.
- `main_pipeline_preview` exposes the persistence plan without mutating the
  proposal queue.
- The processing registry and project goal matrix register the plan as
  `shadow_only` / analysis evidence, not a live executor.
- Focused plan, Product Layer, registry, and capability tests pass.
