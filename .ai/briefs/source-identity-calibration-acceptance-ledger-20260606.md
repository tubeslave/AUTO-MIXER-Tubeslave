# Brief: source-identity-calibration-acceptance-ledger-20260606

## Context

Source identity replay calibration can now surface review-only trust candidates
in `main_pipeline_preview`. The next safe gap is auditability: an operator may
accept or reject a calibration candidate, but that decision must not itself
write classifier models, long-term memory, downstream proposal priority, queues,
or live console state.

## Objective

Add a transport-free source identity calibration acceptance ledger that validates
operator accept/reject records against replay-calibration candidates and exposes
the result in the main pipeline preview and processing registry.

## Safety Constraints

- Do not mutate source identity models or long-term memory.
- Do not change downstream proposal priority automatically.
- Do not import accepted calibration records into the decision queue.
- Do not create mixer write requests, `requested_change`, or live apply paths.
- Treat unsafe authority claims in acceptance records as invalid evidence.
- Preserve `queue_mutation=false`, `model_mutation=false`,
  `memory_mutation=false`, `downstream_priority_mutation=false`, and
  `live_write=false`.

## Expected Evidence

- A backend ledger validates candidate id, source type, recommended trust state,
  operator decision, reviewer, UTC timestamp, and unsafe authority flags.
- `main_pipeline_preview` exposes acceptance-ledger status without mutating the
  proposal queue.
- The processing registry and project goal matrix register the ledger as
  `shadow_only` / analysis evidence, not a live executor.
- Focused ledger, Product Layer, registry, and capability tests pass.
