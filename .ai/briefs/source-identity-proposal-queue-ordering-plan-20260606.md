# Brief: source-identity-proposal-queue-ordering-plan-20260606

## Context

The source identity proposal ordering preview can show a recommended proposal
order, but Product Layer still has no reviewed boundary that turns that view
into explicit queue-ordering/import candidates.

## Objective

Add a read-only queue-ordering plan that consumes source identity ordering
preview rows, validates them, and emits plan records for a future acknowledged
queue import consumer.

## Safety Constraints

- Do not touch WING, dLive, OSC, MIDI, TCP, sockets, or mixer clients.
- Do not reorder Product Layer preview proposals in-place.
- Do not mutate `operator_proposal_queue`.
- Do not import plan records into the decision queue.
- Do not mutate runtime memory, classifier models, source identity trust stores,
  downstream priority state, or live mixer state.
- Plan records are evidence only; live authority and full autonomous readiness
  remain false.
- Preserve `queue_import_allowed_by_plan=false`,
  `queue_reorder_allowed_by_plan=false`, `queue_mutation=false`,
  `can_import_to_queue=false`, and `can_apply_live=false` on safety, summary,
  and generated rows.

## Expected Evidence

- A backend queue-ordering plan module validates ordering-preview envelope,
  summary counts, rows, duplicate IDs, recommended index sequence, UTC
  timestamps, and safety flags.
- Valid ordering-preview rows produce read-only queue-ordering plan records.
- Missing ordering-preview rows stay not-ready.
- Forged queue import, queue mutation, model/runtime-memory mutation, direct
  priority, or live authority is rejected.
- Product Layer, registry, and capability matrix expose the plan as non-live
  decision evidence.
