# Brief: source-identity-proposal-ordering-preview-20260606

## Context

The source identity proposal-priority policy can emit ranked policy records, but
Product Layer still has no separate view that shows the operator how those
records would affect proposal order.

## Objective

Add a read-only proposal ordering preview that consumes source identity
proposal-priority policy output and preview proposals, then emits a recommended
ordered view for inspection.

## Safety Constraints

- Do not touch WING, dLive, OSC, MIDI, TCP, sockets, or mixer clients.
- Do not reorder Product Layer preview proposals in-place.
- Do not import ordering preview rows into `operator_proposal_queue`.
- Do not mutate runtime memory, classifier models, source identity trust stores,
  downstream priority state, or live mixer state.
- Ordering preview rows are evidence only; live authority and full autonomous
  readiness remain false.
- Preserve `proposal_order_mutation=false`, `queue_mutation=false`,
  `downstream_priority_change_allowed_by_policy=false`, and
  `live_widening_allowed=false` on safety, summary, and generated rows.

## Expected Evidence

- A backend ordering-preview module validates priority-policy envelope,
  summary counts, policy records, duplicate IDs, proposal links, UTC timestamps,
  and safety flags.
- Valid ranked policy records produce a separate `ordered_proposals` view.
- Unranked proposals remain visible in original order after ranked records.
- The original `proposals` payload is not mutated.
- Forged ordering, queue, model, runtime-memory, direct-priority, or live
  authority is rejected.
- Product Layer, registry, and capability matrix expose the preview as non-live
  decision evidence.
