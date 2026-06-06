# Brief: source-identity-proposal-priority-policy-20260606

## Context

The source identity memory-trust promotion priority gate can now emit eligible
records for downstream proposal priority. Those records still do not define how
preview proposals should be ranked or when ordering may change.

## Objective

Add a read-only proposal-priority policy projection that consumes eligible
source identity priority-gate records and preview proposals, then emits ranking
evidence for a future ordering consumer.

## Safety Constraints

- Do not touch WING, dLive, OSC, MIDI, TCP, sockets, or mixer clients.
- Do not reorder Product Layer preview proposals in-place.
- Do not import policy records into `operator_proposal_queue`.
- Do not mutate runtime memory, classifier models, source identity trust stores,
  downstream priority state, or live mixer state.
- Ranked policy records are evidence only; live authority and full autonomous
  readiness remain false.
- Preserve `downstream_priority_change_allowed_by_policy=false` and
  `live_widening_allowed=false` on policy safety, summary, and generated records.

## Expected Evidence

- A backend policy module validates priority-gate envelope, summary counts,
  eligible records, duplicate IDs, and safety flags.
- Matching preview proposals produce read-only ranking records with adjusted
  priority scores.
- Missing eligible records or missing matching proposals stay not-ready.
- Forged queue, model, runtime-memory, direct-priority, or live authority is
  rejected.
- Product Layer, registry, and capability matrix expose the policy as non-live
  decision evidence.
