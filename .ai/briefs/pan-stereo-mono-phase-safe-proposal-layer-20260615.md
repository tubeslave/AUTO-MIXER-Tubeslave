# Pan Stereo Mono/Phase-safe Proposal Layer

Date: 2026-06-15
Status: implemented

## Task

Move Pan / Stereo Placement from `proposal_only_needed` to a real
proposal-only layer without widening live console authority.

## Constraints

- Do not contact WING, OSC, MIDI, or any mixer client.
- Do not create live pan writes or import pan proposals into a live queue.
- Non-center pan moves must carry mono/phase evidence before becoming operator
  proposals.
- Pan proposals must remain blocked until current-pan readback, rollback, and
  hardware proof exist.

## Acceptance

- `pan_spatial_proposal` emits non-queueable Product Layer proposals.
- Non-center proposals require fresh mono/phase safety evidence.
- Unsafe or missing phase evidence remains `analysis_only`.
- Backlog, registry, capability matrix, ADR, and tests reflect the boundary.
