# Corrective EQ Protected-band Proposal Layer

Date: 2026-06-15
Status: implemented

## Task

Move EQ from a blocked AutoEQ-only boundary toward a real corrective
proposal-only layer while preserving the current WING live-write boundary.

## Constraints

- Do not contact WING, OSC, MIDI, or mixer clients.
- Do not create live EQ writes, queue imports, or direct apply/reset paths.
- Corrective EQ proposals must require EQ band identity and frequency readback.
- Feedback-notch and manual EQ bands must be preserved.
- Boosts must be bounded by true-peak headroom and feedback risk.
- Phase context and current EQ state must be fresh before operator proposals.

## Acceptance

- `corrective_eq_proposal` emits operator-readable EQ proposals with readback
  and rollback blockers.
- Missing band identity, stale EQ state, stale phase context, feedback-notch
  overwrite, manual-band overwrite, unsafe boost, or missing headroom keeps the
  result `analysis_only`.
- Backlog, registry, capability matrix, ADR, and tests reflect the boundary.
