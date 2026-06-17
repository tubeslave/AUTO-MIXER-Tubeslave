# Compressor Feedback-safe Proposal Layer

Date: 2026-06-15
Status: implemented

## Task

Move compressor work from `proposal_only_needed` to a stricter proposal-only
layer by closing the missing feedback-safety evidence gap while preserving the
current WING live-write boundary.

## Constraints

- Do not contact WING, OSC, MIDI, or mixer clients.
- Do not create live compressor writes, queue imports, or direct apply/reset
  paths.
- Compressor makeup must remain bounded by true-peak headroom.
- High feedback risk must block makeup-bearing compressor proposals.
- Current dynamics freshness, source-identity trust, readback, and rollback
  requirements must remain explicit.

## Acceptance

- `compressor_dynamics_proposal` blocks high-feedback-risk makeup proposals.
- Operator proposals carry old-value and rollback-value readback sources.
- Backlog, registry, capability matrix, ADR, and tests reflect the boundary.
