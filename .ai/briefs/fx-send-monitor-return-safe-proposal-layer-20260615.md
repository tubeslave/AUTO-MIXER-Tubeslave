# FX Send Monitor/Return-safe Proposal Layer

Date: 2026-06-15
Status: implemented

## Task

Move FX / Sends from `proposal_only_needed` to a real proposal-only layer while
preserving the current WING live-write boundary.

## Constraints

- Do not contact WING, OSC, MIDI, or mixer clients.
- Do not create live FX, send, return, insert, or parameter writes.
- FX/send proposals must remain non-queueable and not live-applicable.
- Monitor exposure and return-level safety must be explicit proposal evidence.

## Acceptance

- `fx_send_proposal` emits operator-readable proposals with readback and rollback
  blockers.
- High monitor-safety risk or stale monitor context keeps recommendations
  `analysis_only`.
- Return headroom and return-level safety are surfaced in evidence.
- Backlog, registry, capability matrix, ADR, and tests reflect the boundary.
