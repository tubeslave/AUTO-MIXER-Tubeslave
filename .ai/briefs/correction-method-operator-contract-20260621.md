# Brief: Correction Method Operator Contract

Date: 2026-06-21
Status: active implementation brief

## Task

Add a shared operator-facing correction contract for offline/proposal-only
Automixer method families.

## Context

The user goal requires every correction method to explain its evidence, measured
or audible problem, bounded correction, confidence, safety blockers, applicable
headroom/feedback/true-peak checks, rollback expectation, and future WING
readback contract. Existing proposal modules already stay transport-free and
non-queueable, but `build_proposal_only_method_contracts()` does not yet expose
that full required field set consistently across method families.

## Safety Boundary

This work is offline/proposal-only. It must not contact WING, start a backend
session, send OSC/MIDI, create queueable live changes, or expand live authority
beyond the current supervised fader/gain boundary.

## Deliverables

- Extend proposal-only method contracts with the shared correction fields.
- Keep all non-fader/gain methods non-queueable and `can_apply_live=false`.
- Include method-family applicability for headroom, feedback, true-peak,
  rollback, and future WING readback evidence.
- Add focused tests proving every named correction family is covered.
- Record the decision in `Docs/adr/`.

## Test Plan

- Focused tests for proposal-only method contracts.
- Existing method-family contract tests where touched.
