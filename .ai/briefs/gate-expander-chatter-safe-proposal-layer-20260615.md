# Brief: Gate/Expander Chatter-safe Proposal Layer

Date: 2026-06-15
Status: active implementation brief

## Task

Improve the context-aware gate/expander layer so safe recommendations can be
rendered as Product Layer operator proposals while preserving the no-live-write
boundary.

## Context

The project goal explicitly requires automatic gating and console gate control.
`backend/context_aware_gate.py` already emits transport-free dry-run
recommendations with bleed telemetry, freshness, and source-identity trust
evidence. The remaining gap is an operator proposal surface with explicit
non-queueable/no-live evidence and a hard-gate chatter guard.

## Safety Boundary

This task must stay offline/proposal-only. It must not create queueable live
changes, contact WING, start a backend/WING session, or expand live write
authority.

## Deliverables

- Add a Product Layer proposal shape for context-aware gate/expander output.
- Keep analysis holds out of operator proposal lists.
- Force high-chatter drum recommendations to gentle expander proposals instead
  of hard gate proposals.
- Attach freshness, source-identity, chatter, hysteresis, readback, and rollback
  evidence to the proposal.
- Update contracts, capability wording, backlog, tests, and docs.

## Test Plan

- Focused context-aware gate, capability matrix, registry, and CAIG tests.
- Full pytest if focused validation passes.
