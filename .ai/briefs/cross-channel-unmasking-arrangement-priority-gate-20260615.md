# Brief: Cross-channel Unmasking Arrangement Priority Gate

Date: 2026-06-15
Status: active implementation brief

## Task

Improve the proposal-only cross-channel unmasking layer so linked mirror-EQ
decisions can use fresh arrangement-role evidence when explicit channel
priorities are absent, while preserving the no-live-write boundary.

## Context

The project goal explicitly requires cross-channel unmasking and decisions
based on arrangement context. `backend/cross_adaptive_eq.py` already emits
transport-free linked cut/boost candidates with masking-context freshness,
source-identity trust, rollback/readback evidence, and no live writes. It still
treated `channel_priorities` as the only source of protected-vs-masker priority.

## Safety Boundary

This task must stay offline/proposal-only. It must not create queueable live
changes, contact WING, start a backend/WING session, or expand live write
authority.

## Deliverables

- Add optional arrangement-role context and freshness evidence to unmasking.
- Use fresh arrangement role as a priority hint when explicit channel priority
  is absent.
- Block proposal output when provided arrangement context is stale, missing per
  channel, unknown, non-processable, or below confidence threshold.
- Update proposal metadata, capability wording, tests, and docs.

## Test Plan

- Focused cross-channel unmasking, registry, and capability matrix tests.
- Related source-identity/frequency-separation/de-esser proposal tests.
- Full pytest if focused validation passes.
