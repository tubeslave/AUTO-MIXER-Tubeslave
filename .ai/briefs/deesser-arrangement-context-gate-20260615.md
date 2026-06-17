# Brief: De-esser Arrangement Context Gate

Date: 2026-06-15
Status: active implementation brief

## Task

Improve the proposal-only vocal de-esser so it uses arrangement context before
recommending processing, while preserving the no-live-write boundary.

## Context

The project goal explicitly requires automatic de-essing and arrangement-aware
decisions. `backend/deesser_vocal_sibilance.py` already emits transport-free
proposal-only de-esser recommendations, but it did not distinguish lead vocal
from tucked/backing vocal roles or block stale arrangement context.

## Safety Boundary

This task must stay offline/proposal-only. It must not create queueable live
changes, contact WING, start a backend/WING session, or expand live write
authority.

## Deliverables

- Add arrangement role evidence to de-esser analysis.
- Use a less aggressive de-esser profile for tucked/backing vocals.
- Block de-esser proposals when provided arrangement context freshness is stale.
- Update proposal metadata, capability wording, tests, and docs.

## Test Plan

- Focused de-esser and capability matrix tests.
- Focused related proposal/contract tests.
- Full pytest if focused validation passes.
