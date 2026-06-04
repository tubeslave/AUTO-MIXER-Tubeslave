# Performance Adaptation Proposal Brief

Task id: performance-adaptation-proposal-20260604
Date: 2026-06-04

## Context

The project goal requires adaptation to changing musician levels, concert-time
mix adaptation, and real-time corrections. These rows were still shadow-only.
The current live executor remains limited to supervised fader/gain, so the next
safe step is a proposal layer that can explain persistent level drift without
creating live writes.

## Objective

Add a transport-free performance adaptation proposal layer for level drift and
rate-limited correction candidates. The layer must use baseline/readback,
rollback, cooldown, latency, true-peak headroom, feedback, bleed, and operator
intent evidence.

## Safety Boundary

- No live console writes.
- No queue import.
- No `requested_change` payload.
- No bypass of manual override.
- No upward adaptation without headroom, feedback, and bleed checks.
- Promotion requires replay match against engineer action plus hardware proof.

## Expected Files

- `backend/performance_adaptation_proposal.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_performance_adaptation_proposal.py`
- `tests/test_project_goal_capability_matrix.py`
- `Docs/adr/performance-adaptation-proposal-20260604.md`

## Test Plan

- Compile touched Python modules.
- Run focused performance adaptation tests.
- Run registry, proposal adapter, and capability matrix tests.
- Run broader proposal-boundary tests.
- Run `git diff --check` and smoke-check emitted proposal flags.
