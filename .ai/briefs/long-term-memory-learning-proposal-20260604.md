# Long-Term Memory Learning Proposal Brief

Task id: long-term-memory-learning-proposal-20260604
Date: 2026-06-04

## Context

The project goal requires the Automixer to improve decisions from accumulated
data, logs, tests, and previous concerts. The repository already has
`decision_memory.py`, replay traces, telemetry learning export, and memory
pipeline docs, but `long_term_memory_learning` was still shadow-only. The live
executor remains limited to supervised fader/gain.

## Objective

Add a transport-free long-term memory learning proposal layer that evaluates
verified decisions, offline replay, telemetry sessions, accepted/rejected
operator outcomes, decision evidence packets, references, genre context,
arrangement context, and safety regressions.

## Safety Boundary

- No live console writes.
- No queue import.
- No automatic memory mutation.
- No direct influence over faders, EQ, dynamics, routing, scenes, presets, or
  master bus.
- Learning records require operator review and replay correlation before they
  can influence future proposal priority.

## Expected Files

- `backend/long_term_memory_learning_proposal.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_long_term_memory_learning_proposal.py`
- `tests/test_project_goal_capability_matrix.py`
- `Docs/adr/long-term-memory-learning-proposal-20260604.md`

## Test Plan

- Compile touched Python modules.
- Run focused long-term memory learning tests.
- Run registry, proposal adapter, and capability matrix tests.
- Run broader proposal-boundary tests.
- Run whitespace scan and smoke-check emitted proposal flags.
