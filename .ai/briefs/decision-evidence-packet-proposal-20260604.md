# Decision Evidence Packet Proposal Brief

Task id: decision-evidence-packet-proposal-20260604
Date: 2026-06-04

## Context

The project goal requires every meaningful mix decision to be grounded in
audio, psychoacoustics, statistics, history, references, previous decisions,
inter-channel context, arrangement, and genre. The capability matrix still
treated `decision_basis` as shadow-only. The live executor remains limited to
supervised fader/gain.

## Objective

Add a transport-free decision evidence packet proposal layer that can attach a
single operator-review evidence contract to proposals and rejected shadow
decisions.

## Safety Boundary

- No live console writes.
- No queue import.
- No direct authority over faders, EQ, dynamics, routing, scenes, presets, or
  master bus.
- No apply payload.
- Evidence packets must be replayed against accepted/rejected engineer decisions
  before they can influence live authority.

## Expected Files

- `backend/decision_evidence_packet_proposal.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_decision_evidence_packet_proposal.py`
- `tests/test_project_goal_capability_matrix.py`
- `Docs/adr/decision-evidence-packet-proposal-20260604.md`

## Test Plan

- Compile touched Python modules.
- Run focused decision evidence packet tests.
- Run registry, proposal adapter, and capability matrix tests.
- Run broader proposal-boundary tests.
- Run `git diff --check` and smoke-check emitted proposal flags.
