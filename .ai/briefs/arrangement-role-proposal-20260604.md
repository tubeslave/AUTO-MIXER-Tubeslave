# Arrangement Role Proposal Brief

Task id: arrangement-role-proposal-20260604
Date: 2026-06-04

## Context

The project goal requires automatic determination of instrument roles in the
arrangement. The capability matrix still treated arrangement roles as
shadow-only. The next safe step is an operator-review proposal layer that
explains why a source should be treated as lead, support, anchor, texture,
effect, utility, or reference.

## Objective

Add a transport-free arrangement role proposal layer that uses source-type,
section, inter-channel context, reference role, accepted-history, persistence,
ambiguity, and masking evidence.

## Safety Boundary

- No live console writes.
- No queue import.
- No `requested_change` payload.
- No direct authority over faders, EQ, dynamics, routing, scenes, or presets.
- Role proposals must be replayed against accepted/rejected engineer decisions
  before they can influence downstream action priority.

## Expected Files

- `backend/arrangement_role_proposal.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_arrangement_role_proposal.py`
- `tests/test_project_goal_capability_matrix.py`
- `Docs/adr/arrangement-role-proposal-20260604.md`

## Test Plan

- Compile touched Python modules.
- Run focused arrangement role tests.
- Run registry, proposal adapter, and capability matrix tests.
- Run broader proposal-boundary tests.
- Run `git diff --check` and smoke-check emitted proposal flags.
