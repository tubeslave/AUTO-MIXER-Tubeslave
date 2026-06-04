# Mix Variant Profile Proposal Brief

Task id: mix-variant-profile-proposal-20260604
Date: 2026-06-04

## Context

The project goal requires automatic mix variants for genres and tasks. The
current capability matrix still treated this area as offline-only even though
Assisted mode needs a way to propose a profile before any downstream fader, FX,
master-bus, scene, or preset action is considered.

## Objective

Add a transport-free mix variant profile proposal layer that turns genre, task,
reference metrics, arrangement context, Ayaic level planes, and safety evidence
into a non-queueable operator proposal.

## Safety Boundary

- No live console writes.
- No `requested_change` payload.
- No artist/style clone requests.
- No direct scene/preset apply.
- Follow-up changes must fan out through existing proposal-only layers and keep
  the current fader/gain live boundary unchanged.

## Expected Files

- `backend/mix_variant_profile_proposal.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_mix_variant_profile_proposal.py`
- `tests/test_project_goal_capability_matrix.py`
- `Docs/adr/mix-variant-profile-proposal-20260604.md`

## Test Plan

- Compile touched Python modules.
- Run focused mix variant profile tests.
- Run registry, proposal adapter, and capability matrix tests.
- Run broader proposal-boundary tests.
- Run `git diff --check` and payload smoke checks.
