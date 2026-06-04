# Autonomy Promotion Readiness Brief

Task id: autonomy-promotion-readiness-20260604
Date: 2026-06-04

## Context

The project-goal capability matrix now has proposal coverage for every goal
row except intentionally blocked Full Autonomous mode. The next step toward the
real target is not to widen live writes, but to make proof gaps explicit: which
action families can be considered supervised-only, which are blocked by missing
readback/rollback/hardware proof, and why Full Autonomous remains unavailable.

## Objective

Add a read-only promotion readiness snapshot that combines the project-goal
matrix, safety coverage matrix, write-surface inventory, and read-surface
inventory into one fail-closed proof-gate view.

## Safety Boundary

- No live console writes.
- No queue import.
- No new executor methods.
- No new allowed live write kinds.
- Full Autonomous must remain blocked until proof gates are complete.

## Expected Files

- `backend/autonomy_promotion_readiness.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_autonomy_promotion_readiness.py`
- `tests/test_processing_registry.py`
- `Docs/adr/autonomy-promotion-readiness-20260604.md`

## Test Plan

- Compile touched Python modules.
- Run focused readiness, registry, matrix, safety/read/write inventory tests.
- Run broader proposal-boundary tests.
- Run token and whitespace scans.
- Smoke-check that fader is supervised-only and no row is autonomous-ready.
