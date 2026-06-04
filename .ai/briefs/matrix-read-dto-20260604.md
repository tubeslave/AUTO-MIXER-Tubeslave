# Matrix Read DTO Brief

Task id: matrix-read-dto-20260604
Date: 2026-06-04

## Context

The Automixer goal requires reading and analyzing matrix state before master,
matrix, and output optimization can be trusted. The read-surface inventory still
listed matrices as partial because there was no first-class Product Layer
snapshot.

## Objective

Add a read-only matrix snapshot that normalizes cached matrix basic controls,
EQ, and insert state with explicit coverage metadata.

## Safety Boundary

- No live WING run in this task.
- No OSC/MIDI query execution.
- No console mutation.
- No matrix apply authority.
- Full Autonomous remains blocked.

## Expected Files

- `backend/operator_product_state.py`
- `backend/handlers/product_state_handlers.py`
- `backend/read_surface_inventory.py`
- `tests/test_operator_product_state.py`
- `Docs/adr/matrix-read-dto-20260604.md`

## Test Plan

- Compile touched Python files.
- Run focused operator product-state and read-surface tests.
- Run broader handler, registry, readiness, and mode tests.
- Run the standard full pytest command.
