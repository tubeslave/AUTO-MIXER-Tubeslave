# Channel Processing Read DTO Brief

Task id: channel-processing-read-dto-20260604
Date: 2026-06-04

## Context

The project goal requires the Automixer to read and analyze channel EQ,
compression, gate, and HPF state before any autonomous processing decisions can
be trusted. WING cached state already contains many of these addresses, but the
Product Layer did not expose a dedicated channel-processing read DTO.

## Objective

Add a read-only `channel_processing_snapshot` contract that normalizes cached
HPF, EQ, compressor, and gate state with per-section coverage metadata.

## Safety Boundary

- No live WING run in this task.
- No OSC/MIDI query execution.
- No console mutation.
- No EQ/dynamics/gate apply authority.
- Full Autonomous remains blocked.

## Expected Files

- `backend/operator_product_state.py`
- `backend/handlers/product_state_handlers.py`
- `backend/read_surface_inventory.py`
- `tests/test_operator_product_state.py`
- `Docs/adr/channel-processing-read-dto-20260604.md`

## Test Plan

- Compile touched Python files.
- Run focused operator product-state and read-surface tests.
- Run broader handler, registry, readiness, and mode tests.
- Run the standard full pytest command.
