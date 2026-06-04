# Channel Read Coverage DTO Brief

Task id: channel-read-coverage-dto-20260604
Date: 2026-06-04

## Context

The read-surface inventory shows that channel identity and basic controls are
partially visible through `build_channel_inventory`, but consumers cannot tell
which sections came from cached mixer state, static routing, audio capture, or
no source.

## Objective

Add a read-only Product Layer DTO to each channel inventory row that reports
coverage status, source kind, and missing fields for identity, routing,
metering, basic controls, and cached gate/dynamics sections.

## Safety Boundary

- No live WING run in this task.
- No OSC/MIDI query execution.
- No console mutation.
- No live write or autonomous promotion.
- Preserve existing channel inventory fields.
- Full Autonomous remains blocked.

## Expected Files

- `backend/operator_product_state.py`
- `backend/read_surface_inventory.py`
- `tests/test_operator_product_state.py`
- `Docs/adr/channel-read-coverage-dto-20260604.md`

## Test Plan

- Compile touched Python files.
- Run focused operator product-state and read-surface tests.
- Run broader readiness/mode/registry tests.
- Run the standard full pytest command.
