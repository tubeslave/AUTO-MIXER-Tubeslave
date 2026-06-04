# Routing Sends Read DTO Brief

Task id: routing-sends-read-dto-20260604
Date: 2026-06-04

## Context

The Automixer goal requires reading route and send state before any autonomous
routing, FX-send, or monitor/send decisions can be trusted. Routing is a broad
destructive console surface, so this work must expose observability without
creating apply authority.

## Objective

Add a read-only Product Layer snapshot for static Dante routing intent plus
cached channel sends, main sends, and output route source fields.

## Safety Boundary

- No live WING run in this task.
- No OSC/MIDI query execution.
- No console mutation.
- No routing/send apply authority.
- Full Autonomous remains blocked.

## Expected Files

- `backend/operator_product_state.py`
- `backend/handlers/product_state_handlers.py`
- `backend/read_surface_inventory.py`
- `tests/test_operator_product_state.py`
- `Docs/adr/routing-sends-read-dto-20260604.md`

## Test Plan

- Compile touched Python files.
- Run focused operator product-state and read-surface tests.
- Run broader handler, registry, readiness, and mode tests.
- Run the standard full pytest command.
