# Bus Main DCA Read DTO Brief

Task id: bus-main-dca-read-dto-20260604
Date: 2026-06-04

## Context

The Automixer goal requires reading and analyzing bus, main, group, and DCA
state before broad autonomous balance decisions can be trusted. The existing
`groups_buses_snapshot` is useful for UI grouping, but it is synthetic and does
not expose cached console state for buses, mains, or DCAs.

## Objective

Add a read-only Product Layer snapshot for cached bus, main, and DCA state with
per-section coverage metadata.

## Safety Boundary

- No live WING run in this task.
- No OSC/MIDI query execution.
- No console mutation.
- No bus/main/DCA apply authority.
- Full Autonomous remains blocked.

## Expected Files

- `backend/operator_product_state.py`
- `backend/handlers/product_state_handlers.py`
- `backend/read_surface_inventory.py`
- `tests/test_operator_product_state.py`
- `Docs/adr/bus-main-dca-read-dto-20260604.md`

## Test Plan

- Compile touched Python files.
- Run focused operator product-state and read-surface tests.
- Run broader handler, registry, readiness, and mode tests.
- Run the standard full pytest command.
