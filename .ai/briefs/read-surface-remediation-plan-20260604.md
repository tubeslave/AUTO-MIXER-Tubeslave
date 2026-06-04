# Read Surface Remediation Plan Brief

Task id: read-surface-remediation-plan-20260604
Date: 2026-06-04

## Context

The project goal requires reading and analyzing the full digital console state:
channels, buses, groups, matrices, routing, FX, scenes, and presets. The read
surface inventory is explicit, but many surfaces are still partial or missing.

## Objective

Add a read-only remediation plan that turns partial/missing read inventory rows
into concrete DTO, fixture, and coverage tasks.

## Safety Boundary

- No live WING run in this task.
- No OSC/MIDI query execution.
- No console mutation.
- No live write or autonomous promotion.
- Full Autonomous remains blocked.

## Expected Files

- `backend/read_surface_remediation_plan.py`
- `backend/processing_registry.py`
- `tests/test_read_surface_remediation_plan.py`
- `tests/test_read_surface_inventory.py`
- `tests/test_processing_registry.py`
- `Docs/adr/read-surface-remediation-plan-20260604.md`

## Test Plan

- Compile touched Python files.
- Run focused read-remediation, read-inventory, and registry tests.
- Run broader readiness/mode/proof/no-WING tests.
- Run the standard full pytest command.
