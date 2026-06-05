# Brief: read-surface-fixture-drift-guard-20260605

## Context

`read_surface_fixture_evidence` makes cached DTO fixture coverage visible as
read-only evidence, while still marking physical readback proof incomplete. The
ADR for that layer calls out a practical risk: the declarative fixture map can
drift if pytest functions are renamed or deleted.

## Goal

Add a focused test guard that verifies every fixture path and pytest function
declared by `read_surface_fixture_evidence` exists in the repo.
Expose the same read-only fixture-evidence snapshot in the no-WING release
summary so remaining physical readback fixture gaps stay visible.

## Constraints

- Do not change runtime behavior.
- Do not import mixer clients, websocket services, or OSC/MIDI transports.
- Do not execute live WING/OSC writes.
- Treat cached DTO fixtures as evidence only, not Full Autonomous readiness.

## Validation

- `PYTHONPATH=backend python3 -m pytest tests/test_read_surface_fixture_evidence.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_read_surface_fixture_evidence.py tests/test_read_surface_inventory.py tests/test_processing_registry.py -q`
- `python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer-no-wing-release-check-read-surface --command-timeout-sec 300`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `python3 tools/memory/memory_lint.py`
