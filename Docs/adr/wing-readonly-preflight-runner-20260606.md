# ADR: WING Read-Only Preflight Runner

## Context

The current execution priority is a supervised WING Rack MVP. Before any
supervised fader or gain proof, the project needs read-only connection evidence,
current console state capture, and physical readback evidence.

## Options considered

1. Keep using the proposal validation runner for read-only preflight.
2. Add a dedicated read-only runner that uses the existing backend handlers.
3. Add new backend transport handlers.

## Decision

Use option 2.

## Why this won

The existing backend already exposes `get_operator_mode`, `get_state`, and
`query_mixer_parameter`. A dedicated runner can capture the first proof steps
without touching operator mode, proposals, rollback, emergency stop, connect,
disconnect, or any hard-safety write message.

## Rejected alternatives

- The proposal validation runner is correct for later supervised apply proof,
  but too broad for the first read-only preflight step.
- New backend handlers would increase surface area without need.

## Implementation plan

- Add `tools/wing_readonly_preflight_runner.py`.
- Add fake-WebSocket tests for pass/block cases.
- Register the runner in `backend/write_surface_inventory.py` as proof tooling
  so direct-write-token static scans keep covering its forbidden-message guard.
- Add an operator-visible report and memory note.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_runner.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_runner.py tests/test_wing_operator_proposal_validation_runner.py tests/test_server.py::TestWingDeploymentWriteBoundary::test_query_mixer_parameter_reads_gain_without_writing tests/test_server.py::TestWingDeploymentWriteBoundary::test_query_mixer_parameter_blocks_unsupported_address`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_write_surface_inventory_static_scan.py tests/test_processing_registry.py::test_write_surface_inventory_is_fail_closed_by_default tests/test_processing_registry.py::test_write_surface_inventory_covers_known_legacy_write_paths tests/test_processing_registry.py::test_write_surface_snapshot_status_counts_and_blocked_surfaces`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `python3 -m py_compile tools/wing_readonly_preflight_runner.py`
- `python3 -m py_compile tools/wing_readonly_preflight_runner.py backend/write_surface_inventory.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

The runner may be mistaken for a completed hardware proof. The artifact and
report therefore state that it proves only backend-observed read-only evidence.
Rollback is a normal git revert.
