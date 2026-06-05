# ADR: Matrix Dynamics Read DTO - 2026-06-05

## Context

`matrix_snapshot` is the Product Layer read contract for cached matrix state. It currently covers matrix basic controls, EQ, and pre/post insert state. Matrix dynamics remain listed as a read-surface blocker even though cached `/mtx/N/dyn/*` state can be normalized safely when present.

The approved live write boundary remains supervised Product Layer `fader` and `gain` only. Matrix dynamics control is not part of the live executor.

## Options considered

1. Add cached matrix dynamics to the matrix snapshot.
2. Query matrix dynamics from WING during snapshot building.
3. Leave matrix dynamics as raw cache only.
4. Promote matrix dynamics apply.

## Decision

Choose option 1. Matrix snapshot now exposes cached matrix dynamics fields and a dynamics read-coverage section without querying or mutating hardware.

## Why this won

It closes a concrete read-model blind spot for matrix/master optimization while preserving the current safety boundary. Downstream proposal builders can inspect dynamics evidence, freshness, and coverage before deciding whether a matrix recommendation is sufficiently supported.

## Rejected alternatives

- Active queries were rejected because Product Layer snapshots must not contact hardware.
- Raw cache only was rejected because consumers cannot infer completeness safely.
- Live apply was rejected because dynamics readback, rollback, headroom, and hardware proof are incomplete.

## Implementation plan

1. Add cached matrix dynamics fields to `_matrix_surface_row`.
2. Add `dynamics` to matrix read coverage.
3. Update read-surface inventory and fixture evidence.
4. Add focused regression tests for present and missing dynamics cache.

## Test plan

- `PYTHONPATH=backend python3 -m py_compile backend/operator_product_state.py backend/read_surface_inventory.py backend/read_surface_fixture_evidence.py tests/test_operator_product_state.py tests/test_read_surface_inventory.py tests/test_read_surface_fixture_evidence.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_matrix_snapshot_publishes_cached_console_state_without_queries tests/test_operator_product_state.py::test_matrix_snapshot_marks_missing_cache_without_queries tests/test_read_surface_inventory.py tests/test_read_surface_fixture_evidence.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: cached matrix dynamics may be mistaken for physical proof. The snapshot safety flags still report `queries_console=False`, `live_write=False`, and `full_autonomous_ready=False`, and inventory blockers continue to require physical fixtures and parity.

Rollback is removing the dynamics section, inventory wording, fixture evidence, tests, and this ADR. No live write path is introduced.
