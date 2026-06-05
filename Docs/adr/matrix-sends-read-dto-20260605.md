# ADR: Matrix Sends Read DTO - 2026-06-05

## Context

`routing_sends_snapshot` is the Product Layer read contract for static routing intent and cached send state. It currently covers static Dante routes, channel-to-bus sends, channel-to-main sends, and output route source fields. Matrix sends remain a named blocker for matrix/routing read coverage.

The approved live write boundary remains supervised Product Layer `fader` and `gain` only. Routing and matrix-send control are not part of the live executor.

## Options considered

1. Add cached channel-to-matrix send rows to `routing_sends_snapshot`.
2. Query matrix sends from WING during snapshot building.
3. Embed matrix sends into `matrix_snapshot`.
4. Promote matrix send apply.

## Decision

Choose option 1. `routing_sends_snapshot` now exposes cached channel-to-matrix send rows without querying or mutating hardware.

## Why this won

The known WING addresses are channel send paths (`/ch/N/send/MXK/...`), so the existing routing/sends DTO is the correct Product Layer surface. This closes a concrete read-model gap for matrix-aware routing analysis while preserving the current safety boundary.

## Rejected alternatives

- Active queries were rejected because Product Layer snapshots must not contact hardware.
- Embedding into `matrix_snapshot` was rejected because it would mix matrix strip state with channel routing rows.
- Live apply was rejected because route diffing, rollback, physical readback, and safety proof are incomplete.

## Implementation plan

1. Add cached matrix-send row construction next to bus/main send rows.
2. Add matrix-send rows to routing/sends coverage and summary.
3. Update read-surface inventory and fixture evidence.
4. Add focused regression tests for present and missing matrix-send cache.

## Test plan

- `PYTHONPATH=backend python3 -m py_compile backend/operator_product_state.py backend/read_surface_inventory.py backend/read_surface_fixture_evidence.py tests/test_operator_product_state.py tests/test_read_surface_inventory.py tests/test_read_surface_fixture_evidence.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_routing_sends_snapshot_publishes_static_and_cached_state_without_queries tests/test_operator_product_state.py::test_routing_sends_snapshot_marks_missing_cache_without_queries tests/test_read_surface_inventory.py::test_matrix_read_surface_tracks_cached_matrix_sends_without_write_authority tests/test_read_surface_fixture_evidence.py::test_read_surface_fixture_evidence_snapshot_tracks_cached_dto_fixtures_only -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: cached matrix-send state may be mistaken for physical routing proof. The snapshot safety flags still report `queries_console=False`, `live_write=False`, and `full_autonomous_ready=False`, and inventory blockers continue to require physical fixtures and route diff gates.

Rollback is removing the matrix-send row builder, snapshot fields, inventory wording, fixture evidence, tests, and this ADR. No live write path is introduced.
