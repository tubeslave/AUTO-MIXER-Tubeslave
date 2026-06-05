# ADR: Matrix Cache Freshness - 2026-06-05

## Context

`matrix_snapshot` is the Product Layer read contract for cached matrix state. It exposes matrix fader, mute, pan, EQ, dynamics, and insert state, and matrix send state is represented through the routing/sends DTO. The read-surface inventory still lists stale-cache timestamps as a blocker for matrix state.

The approved live write boundary remains supervised Product Layer `fader` and `gain` only. Matrix EQ, dynamics, inserts, routing, and matrix-send writes remain blocked.

## Options considered

1. Add aggregate cache freshness evidence to `matrix_snapshot`.
2. Query matrix state from WING before building the Product Layer snapshot.
3. Leave matrix freshness only inside the dynamics subsection.
4. Promote matrix apply after adding freshness.

## Decision

Choose option 1. Matrix snapshot now exposes aggregate cached timestamp freshness without querying or mutating hardware.

## Why this won

Freshness is necessary for safe real-time matrix recommendations. Aggregate freshness lets proposal builders and UI consumers reject stale matrix state while preserving cached Product Layer behavior and current live-write limits.

## Rejected alternatives

- Active queries were rejected because Product Layer snapshots must not contact hardware.
- Dynamics-only freshness was rejected because fader, mute, pan, EQ, and inserts also affect matrix behavior.
- Live apply was rejected because physical readback, rollback proof, WING/dLive parity, and safety review remain incomplete.

## Implementation plan

1. Add a matrix address collector for basic controls, EQ, dynamics, and inserts.
2. Add aggregate `cache_freshness` and `summary.cache_freshness_status` to `build_matrix_snapshot`.
3. Update read-surface inventory and fixture evidence.
4. Add focused tests for fresh, missing, and stale cache behavior.

## Test plan

- `PYTHONPATH=backend python3 -m py_compile backend/operator_product_state.py backend/read_surface_inventory.py backend/read_surface_fixture_evidence.py tests/test_operator_product_state.py tests/test_read_surface_inventory.py tests/test_read_surface_fixture_evidence.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_matrix_snapshot_publishes_cached_console_state_without_queries tests/test_operator_product_state.py::test_matrix_snapshot_marks_missing_cache_without_queries tests/test_operator_product_state.py::test_matrix_snapshot_marks_stale_cache_timestamps_without_queries tests/test_read_surface_inventory.py::test_matrix_read_surface_tracks_cache_freshness_without_write_authority tests/test_read_surface_fixture_evidence.py::test_read_surface_fixture_evidence_snapshot_tracks_cached_dto_fixtures_only -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: fresh cached state may be mistaken for physical proof. The snapshot safety flags still report `queries_console=False`, `live_write=False`, and `full_autonomous_ready=False`, and inventory blockers continue to require physical fixtures, WING/dLive parity, route diff proof, and safety review.

Rollback is removing the matrix freshness helper, snapshot fields, inventory wording, fixture evidence, tests, and this ADR. No live write path is introduced.
