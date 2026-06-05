# ADR: Bus Main DCA Cache Freshness - 2026-06-05

## Context

`bus_main_dca_snapshot` is the Product Layer read contract for cached bus, main, and DCA state. It exposes bus identity, controls, EQ, dynamics, main send, DCA assignments, main controls/EQ, and DCA identity/controls. The read-surface inventory still lists stale-cache timestamps as a blocker.

The approved live write boundary remains supervised Product Layer `fader` and `gain` only. Bus/main/DCA writes and grouping changes remain blocked.

## Options considered

1. Add aggregate cache freshness evidence to `bus_main_dca_snapshot`.
2. Query bus/main/DCA state from WING before building the Product Layer snapshot.
3. Leave bus/main/DCA freshness for future physical fixtures.
4. Promote bus/main/DCA apply after adding freshness.

## Decision

Choose option 1. Bus/main/DCA snapshot now exposes aggregate cached timestamp freshness without querying or mutating hardware.

## Why this won

Freshness is necessary for safe real-time balance recommendations. Aggregate freshness lets proposal builders and UI consumers reject stale bus/main/DCA state while preserving cached Product Layer behavior and current live-write limits.

## Rejected alternatives

- Active queries were rejected because Product Layer snapshots must not contact hardware.
- Future-only freshness was rejected because stale-cache timestamps are a current named blocker and the helper already exists.
- Live apply was rejected because physical readback, group membership proof, matrix-send proof, rollback proof, WING/dLive parity, and safety review remain incomplete.

## Implementation plan

1. Add a bus/main/DCA address collector for normalized cached fields.
2. Add aggregate `cache_freshness` and `summary.cache_freshness_status` to `build_bus_main_dca_snapshot`.
3. Update read-surface inventory and fixture evidence.
4. Add focused tests for fresh, missing, and stale cache behavior.

## Test plan

- `PYTHONPATH=backend python3 -m py_compile backend/operator_product_state.py backend/read_surface_inventory.py backend/read_surface_fixture_evidence.py tests/test_operator_product_state.py tests/test_read_surface_inventory.py tests/test_read_surface_fixture_evidence.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_bus_main_dca_snapshot_publishes_cached_console_surfaces_without_queries tests/test_operator_product_state.py::test_bus_main_dca_snapshot_marks_missing_cache_without_queries tests/test_operator_product_state.py::test_bus_main_dca_snapshot_marks_stale_cache_timestamps_without_queries tests/test_read_surface_inventory.py::test_bus_main_dca_read_surface_tracks_cache_freshness_without_write_authority tests/test_read_surface_fixture_evidence.py::test_read_surface_fixture_evidence_snapshot_tracks_cached_dto_fixtures_only -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: fresh cached state may be mistaken for physical proof. The snapshot safety flags still report `queries_console=False`, `live_write=False`, and `full_autonomous_ready=False`, and inventory blockers continue to require physical fixtures, full group membership, matrix send evidence, WING/dLive parity, and safety review.

Rollback is removing the bus/main/DCA freshness helper, snapshot fields, inventory wording, fixture evidence, tests, and this ADR. No live write path is introduced.
