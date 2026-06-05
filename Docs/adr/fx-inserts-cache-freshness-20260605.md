# ADR: FX Inserts Cache Freshness - 2026-06-05

## Context

`fx_inserts_snapshot` is the Product Layer read contract for cached FX rack and insert state. It exposes FX slot metadata, cached numeric FX parameters, and insert assignments across channels, buses, mains, and matrices. The read-surface inventory still lists stale-cache timestamps as a blocker.

The approved live write boundary remains supervised Product Layer `fader` and `gain` only. FX apply and insert assignment remain blocked.

## Options considered

1. Add aggregate cache freshness evidence to `fx_inserts_snapshot`.
2. Query FX/insert state from WING before building the Product Layer snapshot.
3. Leave FX/inserts freshness for future physical fixtures.
4. Promote FX or insert apply after adding freshness.

## Decision

Choose option 1. FX/inserts snapshot now exposes aggregate cached timestamp freshness without querying or mutating hardware.

## Why this won

Freshness is necessary for safe FX recommendations. Aggregate freshness lets proposal builders and UI consumers reject stale FX/insert state while preserving cached Product Layer behavior and current live-write limits.

## Rejected alternatives

- Active queries were rejected because Product Layer snapshots must not contact hardware.
- Future-only freshness was rejected because stale-cache timestamps are a current named blocker and the helper already exists.
- Live apply was rejected because physical readback, FX model schemas, scoped insert diffs, rollback proof, and safety review remain incomplete.

## Implementation plan

1. Add an FX/inserts address collector for normalized cached fields.
2. Add aggregate `cache_freshness` and `summary.cache_freshness_status` to `build_fx_inserts_snapshot`.
3. Update read-surface inventory and fixture evidence.
4. Add focused tests for fresh, missing, and stale cache behavior.

## Test plan

- `PYTHONPATH=backend python3 -m py_compile backend/operator_product_state.py backend/read_surface_inventory.py backend/read_surface_fixture_evidence.py tests/test_operator_product_state.py tests/test_read_surface_inventory.py tests/test_read_surface_fixture_evidence.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_fx_inserts_snapshot_publishes_cached_fx_and_insert_state_without_queries tests/test_operator_product_state.py::test_fx_inserts_snapshot_marks_missing_cache_without_queries tests/test_operator_product_state.py::test_fx_inserts_snapshot_marks_stale_cache_timestamps_without_queries tests/test_read_surface_inventory.py::test_fx_insert_read_surface_tracks_cache_freshness_without_write_authority tests/test_read_surface_fixture_evidence.py::test_read_surface_fixture_evidence_snapshot_tracks_cached_dto_fixtures_only -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: fresh cached state may be mistaken for physical proof. The snapshot safety flags still report `queries_console=False`, `live_write=False`, `fx_apply_allowed=False`, and `insert_assignment_allowed=False`, and inventory blockers continue to require physical fixtures, FX parameter schemas, semantic names, scoped insert diffs, and safety review.

Rollback is removing the FX/inserts freshness helper, snapshot fields, inventory wording, fixture evidence, tests, and this ADR. No live write path is introduced.
