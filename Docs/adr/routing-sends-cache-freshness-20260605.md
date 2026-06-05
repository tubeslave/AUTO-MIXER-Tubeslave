# ADR: Routing Sends Cache Freshness - 2026-06-05

## Context

`routing_sends_snapshot` is the Product Layer read contract for routing and send visibility. It now exposes static Dante routes plus cached channel bus sends, channel main sends, channel matrix sends, and output route source fields. The read-surface inventory still lists stale-cache timestamps as a blocker.

The approved live write boundary remains supervised Product Layer `fader` and `gain` only. Routing, send, matrix, and output-route writes remain blocked.

## Options considered

1. Add cache freshness evidence to routing/sends rows and snapshot summary.
2. Query routing/sends from WING before building the Product Layer snapshot.
3. Leave freshness to future physical fixtures only.
4. Promote route/send apply after adding freshness.

## Decision

Choose option 1. Routing/sends DTOs now expose cached timestamp freshness per row and at the top level without querying or mutating hardware.

## Why this won

Freshness is necessary for safe real-time recommendations. It lets proposal builders and UI consumers reject stale send/routing state while preserving the cached Product Layer architecture and current live-write boundary.

## Rejected alternatives

- Active queries were rejected because Product Layer snapshots must not contact hardware.
- Future-only freshness was rejected because stale cache is a current named blocker and the helper already exists.
- Live apply was rejected because physical readback, route diffing, rollback proof, and safety review remain incomplete.

## Implementation plan

1. Add row-level freshness metadata for bus, main, matrix send, and output route rows.
2. Add aggregate routing/sends cache freshness and summary status.
3. Update read-surface inventory and fixture evidence.
4. Add focused tests for fresh, missing, and stale cache behavior.

## Test plan

- `PYTHONPATH=backend python3 -m py_compile backend/operator_product_state.py backend/read_surface_inventory.py backend/read_surface_fixture_evidence.py tests/test_operator_product_state.py tests/test_read_surface_inventory.py tests/test_read_surface_fixture_evidence.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_routing_sends_snapshot_publishes_static_and_cached_state_without_queries tests/test_operator_product_state.py::test_routing_sends_snapshot_marks_missing_cache_without_queries tests/test_operator_product_state.py::test_routing_sends_snapshot_marks_stale_cache_timestamps_without_queries tests/test_read_surface_inventory.py::test_routing_send_read_surface_tracks_cache_freshness_without_write_authority tests/test_read_surface_fixture_evidence.py::test_read_surface_fixture_evidence_snapshot_tracks_cached_dto_fixtures_only -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: fresh cached state may be mistaken for physical proof. The snapshot safety flags still report `queries_console=False`, `live_write=False`, and `full_autonomous_ready=False`, and inventory blockers continue to require physical fixtures and route diff proof.

Rollback is removing the freshness metadata, inventory wording, fixture evidence, tests, and this ADR. No live write path is introduced.
