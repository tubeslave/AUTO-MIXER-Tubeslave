# ADR: scene-preset-cache-freshness-20260605

## Context

Snapshots, scenes, and user presets are broad console-state surfaces. Existing
Product Layer DTOs expose cached catalog metadata and keep load/save/recall/store
blocked, while `scene_preset_management_proposal` can block stale current-state
evidence. The DTOs did not publish cache freshness, so the read surface and
proposal gate were not connected.

## Options considered

1. Add cache freshness to cached snapshots/scenes and user-presets DTOs.
2. Query the mixer when building Product Layer scene/preset snapshots.
3. Leave freshness as a caller-only proposal input.

## Decision

Use option 1. `snapshots_scenes_snapshot` and `user_presets_snapshot` now publish
top-level `cache_freshness` payloads and `summary.cache_freshness_status`, built
from existing cached state timestamps only.

## Why this won

This makes scene/preset proposal evidence more reliable without contacting the
console or adding apply authority. Stale cached catalogs can be identified before
scene/preset recommendations are treated as proposal-ready.

## Rejected alternatives

Active query was rejected because Product Layer snapshots must be safe in offline
and live views. Caller-only freshness was rejected because it leaves the Product
Layer read contract incomplete.

## Implementation plan

1. Track candidate scene/snapshot catalog, active scene, and show addresses.
2. Track candidate user-preset catalog and per-slot metadata addresses.
3. Publish freshness payloads on both DTOs.
4. Update read-surface inventory and fixture evidence.
5. Add focused fresh/stale/missing tests.

## Test plan

- `PYTHONPATH=backend python3 -m py_compile backend/operator_product_state.py backend/read_surface_inventory.py backend/read_surface_fixture_evidence.py tests/test_operator_product_state.py tests/test_read_surface_inventory.py tests/test_read_surface_fixture_evidence.py`
- Focused pytest for Product Layer scene/preset read DTOs and read-surface metadata.
- Standard repository pytest command before merge.

## Risks and rollback

Risk: fresh cached state may be mistaken for physical scene/preset proof. The DTOs
still report `queries_console=False`, `live_write=False`, and
`full_autonomous_ready=False`, and fixture evidence still reports physical
readback, scoped diff, and rollback gaps.

Rollback is removing the freshness address helpers, DTO fields, tests, inventory
metadata, fixture evidence metadata, and this ADR. No live write path is
introduced.
