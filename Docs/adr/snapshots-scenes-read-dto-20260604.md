# ADR: snapshots-scenes-read-dto-20260604

## Context

Snapshot and scene operations can overwrite broad console state. The system
already quarantines direct snapshot load/save paths, but the Product Layer did
not have a stable read DTO for displaying cached scene catalog information.

## Options considered

1. Leave snapshots/scenes only in `wing_client.py` query helpers.
2. Add a cached-only Product Layer DTO with explicit destructive markers.
3. Add proposal/apply support for scene recall and save.

## Decision

Use option 2. Add a cached-only snapshots/scenes DTO and keep load/save blocked.

## Why this won

The cockpit can show what is known from cache while preserving the current
safety model. The DTO does not query the mixer, does not load snapshots, does
not save snapshots, and does not add queue/apply authority.

## Rejected alternatives

Querying names on every Product snapshot was rejected because it would send OSC.
Scene proposal/apply support was rejected because scoped diff, rollback proof,
and physical readback evidence are incomplete.

## Implementation plan

- Add `build_snapshots_scenes_snapshot`.
- Add `get_snapshots_scenes_snapshot`.
- Update read surface inventory fields and blocker text.
- Add focused tests for cached and missing cache behavior.

## Test plan

- `PYTHONPATH=backend python -m pytest tests/test_operator_product_state.py tests/test_read_surface_inventory.py -q`
- Broader product-state/readiness tests.
- Full suite with `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `git diff --check`

## Risks and rollback

Risk: cache completeness may be overread as physical proof. Mitigation: keep
the DTO `partial`, publish read coverage, and label load/save blocked. Rollback
is removing the builder, handler, tests, and inventory entrypoint additions.
