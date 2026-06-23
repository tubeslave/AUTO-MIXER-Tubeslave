# ADR: Scene / Preset Read-only Inventory

Date: 2026-06-15
Status: accepted

## Context

The Automixer goal requires reading and eventually controlling scenes and user
presets. Existing scene/preset proposal logic remains proposal-only, and
physical readback contracts exist, but the correction backlog still called out
a missing read-only inventory.

## Options Considered

1. Leave Scenes / Presets as `read_only_inventory_needed`.
2. Add a transport-free inventory over cached/static Product Layer catalog rows.
3. Query WING or promote scene/preset load/save/recall/store operations.

## Decision

Use option 2. `scene_preset_read_inventory` is a `shadow_only` method that
normalizes scene and user-preset catalog evidence. Missing or stale surfaces,
missing load/save/recall/store policy markers, missing destructive markers, and
missing preset scope keep the inventory incomplete.

## Why This Won

It advances console-state awareness without expanding live authority. The
inventory gives scene/preset proposal layers a concrete evidence shape while
remaining read-only, non-queueable, and transport-free.

## Rejected Alternatives

- Adding another scene/preset proposal would not close the read-only inventory
  gap.
- Querying hardware or promoting load/save/recall/store operations is
  premature before WING read-only proof, physical fixtures, scoped diff,
  rollback, and manual review are complete.

## Implementation Plan

- Add `backend/scene_preset_read_inventory.py`.
- Register it as a `shadow_only` processing method.
- Add it to console read and scene/preset capability rows.
- Update correction backlog from `read_only_inventory_needed` to
  `read_only_inventory`.
- Add focused tests for read-only safety, ready/incomplete inventory states,
  registry snapshot visibility, and capability visibility.

## Test Plan

- `PYTHONPATH=backend python -m py_compile ...`
- `PYTHONPATH=backend python -m pytest -q tests/test_scene_preset_read_inventory.py tests/test_read_surface_inventory.py tests/test_scene_preset_management_proposal.py tests/test_project_goal_capability_matrix.py tests/test_processing_registry.py`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend`

## Risks And Rollback

The registry-level inventory is incomplete until real Product Layer rows are
passed. Rollback is removing the inventory module, registry entry, capability
references, tests, and documentation. No live transport code is touched.
