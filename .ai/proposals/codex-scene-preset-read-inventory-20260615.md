# Codex Proposal: Scene / Preset Read-only Inventory

## Short Thesis

Add a transport-free inventory that normalizes cached scene and user-preset
catalog rows before downstream scene/preset proposals can claim scoped context.

## What I Understood

The Automixer goal requires future scene and user-preset control, but those
operations can overwrite broad console state. The current safe increment is
catalog awareness, not loading, saving, recalling, storing, or queueing
scene/preset mutations.

## Proposed Solution

- Add `backend/scene_preset_read_inventory.py`.
- Accept already captured Product Layer rows for snapshots/scenes and user
  presets.
- Normalize those rows into scene and preset catalog inventories.
- Require fresh `snapshots_scenes` and `user_presets` surfaces before a ready
  inventory claim.
- Block missing scene load/save policies, missing scene destructive markers,
  missing preset scope, missing preset recall/store policies, and missing
  preset destructive recall markers.
- Register the method as `shadow_only` and wire it into console read and
  scene/preset capability rows.

## Likely Files To Touch

- `backend/scene_preset_read_inventory.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_scene_preset_read_inventory.py`
- `tests/test_project_goal_capability_matrix.py`
- `.ai/project-layer/correction_methods_backlog.md`

## Alternatives Considered

- Expand scene/preset proposals directly: rejected because the missing piece is
  read-only catalog evidence before scoped diff/rollback proposals.
- Query WING or use snapshot load helpers to discover scene contents: rejected
  because this pass must stay no-WING, transport-free, and non-destructive.

## Risks

- The default registry snapshot is intentionally incomplete because no live or
  cached rows are passed at registry build time.
- Consumers still need adapter work to feed real Product Layer catalog rows
  into the inventory.

## Test Plan

- Compile touched Python files.
- Run focused scene/preset inventory, read-surface, scene/preset proposal,
  registry, and capability matrix tests.
- Run full project pytest and no-WING release checks before calling the
  increment validated.

## Where The Other Agent May Disagree

Kimi may prefer physical scene/preset readback artifacts first. The inventory
is still useful now because it defines the non-authoritative read-only evidence
shape that those future physical artifacts must satisfy.
