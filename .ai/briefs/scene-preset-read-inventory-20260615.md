# Scene / Preset Read-only Inventory

Date: 2026-06-15
Status: implemented

## Task

Move Scenes / Presets from `read_only_inventory_needed` to a concrete
read-only inventory artifact while preserving the current WING live-write
boundary.

## Constraints

- Do not contact WING, OSC, MIDI, or mixer clients.
- Do not load, save, recall, store, or queue scene/preset operations.
- Use already captured cached/static Product Layer rows only.
- Surface freshness, load/save policy markers, recall/store policy markers,
  destructive-change markers, and preset scope blockers must be explicit.
- The inventory must not be queueable and must not grant live authority.

## Acceptance

- `scene_preset_read_inventory` produces a read-only scene/preset catalog
  artifact.
- Missing/stale surfaces, missing scene policy markers, missing preset policy
  markers, missing destructive markers, and missing preset scope keep the
  inventory incomplete.
- Registry, capability matrix, backlog, ADR, report, and tests reflect the
  boundary.
