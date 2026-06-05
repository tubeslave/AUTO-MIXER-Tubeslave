# Snapshots/Scenes Physical Readback Contract - 2026-06-05

## Context

`snapshots_scenes` exposes cached scene catalog and active-scene markers, but it
does not prove scene contents, scoped diffs, rollback references, or physical
readback parity. Scene recall/save can overwrite live console state, so cached
catalog visibility cannot be treated as automation proof.

## Task

Add a read-only, transport-free contract for future snapshots/scenes physical
readback artifacts and expose it through registry, no-WING, Product Layer,
frontend Module Status, fixture evidence, remediation planning, and console-read
goal metadata.

## Constraints

- Do not load, save, recall, or query scenes.
- Keep `scene_load_allowed=false` and `scene_save_allowed=false`.
- Preserve `physical_fixture_gap_remains=true`.
- Normalize source live-authority and full-autonomous claims to false.
- Require scene content fingerprint, scoped diff marker, and rollback reference
  before a future artifact can be trusted.

## Expected Outcome

Scene/preset automation has a schema gate for future hardware proof, while all
scene load/save authority remains blocked until an operator-run physical readback
artifact exists.
