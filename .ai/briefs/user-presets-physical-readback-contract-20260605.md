# User Presets Physical Readback Contract

## Task

Add a read-only physical readback artifact contract for the `user_presets`
console read surface.

## Context

The project goal includes full console parameter reading and eventual
management of user presets. Current Product Layer coverage exposes cached user
preset catalog metadata, but physical preset enumeration, preset content,
scoped diff, and rollback proof are still missing.

## Safety Constraints

- Do not recall, store, load, save, or apply presets.
- Do not import WING or dLive transport clients into the contract.
- Do not grant live authority, Full Autonomous readiness, or preset recall/store
  permission.
- Preserve `physical_fixture_gap_remains=true` until an operator-run physical
  readback artifact exists.

## Required Output

- Transport-free backend contract and validator.
- Registry, goal matrix, fixture evidence, remediation, no-WING, Product Layer,
  and frontend visibility.
- Tests proving schema shape, cache/live rejection, read-only registry status,
  no-WING summary integration, Product Layer normalization, and frontend status
  rows.
