# Snapshots/Scenes Physical Readback Contract - 2026-06-05

## Context

The project must eventually manage scenes and presets, but scene recall/save can
overwrite live console state. Current snapshots/scenes coverage is cached catalog
visibility only; it lacks physical readback proof, scene content fingerprints,
scoped diffs, and rollback references.

## Options considered

1. Treat cached scene catalog DTOs as sufficient proof.
2. Add live scene query or recall transport support now.
3. Add a read-only physical readback contract and keep scene load/save blocked.

## Decision

Choose option 3. Define a transport-free snapshots/scenes physical readback
contract and surface it through no-WING, Product Layer, frontend status, registry,
fixture evidence, remediation planning, and the console-read capability matrix.

## Why this won

It advances scene automation proof readiness without changing mixer transport or
granting scene recall authority. The contract requires active-scene identity,
snapshot catalog rows, scene content fingerprints, scoped diff markers, rollback
references, physical sources, and false safety flags for scene load/save.

## Rejected alternatives

Cached catalog promotion was rejected because it cannot prove scene contents or
rollback. Live scene transport work was rejected for this slice because it needs
a separate safety proof and operator-run artifact design.

## Implementation plan

- Add the snapshots/scenes contract builder and artifact validator.
- Register the contract as a shadow-only method.
- Include it in no-WING schema validation and summary output.
- Normalize it in Product Layer proof quality and expose it in frontend Module
  Status.
- Update fixture evidence, remediation expected artifacts, console-read goal
  support paths, and tests.

## Test plan

- `PYTHONPATH=backend python3 -m py_compile ...`
- Focused pytest for contract, registry, fixture evidence, remediation plan,
  no-WING, and Product Layer.
- Frontend adapter tests and frontend build.
- no-WING quick check.
- Memory lint.
- Standard backend pytest.

## Risks and rollback

Risk: contract availability could be misread as permission to load/save scenes.
Mitigation: surfaced rows preserve `scene_load_allowed=false`,
`scene_save_allowed=false`, `live_write=false`, and
`physical_fixture_gap_remains=true`.

Rollback: remove the contract method, no-WING validation/summary wiring, Product
Layer/frontend visibility, fixture/remediation references, and related tests.
No live scene transport or write path is introduced by this ADR.
