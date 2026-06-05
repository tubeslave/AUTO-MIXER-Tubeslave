# Brief: read-surface-fixture-closure-plan-frontend-20260605

## Context

The backend now exposes
`read_surface_physical_readback_fixture_closure_plan` as a no-WING, read-only
manual fixture-baseline update plan. The cockpit module-status adapter already
surfaces artifact intake and the aggregate review queue, but it must also show
the closure plan so operators can see accepted-record progress without
mistaking it for completed fixture closure.

## Goal

Expose the fixture closure plan in the frontend module-status data model as a
dedicated read-only row and compact supervised proof tags.

## Safety Constraints

- Read-only display only.
- Do not add WebSocket commands, mixer transports, fixture writes, baseline
  updates, OSC/MIDI/TCP sends, recall/store/apply, or live authority.
- Preserve `canWriteLive=false`, empty `liveWriteKinds`,
  `fixture_update_allowed=false`, `fixture_closure_allowed=false`,
  `physical_gap_remains=true`, `live_authority=false`, and
  `full_autonomous_ready=false`.
- Preserve missing-plan fallback rows so stale proof summaries do not hide the
  manual fixture-closure gate.

## Acceptance Checks

- `adaptModuleStatusRecords()` appends a
  `read_surface_physical_readback_fixture_closure_plan` row after the artifact
  review queue.
- The supervised proof evidence row includes `fixture_closure_plan_*` tags.
- Contract tests cover present and missing closure-plan payloads.
- Frontend tests and build pass.
