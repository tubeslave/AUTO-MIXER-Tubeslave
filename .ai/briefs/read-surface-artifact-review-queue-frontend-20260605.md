# Brief: read-surface-artifact-review-queue-frontend-20260605

## Context

The backend/no-WING release gate now aggregates all physical readback
artifact-intake surfaces into
`read_surface_physical_readback_artifact_review_queue`. The Product Layer proof
quality summary can expose the queue, but the cockpit module-status adapter
still needs to present it as a separate operator review surface.

## Goal

Expose the aggregate artifact review queue in the frontend module-status data
model so operators can see missing artifacts, review candidates, invalid
candidates, unsafe authority claims, and target-inconsistent surfaces without
misreading the queue as fixture closure or live-console authority.

## Safety Constraints

- Read-only display only.
- Do not add WebSocket commands, mixer transports, OSC/MIDI/TCP sends, recall,
  store, apply, or fixture-closure behavior.
- Keep `canWriteLive=false`, no `liveWriteKinds`, and explicit false tags for
  queue fixture closure, live authority, and full autonomous readiness.
- Preserve missing-queue fallback rows so stale proof summaries do not hide the
  proof gap.

## Acceptance Checks

- `adaptModuleStatusRecords()` includes a
  `read_surface_physical_readback_artifact_review_queue` row after individual
  artifact-intake rows.
- The supervised proof evidence row includes `artifact_queue_*` tags.
- Contract tests cover present and missing queue payloads.
- Frontend tests and build pass.
