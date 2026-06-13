# Brief: WING handoff backend connection confirmation

## Context

The WING read-only handoff packet already declares
`physical_preflight_requires_existing_backend_connection: true`, but the
operator sequence must also make the intended backend/WING connection check a
dedicated step before capture starts.

## Task

Require the no-WING handoff packet and summary to expose a backend connection
confirmation step before `capture_readonly_preflight`.

## Acceptance

- The shared contract defines the backend connection confirmation step and its
  required markers.
- Handoff generation uses the shared step.
- no-WING validation rejects handoff packets missing the step or placing it
  after capture.
- Project Director, WING rails, runbook, and MVP docs surface the new guard.
- No backend, WING, OSC, or live-write behavior is added.
