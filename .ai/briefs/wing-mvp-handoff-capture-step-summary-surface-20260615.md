# Brief: WING MVP Handoff Capture Step Summary Surface

## Task

Expose the read-only preflight capture step as explicit no-WING summary fields.

## Context

The handoff packet validator already computes whether backend/WING connection
confirmation occurs before `capture_readonly_preflight`. The operator-facing
no-WING summary did not expose the capture step text, expected index, or match
verdict as flat fields.

## Scope

- Validate the capture step index and text against the shared handoff contract.
- Promote backend-confirmation and capture-step fields to the no-WING summary.
- Render the new capture-step fields in Markdown.
- Add Project Director baseline markers and tests.

## Out Of Scope

- Contacting WING hardware.
- Starting a backend WebSocket.
- Granting live authority.
- Closing readiness from offline evidence.
