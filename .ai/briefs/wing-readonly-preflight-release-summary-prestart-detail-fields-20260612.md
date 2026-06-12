# Brief: WING Read-Only Preflight Release Summary Prestart Detail Fields

## Task

Render the WING preflight prestart busy-port failure detail fields in the
no-WING release markdown summary wherever the JSON summary already carries them.

## Context

The first supervised WING Rack MVP blocker remains `WING read-only connection
proof`. Recent slices aligned the shared contract, runbook, and handoff markdown
around busy-port detail fields. The no-WING release markdown summary still hid
those fields in the WING preflight evidence, plan artifact, and handoff packet
artifact sections.

## Constraints

- Do not contact WING hardware.
- Do not open a backend WebSocket WING session.
- Do not change JSON contract semantics.
- Do not widen live-write authority.
- Keep the change to markdown rendering and tests.

## Acceptance

- Markdown summary renders the prestart failure detail field list for WING
  evidence, plan artifact, and handoff packet artifact.
- Existing JSON summary fields remain unchanged.
- Tests assert the rendered markdown includes the field list and shared field
  names.
