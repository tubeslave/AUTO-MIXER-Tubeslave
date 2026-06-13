# Brief: WING MVP Backend Listener Confirmation Template

Date: 2026-06-13

## Task

Make the busy backend-port fail-closed path easier and safer for the operator
without clearing the WING read-only blocker from offline evidence.

## Context

The current no-WING gate can accept a manually edited
`wing_backend_port_listener_confirmation.v1` JSON when the backend port already
has the intended Automixer listener. The fail-closed summary exposes listener
PID and command-line diagnostics, but the operator still has to hand-copy the
accepted JSON shape.

## Scope

- Generate a pending confirmation template from the current busy-port listener
  diagnostics.
- Keep the template invalid until the operator edits it.
- Preserve `safe_to_stop_listener_by_gate: false` and
  `live_authority_granted: false`.
- Update docs/tests/report evidence.

## Out Of Scope

- No backend listener start, stop, or reconfiguration.
- No WING hardware contact.
- No OSC/WING writes.
- No readiness closure from the generated template.
