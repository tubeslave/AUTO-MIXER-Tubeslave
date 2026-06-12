# Brief: WING Read-Only Preflight Runbook Prestart Detail Fields

## Task

Keep the WING read-only preflight operator runbook synchronized with the shared
busy-port prestart failure detail contract.

## Context

The first supervised WING Rack MVP blocker remains `WING read-only connection
proof` with status `tooling_ready_waits_real_wing`. The shared contract already
requires the stable busy-port detail fields emitted by the no-WING prestart
gate, but the operator runbook still named only `listener_processes`.

## Constraints

- Do not contact WING hardware.
- Do not open a backend WebSocket WING session.
- Do not stop, kill, or reconfigure local listener processes.
- Do not widen live-write authority.
- Keep the change to documentation and static validation.

## Acceptance

- The runbook names every shared prestart failure detail field.
- The runbook keeps `command_line` as optional diagnostics, not a required
  contract field.
- A test fails if the runbook drifts from the shared contract.
