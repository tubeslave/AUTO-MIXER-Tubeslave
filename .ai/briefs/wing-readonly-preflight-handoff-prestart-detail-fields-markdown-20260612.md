# Brief: WING Read-Only Preflight Handoff Prestart Detail Fields Markdown

## Task

Make the generated markdown handoff packet show the same prestart busy-port
failure detail fields that are already present in the JSON packet and shared
contract.

## Context

The WING read-only connection proof remains the first supervised WING Rack MVP
blocker. The JSON handoff packet includes
`backend_port_prestart_check_failure_detail_fields`, but the markdown operator
packet only rendered the prestart command.

## Constraints

- Do not contact WING hardware.
- Do not open a backend WebSocket WING session.
- Do not change runner request behavior.
- Do not widen live-write authority.
- Keep the change to offline handoff rendering and tests.

## Acceptance

- Markdown handoff output renders
  `backend_port_prestart_check_failure_detail_fields`.
- The handoff markdown test asserts every shared failure detail field appears.
- Existing JSON packet contract remains unchanged.
