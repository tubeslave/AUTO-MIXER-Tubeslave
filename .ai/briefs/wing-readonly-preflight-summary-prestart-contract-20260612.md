# Brief: WING Read-Only Preflight Summary Prestart Contract

## Task

Make real `wing_readonly_preflight.v1` summary artifacts carry and validate the
same backend-port prestart contract as the offline expected plan.

## Context

The first supervised WING Rack MVP blocker remains `WING read-only connection
proof`. Offline plans, handoff packets, runbook, and release summaries now show
the busy-port prestart contract. The real-run summary artifact did not yet echo
those fields, so artifact intake could accept a summary that did not prove which
prestart gate was required before the backend/WING session.

## Constraints

- Do not contact WING hardware.
- Do not open a backend WebSocket WING session.
- Do not add live-write authority.
- Keep the change to summary metadata and offline validation.

## Acceptance

- `wing_readonly_preflight_runner.py` writes the shared prestart contract fields
  into every real-run summary, including fail-closed summaries before WebSocket.
- Evidence intake rejects summaries whose prestart fields are missing or do not
  match the shared contract.
- Runner and evidence tests cover the new summary contract.
