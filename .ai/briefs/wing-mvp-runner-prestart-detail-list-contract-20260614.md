# Brief: WING MVP Runner Prestart Detail List Contract

## Task

Make the real WING read-only preflight runner fail closed when the referenced
no-WING prestart summary carries a stale backend-port failure-detail field list.

## Context

The active WING MVP blocker remains `WING read-only connection proof`. The
offline no-WING release gate records the backend-port failure-detail contract in
`backend_port_free.detail`. The real runner already checks required false fields
before opening WebSocket, but it also needs to reject a stale detail-field list
before accepting the prestart summary.

## Scope

- Validate `backend_port_prestart_check_failure_detail_fields` inside the
  referenced no-WING prestart summary's `backend_port_free.detail`.
- Block before expected-plan validation and before WebSocket on mismatch.
- Keep the operator next step explicit.
- Add focused runner regression coverage and runbook wording.

## Non-goals

- Do not contact WING.
- Do not start, stop, or reconfigure backend/WING listeners.
- Do not grant live authority.
- Do not close the first WING blocker from offline evidence.
