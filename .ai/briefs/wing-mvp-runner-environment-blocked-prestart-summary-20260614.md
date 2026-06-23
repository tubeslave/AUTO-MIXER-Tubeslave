# Brief: WING MVP runner environment-blocked prestart summary

## Task

Make the WING read-only preflight runner preserve no-WING
`environment_blocked_checks` details when a prestart `summary.json` is blocked
by the frontend dev-server bind environment.

## Context

The no-WING quick gate now reports frontend e2e sandbox bind denial as
`environment_blocked` while keeping the overall summary failed. The read-only
runner already blocks any prestart summary whose status is not `passed`, but it
only exposed a generic invalid-prestart next step.

## Constraints

- Do not open the backend/WING WebSocket for an environment-blocked prestart
  summary.
- Do not grant readiness, execution authority, or live authority.
- Preserve the underlying no-WING environment-blocked details for operator
  diagnosis.
- Keep the change local to the read-only runner, tests, and evidence docs.

## Desired outcome

An environment-blocked no-WING prestart summary blocks before WebSocket, records
`prestart_summary_environment_blocked`, preserves
`environment_blocked_check_details`, and tells the operator to rerun no-WING in
an environment that can bind `127.0.0.1:3000`.
