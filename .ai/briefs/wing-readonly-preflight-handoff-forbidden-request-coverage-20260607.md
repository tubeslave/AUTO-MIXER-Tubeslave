# Brief: wing-readonly-preflight-handoff-forbidden-request-coverage-20260607

## Goal

Strengthen the first supervised WING MVP blocker, `WING read-only connection
proof`, by making the offline operator handoff packet list the full forbidden
live request surface for read-only preflight and by making the offline plan
explicitly fail closed when the event log does not prove WING mode.

## Context

The real next step is still the hardware run in
`Docs/runbooks/WING_READONLY_PREFLIGHT.md`. No code-only or report-only work can
close the blocker, but the operator handoff should be explicit about every live
write/proposal/rollback/connect request type that must not appear during the
read-only proof.

## Scope

- Keep the handoff packet offline and transport-free.
- Keep no-WING release validation offline.
- Do not run WING, OSC, WebSocket, or supervised apply.
- Do not close any MVP blocker row.

## Acceptance

- The handoff packet includes all forbidden live write/proposal/rollback/connect
  request types relevant to the read-only preflight.
- The no-WING handoff artifact gate fails if any required forbidden request type
  is missing.
- The plan-only artifact includes `event_log_wing_mode_not_proven` as a
  fail-closed condition.
- Focused tests prove the pass and fail-closed paths.
