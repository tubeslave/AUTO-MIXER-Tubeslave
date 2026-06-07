# Brief: wing-readonly-preflight-transport-timeout-artifact-20260607

## Goal

Strengthen the first supervised WING MVP blocker, `WING read-only connection
proof`, by making read-only preflight transport timeout or connection-loss
failures produce explicit fail-closed evidence instead of a generic transport
error.

## Context

The real next step remains the hardware run in
`Docs/runbooks/WING_READONLY_PREFLIGHT.md`. This slice does not contact WING and
does not close any blocker. It prepares the operator proof trail so a timeout
during the real read-only preflight creates a reviewable blocked artifact.

## Scope

- Keep the runner read-only.
- Keep no-WING release validation offline.
- Do not open a real backend, WING, OSC, or supervised apply session.
- Do not grant readiness or live authority from timeout evidence.

## Acceptance

- The offline preflight plan lists `transport_timeout_or_connection_loss` as a
  fail-closed condition.
- The no-WING release gate requires that condition.
- A runner timeout writes `summary.json` with `status: blocked`,
  `fail_closed: true`, and no write/proposal request flags.
- Focused tests prove the timeout artifact path.
