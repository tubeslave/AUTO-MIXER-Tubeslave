# Brief: WING MVP read-only event log run-id fail-closed contract

## Task

Make the WING read-only preflight plan and handoff packet explicitly list
`event_log_run_id_mismatch` as a fail-closed condition.

## Context

The evidence validator now fails closed when any `events.jsonl` row is missing
the same `run_id` as `summary.json`. The shared plan/handoff fail-closed list
still names only the broader `event_log_missing_or_invalid` condition, so the
operator-facing packet does not explicitly describe the new row-level identity
failure.

## Constraints

- Do not contact WING or backend services.
- Do not add live authority or readiness closure.
- Keep the change on the first MVP blocker: WING read-only connection proof.
- Use the existing shared contract source.

## Acceptance

- `WING_READONLY_PREFLIGHT_PLAN_FAIL_CLOSED_CONDITIONS` includes
  `event_log_run_id_mismatch`.
- Runner, handoff, and no-WING tests prove the condition is visible through the
  shared contract.
- Runbook/report wording explains that a mismatched event-row `run_id` is a
  named fail-closed condition.
