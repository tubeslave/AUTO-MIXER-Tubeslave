# ADR: WING Read-Only Prestart Detail Fail-Closed Condition

## Context

The read-only WING preflight runner now rejects stale expected plans whose
backend-port prestart detail fields omit `listener_confirmation_checklist`.
The offline plan still represented that risk only as a generic expected-plan
mismatch.

## Options considered

1. Keep only `expected_plan_missing_or_mismatch`.
2. Add `stale_backend_port_prestart_detail_contract` to the fail-closed list.
3. Add new runtime networking checks.

## Decision

Use option 2. Offline plans and evidence validation now explicitly list
`stale_backend_port_prestart_detail_contract`.

## Why this won

It makes the operator-facing plan describe the actual prestart risk while
preserving the existing fail-closed behavior. No live authority is widened.

## Rejected alternatives

The generic condition was too vague for the current proof boundary. New runtime
network checks were unnecessary because stale plans are already blocked before
WebSocket contact.

## Implementation plan

- Add the condition to `build_offline_preflight_plan`.
- Add it to `EXPECTED_PLAN_FAIL_CLOSED_CONDITIONS`.
- Update runbook and regression assertions.

## Test plan

- Runner/evidence/no-WING focused tests.
- Adjacent WING/no-WING tests.
- Full repo validation before commit.

## Risks and rollback

The main risk is downstream fixture churn when fail-closed conditions change.
Rollback by removing the condition from runner/evidence constants and related
assertions.
