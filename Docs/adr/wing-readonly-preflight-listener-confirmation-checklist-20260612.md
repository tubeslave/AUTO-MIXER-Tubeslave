# ADR: WING Read-Only Preflight Listener Confirmation Checklist

## Context

The no-WING prestart gate blocks real WING preflight when the backend port
already has a listener. The artifact includes process diagnostics and a next
step, but the manual confirmation procedure was not a structured field in the
shared WING preflight contract.

## Options considered

1. Keep only `operator_next_step`.
2. Add `listener_confirmation_checklist` as a shared failure-detail field.
3. Let the gate stop or reconfigure unexpected listeners automatically.

## Decision

Use option 2. Failed `backend_port_free` results now include
`listener_confirmation_checklist`, and the shared contract plus runbook require
reviewing it before changing local services.

## Why this won

It improves operator-visible evidence for the first WING MVP blocker without
opening a backend session, contacting WING, or changing local processes. The
gate remains fail-closed and diagnostic only.

## Rejected alternatives

Keeping only prose was rejected because it is harder to test and easier to miss
in generated summaries. Automatic process changes were rejected because they
would violate the no-WING prestart boundary.

## Implementation plan

- Extend `WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS`.
- Emit the checklist from `check_port_free`.
- Render the checklist in failed backend-port Markdown details.
- Update runbook and regression tests.

## Test plan

- Focused backend-port/listener/no-WING tests.
- Real local no-WING quick run against the current busy-port environment.
- Full repo validation before commit.

## Risks and rollback

The main risk is noisy summaries. Roll back by shortening checklist item text
while preserving the structured field and fail-closed behavior.
