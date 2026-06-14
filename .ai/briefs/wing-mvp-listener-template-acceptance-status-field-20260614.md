# WING MVP Listener Template Acceptance Status Field

## Context

The no-WING backend-port prestart gate already emits
`listener_confirmation_template_acceptance_status` when it generates a backend
listener confirmation template. The shared
`backend_port_prestart_check_failure_detail_fields` contract does not yet list
that field.

The current first WING MVP blocker remains `WING read-only connection proof`.
If a backend listener exists before the intended WING session, the gate must
fail closed and make the operator-review state explicit.

## Task

Add `listener_confirmation_template_acceptance_status` to the required
backend-port prestart failure-detail field list.

## Constraints

- Do not start or stop the backend listener.
- Do not contact WING or query console state.
- Do not grant live authority, execution authority, or readiness closure.
- Preserve the existing `accepted_by_template` and placeholder-edit gates.
- Keep the first open blocker as `WING read-only connection proof`.

## Test Plan

- Run focused no-WING, WING preflight, Project Director, and rail tests.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Run Project Director and direction guard.
- Run memory lint, `git diff --check`, and the standard full test suite.
