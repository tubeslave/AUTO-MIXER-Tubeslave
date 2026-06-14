# WING MVP Listener Template Path Failure Field

## Context

The no-WING backend-port prestart gate already writes
`listener_confirmation_template_path` when it generates the backend-listener
confirmation template. The WING rails separately require the failed no-WING
summary to expose that path, because the operator needs it for the rerun with
`--backend-listener-confirmation`.

The shared `backend_port_prestart_check_failure_detail_fields` contract did not
yet include the path field.

## Task

Add `listener_confirmation_template_path` to the required backend-port prestart
failure-detail field list.

## Constraints

- Do not start, stop, or reconfigure the backend listener.
- Do not contact WING or query console state.
- Do not grant execution, live authority, or readiness closure.
- Keep the first open blocker as `WING read-only connection proof`.

## Test Plan

- Run focused no-WING, WING preflight, Project Director, and rail tests.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Run Project Director and direction guard.
- Run memory lint, `git diff --check`, and the standard full test suite.
