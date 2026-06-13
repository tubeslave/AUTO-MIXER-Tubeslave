# ADR: WING MVP Backend Listener Template Authority Summary Fields

Date: 2026-06-13
Status: accepted

## Context

When the backend-port prestart check finds an existing listener, the no-WING
gate writes a pending backend-listener confirmation template. The template
contains `execution_allowed_by_template: false` and
`readiness_closure_allowed_by_template: false`, but those denial markers were
not first-class fields in the failed summary detail.

## Options considered

1. Keep the markers only inside the generated template file.
2. Surface only live-authority and accepted-template markers.
3. Surface execution and readiness-closure denial fields directly in the failed
   summary.

## Decision

Use option 3. The failed `backend_port_free` summary now includes
`listener_confirmation_template_execution_allowed_by_template: false` and
`listener_confirmation_template_readiness_closure_allowed_by_template: false`.

## Why this won

The operator can verify from `summary.json` or `summary.md` that the generated
template is pending, cannot start or authorize a backend/WING session, and
cannot close readiness.

## Rejected alternatives

Keeping the state only in the generated template makes the summary less useful
for review. Live-authority denial alone is clear but does not explicitly cover
execution or readiness closure.

## Implementation plan

Add detail fields in `tools/no_wing_release_check.py`, render them in Markdown,
cover JSON/Markdown output in tests, update WING read-only docs, and add the
new report to first-blocker evidence.

## Test plan

Run focused no-WING and WING rail tests, Project Director/direction guard,
memory lint, no-WING quick fail-closed validation, and the standard full pytest
command.

## Risks and rollback

If the summary is too verbose, remove only these extra summary fields; keep the
template's internal authority denials and pending-template validation unchanged.
