# ADR: WING MVP Backend Listener Template Pending Summary Fields

Date: 2026-06-13
Status: accepted

## Context

When the backend-port prestart check finds an existing listener, the no-WING
gate writes a pending backend-listener confirmation template. The template
contains `accepted_by_template: false` and `REPLACE_WITH_*` operator fields, but
those markers were not first-class fields in the failed summary detail.

## Options considered

1. Keep the markers only inside the generated template file.
2. Surface only the template path and required edits.
3. Surface accepted-by-template and operator placeholder fields directly in the
   failed summary.

## Decision

Use option 3. The failed `backend_port_free` summary now includes
`listener_confirmation_template_accepted_by_template: false`, placeholder
operator field values, and `listener_confirmation_template_operator_placeholder_fields`.

## Why this won

The operator can verify from `summary.json` or `summary.md` that the generated
template is pending and cannot clear the preflight gate before opening the
template file.

## Rejected alternatives

Keeping the state only in the generated template makes the summary less useful
for review. Required-edits text alone is clear but does not expose the actual
pending marker values.

## Implementation plan

Add detail fields in `tools/no_wing_release_check.py`, render them in Markdown,
cover JSON/Markdown output in tests, and update the WING read-only runbook.

## Test plan

Run focused no-WING and WING rail tests, Project Director/direction guard,
memory lint, no-WING quick fail-closed validation, and the standard full pytest
command.

## Risks and rollback

If the summary is too verbose, remove only these extra summary fields; keep the
template validation and pending-template acceptance gate unchanged.
