# Brief: WING read-only review next action promotion

## Context

Blocked WING read-only preflight summaries may include `operator_next_step`.
The review report preserved that field per artifact row, but the report-level
`next_required_action` could still remain generic.

## Task

Promote artifact-specific `operator_next_step` guidance into the review
report's primary next action when no clean candidate is available.

## Acceptance

- Blocked review reports expose artifact next steps at top level.
- Markdown renders the promoted next-step list.
- Clean candidates still require manual review and do not grant live authority.
- No WING/backend/OSC contact is added.
