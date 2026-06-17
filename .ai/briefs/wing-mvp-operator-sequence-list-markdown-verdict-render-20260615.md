# Brief: WING MVP Operator Sequence List Markdown Verdict Render

## Task

Render the WING read-only operator-sequence-list Markdown-surface verdict fields
inside the operator-readable no-WING `summary.md`.

## Context

The no-WING check already records
`wing_readonly_operator_sequence_list_markdown_surface_*` verdicts in
`summary.json`. Operators read `summary.md` before any backend/WING session, so
the Markdown artifact must also show the verdict that the required operator
sequence list lines were rendered.

## Scope

- Seed default Markdown-surface verdict fields before `summary.md` is assembled.
- Render those fields in the WING handoff section of `summary.md`.
- Replace the rendered values after the runtime Markdown-surface check runs.
- Add focused assertions that the operator-readable Markdown contains the
  verdict and denial fields.

## Safety Boundary

This is offline and read-only. It does not contact WING, start a backend,
execute handoff commands, grant live authority, or close readiness.

## Validation

Run focused py_compile, focused WING/no-WING tests, no-WING quick with temporary
artifacts, Project Director marker check, the standard full pytest command,
memory lint, and diff whitespace checks.
