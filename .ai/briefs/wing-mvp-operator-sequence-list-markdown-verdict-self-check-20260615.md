# Brief: WING MVP Operator Sequence List Markdown Verdict Self-Check

## Task

Make the no-WING operator-sequence-list Markdown-surface check fail closed when
its own verdict lines disappear from the operator-readable `summary.md`.

## Context

The current WING read-only proof rail requires operators to inspect `summary.md`
before opening a backend/WING session. The previous increment rendered the
`wing_readonly_operator_sequence_list_markdown_surface_*` verdict fields, but
the runtime check still only required the underlying operator sequence list and
summary-surface fields.

## Scope

- Add the Markdown-surface verdict fields to the required Markdown line
  contract.
- Add a direct regression test for a missing verdict line.
- Keep readiness closure and live authority denied.
- Update traceability docs.

## Safety Boundary

This is offline and read-only. It does not contact WING, start a backend,
execute handoff commands, grant live authority, or close readiness.

## Validation

Run focused py_compile, focused no-WING/WING tests, no-WING quick with temporary
artifacts, Project Director marker check, the standard full pytest command,
memory lint, and diff whitespace checks.
