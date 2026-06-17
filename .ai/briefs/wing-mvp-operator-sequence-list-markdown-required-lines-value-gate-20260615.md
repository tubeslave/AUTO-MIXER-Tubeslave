# Brief: WING MVP Operator Sequence List Markdown Required Lines Value Gate

## Task

Fail closed when the operator-readable no-WING `summary.md` renders
`wing_readonly_operator_sequence_list_markdown_surface_required_field_lines`
with a value that does not match the runtime contract.

## Context

The Markdown-surface check already requires the `required_field_lines` line to
exist. A drifted value such as `[]` could still pass because the check only
looked for line presence.

## Scope

- Extract the rendered Markdown field value.
- Compare it with the JSON-serialized runtime required-line contract.
- Record expected, actual, and match verdict fields.
- Fail the no-WING check when the values differ.
- Keep readiness closure and live authority denied.

## Safety Boundary

This is offline and read-only. It does not contact WING, start a backend,
execute handoff commands, grant live authority, or close readiness.

## Validation

Run focused py_compile, focused no-WING/WING tests, no-WING quick with temporary
artifacts, Project Director marker check, the standard full pytest command,
memory lint, and diff whitespace checks.
