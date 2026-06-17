# Brief: WING MVP Operator Sequence List Markdown Value Gate Self-Surface

## Task

Fail closed when the operator-readable no-WING `summary.md` omits or corrupts
the visible value-gate fields for
`wing_readonly_operator_sequence_list_markdown_surface_required_field_lines`.

## Context

The current no-WING check verifies that the rendered `required_field_lines`
value matches the runtime contract. The value-gate result fields
`expected_value`, `actual_value`, and `match_contract` must also be present and
consistent in the Markdown surface operators inspect before WING access.

## Scope

- Add the value-gate self-surface fields to the required Markdown line
  contract.
- Fail closed when those lines are missing.
- Fail closed when their rendered values do not match the computed verdict.
- Update focused tests and traceability docs.

## Safety Boundary

This is offline and read-only. It does not contact WING, start a backend,
execute handoff commands, grant live authority, or close readiness.

## Validation

Run focused py_compile, focused no-WING/WING tests, no-WING quick with temporary
artifacts, Project Director marker check, the standard full pytest command,
memory lint, and diff whitespace checks.
