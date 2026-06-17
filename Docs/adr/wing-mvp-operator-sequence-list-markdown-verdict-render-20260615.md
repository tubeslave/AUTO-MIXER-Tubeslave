# ADR: WING MVP Operator Sequence List Markdown Verdict Render

## Context

The no-WING release summary has a conditional check that fails closed when the
operator-readable `summary.md` omits the WING handoff operator sequence list
fields. The check's own verdict fields were available in `summary.json`, but
the Markdown artifact did not render those verdicts.

## Options considered

1. Leave the verdict in JSON only.
2. Add a separate post-write Markdown rewrite.
3. Seed, render, and replace the verdict fields in the existing summary writer.

## Decision

Seed, render, and replace the
`wing_readonly_operator_sequence_list_markdown_surface_*` verdict fields in the
existing no-WING summary writer.

## Why this won

It keeps the operator-readable artifact aligned with `summary.json` without
adding a second write path or changing live behavior.

## Rejected alternatives

JSON-only visibility hides an important fail-closed verdict from the runbook's
human review path. A post-write rewrite is more fragile than the existing final
Markdown replacement pass.

## Implementation plan

- Add a defaults builder for the Markdown-surface verdict fields.
- Seed defaults before the Markdown line list is assembled.
- Render the verdict fields in the WING handoff section.
- Replace those lines after the runtime Markdown-surface check computes the
  final missing-lines verdict.
- Add focused Markdown assertions.

## Test plan

- Focused py_compile.
- Focused no-WING/WING pytest targets.
- no-WING quick temporary-artifact inspection.
- Project Director marker check.
- Standard full pytest command.
- Memory lint and diff whitespace check.

## Risks and rollback

This is an offline reporting change. If valid no-WING summaries regress, remove
the defaults builder, Markdown lines, replacement loop, test assertions, and
traceability docs while keeping the existing Markdown-surface check intact.
