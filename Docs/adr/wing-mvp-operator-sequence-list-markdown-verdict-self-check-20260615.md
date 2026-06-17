# ADR: WING MVP Operator Sequence List Markdown Verdict Self-Check

## Context

The no-WING operator-readable `summary.md` renders WING handoff operator
sequence list fields and the
`wing_readonly_operator_sequence_list_markdown_surface_*` verdict fields. The
runtime check must fail closed if those verdict lines disappear from Markdown.

## Options considered

1. Rely on tests that inspect current Markdown output.
2. Add a separate verdict-surface checker.
3. Extend the existing Markdown-surface required-line contract.

## Decision

Extend `WING_READONLY_OPERATOR_SEQUENCE_LIST_MARKDOWN_SURFACE_REQUIRED_FIELD_LINES`
to include the Markdown-surface verdict fields.

## Why this won

The existing check already runs at the correct time: after Markdown assembly and
only when a WING handoff artifact is present. Extending its contract gives
runtime fail-closed behavior without another reporting path.

## Rejected alternatives

String assertions alone do not protect real no-WING runs. A separate checker
would duplicate scope and create another set of denial fields to keep aligned.

## Implementation plan

- Add the verdict fields to the required Markdown line tuple.
- Add a regression test that removes one verdict line.
- Keep summary failure routed through
  `wing_readonly_operator_sequence_list_markdown_surface`.
- Update traceability docs and validation report.

## Test plan

- Focused py_compile.
- Focused no-WING/WING pytest targets.
- no-WING quick temporary-artifact inspection.
- Project Director marker check.
- Standard full pytest command.
- Memory lint and diff whitespace check.

## Risks and rollback

This is an offline reporting contract change. If valid no-WING summaries fail,
rollback by removing the verdict fields from the required-line tuple and
removing the regression test and traceability docs.
