# ADR: WING MVP Operator Sequence List Markdown Required Lines Value Gate

## Context

The no-WING operator-readable `summary.md` includes
`wing_readonly_operator_sequence_list_markdown_surface_required_field_lines`.
The existing Markdown-surface check required that line to exist, but did not
verify that its value matched the runtime contract.

## Options considered

1. Keep presence-only validation.
2. Parse the Markdown field as JSON and compare lists.
3. Compare the rendered Markdown value with the runtime `json.dumps` contract.

## Decision

Compare the rendered Markdown value with the runtime serialized required-line
contract and fail the existing Markdown-surface check on mismatch.

## Why this won

The no-WING writer renders the contract with `json.dumps`, so the expected value
is deterministic. Exact comparison catches stale, empty, or manually corrupted
contract lines before a backend/WING session.

## Rejected alternatives

Presence-only validation allows value drift. JSON parsing is more permissive
than this artifact contract needs to be.

## Implementation plan

- Add a Markdown field-value extractor.
- Add expected, actual, and match verdict fields.
- Fail `wing_readonly_operator_sequence_list_markdown_surface` when the values
  differ.
- Add a direct value-drift regression test.
- Update runbook/current-command traceability.

## Test plan

- Focused py_compile.
- Focused no-WING/WING pytest targets.
- no-WING quick temporary-artifact inspection.
- Project Director marker check.
- Standard full pytest command.
- Memory lint and diff whitespace check.

## Risks and rollback

This is an offline reporting contract change. If valid summaries fail, remove
the value comparison fields, helper, regression test, and traceability docs.
