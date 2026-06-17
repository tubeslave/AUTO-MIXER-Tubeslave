# ADR: WING MVP Operator Sequence List Markdown Value Gate Self-Surface

## Context

The no-WING operator sequence list Markdown-surface check now verifies that the
rendered `required_field_lines` value matches the runtime contract. Operators
also need the value-gate result fields in `summary.md` to be present and
trustworthy.

## Options considered

1. Keep value-gate diagnostics JSON-only.
2. Add a separate value-gate Markdown checker.
3. Extend the existing Markdown-surface check to require and verify its
   value-gate lines.

## Decision

Extend the existing Markdown-surface check.

## Why this won

The existing check already runs after Markdown assembly and only for WING
handoff summaries. Keeping value-gate self-surface validation there avoids a
second failure path and preserves the same safety-denial fields.

## Rejected alternatives

JSON-only diagnostics leave the operator-readable artifact weaker. A separate
checker duplicates the same surface and increases maintenance cost.

## Implementation plan

- Add expected, actual, and match-contract lines to the required Markdown field
  list.
- Extract their rendered values during the check.
- Record `values_match_contract` and `value_mismatches`.
- Fail closed on missing or mismatched value-gate lines.
- Update tests, runbook, current command, and validation report.

## Test plan

- Focused py_compile.
- Focused no-WING/WING pytest targets.
- no-WING quick temporary-artifact inspection.
- Project Director marker check.
- Standard full pytest command.
- Memory lint and diff whitespace check.

## Risks and rollback

This is an offline reporting contract change. If valid summaries fail, remove
the self-surface value checks and associated tests/docs while keeping the
required-lines value gate itself.
