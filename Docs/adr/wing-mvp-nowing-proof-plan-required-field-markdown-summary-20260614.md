# ADR: WING MVP no-WING Proof Plan Required Field Markdown Summary

## Context

The no-WING `next_wing_proof_plan` summary already exposes the required and
missing supervised proof-plan summary field lists in `summary.json`. Those lists
come from the handoff-required contract and explain why stale proof plans fail
closed before supervised WING proof work. The operator-readable markdown summary
did not render them.

## Options considered

1. Keep the lists JSON-only.
2. Add a separate markdown section for required summary fields.
3. Render the lists inside `## Next WING Proof Plan`.

## Decision

Choose option 3.

`summary.md` now renders:

- `required_supervised_proof_plan_summary_fields`
- `missing_required_supervised_proof_plan_summary_fields`

inside the existing next WING proof-plan section.

## Why this won

The fields are part of the next-plan contract and should be visible next to the
other plan blockers, violations, and read-only preflight diagnostics. This keeps
operator handoff diagnosis available without requiring a JSON inspection step.

## Rejected alternatives

- JSON-only detail hides a fail-closed reason from the markdown handoff surface.
- A separate section would split the proof-plan contract across the report.

## Implementation plan

- Add both fields to the markdown line list for `next_wing_proof_plan`.
- Extend the success-path summary test to assert the empty missing list is rendered.
- Extend the missing-required-field regression to write `summary.md` and assert
  the concrete missing field appears.

## Test plan

- Focused no-WING summary tests.
- Focused WING handoff/no-WING rail tests.
- Whitespace and memory-lint checks.

## Risks and rollback

Risk: markdown output becomes more verbose.

Rollback: remove the two markdown lines and their assertions. JSON behavior and
fail-closed schema validation would remain unchanged, but the operator markdown
would again be less diagnostic.
