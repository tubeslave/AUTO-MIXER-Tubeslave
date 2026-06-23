# ADR: WING MVP Operator Sequence List Summary Surface Self-Check

## Context

The no-WING handoff summary exposes top-level `operator_sequence` and
`expected_operator_sequence` lists for operator review. Existing checks prove the
handoff packet sequence count and digest, but the promoted flat fields also need
a direct consistency verdict so they cannot silently drift from the handoff
artifact or contract.

## Options considered

1. Rely on existing count/hash/contract checks only.
2. Add a report-only consistency field.
3. Add a fail-closed self-check for the promoted list surface.

## Decision

Add a fail-closed self-check for the promoted top-level operator sequence lists.

## Why this won

The self-check keeps the operator-facing summary review surface trustworthy
without widening authority. If either list is missing, malformed, mismatched
against the handoff artifact, mismatched against the shared contract, or
inconsistent with its count field, the no-WING summary fails.

## Rejected alternatives

Existing count/hash checks remain necessary, but they do not directly prove that
the newly promoted top-level fields are present and aligned. Report-only fields
would surface drift but would not prevent accidental readiness use.

## Implementation plan

- Add `wing_readonly_operator_sequence_list_summary_surface` check metadata.
- Compare top-level lists to the nested handoff artifact and expected contract.
- Emit required fields, mismatches, match verdict, and readiness/live denial
  flags.
- Mark the no-WING summary failed when the match verdict is false.
- Render the self-check into `summary.md`.
- Update tests, blocker docs, runbook docs, generated-progress source, and
  traceability docs.

## Test plan

- Focused py_compile.
- Focused no-WING handoff tests.
- Direct mismatch regression test.
- no-WING quick summary check.
- Project Director current-command marker check.
- Standard full pytest command.
- Memory lint and diff whitespace check.

## Risks and rollback

This is an offline summary validation change. If it blocks a valid no-WING run,
rollback by removing the self-check function, call site, Markdown lines, and
tests while keeping the promoted list fields and existing count/hash checks.
