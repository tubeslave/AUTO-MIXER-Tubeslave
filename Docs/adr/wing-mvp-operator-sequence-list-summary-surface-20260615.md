# ADR: WING MVP Operator Sequence List Summary Surface

## Context

The WING read-only handoff rail already requires sequence count, digest, and
contract match fields before supervised proof planning. The flat no-WING summary
did not expose the actual ordered `operator_sequence` and expected
`expected_operator_sequence` lists, so reviewers had to rely on nested detail
output or hashes alone.

## Options considered

1. Keep the sequence lists nested only.
2. Add the actual list only.
3. Add both actual and expected lists to the flat no-WING summary.

## Decision

Add both `operator_sequence` and `expected_operator_sequence` to the flat
no-WING JSON and Markdown handoff summary surfaces.

## Why this won

The paired lists make the review surface self-contained: reviewers can inspect
the actual ordered checklist and compare it with the expected contract while the
existing count/hash/contract gates still fail closed on drift.

## Rejected alternatives

Keeping nested-only output preserves the existing data but leaves the operator
review surface incomplete. Adding only the actual list makes drift harder to
inspect because the expected contract is not visible beside it.

## Implementation plan

- Promote string-only list copies from the handoff artifact detail into the
  flattened handoff artifact summary.
- Emit both lists through top-level JSON and Markdown no-WING summaries.
- Extend tests to assert the two promoted lists and Markdown labels.
- Update blocker, runbook, generated progress source, index, and report docs.

## Test plan

- Focused py_compile on touched Python files.
- Focused no-WING handoff, packet, and Project Director tests.
- no-WING quick summary check with frontend skipped.
- Project Director selected action marker check.
- Standard full pytest command.
- Memory lint and diff whitespace check.

## Risks and rollback

This is an offline summary-surface change. If it causes summary drift or test
failures, rollback by removing the two promoted fields and their tests/docs
while keeping the existing count, digest, and contract checks intact.
