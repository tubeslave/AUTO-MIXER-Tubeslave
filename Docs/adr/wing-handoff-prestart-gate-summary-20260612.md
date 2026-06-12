# WING Handoff Prestart Gate Summary

## Context

The handoff packet validator exposes whether the operator sequence includes the
supervised prestart gate. The release summary is the artifact operators and
reviewers read, so it should not omit that field.

## Options considered

1. Keep the field only in low-level validation detail.
2. Add the field and marker list to `summary.json` and `summary.md`.

## Decision

Use option 2. The handoff packet artifact summary now carries
`operator_sequence_has_supervised_prestart_gate` and
`required_supervised_prestart_gate_markers`.

## Why this won

It makes the same fail-closed gate visible at the release handoff boundary.

## Rejected alternatives

Keeping the field only in `CheckResult.detail` was rejected because reviewers
usually inspect the release summary, not raw internal check detail.

## Implementation plan

- Extend handoff packet artifact summary.
- Render both fields in Markdown.
- Add summary regression assertions.

## Test plan

- Focused summary test.
- Affected no-WING/WING tests.
- Full pytest suite.
- `git diff --check`.

## Risks and rollback

Risk is limited to additional summary fields. Rollback is safe by removing the
fields and assertions, but that would reduce operator visibility.
