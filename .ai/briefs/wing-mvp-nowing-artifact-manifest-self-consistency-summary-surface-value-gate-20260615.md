# Brief: WING MVP no-WING Artifact Manifest Self-Consistency Summary Surface Value Gate

## Task

Make the no-WING summary surfaces fail closed when the manifest
self-consistency verdict fields are present but carry stale or unsafe values.

## Context

The current MVP blocker remains `WING read-only connection proof`. The no-WING
`summary.json` and `summary.md` now render the manifest self-consistency verdict,
but a presence-only check is not enough for an operator handoff surface. The
summary must also reject edited or stale values that imply readiness or live
authority.

## Scope

- Add a summary-surface value check for manifest self-consistency fields.
- Surface JSON, Markdown, and combined mismatch lists.
- Fail closed when expected self-consistency values are missing, stale, or unsafe.
- Keep readiness and live authority denied.

## Non-Goals

- Do not contact WING.
- Do not query or mutate mixer state.
- Do not close readiness.
- Do not grant live authority.

## Validation

- Targeted no-WING manifest summary value-gate tests.
- no-WING quick artifact inspection.
- Focused WING/no-WING/Director tests.
- Full standard pytest before merge.
