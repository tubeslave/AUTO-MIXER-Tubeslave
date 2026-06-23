# Brief: no-WING Manifest Cross-Surface Summary

## Task

Make the no-WING operator summaries expose the manifest cross-surface digest
self-check fields already required by the Current First Command.

## Context

The no-WING artifact manifest already computes whether `summary.json` and
`summary.md` carry the canonical manifest contract digest. The Current First
Command also requires the operator-facing no-WING summaries to show the digest
match fields and an empty mismatch list before any supervised proof planning.

## Constraints

- Do not contact WING.
- Do not grant live authority.
- Do not change readiness rows.
- Keep the change limited to offline no-WING summary/reporting surfaces.

## Acceptance

- `summary.json` exposes the required manifest cross-surface SHA match fields.
- `summary.md` renders the same fields for operator review.
- Missing Markdown lines fail the summary surface self-check.
- Tests prove the added surface remains offline and fail-closed.
