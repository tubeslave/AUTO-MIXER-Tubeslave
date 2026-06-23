# Brief: WING MVP no-WING Artifact Manifest Self-Consistency Summary Surface Sidecar Binding

## Task

Make `artifact-manifest.json` carry the same manifest self-consistency
summary-surface value-gate verdict that `summary.json` and `summary.md` carry.

## Context

The current MVP blocker remains `WING read-only connection proof`. The no-WING
summary surfaces now fail closed when manifest self-consistency values drift,
but the detached manifest sidecar should also record that summary-surface verdict
because it binds the summary artifact hashes for operator review.

## Scope

- Recompute the manifest self-consistency summary-surface value gate in the
  detached manifest writer.
- Write the JSON, Markdown, and combined mismatch lists into
  `artifact-manifest.json`.
- Keep readiness and live authority denied.

## Non-Goals

- Do not contact WING.
- Do not query or mutate mixer state.
- Do not close readiness.
- Do not grant live authority.

## Validation

- Targeted no-WING manifest tests.
- no-WING quick artifact inspection.
- Focused WING/no-WING/Director tests.
- Full standard pytest before merge.
