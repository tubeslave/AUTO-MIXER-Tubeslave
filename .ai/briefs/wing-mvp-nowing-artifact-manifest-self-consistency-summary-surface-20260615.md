# Brief: WING MVP no-WING Artifact Manifest Self-Consistency Summary Surface

## Task

Expose the no-WING `artifact-manifest.json` self-consistency verdict in the
operator-facing `summary.json` and `summary.md` surfaces.

## Context

The current MVP blocker remains `WING read-only connection proof`. The detached
manifest now proves its own required artifact IDs, paths, SHA fields, and
safety-denial flags, but the no-WING handoff summary still requires an operator
to open the sidecar to see that verdict.

## Scope

- Add manifest self-consistency fields to the required no-WING summary surface.
- Render the same fields in `summary.md`.
- Keep the detached `artifact-manifest.json` as the SHA-bound source artifact.
- Keep all readiness and live-authority flags denied.

## Non-Goals

- Do not contact WING.
- Do not query or mutate mixer state.
- Do not close readiness.
- Do not grant live authority.

## Validation

- Targeted no-WING manifest summary surface tests.
- no-WING quick artifact inspection.
- Focused WING/no-WING/Director tests.
- Full standard pytest before merge.
