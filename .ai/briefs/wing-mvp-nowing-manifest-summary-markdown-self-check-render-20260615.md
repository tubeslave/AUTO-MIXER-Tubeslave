# Brief: no-WING Manifest Summary Markdown Self-Check Render

## Task ID

wing-mvp-nowing-manifest-summary-markdown-self-check-render-20260615

## Context

The active supervised WING Rack MVP blocker is still `WING read-only connection
proof`. The no-WING release summary is an offline operator gate before any real
read-only WING session, so its JSON and Markdown surfaces must not overstate
their own completeness.

The manifest summary surface check already recomputed missing Markdown field
lines, but the final `summary.md` writer kept rendering the
`artifact_manifest_summary_surface_missing_field_lines` and
`artifact_manifest_summary_surface_field_lines_rendered` rows from static
success values.

## Goal

Make the operator-readable `summary.md` render the computed manifest summary
surface line-check result after validation.

## Safety boundary

Offline no-WING reporting only. Do not contact WING, query a console, mutate
mixer state, close readiness, or grant live authority.

## Expected validation

- Focused manifest summary surface tests.
- no-WING quick.
- Project Director guard.
- Standard pytest before commit.
