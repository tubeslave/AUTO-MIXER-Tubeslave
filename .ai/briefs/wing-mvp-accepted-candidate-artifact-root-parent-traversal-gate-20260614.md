# Brief: WING MVP Accepted Candidate Artifact Root Parent Traversal Gate

Date: 2026-06-14

## Objective

Keep the first WING MVP blocker moving by tightening the accepted-candidate
artifact-root gate before any physical supervised proof planning.

## Context

The accepted-candidate artifact-root helper already rejects summary, event, and
expected artifact paths that plainly leave the declared root. It still used a
simple prefix check, so a path such as
`artifacts/wing_readonly_preflight/../stale_preflight/run/summary.json` could
look root-scoped before normalization.

## Safety Boundary

This is offline/read-only validation only. It must not contact WING, mutate a
console, close readiness rows, or grant live authority.

## Acceptance

- Accepted candidate artifact paths containing parent traversal fail closed.
- The supervised proof plan stays at `manual_review_ready` with zero physical
  proof steps when an accepted candidate escapes its artifact root through
  `..`.
- The no-WING next-proof-plan validator rejects stale plans that miss the new
  artifact-root violations.
- Tests and release checks pass.
