# Brief: WING MVP Accepted Candidate Artifact Root Canonical Root Gate

Date: 2026-06-14

## Objective

Keep the WING read-only connection proof rail fail-closed by rejecting accepted
candidate metadata whose declared artifact root contains parent traversal.

## Context

The accepted-candidate artifact-root helper now rejects summary, event, and
expected artifact paths that escape a declared root. A separate gap remains
when the root itself contains `..`, because normalization can widen the trusted
root even if all child paths look valid.

## Safety Boundary

This is offline/read-only validation. It must not contact WING, mutate console
state, close readiness rows, or grant live authority.

## Acceptance

- Accepted candidate `artifact_root` values containing `..` fail the shared
  artifact-root contract.
- Supervised proof planning stays blocked when a reviewed candidate root
  contains parent traversal.
- The no-WING next-proof-plan validator recomputes and rejects stale summaries
  that miss the root traversal violation.
- Validation evidence is recorded in `Docs/reports/`.
