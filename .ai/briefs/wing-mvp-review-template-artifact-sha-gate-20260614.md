# Brief: WING MVP Review Template Artifact SHA Gate

## Task ID

wing-mvp-review-template-artifact-sha-gate-20260614

## Context

Evidence intake and the review-decision ledger reject malformed artifact hashes,
but the review decision template builder also accepts already-built report
mappings. A hand-built or stale report object could carry malformed
`summary_sha256` or `events_sha256` values into pending decision template
generation.

## Goal

Make the review decision template fail closed unless artifact digests are valid
SHA-256 hex strings.

## Safety Boundary

This is offline/read-only review tooling. It must not contact WING, grant live
authority, mutate readiness, or widen supervised write permissions.

## Test Plan

- Focused WING evidence/review/no-WING tests.
- Standard repository pytest before any commit attempt.
