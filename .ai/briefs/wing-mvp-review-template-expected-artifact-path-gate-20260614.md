# Brief: WING MVP Review Template Expected Artifact Path Gate

## Task ID

wing-mvp-review-template-expected-artifact-path-gate-20260614

## Context

Evidence intake records whether `summary_path` and `events_path` match the
expected artifact paths, but the review decision template builder also accepts
already-built report mappings. A hand-built or stale report object could carry
unbound summary/events paths into pending decision template generation.

## Goal

Make the review decision template fail closed unless summary and event artifact
paths are bound to the expected artifact paths.

## Safety Boundary

This is offline/read-only review tooling. It must not contact WING, grant live
authority, mutate readiness, or widen supervised write permissions.

## Test Plan

- Focused WING evidence/review/no-WING tests.
- Standard repository pytest before any commit attempt.
