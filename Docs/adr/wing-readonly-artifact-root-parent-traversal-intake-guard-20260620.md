# WING Read-Only Artifact Root Parent Traversal Intake Guard

## Context

M001 is waiting for a real WING Rack read-only artifact. M095-M099 blocked
reserved roots, declared path-shape mismatches, non-canonical roots, copied
source summaries and path-like run ids. A remaining file-intake gap allowed a
self-consistent candidate to declare `artifact_root` as a path such as
`tmp/container/../artifacts/wing_readonly_preflight`. The string still ended in
the canonical WING read-only root, but it contained parent traversal before the
accepted-candidate boundary. The downstream review contract already reports
this as `artifact_root_contains_parent_traversal`.

## Options Considered

1. Leave artifact-root parent traversal to downstream review.
2. Reject file-intake candidates whose `artifact_root` contains parent
   traversal.
3. Normalize the root path before comparison and accept normalized matches.

## Decision

Use option 2. Candidate intake now reports
`artifact_root_contains_parent_traversal` when the declared WING read-only
artifact root contains `..` as a path segment.

## Why This Won

The first accepted-candidate boundary should not allow a path-escaped artifact
root to look canonical through suffix matching. Rejecting parent traversal at
intake aligns the file evidence gate with the existing downstream artifact-root
contract and keeps the operator review target unambiguous.

## Rejected Alternatives

Manual-only review was rejected because the automated intake can cheaply catch
the path escape before a bad candidate looks ready. Normalizing and accepting
the root was rejected because it can hide that the payload attempted to leave
the declared path shape.

## Implementation Plan

- Add a parent-traversal detector for WING read-only evidence artifact roots.
- Add focused regression coverage for a self-consistent parent-traversal root
  that previously validated as an accepted candidate.
- Add rail coverage and update state/report surfaces.

## Test Plan

- Focused WING read-only evidence tests.
- M100 rail.
- Full WING MVP rail.
- JSON/JSONL validation and memory lint before commit.

## Risks and Rollback

The main risk is rejecting manually copied artifact roots with path traversal.
That is intended for M001: the real WING proof root must be a stable canonical
root, not a normalized escape. Roll back by reverting this ADR, the M100
plan/report/state entries, the test and the artifact-root traversal guard. No live transport or console state is affected.
