# WING Read-Only Run ID Path Segment Intake Guard

## Context

M001 is waiting for a real WING Rack read-only artifact. M095-M098 blocked
reserved roots, declared path-shape mismatches, non-canonical WING artifact
roots and copied source-summary files. A remaining intake gap allowed a
self-consistent candidate to declare `run_id` as a path such as
`../stale_wing_readonly_preflight`, making
`artifact_root/<run_id>/summary.json` appear string-consistent while resolving
outside the intended canonical run directory.

## Options Considered

1. Leave `run_id` path sanity to later manual review.
2. Reject candidates whose `run_id` is not a single safe path segment.
3. Normalize paths before comparison and accept normalized matches.

## Decision

Use option 2. Candidate intake now reports
`run_id_not_single_safe_path_segment` when `run_id` is `.`, `..`, or contains
`/` or `\`.

## Why This Won

The first accepted-candidate boundary should not allow path traversal inside
the run identifier. Requiring one safe segment preserves the runner's normal
`<timestamp>_wing_readonly_preflight` shape and keeps artifact-root binding
simple enough for operator review.

## Rejected Alternatives

Manual-only review was rejected because the automated intake can cheaply catch
the path escape before a bad candidate looks ready. Normalizing and accepting
paths was rejected because it can hide that the payload attempted to leave the
declared run directory.

## Implementation Plan

- Add a single-segment `run_id` guard to WING read-only evidence intake.
- Add focused regression coverage for a parent-traversal `run_id` candidate
  that previously validated as an accepted candidate.
- Add rail coverage and update state/report surfaces.

## Test Plan

- Focused WING read-only evidence tests.
- M099 rail.
- Full WING MVP rail.
- JSON/JSONL validation and memory lint before commit.

## Risks and Rollback

The main risk is rejecting manually named run ids that use path separators. That
is intended for M001: the real WING proof run id must name one directory under
the canonical artifact root. Roll back by reverting this ADR, the M099
plan/report/state entries, the test and the run-id path-segment guard. No live transport or console state is affected.
