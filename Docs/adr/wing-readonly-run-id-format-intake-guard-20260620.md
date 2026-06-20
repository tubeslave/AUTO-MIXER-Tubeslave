# WING Read-Only Run ID Format Intake Guard

## Context

M001 is waiting for a real WING Rack read-only artifact. M095-M101 blocked
reserved roots, declared path-shape mismatches, non-canonical roots, copied
source summaries, path-like run ids, parent traversal in artifact roots and
artifact-root whitespace. A remaining file-intake gap allowed a self-consistent
candidate to declare a manual `run_id` such as `manual_wing_readonly_preflight`
or a whitespace-padded run id while still matching
`artifact_root/<run_id>/summary.json`.

The real WING read-only runner creates capture directories with
`YYYYMMDDTHHMMSSZ_wing_readonly_preflight`. Manual or renamed run ids weaken
the audit trail before an operator can decide whether a candidate is real WING
evidence.

## Options Considered

1. Leave run-id shape to manual review.
2. Reject candidates whose `run_id` is not the runner-generated UTC capture
   format.
3. Accept any single path segment and rely on declared path consistency.

## Decision

Use option 2. Candidate intake now reports
`run_id_not_wing_readonly_capture_format` when `run_id` is not
`YYYYMMDDTHHMMSSZ_wing_readonly_preflight`.

## Why This Won

The first accepted-candidate boundary should identify runner-produced WING
capture bundles, not arbitrary directory names that only happen to be
self-consistent. The format guard preserves the timestamped audit trail and
prevents renamed or hand-built bundles from looking equivalent to a real
runner capture before manual review.

## Rejected Alternatives

Manual-only review was rejected because the automated intake can cheaply catch
the mismatch before a bad candidate looks ready. Accepting any safe path
segment was rejected because M099 already handles path safety; M102 is about
provenance and auditability, not only filesystem traversal.

## Implementation Plan

- Add a canonical WING read-only run-id format check.
- Add focused regression coverage for a self-consistent manual-run-id candidate
  that previously validated as an accepted candidate.
- Keep review-decision multi-candidate fixtures canonical so they continue to
  test explicit source selection rather than invalid run-id rejection.
- Add rail coverage and update state/report surfaces.

## Test Plan

- Focused WING read-only evidence tests.
- Affected review-decision template fixture test.
- M102 rail.
- Full WING MVP rail.
- JSON/JSONL validation and memory lint before commit.

## Risks and Rollback

The main risk is rejecting manually renamed real-run folders. That is intended
for M001: real hardware proof should keep the runner-generated capture id.
Roll back by reverting this ADR, the M102 plan/report/state entries, the test
and the run-id format guard. No live transport or console state is affected.
