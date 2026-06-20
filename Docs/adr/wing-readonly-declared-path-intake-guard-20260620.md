# WING Read-Only Declared Path Intake Guard

## Context

M001 is waiting for a real WING Rack read-only artifact. M095 rejected
candidates from reserved no-WING/control-plane roots. A remaining intake risk
was a direct candidate payload whose `summary_path` or `events_path` pointed
outside the declared `artifact_root/<run_id>/...` shape while still carrying
otherwise plausible read-only fields.

## Options Considered

1. Leave path-shape enforcement to later review-decision gates.
2. Require declared path shape at the first read-only evidence intake boundary.
3. Add a report-only warning without blocking the candidate.

## Decision

Use option 2. The first WING read-only evidence intake now fails candidates
whose summary or event paths do not match the declared artifact root and run id.

## Why This Won

The operator should not have to discover stale or cross-run artifact paths at a
later manual-review step. Failing at intake preserves the real WING handoff and
keeps M001 pointed at hardware proof.

## Rejected Alternatives

Leaving the check to review-decision gates was rejected because it lets a bad
candidate look accepted earlier in the evidence chain. A report-only warning
was rejected because readiness-adjacent artifact intake must fail closed.

## Implementation Plan

- Validate `summary_path` and `events_path` against
  `artifact_root/<run_id>/summary.json|events.jsonl`.
- Add focused tests for mismatched direct candidate payloads.
- Add M096 roadmap, current-state, report and rail coverage.

## Test Plan

- Focused WING read-only evidence tests.
- M096 rail.
- Full WING MVP rail.
- JSON/JSONL validation and memory lint before commit.

## Risks and Rollback

The main risk is rejecting a non-standard real artifact layout. The contract and
handoff already name the canonical layout, so failing closed is correct. Roll
back by reverting this ADR, the M096 plan/report/state entries, the test, and
the path-shape guard. No live transport or console state is affected.
