# ADR: read-surface-artifact-review-queue-frontend-20260605

## Context

AUTO-MIXER is expanding physical-readback proof coverage before any broader
console-control authority can be considered. The backend release gate now
aggregates per-surface artifact-intake results into
`read_surface_physical_readback_artifact_review_queue`. The cockpit should make
that aggregate operator work visible without changing the system's live-control
boundary.

## Options Considered

1. Keep the queue only in no-WING release summaries.
2. Add queue tags only to the existing supervised proof evidence row.
3. Add queue tags plus a dedicated read-only module-status row.

## Decision

Use option 3. The frontend adapter exposes the queue as a dedicated
`read_surface_physical_readback_artifact_review_queue` row and also adds
`artifact_queue_*` tags to the supervised proof evidence row.

## Why This Won

A dedicated row gives operators and reviewers one place to find cross-surface
artifact review work: missing surfaces, valid review candidates, invalid
candidates, unsafe authority claims, target inconsistencies, and blockers. The
proof-evidence tags preserve filterable summary data for the existing cockpit
view.

## Rejected Alternatives

- Backend-only visibility was rejected because cockpit users would need to open
  generated release artifacts to understand the proof queue.
- Tags-only visibility was rejected because a cross-surface operator queue is a
  first-class review surface, not just a scalar summary.
- Any automatic fixture closure was rejected because accepting physical proof
  remains a separate manual review decision.

## Implementation Plan

1. Parse `read_surface_physical_readback_artifact_review_queue` from proof
   quality summaries.
2. Add queue status/count/safety tags to supervised proof evidence.
3. Append a read-only module-status row after per-surface artifact intake rows.
4. Preserve missing-queue fallback behavior for stale or incomplete proof
   summaries.
5. Cover present and missing payloads in frontend contract tests.

## Test Plan

- `node --test frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend test`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_no_wing_release_check.py tests/test_read_surface_physical_readback_artifact_review_queue.py -q`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks And Rollback

Risk: operators may treat queue review candidates as accepted physical fixtures.
Mitigation: the row and tags explicitly keep `fixture_closure_allowed=false`,
`live_authority=false`, and `full_autonomous_ready=false`.

Rollback: remove the queue row and `artifact_queue_*` tags from
`operatorAdapters.ts`, then remove the related contract expectations. Backend
release summaries and Product Layer queue data can remain unchanged.
