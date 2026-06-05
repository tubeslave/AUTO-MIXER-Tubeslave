# ADR: read-surface-artifact-review-queue-release-gate-20260605

## Context

Physical readback artifact intake is now split across many read surfaces. Each
surface can report missing artifacts, invalid artifacts, unsafe authority
claims, target mismatch, or valid manual-review candidates. Operators need one
release-evidence view that aggregates this status without changing the safety
meaning of any individual intake.

## Options Considered

1. Keep only per-surface intake blocks.
2. Add an aggregate queue only to the processing registry.
3. Add the aggregate queue to the registry, no-WING release checks, JSON
   summary, and Markdown summary.

## Decision

Use option 3. The read-surface physical readback artifact review queue is a
mandatory no-WING release evidence gate after all per-surface intake gates,
including dLive-state intake. It aggregates operator review candidates, missing
artifact surfaces, invalid surfaces, unsafe authority claims, target drift, and
blockers.

## Why This Won

The queue gives operators a single read-only review status while preserving the
per-surface evidence. It makes missing or unsafe proof visible before any
promotion discussion and avoids treating valid candidate files as accepted
physical fixtures.

## Rejected Alternatives

- Per-surface-only reporting was rejected because it makes release review harder
  and easier to misread.
- Registry-only exposure was rejected because no-WING release artifacts are the
  repeatable gate used before readiness claims.
- Automatic fixture closure from queue candidates was rejected because manual
  review and physical proof acceptance remain separate.

## Implementation Plan

1. Keep the queue transport-free and `shadow_only` in the processing registry.
2. Run the queue validator after dLive-state artifact intake in
   `run_release_checks()`.
3. Publish queue status and safety flags in `summary.json` and `summary.md`.
4. Cover direct validation, release ordering, summary rendering, and unsafe
   authority normalization in tests.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_read_surface_physical_readback_artifact_review_queue.py tests/test_no_wing_release_check.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile tools/no_wing_release_check.py backend/read_surface_physical_readback_artifact_review_queue.py`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks And Rollback

Risk: operators may overread queue candidates as accepted physical fixtures.
Mitigation: the queue and summaries keep `execution_allowed_by_queue=false`,
`fixture_closure_allowed_by_queue=false`, `live_authority_granted=false`, and
`full_autonomous_ready=false`.

Rollback: remove the queue validator from `run_release_checks()` and remove the
queue summary block. The registry and per-surface gates can remain, but release
evidence would again lose aggregate review visibility.
