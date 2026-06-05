# Codex Proposal: snapshots-scenes-physical-readback-artifact-intake-20260605

## Short thesis

Expose snapshots/scenes physical readback artifacts as read-only manual-review
candidates, and make invalid or authority-claiming artifacts fail closed.

## What I understood

The project needs a path to inspect future operator-captured scene/snapshot
readback artifacts, but scene recall/save surfaces are destructive. The intake
can help reviewers see artifact quality; it cannot close fixture gaps or grant
live scene/preset authority.

## Proposed solution

- Keep `backend/snapshots_scenes_physical_readback_artifact_intake.py`
  transport-free.
- Register the method as `shadow_only` and non-executor in the processing
  registry and capability matrix.
- Surface normalized proof quality through Product Layer, no-WING summaries,
  and frontend module status rows.
- Treat any invalid artifact in a candidate set as a failed intake.
- Add regression coverage for mixed candidates and frontend read-only row
  visibility.

## Likely files to touch

- `backend/snapshots_scenes_physical_readback_artifact_intake.py`
- `backend/processing_registry.py`
- `backend/operator_product_state.py`
- `backend/project_goal_capability_matrix.py`
- `tools/no_wing_release_check.py`
- `tests/test_snapshots_scenes_physical_readback_artifact_intake.py`
- `tests/test_no_wing_release_check.py`
- `frontend/src/services/operatorAdapters.ts`
- `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`

## Alternatives considered

- Build a scene/snapshot capture runner now. Rejected: that would need a
  separate operator-approved hardware design and could be mistaken for scene
  recall/save permission.
- Treat any valid artifact as fixture closure. Rejected: manual review,
  freshness proof, rollback proof, and physical fixture governance are still
  required.
- Keep only the schema contract. Rejected: operators need artifact intake
  visibility once captured evidence appears.

## Risks

- A valid-looking JSON artifact could be mistaken for production readiness.
  Mitigation: summary and UI preserve fixture gap, live authority, and scene
  load/save denial.
- A mixed artifact directory could hide a bad artifact next to a good one.
  Mitigation: invalid artifacts dominate status and prevent candidate
  acceptance.
- Existing worktree changes may be from adjacent slices. Mitigation: keep this
  cycle scoped and do not revert unrelated files.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_snapshots_scenes_physical_readback_artifact_intake.py tests/test_no_wing_release_check.py -q`
- `node --test frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where the other agent may disagree

Another agent may want to promote scene/preset artifacts immediately when a
valid artifact is present. That should stay blocked: intake only creates a
manual-review candidate, not live authority or fixture closure.
