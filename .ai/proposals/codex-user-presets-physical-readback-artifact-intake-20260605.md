# Codex Proposal: user-presets-physical-readback-artifact-intake-20260605

## Short thesis

Expose user-preset physical readback artifacts as read-only manual-review
candidates, and make invalid or authority-claiming artifacts fail closed.

## What I understood

The project needs a path to inspect future operator-captured preset readback
artifacts, but preset recall and store surfaces are destructive. The intake can
help reviewers see artifact quality; it cannot close fixture gaps or grant live
scene/preset authority.

## Proposed solution

- Keep `backend/user_presets_physical_readback_artifact_intake.py`
  transport-free.
- Register the method as `shadow_only` and non-executor in the processing
  registry and capability matrix.
- Add no-WING validation and summary output.
- Treat any invalid artifact in a candidate set as a failed intake.
- Add regression coverage for invalid artifacts, discovery, registry status,
  and release-gate safety flags.

## Likely files to touch

- `backend/user_presets_physical_readback_artifact_intake.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tools/no_wing_release_check.py`
- `tests/test_user_presets_physical_readback_artifact_intake.py`
- `tests/test_no_wing_release_check.py`
- `tests/test_processing_registry.py`
- `tests/test_project_goal_capability_matrix.py`

## Alternatives considered

- Build a preset capture runner now. Rejected: that would need a separate
  operator-approved hardware design and could be mistaken for preset
  recall/store permission.
- Treat any valid artifact as fixture closure. Rejected: manual review,
  freshness proof, rollback proof, and physical fixture governance are still
  required.
- Keep only the schema contract. Rejected: operators need artifact intake
  visibility once captured evidence appears.

## Risks

- A valid-looking JSON artifact could be mistaken for production readiness.
  Mitigation: summary preserves fixture gap, live authority, preset recall, and
  preset store denial.
- A mixed artifact directory could hide a bad artifact next to a good one.
  Mitigation: invalid artifacts dominate status and prevent candidate
  acceptance.
- Existing worktree changes may be from adjacent slices. Mitigation: keep this
  cycle scoped and do not revert unrelated files.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_user_presets_physical_readback_artifact_intake.py tests/test_no_wing_release_check.py -q`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where the other agent may disagree

Another agent may want to promote preset artifacts immediately when a valid
artifact is present. That should stay blocked: intake only creates a
manual-review candidate, not live authority or fixture closure.
