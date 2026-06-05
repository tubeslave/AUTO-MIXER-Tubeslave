# Proposal: read-surface-artifact-review-queue-release-gate-20260605

## Short Thesis

The aggregate artifact review queue should become a mandatory no-WING release
evidence gate, but it must remain an operator-review view only.

## What I Understood

Automixer has many physical readback artifact-intake surfaces. The safe pattern
is consistent: a valid artifact can become a manual-review candidate, but intake
does not close fixture gaps and never grants live authority. A release report
needs one aggregate view to show whether any surface has valid candidates,
invalid artifacts, unsafe authority claims, target drift, or missing artifacts.

## Proposed Solution

- Keep `backend/read_surface_physical_readback_artifact_review_queue.py`
  transport-free and registry-visible as `shadow_only`.
- Execute
  `validate_read_surface_physical_readback_artifact_review_queue_snapshot()` in
  `run_release_checks()` after dLive-state artifact intake.
- Publish `read_surface_physical_readback_artifact_review_queue` in
  `summary.json` and `summary.md`.
- Test direct validation, release ordering, summary rendering, and fail-closed
  normalization of queue authority claims.

## Likely Files To Touch

- `backend/read_surface_physical_readback_artifact_review_queue.py`
- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `tests/test_read_surface_physical_readback_artifact_review_queue.py`
- `.ai/memory/project.md`
- `Docs/adr/read-surface-artifact-review-queue-release-gate-20260605.md`

## Alternatives Considered

- Leave only per-surface intake blocks. Rejected because it hides cross-surface
  review status from release evidence.
- Let the queue close fixture gaps when candidates are valid. Rejected because
  candidate review is not physical proof acceptance.
- Trigger artifact capture from the queue. Rejected because this cycle must stay
  no-WING, no-network, read-only, and operator-review-only.

## Risks

- The queue could be mistaken for permission to widen live controls. Mitigate
  with false safety flags in code, summary output, and tests.
- A malformed per-surface intake could pollute the aggregate. Mitigate by
  treating invalid, unsafe, or target-inconsistent items as replacement-required
  evidence.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_read_surface_physical_readback_artifact_review_queue.py tests/test_no_wing_release_check.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile tools/no_wing_release_check.py backend/read_surface_physical_readback_artifact_review_queue.py`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where The Other Agent May Disagree

Kimi might prefer delaying the aggregate until real operator-captured artifacts
exist. I would still add the queue now because missing-artifact status is useful
release evidence and does not reduce the safety boundary.
