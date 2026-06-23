# WING MVP Current Command Markdown Summary Surface

## Summary

Project Director / Current First Command now explicitly require the
operator-readable no-WING `summary.md` proof-plan section to render the canonical
supervised proof-plan helper fields before any supervised fader/gain proof is
planned.

## Changed

- `tools/codex_autoloop/update_mvp_progress.py` adds
  `operator-readable no-WING summary.md` to Current First Command baseline
  coverage.
- `.ai/project-layer/wing_mvp_blockers.md` names the operator-readable
  `summary.md` proof-plan section in the active Current First Command.
- `tools/codex_autoloop/codex_cycle_prompt.md` and
  `tools/codex_autoloop/README.md` carry the same operator-facing requirement.
- `tests/test_project_director_agent.py` and `tests/test_wing_mvp_rails.py`
  assert the requirement and fail baseline coverage when it is removed.

## Safety Boundary

This is an offline rail/documentation/test change. It does not contact WING,
query the console, grant live authority, or authorize fader/gain movement.

## Remaining Blocker

The WING MVP remains blocked on a real WING read-only preflight capture and an
operator-reviewed accepted candidate before any supervised fader/gain proof run.
