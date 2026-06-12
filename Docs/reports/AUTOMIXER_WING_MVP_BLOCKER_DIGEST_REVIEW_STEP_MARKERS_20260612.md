# Automixer WING MVP Blocker Digest Review Step Markers

Date: 2026-06-12

## Summary

The first open WING MVP blocker now explicitly requires
`operator_sequence_mvp_blocker_anchor_digest_review_step` to include:

- `mvp_blocker_anchor`
- `mvp_blocker_anchor_sha256`
- `current first open blocker`

## Why

The digest-review step is the operator-visible proof that the handoff packet
still points at the current first WING blocker and digest. Project Director and
the autoloop prompt now carry the same marker set as the read-only runbook and
handoff validator.

## Changed

- `.ai/project-layer/wing_mvp_blockers.md` names the digest-review step markers
  in the first blocker next action and current command.
- `Docs/reports/MVP_PROGRESS_LATEST.md` was regenerated from the blocker table.
- `tools/codex_autoloop/codex_cycle_prompt.md` and
  `tools/codex_autoloop/README.md` document the same pre-proof requirement.
- `tests/test_project_director_agent.py` and `tests/test_wing_mvp_rails.py`
  assert the marker strings in selected-action and autoloop surfaces.

## Boundary

This report does not grant live authority. It does not contact WING, does not
start a backend/WING session, and does not close the read-only connection proof
blocker.

## Validation

- `PYTHONPATH=backend python3 tools/codex_autoloop/update_mvp_progress.py --blockers .ai/project-layer/wing_mvp_blockers.md --output Docs/reports/MVP_PROGRESS_LATEST.md`
  - PASS
- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py tests/test_codex_direction_guard.py`
  - `23 passed`
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --format json --print`
  - `selected_blocker: WING read-only connection proof`
  - selected action includes `operator_sequence_mvp_blocker_anchor_digest_review_step`
  - selected action includes `mvp_blocker_anchor`
  - selected action includes `mvp_blocker_anchor_sha256`
  - selected action includes `current first open blocker`
  - `guard_status: on_target`
  - `guard_stop_the_line: false`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
  - `status: on_target`
  - `stop_the_line: false`
- `/opt/homebrew/bin/git diff --check`
  - PASS
- `python3 tools/memory/memory_lint.py`
  - PASS
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
  - `3205 passed, 1 skipped, 1 warning`
