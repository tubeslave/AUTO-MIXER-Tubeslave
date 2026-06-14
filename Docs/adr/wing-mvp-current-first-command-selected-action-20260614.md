# ADR: WING MVP Current First Command Selected Action

## Context

The WING read-only connection proof remains the first open MVP blocker. The
progress tool combines `## Current First Command` with the blocker table
baseline whenever the strings differ, even when the current command already
contains the safety/proof requirements needed for the next step.

## Options considered

1. Keep combining both texts.
2. Trust `Current First Command` unconditionally.
3. Trust `Current First Command` only when it covers an explicit marker set.

## Decision

Use `Current First Command` directly when it contains the required baseline
coverage markers. If any marker is missing, retain the existing
`Current First Command + blocker table baseline` fallback.

## Why this won

It keeps Project Director focused and reduces duplicated action text without
removing the conservative fallback. The marker list names the safety and proof
constraints that must remain present before the shorter source is allowed.

## Rejected alternatives

Keeping the duplicate baseline makes every Goal-mode iteration noisier and
less precise. Trusting `Current First Command` unconditionally would make future
drift harder to detect.

## Implementation plan

- Add `CURRENT_FIRST_COMMAND_BASELINE_COVERAGE_MARKERS`.
- Add `current_first_command_covers_baseline()`.
- Switch `selected_action_text()` to direct current-command selection only when
  coverage passes.
- Add listener-template denial fields that were previously recovered from the
  blocker-row baseline.
- Update tests and regenerate latest progress.

## Test plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short -x`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/update_mvp_progress.py --blockers .ai/project-layer/wing_mvp_blockers.md --output Docs/reports/MVP_PROGRESS_LATEST.md`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --artifact-root /tmp/automixer_goal_current_first_command_selected_action_nowing_20260614`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk is limited to offline Project Director action selection. Rollback removes
the marker coverage check and restores the previous always-combine behavior.
No backend startup, WING connection, console query, console write, readiness
closure, or live authority is introduced.
