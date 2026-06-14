# ADR: WING MVP Director Action Coverage Digest

## Context

The WING read-only connection proof remains the first open MVP blocker. Project
Director now selects `Current First Command` directly when that command covers
the WING safety/proof baseline. The selected action was auditable by reading
long text, but it did not publish a compact coverage verdict or digest.

## Options considered

1. Keep coverage implicit in selected action text.
2. Always append the blocker-row baseline again.
3. Publish explicit coverage and digest fields while keeping fallback behavior.

## Decision

Project Director publishes:

- `selected_action_sha256`
- `current_first_command_baseline_coverage_required`
- `current_first_command_baseline_coverage_passed`
- `current_first_command_baseline_coverage_marker_count`
- `current_first_command_baseline_missing_markers`

The progress helper now exposes the missing coverage markers used by the
coverage pass/fail verdict.

## Why this won

It makes the shorter selected action auditable without expanding the action
text again. If a marker drifts, the existing fallback still appends the blocker
baseline and the missing marker list explains why coverage failed.

## Rejected alternatives

Implicit text-only coverage is harder for later automation to verify. Reverting
to always-appended baseline would undo the previous action-focus improvement.

## Implementation plan

- Add `current_first_command_missing_baseline_markers()`.
- Emit selected-action SHA256 and coverage fields from Project Director.
- Render the fields in markdown output.
- Document the fields in the autoloop prompt and README.
- Add focused tests around the new metadata.

## Test plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short -x`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --artifact-root /tmp/automixer_goal_director_action_coverage_digest_nowing_20260614`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk is limited to offline Project Director metadata. Rollback removes the new
fields and helper. No backend startup, WING connection, console query, console
write, readiness closure, or live authority is introduced.
