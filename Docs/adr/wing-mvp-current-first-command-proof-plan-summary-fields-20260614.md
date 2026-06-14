# ADR: WING MVP Current First Command Proof-Plan Summary Fields

## Context

The WING read-only connection proof remains the first open MVP blocker. The
handoff packet and no-WING gate now require the supervised proof-plan summary
field list, and the autoloop prompt/README also name it. The explicit
`## Current First Command` section lagged behind and did not include those
fields.

## Options considered

1. Leave `Current First Command` unchanged and rely on the blocker row
   baseline.
2. Update only the progress report.
3. Update `Current First Command` and protect it with tests.

## Decision

Add the supervised proof-plan summary field requirements to
`## Current First Command`, including the declared field list, expected field
list, contract-match flag, expected accepted-candidate source IDs, and expected
violations map.

## Why this won

Project Director uses `Current First Command` as the primary selected-action
source. Making it self-contained prevents weaker action text from steering the
next Goal-mode run when the blocker row has stricter requirements.

## Rejected alternatives

Relying on the blocker row baseline still lets the primary command text drift.
Updating only generated progress would not fix the source of truth.

## Implementation plan

- Add the proof-plan summary field contract to `## Current First Command`.
- Extend WING rail tests around `current_first_command()`.
- Extend Project Director selected-action assertions.
- Record the report in the first WING blocker evidence row and docs index.

## Test plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short -x`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/update_mvp_progress.py --blockers .ai/project-layer/wing_mvp_blockers.md --output Docs/reports/MVP_PROGRESS_LATEST.md`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --artifact-root /tmp/automixer_goal_current_first_command_proof_plan_summary_fields_nowing_20260614`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk is limited to stricter offline command text and static tests. Rollback
removes the added first-command requirements and assertions. No backend
startup, WING connection, console query, console write, readiness closure, or
live authority is introduced.
