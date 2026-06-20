# ADR: Project Director Post-M105 Intake Loop Persistence

## Context

M105 correctly added a Project Director guard against endlessly selecting
generic WING read-only intake-hardening milestones after M104. After the M105
commit landed, the guard no longer activated in the live directive because the
streak helper only inspected the newest commit prefix. The newest commit became
`Guard project director intake loop`, so the M095-M104 WING read-only streak was
still present but no longer at the prefix.

## Options Considered

1. Accept that the guard only works immediately before the M105 commit.
2. Add a special-case subject exception for `Guard project director intake loop`.
3. Make Project Director find the longest consecutive WING read-only intake
   guard streak inside recent history.

## Decision

Use option 3. The streak helper now returns the longest WING read-only intake
guard streak in recent commit subjects, instead of only the prefix streak.

## Why This Won

The guard is about historical direction, not only the latest subject. Searching
inside recent history keeps M105 active after direction-guard commits while
still requiring a real consecutive M095-M104-style WING read-only guard streak.

## Rejected Alternatives

Option 1 was rejected because it made M105 self-invalidating. Option 2 was
rejected because it would need another exception for each future direction-guard
commit.

## Implementation Plan

1. Update the Project Director streak helper to track the longest matching
   streak.
2. Extend the focused Project Director test with a leading direction-guard
   commit.
3. Add an M106 rail proving current Project Director output remains
   `external_validation_wait_or_named_gap_only`.
4. Record M106 state/report evidence without changing WING readiness.

## Test Plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_project_director_agent.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m106_project_director_post_m105_intake_loop_guard_persists -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py --root .`

## Risks and Rollback

No live transport is introduced. Rollback is to revert the M106 commit, which
restores prefix-only Project Director streak detection and M106 state/report
rows. The risk is overmatching older streaks inside the lookback window; the
guard still only narrows work while M001 is waiting for external validation and
the selected action requires real WING evidence.
