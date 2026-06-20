# ADR: Project Director Intake Loop Lookback Floor

## Context

M105 added a Project Director guard against endlessly selecting generic WING
read-only intake-hardening milestones after M104. M106 made the guard find the
longest WING read-only streak inside recent history instead of only at the
newest commit prefix.

After M106, the default Project Director evidence window still had a horizon
gap. The default request used `lookback_commits=12`. With M106 and M105 above
the ten WING read-only intake commits, the streak was still visible at exactly
12 subjects. One more direction-guard commit would push part of the WING streak
outside that requested window, turning the guard off again and reopening the
same class of repeated intake-hardening work.

## Options Considered

1. Require callers to remember a larger `--lookback-commits` value.
2. Increase the CLI default only.
3. Give normal Project Director calls an internal effective lookback floor for
   the intake-loop guard.

## Decision

Use option 3. Project Director now keeps the explicit requested lookback value
for reporting, but raises normal/default evidence collection to the
post-M104 intake-loop guard lookback floor.

## Why This Won

The guard is a safety and direction-control mechanism. It should not depend on
each caller remembering a fragile horizon value. An internal floor keeps the
M105/M106 behavior stable for normal Project Director runs while preserving
small artificial lookbacks used by focused tests.

## Rejected Alternatives

Option 1 was rejected because it would make the same gap recur in every caller
that used the old default. Option 2 was rejected because it only fixes the CLI
surface and not direct `build_directive` callers.

## Implementation Plan

1. Add a Project Director constant for the default lookback and a lookback
   floor derived from the M105 guard streak length plus buffer.
2. Collect repo evidence with the effective lookback for normal/default calls.
3. Surface requested lookback, effective lookback and floor in the directive.
4. Add focused and WING rail tests proving the M105 guard remains active.
5. Record M107 state/report evidence without changing WING readiness.

## Test Plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_project_director_agent.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m107_project_director_intake_loop_guard_lookback_floor_persists -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py --root .`

## Risks and Rollback

No live transport is introduced. Rollback is to revert the M107 commit, which
restores caller-provided Project Director lookback behavior and M107
state/report rows. The risk is collecting a few more commit subjects than the
caller requested during normal runs; the guard still only narrows work while
M001 is waiting for external validation and the selected action requires real
WING evidence.
