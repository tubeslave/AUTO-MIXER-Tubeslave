# project-goal-capability-matrix-20260603

## Context

The active project goal lists a full AI Mixing Engineer target: automatic
analysis, source/role recognition, gain, EQ, dynamics, gate, de-essing, panning,
phase, FX, mix balance, live adaptation, master-bus work, full console control,
four runtime modes, and long-term learning. The repo has many partial systems,
but the target goal needs a read-only map that makes missing and blocked
capabilities explicit.

## Scope

- Add a declarative project-goal capability matrix.
- Cover every explicit project-goal capability at a coarse product level.
- Separate current live-write kinds from target console surfaces.
- Keep the matrix read-only and fail-closed.
- Add tests that prevent missing capabilities like de-essing from being ignored.

## Non-goals

- No live OSC/WING writes.
- No promotion of broad autonomous apply.
- No edits to existing dirty write-surface or safety-coverage files.
