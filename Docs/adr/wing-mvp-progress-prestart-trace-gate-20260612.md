# WING MVP Progress Prestart Trace Gate

## Context

The supervised WING proof plan now requires accepted read-only preflight source
ids to have a valid accepted-candidate prestart trace before physical proof
steps can proceed. The blocker table and generated MVP progress report are the
Project Director source for the next Codex task, so they must expose the same
gate.

## Options considered

1. Keep the MVP progress rails generic.
2. Add accepted-candidate prestart trace wording to the first blocker next
   action and current MVP stop rules.
3. Add a separate prestart trace blocker row.

## Decision

Use option 2. The first blocker remains `WING read-only connection proof` with
status `tooling_ready_waits_real_wing`, but its next action now states that
accepted review source ids must have a valid accepted-candidate prestart trace
before any supervised proof.

## Why this won

It keeps the blocker sequence stable while making the current fail-closed proof
gate visible to the operator and Project Director agent.

## Rejected alternatives

- Generic progress wording was rejected because it can plan supervised proof
  without naming the prestart trace prerequisite.
- A separate blocker row was rejected as too much structure for a prerequisite
  already enforced by `backend/supervised_wing_proof_plan.py` and
  `tools/no_wing_release_check.py`.

## Implementation plan

- Update `.ai/project-layer/wing_mvp_blockers.md`.
- Update `Docs/MVP_WING_SUPERVISED.md`.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Add regression tests for MVP rails and Project Director selected action.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_mvp_rails.py tests/test_project_director_agent.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk is wording-only. Rollback is safe by restoring the previous next action,
removing the focused assertions, and regenerating the progress report. No
runtime console behavior or live write authority changes.
