# Project Goal Supervised AutoMix WING Prestart Trace

## Context

The current WING proof path requires accepted read-only preflight source ids to be bound to candidate review rows with the backend-port prestart contract before physical proof steps are considered acceptable. The project goal capability matrix is the high-level map from the long-term Automixer objective to current repo surfaces.

## Options considered

1. Leave Supervised AutoMix next-step wording generic.
2. Add the accepted-candidate prestart trace prerequisite to `mode_supervised_automix.next_step`.
3. Add a new WING-specific goal row.

## Decision

Use option 2. `mode_supervised_automix` now says WING physical proof steps require accepted-candidate prestart trace from read-only preflight review before widening.

## Why this won

It aligns high-level goal reporting with the current WING proof safety gate without changing runtime behavior or widening live authority.

## Rejected alternatives

- Generic wording was rejected because it can hide the new prerequisite.
- A new goal row was too heavy for one mode-level prerequisite.

## Implementation plan

- Update `backend/project_goal_capability_matrix.py`.
- Add a focused regression in `tests/test_project_goal_capability_matrix.py`.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_goal_capability_matrix.py -k "supervised_automix_goal_requires_wing_candidate_prestart_trace"`
- Affected matrix/registry/WING proof tests.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`

## Risks and rollback

Risk is limited to status wording. Rollback is safe by restoring the previous next-step text and removing the focused test. No console authority changes.
