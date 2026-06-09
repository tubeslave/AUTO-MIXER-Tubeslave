# project-goal-objective-readiness-by-goal-area-breakdown-20260609

## Context

A single readiness score in `project_goal_capability_objective_alignment_snapshot`
summarizes all required goal capabilities, but operators need fast visibility into
which capability families (goal areas) are carrying the remaining blockers.

## Decision

Extend the objective-alignment snapshot summary with
`autonomy_readiness_by_goal_area`.

The payload is keyed by `goal_area`, and each value contains:
- `required_row_count`
- `ready_row_count`
- `gap_row_count`
- `readiness_score`
- `status_counts`
- `ready_row_ids`
- `gap_row_ids`

The snapshot also exposes:
- `autonomy_readiness_goal_area_priorities` as a list sorted descending by
  `gap_row_count`, then ascending by `readiness_score`, then by `goal_area`.

The progress snapshot also mirrors this readiness partition in two summary fields:
- `project_goal_objective_autonomy_readiness_by_goal_area`
- `project_goal_objective_autonomy_readiness_goal_area_priorities`

For planning action, it also exposes:
- `project_goal_objective_autonomy_readiness_hotspots`, a gap-first list of
  areas with:
  - `goal_area`
  - `gap_row_count`
  - `readiness_score`
  - `blocker_row_count`
  - `blocker_row_ids`

These hotspot fields are also surfaced in:
- processing registry snapshot summary as
  `project_goal_capability_objective_readiness_hotspots`;
- processing pipeline preview summary under the same key;
- operator product state snapshots through registry and pipeline payload summaries.

The readiness calculation in each area uses the same rule as the global score:
`int((ready_row_count / required_row_count) * 100)` with required set
`{canonical, supervised_narrow}` as ready.

## Why this won

A flat global percentage is useful, but area-level breakdown aligns better with the
project goal architecture, which is organized by capability families (`processing`,
`console_control`, `runtime_mode`, etc.). This helps prioritize engineering effort
without changing any live-write boundary.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_project_goal_capability_matrix.py -x --tb=short -q`

## Risks and rollback

No write or transport behavior changes; this is metadata-only. Rollback is removing
this map from `backend/project_goal_capability_matrix.py`, dropping the test
assertions, and deleting this ADR if the format is later redesigned.
