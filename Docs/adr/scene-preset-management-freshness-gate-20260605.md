# ADR: scene-preset-management-freshness-gate-20260605

## Context

Scene/preset proposals use current state readback, target previews, scoped
diffs, rollback snapshots, activity checks, scope limits, and operator intent.
They also need to account for whether current scene/preset state is fresh. A
stale scene/preset cache can make a scoped scene recall or preset save unsafe,
especially when the active scene, target preview, or rollback snapshot has
changed.

## Options considered

1. Keep scene/preset proposals based only on readback flags, target preview,
   scoped diff, rollback, and scope checks.
2. Gate scene/preset proposals when provided current-state freshness is not
   fresh.
3. Require physical full-state readback before any proposal-only output.

## Decision

Use option 2. The proposal module remains transport-free. Fresh current-state
cache is recorded as evidence; stale, unknown, or missing provided freshness
demotes the item to `analysis_only`.

## Why this won

It consumes read-only freshness evidence without expanding live write authority.
It prevents known stale scene/preset state from producing confident scene,
snapshot, or preset recommendations.

## Rejected alternatives

Option 1 ignores available safety evidence. Option 3 is the future promotion
bar, but it would break legacy proposal generation before all callers can pass
readback freshness.

## Implementation plan

- Add optional `current_state_cache_freshness` to scene/preset management
  analysis input.
- Normalize freshness into recommendation evidence.
- Add `scene_preset_current_state_cache_not_fresh` when provided freshness is
  not fresh.
- Update proposal contract metadata to list current scene/preset freshness
  evidence.
- Add tests for fresh and stale current-state freshness.

## Test plan

- `PYTHONPATH=backend python -m py_compile backend/scene_preset_management_proposal.py backend/processing_registry.py backend/processing_proposal_adapters.py backend/project_goal_capability_matrix.py tests/test_scene_preset_management_proposal.py`
- Focused scene/preset and registry tests.
- Full suite with `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: callers that pass stale freshness produce fewer scene/preset proposals.
Rollback is removing the freshness field, blocker, tests, registry/contract
wording, and this ADR. No live write behavior is affected.
