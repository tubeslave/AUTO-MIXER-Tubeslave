# ADR: read-surface-fixture-closure-plan-20260605

## Context

Automixer now has read-surface artifact intake, an aggregate artifact review
queue, and an operator acceptance ledger. Accepted records identify candidate
physical readback evidence, but they do not update fixture baselines and do not
prove live readiness. A separate read-only plan is needed before any future
manual fixture-baseline update.

## Options Considered

1. Treat accepted ledger records as closed fixture gaps.
2. Write fixture baselines directly from accepted records.
3. Add a separate read-only manual fixture closure plan.

## Decision

Use option 3. The fixture closure plan exposes manual closure candidates from
valid accepted records while preserving physical fixture gaps and live authority
blockers.

## Why This Won

The plan creates a clear audit boundary: artifacts are captured, candidates are
reviewed, accepted records become manual closure candidates, and only a later
manual baseline update can close the physical fixture gap.

## Rejected Alternatives

- Automatic closure from acceptance was rejected because it would overstate
  proof quality.
- Direct fixture file writes were rejected because this cycle is no-WING,
  read-only, and should not mutate baseline evidence.
- Hiding the plan from no-WING release output was rejected because operators need
  release-visible proof progress and blockers.

## Implementation Plan

1. Add a transport-free backend closure-plan builder.
2. Include only valid accepted ledger records as closure candidates.
3. Register the plan as `shadow_only` evidence.
4. Add no-WING validator, JSON summary, Markdown summary, and run ordering.
5. Add tests for accepted, rejected, invalid, unreviewed, registry, and release
   summary behavior.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_read_surface_physical_readback_fixture_closure_plan.py tests/test_no_wing_release_check.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/read_surface_physical_readback_fixture_closure_plan.py tools/no_wing_release_check.py`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer-no-wing-release-check-codex`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks And Rollback

Risk: manual closure candidates may be mistaken for completed fixture closure or
live readiness. Mitigation: the plan and release summaries force
`fixture_update_allowed_by_plan=false`, `fixture_closure_allowed_by_plan=false`,
`physical_fixture_gap_remains=true`, `live_authority_granted=false`, and
`full_autonomous_ready=false`.

Rollback: remove the closure-plan validator from `run_release_checks()` and
remove the registry/capability row. The acceptance ledger and artifact review
queue can remain unchanged.
