# Proposal: read-surface-fixture-closure-plan-20260605

## Short Thesis

Add a read-only closure plan after the acceptance ledger so accepted physical
readback records become explicit manual fixture-baseline update candidates
without automatically closing proof gaps.

## What I Understood

The project needs physical readback proof before broad console automation can be
trusted. Candidate intake and operator acceptance are not enough by themselves:
the repository still needs a separate, auditable manual fixture-baseline update
step before `physical_fixture_gap_remains` can be cleared.

## Proposed Solution

- Add `backend/read_surface_physical_readback_fixture_closure_plan.py`.
- Build closure candidates only from valid accepted ledger records.
- Preserve fail-closed safety fields:
  `fixture_update_allowed_by_plan=false`,
  `fixture_closure_allowed_by_plan=false`,
  `physical_fixture_gap_remains=true`, `live_authority_granted=false`, and
  `full_autonomous_ready=false`.
- Register the plan as `shadow_only` and add it to the console read-parameter
  capability row.
- Validate and publish the plan in no-WING release evidence after the acceptance
  ledger.

## Likely Files To Touch

- `backend/read_surface_physical_readback_fixture_closure_plan.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tools/no_wing_release_check.py`
- `tests/test_read_surface_physical_readback_fixture_closure_plan.py`
- `tests/test_no_wing_release_check.py`
- `.ai/memory/project.md`
- `Docs/adr/read-surface-fixture-closure-plan-20260605.md`

## Alternatives Considered

- Let the acceptance ledger imply fixture closure. Rejected because accept/reject
  review is not the same as updating fixture baselines.
- Actually write fixture files from the plan. Rejected because this cycle must
  stay read-only and no-WING.
- Skip release-check exposure. Rejected because release artifacts are where
  read-surface proof gaps and blockers are audited.

## Risks

- Operators may overread a closure candidate as completed proof. Mitigate by
  forcing all closure and authority flags false and keeping
  `physical_fixture_gap_remains=true`.
- A rejected or invalid record could create a closure candidate. Mitigate by
  filtering candidates to valid accepted records only.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_read_surface_physical_readback_fixture_closure_plan.py tests/test_no_wing_release_check.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/read_surface_physical_readback_fixture_closure_plan.py tools/no_wing_release_check.py`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer-no-wing-release-check-codex`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where The Other Agent May Disagree

Kimi might prefer jumping directly to fixture file updates. I would keep this
manual plan layer first because it creates an auditable boundary between
operator acceptance and baseline mutation.
