# Proposal: read-surface-fixture-acceptance-ledger-20260605

## Short Thesis

Add a read-only acceptance ledger after the artifact review queue so operator
accept/reject decisions are recorded safely without closing fixtures or granting
live authority.

## What I Understood

Valid physical readback artifacts are only manual-review candidates. The system
needs a place to validate operator decisions about those candidates, but that
decision is still not the same as adding a fixture to the test baseline or
unlocking live control surfaces.

## Proposed Solution

- Add `backend/read_surface_physical_readback_fixture_acceptance_ledger.py`.
- Validate acceptance records against queue candidates by surface, artifact
  source, contract method, decision, reviewer, and timestamp.
- Reject records that claim closure, live writes, console queries, test
  execution, scene/preset authority, dLive authority, or autonomous readiness.
- Register the ledger as `shadow_only` and expose it in capability evidence.
- Add the ledger schema gate to no-WING release checks and summaries.

## Likely Files To Touch

- `backend/read_surface_physical_readback_fixture_acceptance_ledger.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tools/no_wing_release_check.py`
- `tests/test_read_surface_physical_readback_fixture_acceptance_ledger.py`
- `tests/test_no_wing_release_check.py`
- `.ai/memory/project.md`
- `Docs/adr/read-surface-fixture-acceptance-ledger-20260605.md`

## Alternatives Considered

- Fold accept/reject state into the queue. Rejected because the queue should
  aggregate artifact status, not operator decisions.
- Treat accepted records as fixture closure. Rejected because fixture closure
  requires a separate manual baseline decision.
- Delay the ledger until real artifacts exist. Rejected because the safety
  contract and release evidence can be tested before live proof artifacts exist.

## Risks

- Operators may overread accepted records as live readiness. Mitigate by
  forcing closure, live authority, and full autonomy flags to false in code,
  release summaries, and tests.
- Bad records could point at the wrong candidate. Mitigate by validating
  surface, artifact source, and contract method consistency.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_read_surface_physical_readback_fixture_acceptance_ledger.py tests/test_no_wing_release_check.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/read_surface_physical_readback_fixture_acceptance_ledger.py tools/no_wing_release_check.py`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where The Other Agent May Disagree

Kimi might prefer one combined queue plus acceptance state. I would keep a
separate ledger because it preserves the difference between candidate discovery,
operator review, and eventual fixture baseline closure.
