# Proposal: read-surface-fixture-closure-plan-hardening-20260606

## Short Thesis

Make the fixture closure plan distrust its input ledger and fail closed unless
the ledger envelope, summary counts, surface-id lists, and accepted records are
internally consistent.

## What I Understood

The project is moving toward autonomous console control, but physical readback
fixture closure is still a proof boundary. A forged or stale acceptance ledger
must not be able to manufacture a manual fixture-closure candidate, because that
would overstate readiness for later live-control promotion.

## Proposed Solution

- Validate the acceptance ledger envelope type, method id, status, UTC
  timestamp, and false safety flags.
- Validate accepted records for required identity fields, valid UTC
  `reviewed_at`, `decision=accepted`, no conflicting rejected flag, candidate
  linkage, and false authority fields.
- Compare ledger summary counts and accepted/rejected surface-id lists against
  the records before building candidates.
- Treat duplicate or conflicting review records for the same surface as
  replacement-required input.
- Keep all plan authority flags false even when accepted records are otherwise
  valid.

## Likely Files To Touch

- `backend/read_surface_physical_readback_fixture_closure_plan.py`
- `tests/test_read_surface_physical_readback_fixture_closure_plan.py`
- `.ai/memory/project.md`
- `Docs/adr/read-surface-fixture-closure-plan-hardening-20260606.md`

## Alternatives Considered

- Trust the acceptance ledger builder because it normally validates records.
  Rejected because the closure plan accepts an injected ledger mapping in tests
  and future integrations.
- Collapse all malformed input into a generic missing-candidate state. Rejected
  because operators and release artifacts need replacement-required evidence.
- Write fixture baselines after validation. Rejected because this remains a
  read-only proof-planning step.

## Risks

- The hardening could reject a legitimate candidate if the ledger summary format
  drifts. Mitigate with focused tests and explicit violation ids.
- Operators could misread a replacement-required plan as a live fault. Mitigate
  by keeping the blocker text manual and fixture-specific.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_read_surface_physical_readback_fixture_closure_plan.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_no_wing_release_check.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/read_surface_physical_readback_fixture_closure_plan.py tools/no_wing_release_check.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where The Other Agent May Disagree

Kimi might argue this belongs entirely in the acceptance ledger. I would keep
the second barrier because the closure plan is the object that produces manual
fixture-closure candidates, so it should independently reject forged or stale
ledger input.
