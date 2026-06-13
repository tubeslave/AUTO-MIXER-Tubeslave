# Codex Proposal: WING MVP Fail-Closed Contract Source

## Short Thesis

Centralize the WING readonly preflight expected-plan fail-closed conditions in
the shared preflight contract module and keep local names as aliases.

## What I Understood

The WING readonly MVP rail depends on strict plan validation before any real
backend/WING session. The fail-closed list must not drift between the runner,
evidence intake, and no-WING release check.

## Proposed Solution

Add `WING_READONLY_PREFLIGHT_PLAN_FAIL_CLOSED_CONDITIONS` to
`backend/wing_readonly_preflight_contract.py`. Import it in the runner and use
it when building offline plans. Import it in evidence and release-check modules
while preserving `EXPECTED_PLAN_FAIL_CLOSED_CONDITIONS` and
`WING_READONLY_PREFLIGHT_PLAN_REQUIRED_FAIL_CLOSED_CONDITIONS` as compatibility
aliases.

## Likely Files To Touch

- `backend/wing_readonly_preflight_contract.py`
- `tools/wing_readonly_preflight_runner.py`
- `backend/wing_readonly_preflight_evidence.py`
- `tools/no_wing_release_check.py`
- focused tests and WING MVP docs/reports

## Alternatives Considered

Leaving the lists duplicated is simpler today but already caused a release gate
mismatch. Importing from one tool module into another would create tool-to-tool
coupling; the shared contract module is the correct boundary.

## Risks

The main risk is accidentally changing list order. Tests should compare exact
lists and the no-WING quick check should prove generated plans still validate.

## Test Plan

- Compile touched Python modules.
- Focused runner/evidence/no-WING tests.
- no-WING quick on an unused backend port.
- Project Director and direction guard.
- Full pytest.

## Where The Other Agent May Disagree

Kimi may argue this is a refactor rather than blocker progress. The counterpoint
is that this exact duplication already broke the preflight release gate; removing
the duplication strengthens the blocker rail without changing readiness status.
