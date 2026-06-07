# Codex Proposal: WING Read-Only Preflight Plan Policy Flags Release Gate

## Short Thesis

The offline WING read-only preflight plan should fail no-WING release validation
unless it proves both expected-plan enforcement and the existing backend/WING
connection boundary.

## What I Understood

The first WING MVP blocker remains the real read-only connection proof. Current
tooling has stronger runner/evidence contracts, but the release checker still
does not enforce the same policy flags on the offline plan artifact.

## Proposed Solution

Update `validate_wing_readonly_preflight_plan_artifact_snapshot` to require:

- `expected_plan_required_for_real_run: true`
- `physical_preflight_requires_existing_backend_connection: true`

Expose both in summary JSON/Markdown, add the existing-backend boundary to the
offline plan fail-closed list, and update focused tests and runbook wording.

## Likely Files To Touch

- `tools/no_wing_release_check.py`
- `tools/wing_readonly_preflight_runner.py`
- `tests/test_no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_runner.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `Docs/adr/wing-readonly-preflight-plan-policy-flags-release-gate-20260607.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_PLAN_POLICY_FLAGS_RELEASE_GATE_20260607.md`

## Alternatives Considered

- Rely only on runner/evidence tests. That would leave release handoff summaries
  unable to catch a stale or hand-edited plan.
- Require only the new existing-backend field. That would leave expected-plan
  enforcement weaker at the same release boundary.

## Risks

Stale plans without these fields will fail no-WING validation. That is the
intended behavior for the current MVP because the next real WING run should use
current tooling only.

## Test Plan

- `python3 -m py_compile tools/no_wing_release_check.py tools/wing_readonly_preflight_runner.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_runner.py`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root artifacts/no_wing_release_check_current_cycle`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Where The Other Agent May Disagree

Kimi may prefer to validate only runtime summaries. The stronger position is
that the offline plan is the operator-reviewed contract before the WebSocket
opens, so it must carry the same policy boundaries.
