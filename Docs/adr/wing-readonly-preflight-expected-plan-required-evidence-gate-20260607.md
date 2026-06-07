# ADR: Evidence Requires Expected-Plan Requirement Flag

## Context

The WING read-only preflight runner requires an offline expected plan before a
real WebSocket run. Accepted artifacts should prove that policy in structured
form.

## Options Considered

1. Require only `expected_plan_path`, `expected_plan_sha256`, and
   `expected_plan_match`.
2. Also require `expected_plan_required_for_real_run: true`.
3. Accept either shape for backward compatibility.

## Decision

Use option 2. Evidence intake now rejects artifacts that omit or falsify
`expected_plan_required_for_real_run`.

## Why This Won

The current MVP values proof quality over backward compatibility. The accepted
artifact should show that the real run was bound to a pre-approved offline plan
as a runner policy, not only as incidental metadata.

## Rejected Alternatives

- Path/SHA-only proof is weaker and can be hand-written around the runner.
- Backward compatibility would keep stale artifacts eligible as candidates.

## Implementation Plan

- Add the validation check in `backend/wing_readonly_preflight_evidence.py`.
- Update valid WING preflight fixtures.
- Extend the missing expected-plan regression test.

## Test Plan

- `python3 -m py_compile backend/wing_readonly_preflight_evidence.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_autonomy_promotion_readiness.py`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_no_wing_expected_plan_required_evidence_gate_20260607`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks And Rollback

Rollback would remove the check and allow older artifact summaries again. That
should only happen if real WING evidence captured by the current runner cannot
include the field, which current runner tests contradict.
