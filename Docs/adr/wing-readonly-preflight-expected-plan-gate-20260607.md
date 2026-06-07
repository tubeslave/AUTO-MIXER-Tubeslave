# ADR: WING Read-Only Preflight Expected-Plan Gate - 2026-06-07

## 1. Context

The WING read-only preflight runbook has two steps: preview the read-only
request plan offline, then run the real capture against the intended
backend/WING session. Before this ADR, the real artifact did not prove that its
runtime arguments and request sequence matched the previewed plan.

## 2. Options considered

1. Leave the plan as runbook guidance only.
2. Compare only channel and value-type arguments.
3. Validate the real run against the full runtime-relevant plan before opening
   WebSocket and record the plan SHA256 in the artifact.

## 3. Decision

Use option 3.

## 4. Why this won

The first WING proof artifact should be auditable after the fact. Binding the
real run to a plan hash proves the operator did not accidentally run a different
channel, request sequence, artifact root, or fail-closed contract.

## 5. Rejected alternatives

Runbook-only guidance leaves the proof dependent on operator memory. Comparing
only channels and value types misses URL, artifact root, request id policy, and
review-command drift.

## 6. Implementation plan

- Add `--expected-plan` to the runner.
- Load and hash the expected plan before WebSocket.
- Compare runtime-relevant plan fields against current args.
- Record plan path, hash, match status, and violations in summary artifacts.
- Require those fields in offline evidence validation.
- Update handoff/no-WING/runbook commands to use the same plan file.

## 7. Test plan

- `python3 -m py_compile tools/wing_readonly_preflight_runner.py backend/wing_readonly_preflight_evidence.py tools/no_wing_release_check.py tools/build_wing_readonly_preflight_handoff_packet.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_runner.py tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py tests/test_autonomy_promotion_readiness.py`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_no_wing_expected_plan_gate_20260607`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## 8. Risks and rollback

The main risk is rejecting an otherwise clean artifact that was captured without
`--expected-plan`. Rollback is to remove expected-plan fields from evidence
validation and keep the plan as runbook guidance.
