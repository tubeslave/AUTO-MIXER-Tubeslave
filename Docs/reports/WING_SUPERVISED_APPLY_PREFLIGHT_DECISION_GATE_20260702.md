# WING Supervised Apply Preflight Decision Gate

Generated: `2026-07-02T07:42:41Z`

## Summary

The supervised WING proof runner now fails closed before any backend WebSocket
connection when `--execute` is requested without an accepted read-only preflight
decision.

This does not grant live authority. It only prevents a direct runner invocation
from skipping the operator decision gate.

## Changed Behavior

- `--execute` now requires `--accepted-readonly-source-id`.
- The runner rebuilds the WING read-only decision review from:
  - `artifacts/wing_readonly_preflight`
  - `artifacts/wing_readonly_preflight_review_decisions`
- If the review status is not `accepted_decision_ready`, the runner exits before
  contacting backend.
- If the accepted source id does not match the operator-accepted decision, the
  runner exits before contacting backend.
- The generated supervised proof plan and runbook commands now include the
  accepted source-id binding.

## Current Real-Apply Status

- Live apply authority: `false`
- Current decision review status: `operator_review_required`
- Accepted decision records: `0`
- Current first manual step: operator must review the real read-only candidate
  and record an accepted or rejected decision.

## Verification

- `PYTHONPATH=backend python3 -m pytest tests/test_wing_operator_proposal_validation_runner.py tests/test_supervised_wing_proof_plan.py -q`: `203 passed`
- `PYTHONPATH=backend python3 -m pytest tests/test_no_wing_release_check.py::test_no_wing_release_check_validates_wing_runner_physical_readback_policy tests/test_no_wing_release_check.py::test_no_wing_release_check_rejects_wing_runner_cache_as_physical tests/test_wing_mvp_rails.py::test_supervised_fader_gain_proof_stays_blocked_until_m002_and_operator_approval -q`: `3 passed`
- `python3 -m py_compile tools/wing_operator_proposal_validation_runner.py backend/supervised_wing_proof_plan.py tools/no_wing_release_check.py`: passed
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick`: passed

## Boundary

Still blocked:

- live fader apply
- live gain/trim apply
- EQ, compressor, gate, FX, pan, routing, scene, mute, DCA, matrix, master writes

Allowed now:

- read-only evidence review
- operator decision recording
- dry-run planning
- fail-closed pre-apply validation
