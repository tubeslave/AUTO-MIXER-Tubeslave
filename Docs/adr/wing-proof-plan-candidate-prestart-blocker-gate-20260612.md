# WING Proof Plan Candidate Prestart Blocker Gate

## Context

The WING proof plan requires accepted read-only preflight decisions to match `candidate_reviews` rows carrying the backend-port prestart contract. Without a specific blocker, an operator could see a generic manual-review-ready state and miss the concrete remediation.

## Options considered

1. Keep only the generic accepted-review-decision blocker.
2. Require a specific candidate-prestart blocker when accepted source ids exist but the candidate prestart trace is invalid.
3. Reject all manual-review-ready proof plans with accepted source ids.

## Decision

Use option 2. The no-WING release gate now fails when accepted source ids exist, candidate prestart contract validation is false, and the proof plan omits `accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof`.

## Why this won

It preserves a valid blocked state while making the operator remediation explicit. It also keeps the boundary read-only and does not add any WING transport, backend startup, or live-write behavior.

## Rejected alternatives

- Generic-only blocking was rejected because it weakens the handoff.
- Rejecting all manual-review-ready plans with accepted source ids was too broad; accepted ids can be present while the plan is correctly blocked pending prestart trace repair.

## Implementation plan

- Add a no-WING release-check blocker constant.
- Enforce the blocker in `validate_supervised_proof_report_next_wing_plan`.
- Add tests for missing blocker failure and valid blocked state.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py -k "next_wing_plan_without_candidate_prestart_trace or next_wing_plan_blocked_by_candidate_prestart_trace or validates_next_wing_proof_plan_schema"`
- Affected WING/no-WING tests.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`

## Risks and rollback

The main risk is stricter validation for older report snapshots. Rollback is safe by removing the blocker requirement and the two regression assertions. No live behavior or console authority is affected.
