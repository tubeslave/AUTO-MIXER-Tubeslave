# ADR: Quality threshold policy contract

## Context

M011 requires approved quality thresholds before replay-level promotion, and
the Project Completion Gate requires quality evaluation against approved
thresholds before any autonomous-completion claim. The repo had replay,
fault-recovery and integration bundle contracts, but no focused contract for
the threshold policy evidence itself.

## Options Considered

1. Keep threshold approval as a loose field inside replay manifests.
2. Add a focused offline threshold-policy contract.
3. Treat complete candidate thresholds as approved thresholds.

## Decision

Use option 2. Add `backend/quality_threshold_policy_contract.py` as a
fail-closed offline contract for candidate threshold policies.

## Why This Won

The focused contract makes threshold readiness auditable without granting live
authority. It separates two states that must not be conflated: complete
candidate evidence ready for owner review, and owner-approved thresholds ready
for replay or integration promotion.

## Rejected Alternatives

Loose replay-manifest fields were rejected because they hide missing axes,
missing policy checks and missing approval fields. Auto-approving complete
candidate thresholds was rejected because threshold selection depends on owner,
listening and expert judgment outside unattended automation.

## Implementation Plan

1. Add `evaluate_quality_threshold_policy_contract`.
2. Require every replay quality axis, policy checks, evidence artifacts,
   rationale and rollback policy.
3. Return `ready_for_owner_review` separately from
   `ready_for_threshold_policy_use`.
4. The contract always returns `live_authority_granted: false` and
   `readiness_closure_allowed: false`.
5. Bind M014 through roadmap, capability, current-state and autopilot reports.

## Test Plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_quality_threshold_policy_contract.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m014_quality_threshold_policy_contract_stays_owner_approved_and_offline -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk: a future caller may mistake `ready_for_owner_review` for approval.
Mitigation: the contract returns separate booleans and keeps
`ready_for_threshold_policy_use` false until approval fields exist.

Rollback: revert the M014 commit. This removes the threshold-policy contract,
tests and documentation bindings without touching live transport or runtime
mixer behavior.
