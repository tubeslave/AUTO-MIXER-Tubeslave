# ADR: Recovery policy contract

## Context

M012 requires an approved recovery policy before simulator fault-recovery
promotion, and the Project Completion Gate requires recovery, rollback,
emergency-stop and audit behavior before any completion claim. The repo had a
fault-recovery evidence contract, but the policy itself was still represented
as a loose approval field.

## Options Considered

1. Keep recovery approval as a loose field inside fault-recovery manifests.
2. Add a focused offline recovery-policy contract.
3. Treat complete recovery policy candidates as approved recovery policies.

## Decision

Use option 2. Add `backend/recovery_policy_contract.py` as a fail-closed
offline contract for candidate recovery policies.

## Why This Won

The focused contract makes recovery policy readiness auditable without
granting live authority. It separates complete candidate evidence ready for
owner review from owner-approved recovery policy evidence that future
fault-recovery promotion bindings can consume.

## Rejected Alternatives

Loose fault-recovery manifest fields were rejected because they hide missing
controls, missing evidence bindings and missing approval fields. Auto-approving
complete candidate recovery policies was rejected because recovery policy
selection depends on owner, operator and safety review outside unattended
automation.

## Implementation Plan

1. Add `evaluate_recovery_policy_contract`.
2. Require recovery controls, recovery evidence bindings and owner approval
   fields.
3. Return `ready_for_owner_review` separately from
   `ready_for_recovery_policy_use`.
4. The contract always returns `live_authority_granted: false` and
   `readiness_closure_allowed: false`.
5. Bind M016 through roadmap, capability, current-state and autopilot reports.

## Test Plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_recovery_policy_contract.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m016_recovery_policy_contract_stays_owner_approved_and_offline -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk: a future caller may mistake `ready_for_owner_review` for approved policy
use. Mitigation: the contract returns separate booleans and keeps
`ready_for_recovery_policy_use` false until approval fields exist.

Rollback: revert the M016 commit. This removes the recovery-policy contract,
tests and documentation bindings without touching live transport or runtime
mixer behavior.
