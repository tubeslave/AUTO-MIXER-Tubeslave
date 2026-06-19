# ADR: Integration policy contract

## Context

M013 requires an approved integration policy before offline integration
promotion, and the Project Completion Gate requires end-to-end offline, replay,
fault-injection and integration evaluations. The repo had replay, recovery,
threshold and fault-recovery policy contracts, but integration policy evidence
itself was still represented as a loose approval field.

## Options Considered

1. Keep integration approval as a loose field inside integration bundles.
2. Add a focused offline integration-policy contract.
3. Treat complete integration bundle evidence as approved integration policy.

## Decision

Use option 2. Add `backend/integration_policy_contract.py` as a fail-closed
offline contract for candidate integration policies.

## Why This Won

The focused contract makes integration policy readiness auditable without
granting live authority. It separates complete candidate evidence ready for
owner review from owner-approved integration policy evidence that future
integration bundle promotion bindings can consume.

## Rejected Alternatives

Loose integration bundle fields were rejected because they hide missing
controls, missing child-result bindings, missing reproducibility evidence and
missing approval fields. Auto-approving complete integration bundles was
rejected because bundle completeness proves evidence assembly, not owner
approval of the integration policy.

## Implementation Plan

1. Add `evaluate_integration_policy_contract`.
2. Require integration controls, integration evidence bindings and owner
   approval fields.
3. Return `ready_for_owner_review` separately from
   `ready_for_integration_policy_use`.
4. The contract always returns `live_authority_granted: false` and
   `readiness_closure_allowed: false`.
5. Bind M018 through roadmap, capability, current-state and autopilot reports.

## Test Plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_integration_policy_contract.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m018_integration_policy_contract_stays_owner_approved_and_offline -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk: a future caller may mistake `ready_for_owner_review` for approved policy
use. Mitigation: the contract returns separate booleans and keeps
`ready_for_integration_policy_use` false until approval fields exist.

Rollback: revert the M018 commit. This removes the integration-policy contract,
tests and documentation bindings without touching live transport or runtime
mixer behavior.
