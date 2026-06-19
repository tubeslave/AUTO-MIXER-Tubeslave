# ADR: Fault recovery policy binding

## Context

M016 introduced a focused recovery-policy contract, but M012 fault-recovery
promotion still accepted a loose `recovery_policy` status object. That left the
same class of ambiguity M015 removed from replay threshold policy: a manifest
could look approved without carrying the evidence-bound policy contract result.

## Options Considered

1. Keep accepting loose `recovery_policy` fields inside fault-recovery
   manifests.
2. Require `recovery_policy_manifest` to pass the M016 contract before
   simulator fault-recovery promotion.
3. Treat complete fault scenarios as sufficient recovery-policy approval.

## Decision

Use option 2. `backend/fault_injection_recovery_contract.py` now consumes the
M016 recovery-policy contract result before reporting simulator fault-recovery
promotion readiness.

## Why This Won

The binding keeps recovery-policy approval evidence explicit and auditable. It
also lets integration bundles surface recovery-policy failures through the
existing `fault_recovery:` child prefix instead of burying them behind a loose
status field.

## Rejected Alternatives

Loose manifest fields were rejected because they can omit required controls,
evidence bindings and approval fields. Auto-approving complete fault scenarios
was rejected because scenario evidence proves recovery behavior, not policy
approval.

## Implementation Plan

1. Evaluate `recovery_policy_manifest` with `evaluate_recovery_policy_contract`.
2. Keep legacy `recovery_policy` only as summary context.
3. Block promotion with `recovery_policy_contract_not_ready` unless the M016
   result is ready for recovery policy use.
4. Prefix recovery-policy blockers for integration bundle consumers.
5. The contract always returns `live_authority_granted: false` and
   `readiness_closure_allowed: false`.

## Test Plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_fault_injection_recovery_contract.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_evaluation_integration_bundle_contract.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m017_fault_recovery_policy_binding_requires_m016_manifest -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk: existing simulator manifests that only carry loose `recovery_policy`
fields will now fail closed. Mitigation: the blocker explicitly names
`legacy_recovery_policy_without_contract_manifest` and points callers to the
M016 manifest shape.

Rollback: revert the M017 commit. This restores the previous loose-policy
behavior without touching live transport or runtime mixer behavior.
