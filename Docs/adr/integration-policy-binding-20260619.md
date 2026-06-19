# ADR: Integration policy binding

## Context

M018 introduced a focused integration-policy contract, but M013 integration
bundle promotion still accepted a loose `integration_policy` status object.
That left the same bypass class removed by M015 and M017: a bundle could look
approved without carrying the evidence-bound policy contract result.

## Options Considered

1. Keep accepting loose `integration_policy` fields inside integration bundles.
2. Require `integration_policy_manifest` to pass the M018 contract before
   offline integration promotion.
3. Treat complete integration checks as sufficient policy approval.

## Decision

Use option 2. `backend/evaluation_integration_bundle_contract.py` now consumes
the M018 integration-policy contract result before reporting offline integration
promotion readiness.

## Why This Won

The binding keeps integration-policy approval evidence explicit and auditable.
It also lets integration bundle consumers see policy failures through the
`integration_policy:` child prefix instead of trusting a loose status field.

## Rejected Alternatives

Loose manifest fields were rejected because they can omit required integration
controls, evidence bindings and approval fields. Auto-approving complete
integration checks was rejected because check coverage proves bundle evidence,
not policy approval.

## Implementation Plan

1. Evaluate `integration_policy_manifest` with
   `evaluate_integration_policy_contract`.
2. Keep legacy `integration_policy` only as summary context.
3. Block promotion with `integration_policy_contract_not_ready` unless the M018
   result is ready for integration policy use.
4. Prefix integration-policy blockers for integration bundle consumers.
5. The contract always returns `live_authority_granted: false` and
   `readiness_closure_allowed: false`.

## Test Plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_evaluation_integration_bundle_contract.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_integration_policy_contract.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m019_integration_policy_binding_requires_m018_manifest -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk: existing integration bundles that only carry loose `integration_policy`
fields will now fail closed. Mitigation: the blocker explicitly names
`legacy_integration_policy_without_contract_manifest` and points callers to the
M018 manifest shape.

Rollback: revert the M019 commit. This restores the previous loose-policy
behavior without touching live transport or runtime mixer behavior.
