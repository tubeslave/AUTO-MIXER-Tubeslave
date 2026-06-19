# ADR: Autonomy Readiness Integration Bundle Overlay

## Context

M011-M025 define and bind offline replay, recovery, policy, safety-kernel,
no-live-transport and operator-report contracts into the offline integration
bundle. `backend/autonomy_promotion_readiness.py` previously reported
promotion readiness gaps without consuming that bundle status.

M001 still waits for a real WING read-only artifact and operator review. This
decision is offline/control-plane only.

## Options Considered

1. Leave autonomy promotion readiness independent from the integration bundle.
2. Add a read-only integration bundle overlay to the readiness snapshot.
3. Treat passing offline integration bundle evidence as readiness closure.

## Decision

Add a read-only `offline_integration_bundle` overlay to autonomy promotion
readiness. The overlay evaluates an optional
`offline_integration_bundle_manifest` through the existing integration bundle
contract and surfaces `offline_integration_bundle_contract_not_ready` when it
is missing or blocked.

The overlay always returns `live_authority_granted: false`.

## Why This Won

Autonomy promotion planning should not ignore the offline integration evidence
gates already built in M011-M025. Reusing the existing integration bundle
contract keeps authority and evidence semantics centralized while keeping the
readiness snapshot report-only.

## Rejected Alternatives

Leaving readiness independent from integration evidence was rejected because it
lets promotion planning proceed without showing whether integration evidence is
contract-ready.

Treating a passing offline bundle as readiness closure was rejected because
offline, replay, simulator and report evidence cannot replace real hardware,
listening, expert, owner, rollback or emergency-stop proof.

## Implementation Plan

- Import and call `evaluate_integration_bundle_contract()` from
  `backend/autonomy_promotion_readiness.py`.
- Add `offline_integration_bundle` fields to the readiness snapshot summary.
- Add the global blocker
  `offline_integration_bundle_contract_not_ready` when the bundle is missing or
  blocked.
- Add tests for missing, passing and live-claiming bundle manifests.
- Update roadmap, completion gate, capability matrix, current state and
  autopilot evidence.

## Test Plan

- `PYTHONPATH=backend python -m pytest tests/test_autonomy_promotion_readiness.py -q`
- `PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py::test_m026_autonomy_readiness_consumes_integration_bundle_status -q`
- `PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk: readiness snapshots without an integration bundle now expose an additional
blocker. This is intended and keeps promotion planning fail-closed.

Rollback: revert the M026 overlay and related tests/docs. M011-M025 integration
contracts remain intact.
