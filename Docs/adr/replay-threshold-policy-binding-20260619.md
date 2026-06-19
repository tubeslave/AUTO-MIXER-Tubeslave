# ADR: Replay threshold policy binding

## Context

M014 introduced a focused quality threshold policy contract, but M011 replay
evaluation still accepted a loose `threshold_policy` status object. That left a
gap: replay promotion could appear ready without the complete threshold axes,
policy checks, evidence artifacts, rollback policy and owner approval required
by the M014 contract.

## Options Considered

1. Keep M014 as documentation-only guidance for replay manifests.
2. Bind M011 replay evaluation to the M014 contract result.
3. Merge the threshold policy contract into the replay evaluator.

## Decision

Use option 2. `backend/offline_replay_evaluation_contract.py` now consumes
`quality_threshold_policy_manifest` and requires
`ready_for_threshold_policy_use` before replay promotion.

## Why This Won

The binding removes the loose-status bypass while preserving separate concerns:
M014 owns threshold-policy evidence quality, and M011 owns replay scenario,
safety and quality-axis coverage.

## Rejected Alternatives

Documentation-only guidance was rejected because it would not fail closed.
Merging M014 into M011 was rejected because threshold policy evidence is also
needed by integration and completion gates.

## Implementation Plan

1. Evaluate `quality_threshold_policy_manifest` from the replay manifest.
2. Block replay promotion when M014 reports anything other than
   `ready_for_threshold_policy_use`.
3. Prefix threshold-policy blockers as `quality_threshold_policy:*`.
4. Keep legacy threshold status visible only as summary context, not approval.
5. The contract always returns `live_authority_granted: false` and
   `readiness_closure_allowed: false`.

## Test Plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_offline_replay_evaluation_contract.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_evaluation_integration_bundle_contract.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m015_replay_threshold_policy_binding_requires_m014_manifest -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk: older replay fixtures that only include loose `threshold_policy` fields
will now fail. This is intentional: they are incomplete evidence after M014.

Rollback: revert the M015 commit. This restores the previous loose threshold
approval behavior without touching live transport or runtime mixer behavior.
