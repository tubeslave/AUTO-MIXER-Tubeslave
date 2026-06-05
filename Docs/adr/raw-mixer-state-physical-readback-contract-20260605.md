# ADR: raw-mixer-state-physical-readback-contract-20260605

## Context

`raw_mixer_state_cache` is the next physical readback target in read-surface
remediation. The repo needs a stable artifact contract before a hardware proof
runner can produce evidence that release checks, Product Layer snapshots, and
operators can all interpret the same way.

## Options considered

1. Add a transport-free artifact contract and validator.
2. Wait for a physical readback runner before defining the artifact.
3. Treat the existing cached DTO fixture as physical proof.

## Decision

Use option 1. Add `raw_mixer_state_physical_readback_contract` as a shadow-only
evaluation method. It declares the future artifact schema and validates physical
readback artifacts without contacting hardware or granting runtime authority.

## Why this won

The remediation action becomes concrete while preserving the safety boundary:
schema readiness is not physical proof, and physical proof is not live write
permission.

## Rejected alternatives

Waiting for a runner leaves the next remediation action underspecified.
Treating cached DTO fixtures as physical proof would blur cache evidence with
hardware verification and could create unsafe readiness claims.

## Implementation plan

1. Add the contract builder and artifact validator.
2. Register the method as `shadow_only` and expose it in registry snapshots.
3. Reference the contract from fixture evidence and goal support paths without
   adding `physical_readback_fixture`.
4. Add the contract test file to the no-WING read-surface quick check.
5. Add focused tests for valid physical artifacts, cached/live rejection,
   registry visibility, and remediation expected artifacts.

## Test plan

- `PYTHONPATH=backend python3 -m py_compile backend/raw_mixer_state_physical_readback_contract.py tests/test_raw_mixer_state_physical_readback_contract.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_raw_mixer_state_physical_readback_contract.py tests/test_read_surface_fixture_evidence.py tests/test_read_surface_remediation_plan.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py tests/test_no_wing_release_check.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: the contract is mistaken for completed physical readback. Mitigation:
fixture evidence keeps `physical_readback_fixture_missing`, the contract summary
sets `physical_fixture_gap_remains=true`, and validation always reports
`live_authority_granted=false` and `full_autonomous_ready=false`.

Rollback is removing the contract module, registry/goal references, fixture
evidence additions, tests, no-WING quick-check inclusion, and this ADR.
