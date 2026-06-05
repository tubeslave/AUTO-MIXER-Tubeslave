# FX Slots/Inserts Physical Readback Artifact Intake

## Context

FX slots/inserts physical readback now has a schema contract, but future
operator-captured artifacts need a safe intake path. Cached mixer DTOs and JSON
files are not live proof by themselves.

## Options considered

1. Add a read-only artifact intake that validates JSON and exposes manual-review
   candidates.
2. Treat valid JSON as fixture closure.
3. Wait for a hardware capture runner.

## Decision

Use option 1. Add `fx_slots_inserts_physical_readback_artifact_intake` as a
transport-free artifact intake.

## Why this won

The intake makes operator evidence visible and drift-checked without turning it
into live authority. It also keeps the schema contract, no-WING gate, Product
Layer, and frontend aligned around the same fail-closed status.

## Rejected alternatives

Treating valid JSON as fixture closure would overstate proof quality and could
hide stale or wrong-console captures. Waiting for a hardware runner would leave
artifact handling undefined and make future review more ambiguous.

## Implementation plan

- Build the intake module around the existing FX slots/inserts contract
  validator.
- Add registry and goal-capability references.
- Add release-gate validation and summary embedding.
- Add Product Layer proof-quality normalization.
- Add frontend Module Status evidence.
- Add focused backend/frontend tests.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_fx_slots_inserts_physical_readback_artifact_intake.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_no_wing_release_check.py tests/test_operator_product_state.py -q`
- `node --test frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk: valid artifacts could be misread as fixture closure or live readiness.
Mitigation: intake, release summary, Product Layer, and frontend all force
`fixture_closure_allowed_by_intake=false`, `physical_fixture_gap_remains=true`,
`live_authority_granted=false`, and `full_autonomous_ready=false`. Rollback is
to remove the intake wiring while keeping the schema-only FX contract.
