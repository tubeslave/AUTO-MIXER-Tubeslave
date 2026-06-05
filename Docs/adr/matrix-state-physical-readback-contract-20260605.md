# Matrix State Physical Readback Contract

## Context

Full autonomous console awareness requires physical readback evidence for every
console surface. `matrix_state` already exposes cached Product Layer DTOs, but
matrix controls, EQ, dynamics, inserts, and metering cannot be treated as
physical proof until an operator-run artifact validates them.

## Options considered

1. Add a dedicated matrix-state physical readback contract.
2. Reuse the bus/group/main/DCA contract for matrix rows.
3. Wait for a physical readback runner before defining the artifact schema.

## Decision

Use option 1. Add `matrix_state_physical_readback_contract` as a schema-only,
transport-free contract.

## Why this won

Matrix state has different proof needs from bus/main/DCA: 8-band matrix EQ,
matrix dynamics, pre/post inserts, and matrix metering. A dedicated contract
makes those requirements visible without creating live authority.

## Rejected alternatives

Reusing the bus/main/DCA contract would hide matrix-specific EQ/dynamics/insert
requirements. Waiting for a hardware runner would leave future physical
artifacts underspecified.

## Implementation plan

- Add backend contract and validator.
- Add fixture-evidence/remediation expected artifacts.
- Add processing-registry and goal-capability references.
- Add no-WING release validation and summary embedding.
- Add Product Layer and frontend proof-quality visibility.
- Add focused tests for contract validity, drift, and safety normalization.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_matrix_state_physical_readback_contract.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_read_surface_fixture_evidence.py tests/test_read_surface_remediation_plan.py tests/test_processing_registry.py tests/test_no_wing_release_check.py tests/test_operator_product_state.py -q`
- `npm --prefix frontend test -- --watch=false`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk: schema readiness could be misread as completed physical proof. Mitigation:
the contract, no-WING summary, Product Layer, and frontend all keep
`physical_fixture_gap_remains=true`, `live_authority_granted=false`, and
`full_autonomous_ready=false`. Rollback is to remove the matrix contract wiring
and return `matrix_state` to cached DTO evidence only.
