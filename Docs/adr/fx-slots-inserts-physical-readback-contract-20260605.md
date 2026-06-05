# FX Slots/Inserts Physical Readback Contract

## Context

Full console read awareness needs physical readback evidence for FX slots and
insert assignments. `fx_slots_and_inserts` already has cached Product Layer DTOs
for FX slot identity, model hints, parameters, and insert state, but cached
evidence cannot be treated as physical proof.

## Options considered

1. Add a dedicated FX slots/inserts physical readback contract.
2. Reuse matrix insert contract fields.
3. Wait for a physical readback runner before defining the artifact schema.

## Decision

Use option 1. Add `fx_slots_inserts_physical_readback_contract` as a schema-only,
transport-free contract.

## Why this won

FX slots require proof of slot identity, model-specific parameter schema,
parameter values, and scoped insert assignment state. Matrix insert proof does
not cover FX model/parameter requirements. A dedicated contract makes the
requirements visible without creating live authority.

## Rejected alternatives

Reusing matrix insert fields would hide FX parameter schema gaps. Waiting for a
hardware runner would leave future artifacts underspecified and harder to
review safely.

## Implementation plan

- Add backend contract and validator.
- Add fixture-evidence/remediation expected artifacts.
- Add processing-registry and goal-capability references.
- Add no-WING release validation and summary embedding.
- Add Product Layer and frontend proof-quality visibility.
- Add focused tests for contract validity, drift, incomplete rows, and safety
  normalization.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_fx_slots_inserts_physical_readback_contract.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_read_surface_fixture_evidence.py tests/test_read_surface_remediation_plan.py tests/test_processing_registry.py tests/test_no_wing_release_check.py tests/test_operator_product_state.py -q`
- `npm --prefix frontend test -- --watch=false`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk: schema readiness could be misread as completed physical proof. Mitigation:
the contract, no-WING summary, Product Layer, and frontend all keep
`physical_fixture_gap_remains=true`, `live_authority_granted=false`, and
`full_autonomous_ready=false`. Rollback is to remove the FX contract wiring and
return `fx_slots_and_inserts` to cached DTO evidence only.
