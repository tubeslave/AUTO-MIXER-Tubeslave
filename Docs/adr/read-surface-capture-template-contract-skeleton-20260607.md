# ADR: Read-Surface Capture Template Contract Skeleton

## Context

The read-surface capture-template artifact gives operators a pending record to
fill before artifact-review queue intake. It includes the required contract id,
but the actual `physical_readback_payload` field is still a generic placeholder.
That leaves the precise JSON payload shape implicit.

## Options considered

1. Keep the generic payload placeholder only.
2. Generate separate contract-specific skeleton artifacts.
3. Embed a contract-derived payload skeleton inside each pending capture record.

## Decision

Use option 3. Each pending capture record now includes
`physical_readback_payload_template`, derived from the target's required
physical readback contract.

## Why this won

The capture-template artifact is already the operator's next editing surface.
Embedding a skeleton keeps the expected payload shape next to the capture target,
while the no-WING validator can still reject any attempt to treat the skeleton as
live authority, fixture candidacy, or completed proof.

## Rejected alternatives

- Generic placeholder only keeps the contract shape hidden from the operator.
- Separate skeleton artifacts add another handoff file without a stronger safety
  boundary.
- Treating the skeleton as a candidate artifact would blur pending-template and
  physical-proof states.

## Implementation plan

1. Resolve each record's `required_contract_method_id` through the existing
   physical readback contract builders.
2. Copy artifact type, schema id, required top-level fields, and nested required
   field sets into the template.
3. Build a placeholder payload from those fields, preserving pending operator
   values and nested read-only safety flags.
4. Extend no-WING artifact validation and summary output for the nested payload
   template.
5. Cover builder, unsafe-payload, and summary behavior in focused tests.

## Test plan

- `python3 -m py_compile tools/build_read_surface_physical_readback_capture_template.py tools/no_wing_release_check.py`
- Focused capture-template/no-WING pytest.
- Broader read-surface/no-WING pytest.
- no-WING quick run.
- Memory lint and diff check.
- Full backend pytest.

## Risks and rollback

This change does not contact a mixer, accept physical readback, update fixtures,
close read-surface gaps, or widen live authority. Rollback is to remove the
nested payload skeleton, validator checks, summary fields, tests, and docs row;
the console transport path is unaffected.
