# Routing/Sends Physical Readback Contract - 2026-06-05

## Context

Routing and send state currently has partial cached/static DTO coverage, but no
physical readback artifact contract. Promotion families that depend on routing,
FX sends/inserts, bus/matrix balance, scene/preset, mix-balance, or performance
adaptation need a stable proof schema before hardware evidence can be trusted.

## Options considered

1. Trust cached/static routing and send DTOs as proof.
2. Add live routing/sends console queries now.
3. Add a read-only physical readback contract and keep the physical fixture gap
   open.

## Decision

Choose option 3. Define a transport-free routing/sends physical readback
contract and surface it through no-WING, Product Layer, frontend status, the
processing registry, and the console-read goal matrix.

## Why this won

It advances proof readiness without changing live mixer behavior. The contract
names required static routes, bus/main/matrix sends, output routes, coverage
flags, minimum counts, accepted physical sources, and forbidden mutation fields
while preserving `physical_fixture_gap_remains=true`.

## Rejected alternatives

Cached/static DTO promotion was rejected because cache parity is not hardware
proof. Live console read work was rejected for this slice because transport and
operator-supervised hardware execution need a separate safety review.

## Implementation plan

- Add the routing/sends contract builder and artifact validator.
- Register the contract as a shadow-only analysis method.
- Include it in no-WING schema validation and summary output.
- Normalize it in Product Layer proof quality and expose it in Module Status.
- Update fixture evidence, remediation planning, goal support paths, and tests.

## Test plan

- `PYTHONPATH=backend python3 -m py_compile ...`
- Focused pytest for routing/sends contract, registry, fixture evidence,
  remediation plan, no-WING, and Product Layer proof quality.
- Frontend adapter tests and frontend build.
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend`
- `python3 tools/memory/memory_lint.py`
- Standard backend pytest.

## Risks and rollback

Risk: operators may misread contract availability as physical proof. Mitigation:
all surfaced rows preserve `physical_fixture_gap_remains=true`, `live_write=false`,
and `live_authority_granted=false`.

Rollback: remove the contract method, no-WING summary/check wiring, Product Layer
normalizer row, frontend status row, and related tests/docs. No live transport
behavior or mixer write path is introduced by this ADR.
