# ADR: read-surface-remediation-physical-target-20260605

## Context

`read_surface_fixture_evidence` reports physical readback gaps and the next
surface to prove. `read_surface_remediation_plan` previously used only inventory
status, so a surface with cached DTO coverage could be absent from remediation
even when physical proof was missing.

## Options considered

1. Merge fixture physical gaps into remediation actions.
2. Keep remediation status-only.
3. Change cached inventory rows to partial.

## Decision

Use option 1. The plan creates actions for inventory-incomplete surfaces and
for any surface with `physical_readback_fixture_missing`.

## Why this won

It keeps cache coverage and physical proof separate while making the next
readback task visible to Product Layer and release evidence.

## Rejected alternatives

Status-only remediation hides physical proof gaps. Changing cached rows to
partial blurs the difference between cached read support and missing hardware
proof.

## Implementation plan

1. Read fixture evidence inside the remediation builder by default.
2. Add physical fixture metadata and next-target markers to actions.
3. Expose summary fields for physical gap IDs and next action ID.
4. Add tests for `raw_mixer_state_cache` as the next physical target.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_read_surface_remediation_plan.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_read_surface_inventory.py tests/test_read_surface_fixture_evidence.py tests/test_processing_registry.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk: physical fixture actions may be mistaken for live readback. The payload
continues to set `read_only=true`, `live_write=false`,
`mutates_console_state=false`, and `full_autonomous_ready=false`.

Rollback is removing the fixture-evidence merge fields, tests, and this ADR.
