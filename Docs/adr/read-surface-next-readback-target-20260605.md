# ADR: read-surface-next-readback-target-20260605

## Context

Read-surface proof quality exposes physical gap IDs, but downstream consumers
must infer the next hardware proof target from list order. That inference should
belong to Product Layer normalization so UI surfaces can display one stable
read-only field.

## Options considered

1. Add `next_physical_readback_surface_id` to Product Layer proof quality.
2. Keep deriving the target in Settings.
3. Build a ranked remediation workflow now.

## Decision

Use option 1. Product Layer will expose the first missing physical readback
surface as `next_physical_readback_surface_id`; frontend will render it as
`Next read proof`.

## Why this won

It makes the next proof target explicit without adding live hardware access,
write authority, or a premature remediation workflow.

## Rejected alternatives

Frontend-only derivation duplicates Product Layer artifact logic. A ranked
remediation workflow should wait until the physical proof process is specified.

## Implementation plan

1. Add the normalized field in `_read_surface_proof_quality_from_summary`.
2. Extend backend proof-quality tests.
3. Carry the field through frontend adapter profiles and next-step text.
4. Render the field in Settings and assert it in e2e.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py -q`
- `node frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend test`
- `npm --prefix frontend run build`
- `npm --prefix frontend run test:e2e -- automixer-prototype.spec.ts`
- `python3 tools/memory/memory_lint.py`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk: a displayed next target could be mistaken for permission to query or
write hardware. The field remains inside read-only proof evidence and does not
change live authority.

Rollback is removing the normalized field, adapter profile, Settings row, tests,
and this ADR.
