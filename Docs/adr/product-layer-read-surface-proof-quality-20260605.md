# ADR: product-layer-read-surface-proof-quality-20260605

## Context

No-WING `summary.json` exposes read-surface proof-quality facts after the
release gate validates summary recomputation, fixture declarations, and
contract/gap consistency. The Product Layer supervised proof evidence snapshot
does not currently surface those facts to Settings.

## Options considered

1. Read sibling `summary.json` in the Product Layer and expose sanitized
   read-surface proof quality.
2. Let frontend parse no-WING summary artifacts directly.
3. Require summary proof quality before displaying supervised proof artifacts.

## Decision

Use option 1. Product Layer will normalize sibling summary metadata and expose it
under the existing `supervised_proof_evidence.summary`.

## Why this won

It preserves backend ownership of artifact normalization, keeps UI rows
read-only, and makes proof quality visible without implying promotion readiness.

## Rejected alternatives

Frontend direct parsing duplicates artifact logic. Requiring `summary.json` would
break older proof artifacts and is unnecessary because supervised proof report
availability is separate from read-surface proof-quality metadata.

## Implementation plan

1. Add a helper to read sibling no-WING `summary.json`.
2. Normalize only read-only proof-quality fields.
3. Add the normalized object to `supervised_proof_evidence.summary`.
4. Extend frontend module status profiles and next-step text.
5. Add backend and frontend contract tests.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py -q`
- `node frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend test`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk: proof-quality flags could be mistaken for hardware proof. The exposed
metadata includes `physical_gap_surface_count` and keeps
`full_autonomous_ready=false`.

Rollback is removing the sibling-summary helper, adapter additions, and tests.
