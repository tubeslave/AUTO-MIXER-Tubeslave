# Proposal: product-layer-read-surface-proof-quality-20260605

## 1. Short thesis

Propagate no-WING read-surface proof-quality metadata into the Product Layer and
Settings status row so operators can see why Full Autonomous read awareness is
still blocked.

## 2. What I understood

The release summary now contains useful proof-quality facts, but cockpit
consumers only see supervised proof artifact freshness and overlay status. This
leaves read-surface evidence quality hidden unless someone opens `summary.json`
manually.

## 3. Proposed solution

When reading a supervised proof report artifact, look for sibling
`summary.json`. If present, sanitize `read_surface_fixture_evidence` into a
small read-only `read_surface_proof_quality` object under
`supervised_proof_evidence.summary`. Then include the same facts in frontend
module status pipeline profiles and next-step text.

## 4. Likely files to touch

- `backend/operator_product_state.py`
- `tests/test_operator_product_state.py`
- `frontend/src/services/operatorAdapters.ts`
- `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `.ai/briefs/product-layer-read-surface-proof-quality-20260605.md`
- `Docs/adr/product-layer-read-surface-proof-quality-20260605.md`

## 5. Alternatives considered

- Parse the release summary only in frontend. Rejected because the Product Layer
  owns backend artifact normalization.
- Require summary.json before showing proof evidence. Rejected because older
  proof reports should remain displayable as supervised artifact evidence.
- Promote read surfaces based on summary flags. Rejected because flags prove
  validation, not physical readback.

## 6. Risks

Operators might confuse proof-quality visibility with readiness. Mitigation:
include physical gap count, keep rows read-only, and keep Full Autonomous
blocked.

## 7. Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py -q`
- `node frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend test`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## 8. Where the other agent may disagree

Kimi may prefer adding a dedicated read-surface proof row. I would first enrich
the existing source row to keep source availability and target readiness
separated without increasing UI surface area.
