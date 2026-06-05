# Proposal: read-surface-next-readback-target-20260605

## 1. Short thesis

Make the next read-surface hardware proof target explicit in Product Layer and
Settings.

## 2. What I understood

The project is closing the gap between cached read DTO coverage and trusted
physical console readback. Operators can now see gap counts and IDs, but the
next proof target is still implicit.

## 3. Proposed solution

Normalize `next_physical_readback_surface_id` from the first
`physical_fixture_missing_surface_ids` entry in the Product Layer proof-quality
object. Carry it through frontend module status profiles and render it in the
Settings Module Status inspector as read-only evidence.

## 4. Likely files to touch

- `backend/operator_product_state.py`
- `tests/test_operator_product_state.py`
- `frontend/src/services/operatorAdapters.ts`
- `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `frontend/src/views/SettingsView.tsx`
- `frontend/e2e/automixer-prototype.spec.ts`
- `.ai/briefs/read-surface-next-readback-target-20260605.md`
- `Docs/adr/read-surface-next-readback-target-20260605.md`
- `.ai/memory/project.md`

## 5. Alternatives considered

- Let the frontend keep deriving the target from the first gap ID. Rejected
  because Product Layer should own artifact normalization.
- Add a remediation workflow now. Rejected because this slice is about evidence
  visibility, not hardware proof execution.
- Use a hardcoded priority list. Rejected until physical proof strategy is
  agreed and documented.

## 6. Risks

Operators may treat the target as an executable instruction. Mitigation: keep it
inside proof evidence, label it as read proof, and keep all rows read-only.

## 7. Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py -q`
- `node frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend test`
- `npm --prefix frontend run build`
- `npm --prefix frontend run test:e2e -- automixer-prototype.spec.ts`
- `python3 tools/memory/memory_lint.py`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## 8. Where the other agent may disagree

Kimi may prefer a ranked priority algorithm. I would first make the existing
release evidence explicit and only add ranking once hardware proof workflows
exist.
