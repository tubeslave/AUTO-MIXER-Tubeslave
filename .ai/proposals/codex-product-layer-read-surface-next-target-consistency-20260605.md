# Proposal: product-layer-read-surface-next-target-consistency-20260605

## 1. Short thesis

Product Layer should validate no-WING next-readback target consistency before
showing it to the operator cockpit.

## 2. What I understood

The release artifact now owns `next_physical_readback_surface_id`. If Product
Layer ignores that source value and derives its own target, cockpit evidence can
look clean even when an artifact is malformed or stale.

## 3. Proposed solution

Read the source target from sibling no-WING `summary.json`, compare it with the
first `physical_fixture_missing_surface_ids` entry, and expose both the source
target and `next_physical_readback_target_consistent`. Keep the displayed target
as a safe fallback to the first physical gap when the source is missing or
mismatched. Surface the consistency flag in frontend module profiles and the
Settings inspector.

## 4. Likely files to touch

- `backend/operator_product_state.py`
- `tests/test_operator_product_state.py`
- `frontend/src/services/operatorAdapters.ts`
- `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `frontend/src/views/SettingsView.tsx`
- `frontend/e2e/automixer-prototype.spec.ts`
- `.ai/briefs/product-layer-read-surface-next-target-consistency-20260605.md`
- `Docs/adr/product-layer-read-surface-next-target-consistency-20260605.md`
- `.ai/memory/project.md`

## 5. Alternatives considered

- Trust the source field blindly. Rejected because artifact drift should be
  visible.
- Ignore the source field. Rejected because no-WING summary is now the proof
  evidence source.
- Block display entirely on mismatch. Rejected because showing the fallback
  target plus mismatch is more actionable.

## 6. Risks

Operators may misread a fallback target as validated. Mitigation: Settings shows
`Read target gate` separately.

## 7. Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py -q`
- `node frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend test`
- `npm --prefix frontend run build`
- `npm --prefix frontend run test:e2e -- automixer-prototype.spec.ts`
- `python3 tools/memory/memory_lint.py`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## 8. Where the other agent may disagree

Kimi may prefer failing the Product Layer snapshot entirely on mismatch. I would
keep the snapshot available but explicitly mark the consistency gate as failed
so operators can still see remediation evidence.
