# Proposal: promotion-readiness-evidence-requirements-20260605

## 1. Short thesis

Expose promotion evidence requirements as machine-readable IDs in the readiness
snapshot so autonomous planning can reason about missing proof without parsing
free-form text.

## 2. What I understood

The project goal requires explainable, predictable automation. Current readiness
rows have blockers and human-readable next steps, but not stable proof
requirement keys. Scene/preset work now has explicit evidence contracts, so the
promotion layer should name them directly.

## 3. Proposed solution

Add `evidence_requirement_ids` to `PromotionTarget` and
`PromotionReadinessRow`. Populate the new field for broad action families where
the missing proof is already known, especially:

- scene/preset: Product Layer snapshot freshness, read-only scoped diff,
  read-only rollback evidence, physical fixture proof, hardware readback and
  rollback proof.
- routing: current route freshness, route diff evidence, source identity trust,
  physical fixtures, hardware readback and rollback proof.

## 4. Likely files to touch

- `backend/autonomy_promotion_readiness.py`
- `tests/test_autonomy_promotion_readiness.py`
- `.ai/briefs/promotion-readiness-evidence-requirements-20260605.md`
- `Docs/adr/promotion-readiness-evidence-requirements-20260605.md`

## 5. Alternatives considered

- Keep free-form text only. Rejected because agents cannot reliably gate future
  promotion from prose.
- Add a separate evidence registry module. Rejected for this slice because the
  readiness target already owns the action-family boundary.
- Change UI now. Deferred unless tests show the adapter requires it.

## 6. Risks

Evidence IDs could be mistaken for proof completion. Mitigation: the row still
has blockers, safety status, read/write surface statuses, and
`supervised_live_allowed=False` for unsafe families.

## 7. Test plan

- Focused readiness tests.
- Registry snapshot tests that include readiness.
- Standard pytest command.

## 8. Where the other agent may disagree

Kimi may prefer a full evidence ontology first. I would land stable IDs now and
only extract an ontology after more action families need richer relationships.
