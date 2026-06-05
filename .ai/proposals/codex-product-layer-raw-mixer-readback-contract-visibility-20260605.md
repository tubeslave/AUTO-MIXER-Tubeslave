# Proposal: product-layer-raw-mixer-readback-contract-visibility-20260605

## 1. Short thesis

Surface raw mixer readback contract quality in Product Layer and cockpit so the
next physical-readback target is visible end to end.

## 2. What I understood

The release check now validates the raw mixer-state artifact contract, but the
UI-facing proof evidence still only exposes fixture/gap metadata. That leaves a
visibility gap between release artifacts and operator cockpit.

## 3. Proposed solution

Extend `_read_surface_proof_quality_from_summary()` to normalize
`raw_mixer_state_physical_readback_contract` from sibling no-WING `summary.json`.
Add display-only adapter tags for contract status, gap remaining, live authority,
and target consistency. Add Settings inspector rows and e2e assertions.

## 4. Likely files to touch

- `backend/operator_product_state.py`
- `tests/test_operator_product_state.py`
- `frontend/src/services/operatorAdapters.ts`
- `frontend/src/views/SettingsView.tsx`
- `frontend/e2e/automixer-prototype.spec.ts`
- `.ai/briefs/product-layer-raw-mixer-readback-contract-visibility-20260605.md`
- `Docs/adr/product-layer-raw-mixer-readback-contract-visibility-20260605.md`
- `.ai/memory/project.md`

## 5. Alternatives considered

- Leave contract visibility only in no-WING artifacts. Rejected because cockpit
  operators need the same planning facts.
- Add a separate Product Layer endpoint. Rejected because the existing
  supervised proof evidence snapshot already reads the no-WING summary.
- Treat contract pass as fixture proof. Rejected because schema readiness and
  physical readback evidence are separate.

## 6. Risks

Operators may mistake contract visibility for hardware proof. Mitigation: copy
only normalized display fields and keep `physical_fixture_gap_remains=true`,
`live_authority_granted=false`, and `full_autonomous_ready=false`.

## 7. Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py -q`
- `npm --prefix frontend run build`
- `npm --prefix frontend test -- --watch=false`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## 8. Where the other agent may disagree

Kimi may prefer keeping proof-contract metadata out of the cockpit until a
physical runner exists. I would expose it now because it helps operators see the
next safe proof task without granting execution authority.
