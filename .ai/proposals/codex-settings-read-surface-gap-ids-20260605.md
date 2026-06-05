# Proposal: settings-read-surface-gap-ids-20260605

## 1. Short thesis

Carry read-surface physical gap IDs into Settings so operators can see what
still lacks hardware readback proof.

## 2. What I understood

The backend already normalizes `physical_fixture_missing_surface_ids`, but the
frontend adapter drops the list and Settings only shows a count. A count is not
actionable enough for closing console-read coverage gaps.

## 3. Proposed solution

Add a compact `read_surface_gap_ids` profile to the supervised proof evidence
module row, include the list in next-step text, and render the first missing
surface plus remaining count in Settings.

## 4. Likely files to touch

- `frontend/src/services/operatorAdapters.ts`
- `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `frontend/src/views/SettingsView.tsx`
- `frontend/e2e/automixer-prototype.spec.ts`
- `.ai/briefs/settings-read-surface-gap-ids-20260605.md`
- `Docs/adr/settings-read-surface-gap-ids-20260605.md`
- `.ai/memory/project.md`

## 5. Alternatives considered

- Render the full JSON list in Settings. Rejected because the inspector should
  stay compact.
- Add a separate remediation table. Rejected for this slice because no physical
  proof workflow changes are being implemented.
- Keep the list backend-only. Rejected because operators need visible,
  actionable proof gaps.

## 6. Risks

Operators may interpret a listed surface as live-readable now. Mitigation: keep
the row read-only and label these as physical gaps.

## 7. Test plan

- `node frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend test`
- `npm --prefix frontend run build`
- `npm --prefix frontend run test:e2e -- automixer-prototype.spec.ts`
- `python3 tools/memory/memory_lint.py`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## 8. Where the other agent may disagree

Kimi may prefer a full remediation matrix. I would keep this slice focused on
making the existing backend evidence visible before adding a new workflow.
