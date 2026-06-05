# Proposal: settings-read-surface-proof-quality-20260605

## 1. Short thesis

Expose read-surface proof-quality facts in the Settings inspector as read-only
operator evidence.

## 2. What I understood

Read-surface proof quality is now normalized by the Product Layer and present in
adapter pipeline profiles. Operators still need a browser-visible summary that
explains why cached DTO fixture coverage is not physical readback proof.

## 3. Proposed solution

Extend `proofEvidenceInspectorRows` to parse the existing profile tags:
`read_surface_quality`, `read_surface_gaps`, `fixture_tests`, and
`contract_gap`. Render them as inspector rows under Module Status. Update the
backend DTO e2e fixture to include proof-quality metadata and assert the
Settings inspector displays it.

## 4. Likely files to touch

- `frontend/src/views/SettingsView.tsx`
- `frontend/e2e/automixer-prototype.spec.ts`
- `.ai/briefs/settings-read-surface-proof-quality-20260605.md`
- `Docs/adr/settings-read-surface-proof-quality-20260605.md`
- `.ai/memory/project.md`

## 5. Alternatives considered

- Add a dedicated proof page. Rejected because the existing Settings module
  status inspector already presents proof evidence and write authority.
- Parse artifact JSON in the browser. Rejected because Product Layer owns
  artifact normalization.
- Only rely on the adapter row next-step text. Rejected because it is searchable
  table content, not the operator summary surface.

## 6. Risks

Operators may read "proof quality available" as "ready for Full Auto." Mitigate
by showing physical gap count and keeping the row read-only.

## 7. Test plan

- `npm --prefix frontend test`
- `npm --prefix frontend run build`
- `npm --prefix frontend run test:e2e -- automixer-prototype.spec.ts`
- `python3 tools/memory/memory_lint.py`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## 8. Where the other agent may disagree

Kimi may prefer a separate read-surface readiness panel. I would keep this slice
inside Settings until there is physical hardware readback evidence to justify a
larger readiness surface.
