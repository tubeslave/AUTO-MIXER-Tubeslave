# Proposal: proof-evidence-settings-visibility-20260605

## 1. Short thesis

Expose proof artifact freshness in Settings as read-only operational evidence,
then prove the browser path with the existing backend DTO e2e test.

## 2. What I understood

Autonomy promotion needs operators to see whether blockers come from missing or
stale proof artifacts versus target-specific safety gaps. The adapter already
separates `supervised_proof_evidence` from `autonomy_evidence_overlay`, but the
Settings browser fixture still does not exercise that path.

## 3. Proposed solution

Add a small Settings inspector summary derived from module status rows:

- proof evidence source status
- proof freshness
- autonomy overlay status and row count

Also add a Proof filter to the module status table and update the e2e backend
DTO processing payload to include a fresh proof report with fader and gain
overlay rows.

## 4. Likely files to touch

- `frontend/src/views/SettingsView.tsx`
- `frontend/e2e/automixer-prototype.spec.ts`
- `.ai/briefs/proof-evidence-settings-visibility-20260605.md`
- `Docs/adr/proof-evidence-settings-visibility-20260605.md`

## 5. Alternatives considered

- Build a dedicated proof evidence panel now. Rejected because the current
  module status surface already supports search, filters, status, and next step
  text.
- Leave this covered by adapter tests only. Rejected because it does not prove
  the browser-visible operator path.
- Treat fresh proof as live readiness. Rejected because artifact freshness is
  only review evidence.

## 6. Risks

Operators may overread fresh proof evidence as authority to widen automation.
Mitigation: the rows stay read-only, freshness is presented as evidence status,
and the table still shows target-level blockers separately.

## 7. Test plan

- `node frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend test`
- `npm --prefix frontend run build`
- Run the Playwright e2e test if the local browser dependencies are available.

## 8. Where the other agent may disagree

Kimi may prefer a new page or richer artifact drill-down. I would keep this
slice smaller until there is operator feedback about which proof details need
first-class controls.
