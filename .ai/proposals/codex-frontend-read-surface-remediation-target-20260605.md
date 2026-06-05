# Proposal: frontend-read-surface-remediation-target-20260605

## 1. Short thesis

The cockpit should show the next read-surface remediation action as operator
evidence, not hide it inside the processing registry payload.

## 2. What I understood

The backend now links fixture physical readback gaps to remediation actions.
Without a frontend Module Status row, the operator can see proof quality but not
the concrete `remediate_*` action that moves the project toward complete console
read coverage.

## 3. Proposed solution

Read `read_surface_remediation_plan` from the processing snapshot in
`adaptModuleStatusRecords()`. Add one read-only row with summary profiles for
action count, physical gap count, next physical readback target, and next action
ID. Cover it in the adapter contract and the backend Product Layer e2e fixture.

## 4. Likely files to touch

- `frontend/src/services/operatorAdapters.ts`
- `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `frontend/e2e/automixer-prototype.spec.ts`
- `.ai/briefs/frontend-read-surface-remediation-target-20260605.md`
- `Docs/adr/frontend-read-surface-remediation-target-20260605.md`
- `.ai/memory/project.md`

## 5. Alternatives considered

- Keep remediation visible only in registry JSON. Rejected because cockpit users
  need the next target in Module Status.
- Merge remediation details into supervised proof evidence text. Rejected
  because proof artifact freshness and remediation planning are distinct.

## 6. Risks

The row could look like an executable command. Mitigation: keep `canWriteLive`
false, empty live write kinds, and read-only wording.

## 7. Test plan

- `node frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend test`
- `npm --prefix frontend run build`
- `npm --prefix frontend run test:e2e -- automixer-prototype.spec.ts`

## 8. Where the other agent may disagree

Kimi may prefer adding a dedicated Settings card. I would start with Module
Status because it already aggregates read-only proof, gate, and promotion rows.
