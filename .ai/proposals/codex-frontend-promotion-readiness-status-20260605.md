# Proposal: frontend-promotion-readiness-status-20260605

## 1. Short thesis

Expose backend promotion-readiness rows in the existing Settings status table as
read-only evidence, not as live executors.

## 2. What I understood

The long-term goal needs explainable autonomous planning. Backend readiness now
contains target-level blockers and evidence requirement IDs, but the UI only
shows method registry, write surfaces, and safety coverage. This hides the
promotion gates from operators.

## 3. Proposed solution

Extend `adaptModuleStatusRecords` to append rows from
`autonomy_promotion_readiness.rows`. Map them to `ModuleStatusRecord` with:

- `domain = "promotion_readiness"`
- `uiSurface = "Autonomy promotion readiness"`
- `canWriteLive = false`
- blockers rendered as `blockedReason`
- evidence requirement IDs appended to `nextStep`

## 4. Likely files to touch

- `frontend/src/services/operatorAdapters.ts`
- `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `.ai/briefs/frontend-promotion-readiness-status-20260605.md`
- `Docs/adr/frontend-promotion-readiness-status-20260605.md`

## 5. Alternatives considered

- Build a new readiness panel. Rejected as unnecessary for this slice.
- Hide readiness until fully proven. Rejected because explainability requires
  showing why targets are blocked.
- Mark fader readiness as write-live. Rejected because this table row is a
  read-only promotion status, not the executor.

## 6. Risks

Operators may confuse readiness rows with executable modules. Mitigation:
`canWriteLive=false`, status-only UI surface, and no apply actions.

## 7. Test plan

- Frontend adapter contract test.
- `npm --prefix frontend test`.
- Backend focused tests if backend files are touched; they are not expected.

## 8. Where the other agent may disagree

Kimi may prefer a dedicated readiness UI. I would defer that until operators need
sorting/filtering beyond the existing status table.
