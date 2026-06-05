# Proposal: read-surface-fixture-closure-plan-frontend-20260605

## Short Thesis

Surface the manual fixture closure plan in cockpit module status as read-only
operator evidence, not as fixture mutation or live readiness.

## What I Understood

Accepted physical readback records are only candidates for a future manual
fixture-baseline update. Operators need this status in the cockpit because it is
the next visible step after artifact review and acceptance ledger work. The UI
must keep the safety boundary explicit.

## Proposed Solution

Update `frontend/src/services/operatorAdapters.ts` to:

- read `proofQuality.read_surface_physical_readback_fixture_closure_plan`;
- add compact `fixture_closure_plan_*` tags to supervised proof evidence;
- append a dedicated module-status row after the artifact review queue;
- force `canWriteLive=false`, no live write kinds, and false fixture update,
  closure, live-authority, and full-auto tags;
- preserve a missing-plan row when proof summaries are stale or incomplete.

Update `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs` to
cover present and missing closure-plan payloads.

## Likely Files To Touch

- `frontend/src/services/operatorAdapters.ts`
- `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `.ai/memory/project.md`
- `Docs/adr/read-surface-fixture-closure-plan-frontend-20260605.md`

## Alternatives Considered

- Keep the closure plan only in no-WING release artifacts. Rejected because
  cockpit users would not see closure progress during normal review.
- Merge closure status into the artifact queue row. Rejected because artifact
  review and fixture-baseline planning are separate lifecycle stages.
- Hide missing closure plans. Rejected because missing evidence must remain
  visible as a proof gap.

## Risks

- Operators may overread a closure candidate as completed fixture proof.
- More proof tags make the module-status row denser.
- Stale summaries may not include the new plan until no-WING checks are rerun.

## Test Plan

- `node --test frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend test`
- `npm --prefix frontend run build`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where The Other Agent May Disagree

Kimi may prefer only summary tags and no separate row. I would keep the row
because manual fixture-baseline planning is operator work with its own blockers,
not just a scalar proof metric.
