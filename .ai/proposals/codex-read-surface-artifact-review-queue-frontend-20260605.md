# Proposal: read-surface-artifact-review-queue-frontend-20260605

## Short Thesis

Surface the aggregate physical-readback artifact review queue in cockpit module
status as read-only operator evidence, not as live readiness or fixture
acceptance.

## What I Understood

The backend queue collects all per-surface physical readback artifact-intake
statuses. Operators need this aggregate view in the frontend because reviewing
ten independent intake rows is error-prone. The queue must remain safety
negative: it can show work to do, but cannot close fixture gaps or grant live
authority.

## Proposed Solution

Update `frontend/src/services/operatorAdapters.ts` to:

- read `proofQuality.read_surface_physical_readback_artifact_review_queue`;
- add compact `artifact_queue_*` tags to the supervised proof evidence row;
- append a dedicated module-status row for the queue after dLive artifact
  intake and before read-surface remediation;
- force `canWriteLive=false` and an empty `liveWriteKinds` list;
- publish queue safety fields in `pipelineProfiles`, `blockedReason`, and
  `nextStep`.

Update `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs` to
cover both present and missing queue payloads.

## Likely Files To Touch

- `frontend/src/services/operatorAdapters.ts`
- `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `.ai/memory/project.md`
- `Docs/adr/read-surface-artifact-review-queue-frontend-20260605.md`

## Alternatives Considered

- Only rely on the backend/no-WING Markdown section. Rejected because cockpit
  users would still lack the aggregate status in the operational UI model.
- Merge queue details into the dLive intake row. Rejected because the queue is
  cross-surface evidence and should not appear to belong to one console family.
- Hide missing queue payloads. Rejected because stale or incomplete summaries
  must not make proof gaps disappear.

## Risks

- Operators might overread review candidates as accepted fixtures.
- Long `pipelineProfiles` arrays can be noisy.
- Stale proof summaries may omit the queue until no-WING checks are rerun.

## Test Plan

- `node --test frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend test`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_no_wing_release_check.py tests/test_read_surface_physical_readback_artifact_review_queue.py -q`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where The Other Agent May Disagree

Kimi may prefer a smaller UI surface that only tags the existing supervised
proof evidence row. I would keep the separate row because a cross-surface queue
is operator work, not just proof metadata, and it needs a stable module ID for
filtering and review.
