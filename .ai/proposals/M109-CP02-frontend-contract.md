# Frontend proposal — M109-CP02 operator truth flow

## 1. Short thesis

The active frontend must consume v2 as server truth: paginate without local
reordering, verify detail before apply, stop permanently on close 1009, and
never substitute demo data for a failed live connection.

## 2. What I understood

The active entry is `main.tsx` through `AutomixerPrototype` and
`useOperatorBackend`, not the legacy `App.js` path. Today WebSocket close
metadata is discarded, reconnect continues, adapters can return mock tasks,
and previews fabricate peak/rollback claims.

## 3. Proposed solution

- Add a pure v2 queue-contract reducer/parser. Preserve server order and dedupe
  only by `(proposal_id, review_contract_version)`.
- On stale cursor, clear once and make one first-page request; never loop.
- Represent counts-only overflow as degraded/unavailable state, not an empty
  queue.
- Fetch all detail chunks on explicit Details, retain only a small memory LRU,
  verify SHA-256 and parsed ID/version, then mark the review ready.
- Enable Apply only when connected supervised state, current proposal state,
  verified detail digest and `can_apply=true` agree. Send the server detail
  digest, never a client-derived/list-only digest.
- Treat detail mismatch/expiry/eviction as a red stale state requiring explicit
  refetch. Never auto-reapply.
- Surface all blocker codes/counts; do not silently slice them.
- Preserve mocks only behind explicit demo provenance. Real empty is `[]`;
  disconnected/absent data is unavailable or explicitly stale last-real data.
- Preserve close code/reason. Code 1009 is terminal until manual retry and shows
  that no current live state is available.

## 4. Likely files

- `frontend/src/services/websocketCore.mjs`
- `frontend/src/services/websocket.js`
- a new pure decision-queue contract module
- `frontend/src/hooks/useOperatorBackend.ts`
- `frontend/src/services/operatorAdapters.ts`
- active prototype/task queue/detail views and tests

## 5. Alternatives considered

Automatic reconnect and mock fallback are rejected because they misrepresent
transport failure as live operator truth. Applying from a compact list digest
is rejected because it does not prove that detail was reviewed.

## 6. Risks

Chunk assembly can retain large content too long. Legacy inactive components
may preserve old signatures and confuse later maintainers. Stale-cursor retry
can loop if not modeled as a single state transition.

## 7. Test plan

Test close 1009 terminal behavior, no timers/no mocks, Unicode parsing, empty
queue, counts-only overflow, cursor reset once, chunk order/reassembly/digest,
detail mismatch, stale review, and missing-binding apply performing no send.

## 8. Where another reviewer may disagree

The strict requirement to open and verify detail before Apply adds one explicit
operator step. That friction is intentional because a compact row is not the
reviewed action.
