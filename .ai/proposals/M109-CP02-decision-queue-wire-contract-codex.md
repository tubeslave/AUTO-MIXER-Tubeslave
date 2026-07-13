# Codex proposal — M109-CP02 decision queue v2

## 1. Short thesis

Add a separate v2 wire projection and detail endpoint, but enforce review
version/digest at the existing single apply gate so v1 cannot bypass it.

## 2. What I understood

The internal proposal object is intentionally rich. The defect is exporting it
unchanged into a periodic list message. Row-count slicing cannot bound bytes,
and a transport failure currently degrades operator truth.

## 3. Proposed solution

- Add immutable `review_contract_version=2` and a canonical SHA-256 review projection
  to `OperatorProposal`. The projection includes every apply-authoritative
  field: ID/version, source/mode, requested change(s), expiry and the complete
  supervised transaction plan. Dynamic status/result are excluded.
- Keep `to_dict()` for internal/legacy full data. Add compact list and review
  detail serializers with explicit schemas.
- Add `build_decision_queue_v2` with a 128 KiB complete-payload budget. Pin
  global hazards, then `can_apply=false`, then critical/high rows; paginate the
  remainder newest-first under a snapshot revision.
- Exclude `raw`, `result`, unbounded agent status and derived nested plans from
  list rows. Return exact counts and bounded blocker-code aggregates.
- Add `get_operator_proposal_detail` returning the complete immutable review
  projection as digest-bound 64 KiB canonical-JSON chunks. This is necessary
  because `raw.supervised_transaction_plan` can be apply-authoritative even
  when the rest of `raw` is merely evidence.
- Require server-side review-contract version and `expected_review_content_digest` in the existing
  `apply_operator_proposal` gate before idempotency reservation or claim.
- Migrate every internal executor and frontend request to send the binding.
  Digestless legacy apply returns `reviewed_digest_required`; there is no
  compatibility escape hatch for live apply.
- Keep legacy `get_decision_queue` temporarily for read-only consumers, while
  the frontend requests v2. Legacy list removal is a later deprecation.

## 4. Likely files

- `backend/operator_proposal_queue.py`
- `backend/operator_product_state.py`
- `backend/handlers/product_state_handlers.py`
- `backend/ws_transport.py` only for shared byte measurement/constants
- `frontend/src/services/websocket.js`
- `frontend/src/services/websocketCore.mjs`
- `frontend/src/hooks/useOperatorBackend.ts`, adapters, active task views and
  the smallest review-detail state/view seam
- focused backend/frontend tests

## 5. Alternatives considered

- Truncate `raw`: rejected because it is not a byte guarantee and may hide
  apply-authoritative fields.
- Raise WebSocket limits: rejected because payload growth remains unbounded.
- V2-only digest enforcement: rejected because v1 becomes a bypass.
- Return full 1 MiB detail in one WebSocket frame: rejected; the immutable
  projection is returned only through individually byte-bounded chunks.

## 6. Risks

- Digest projection can omit a field used by apply.
- Critical rows alone can exceed the budget.
- Status/expiry changes can make a cursor stale.
- Broad test updates can accidentally create a test-only bypass.
- UI may send a list digest without showing detail.

## 7. Test plan

- Projection stability and apply-field coverage tests.
- 1 MiB raw, Unicode, critical-row flood and agent-status adversarial cases.
- Exact serialized byte assertions through `ws_transport.serialize_message`.
- Stale cursor/detail eviction and all digest error paths.
- Assert no idempotency reservation/claim/manual write on binding failure.
- Frontend v2 requests, detail-before-apply and close-1009 tests.
- Focused suites, standard full suite, frontend unit/build/E2E as applicable.

## 8. Where another reviewer may disagree

The main pressure point is migration: strict digest enforcement changes the
apply request contract. I prefer a deliberate hard break at the safety gate;
read-only legacy list compatibility can remain, but live apply cannot.
