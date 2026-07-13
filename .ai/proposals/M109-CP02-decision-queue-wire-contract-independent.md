# Independent proposal — M109-CP02 decision queue v2

## 1. Short thesis

Treat CP02 as a control-plane integrity change: freeze review content at
proposal creation, list only compact rows under an exact 128 KiB transport
budget, return full review content in verified chunks, and require its digest
at every apply path.

## 2. What I understood

The present list performs full `to_dict()` deep copies before row slicing and
then appends unbounded MixingAgent state. The apply gate binds only an ID and
approval/idempotency data. Therefore the defect is both an availability fault
(close 1009) and a review-to-write integrity gap.

## 3. Proposed solution

- At creation, canonicalize an immutable v2 projection containing ID, source,
  mode, timestamps/TTL, title/reason/target/kind/severity/confidence,
  requested change(s), and full `raw`, including any supervised transaction
  plan. Store canonical bytes and `sha256:<hex>` on the proposal.
- Never call full `to_dict()` to build a wire list. Generate bounded compact
  metadata directly from proposal attributes and stable analysis-only rows.
- Add `get_decision_queue` v2 with safety-first deterministic priority and an
  exact size oracle using the production serializer. The final UTF-8 payload
  must be at most 131,072 bytes.
- Freeze compact snapshots for 30 seconds and issue connection-bound opaque
  cursors. Snapshot eviction never evicts proposals.
- Add authenticated `get_operator_proposal_detail`. Return canonical bytes in
  base64 chunks of at most 65,536 raw bytes, each response independently under
  131,072 serialized bytes. The client reassembles and verifies SHA-256.
- In canonical apply, branch on the proposal's server-side review version and
  validate the supplied digest before expiry mutation, idempotency reservation,
  claim, pre-read, backend contact or console send. Pre-v2 rows are read-only
  and must be regenerated; request-side v1 cannot downgrade this rule.
- Propagate the same digest through linked, sequence, dry-run, gate and executor
  paths. There is no trusted internal exception.
- Keep only a compact bounded v1 read shim during migration. It cannot grant
  write authority.

## 4. Likely files

- `backend/operator_proposal_queue.py`
- a dedicated backend wire-contract/snapshot module
- `backend/handlers/product_state_handlers.py`
- `backend/ws_transport.py` and server connection state
- MixingAgent compact-identity producer
- active frontend WebSocket, hook, adapter and task-review components

## 5. Alternatives considered

- Increasing WebSocket size leaves memory and deep-copy growth unbounded.
- Truncating arbitrary nested data can hide the actual plan.
- Returning full detail in one frame reproduces close 1009.
- Allowing pre-v2/no-digest apply creates a downgrade bypass.

## 6. Risks

- A digest projection can omit a write-authoritative field.
- Building a frozen snapshot can itself consume excessive memory.
- Unstable MixingAgent indices can collide across pages.
- Cursor ownership and TTL errors can become ambiguous to the UI.

## 7. Test plan

Exercise 1 MiB nested raw/result, Cyrillic and emoji, pinned overflow, MixingAgent
collisions, cursor TTL/ownership/eviction, detail reassembly, immutable-field
mutation, status-only mutation, every apply entrypoint, downgrade attempts, and
spies proving zero reservation/claim/backend/send on binding failure.

## 8. Where the other agent may disagree

The main disagreement is whether detail may omit large raw evidence. Because
`raw` currently contains the supervised transaction plan, omission is unsafe;
chunking is preferable to guessing which nested fields are authoritative.

## Provenance

Independent council fallback was used because Kimi CLI was unavailable with an
HTTP 402. This proposal is read-only architecture analysis, not implementation.
