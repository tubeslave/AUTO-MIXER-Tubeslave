# ADR M109-CP02 — byte-bounded decision queue and review-bound apply

Status: accepted for implementation after independent engineering GO on
2026-07-13.

## 1. Context

`get_decision_queue` currently deep-copies full proposal `raw`/`result` before
row slicing, appends unbounded MixingAgent state, and sends the object without
an outbound byte invariant. One 1 MiB proposal exceeds the WebSocket limit and
can cause close 1009. Separately, apply is bound to proposal ID but not to the
immutable content the operator reviewed.

This repository controls a real WING. A compact list may aid navigation, but it
must never become live write authority.

## 2. Options considered

1. Increase WebSocket limits.
2. Truncate nested `raw`/`result` fields in the existing response.
3. Add a compact list only and retain ID-only apply.
4. Add a versioned compact list, digest-bound chunked detail and mandatory
   review digest at the canonical apply boundary.

## 3. Decision

Choose option 4 and ship backend plus active frontend as one compatibility
boundary.

### Immutable review content

Every new proposal receives `review_contract_version=2`. Freeze canonical
UTF-8 JSON using `ensure_ascii=False`, sorted keys, compact separators and
`allow_nan=False`, and store `sha256:<64 lowercase hex>`.

The projection schema is
`automixer.operator_proposal.review_content.v2` and includes proposal ID,
review-contract version, title, target, kind, severity, confidence, source,
reason, created/expiry/TTL, mode at creation, requested change(s), and complete
`raw` including any supervised transaction plan. Mutable status, updated time,
derived blockers, expiry state and result are excluded. Non-canonical JSON fails
closed.

Creation stores the immutable canonical bytes, digest and version in private,
write-once fields. Apply enforcement branches only on the private stored
version, never on a request field or mutable/public projection. A missing or
non-v2 stored version is `proposal_review_contract_upgrade_required`.

For a stored v2 proposal, apply
also rebuilds the projection from the exact current proposal object immediately
before any mutation. It requires all three values to agree: recomputed current
version/digest, private stored version/digest and requested version/digest, and
all versions must equal 2. A post-creation change to any authoritative field
therefore fails as `proposal_review_content_integrity_error` instead of
executing stale reviewed bytes.

Validation alone is not execution authority. Under a per-proposal lock, the
queue atomically revalidates current/stored/requested identity, claims the
proposal and constructs `VerifiedProposalExecution` by parsing a fresh private
copy of the verified stored canonical bytes. That immutable snapshot contains
the exact ID/version/digest/deadline, requested change(s), proof fields and full
transaction plan. The lock is released only after claim plus snapshot creation.
Direct, linked and transaction apply consume only this snapshot; they never
reread public `proposal.raw`, requested fields or any caller payload after
binding. Batch/sequence/plan/gate/executor carry identity, but the final
canonical executor always reacquires its own atomic verified snapshot. This
closes mutation and await-time TOCTOU gaps.

Admission is bounded before insertion: at most 2,097,152 canonical review bytes
per proposal, 128 pending proposals and 16,777,216 canonical review bytes across
pending proposals. Typed rejections are `proposal_review_content_too_large`,
`operator_proposal_pending_capacity_exceeded` and
`operator_proposal_pending_byte_capacity_exceeded`; rejected IDs are not marked
seen. The consumed-ID ledger is bounded at 4,096 unique IDs; once full, new IDs
are rejected as `operator_proposal_id_ledger_capacity_exceeded` until an
attended restart/maintenance cycle. IDs are never forgotten or reused within a
server lifetime.

Full terminal rows retain at most 200 proposals and 16,777,216 total retained
bytes, counted as canonical review bytes plus canonical result bytes. An inline
result is at most 262,144 bytes. Larger results, every result with
`write_state_unknown`/`manual_review_required`, and any full row evicted for the
global byte cap are first written atomically to
`artifacts/operator_proposal_queue/results/<sha256>.json`; the filename is only
the canonical content digest, so user-controlled IDs never enter paths. Result
bytes use the same strict JSON normalization after the transport converter.
Writes use a same-directory exclusive 0600 temp file, file fsync, atomic
`os.replace`, then parent-directory fsync; symlinks and non-regular targets are
rejected. The in-memory row then keeps only bounded safety truth and a
digest/reference. Oldest terminal full rows may become bounded ID tombstones
only after required evidence persistence succeeds. Pending rows are never
silently evicted.

The result store is limited to 8,388,608 bytes per artifact, 1,024 artifacts
and 268,435,456 total bytes, while preserving at least 536,870,912 free disk
bytes. Before any possible console send, apply reserves one 8 MiB evidence slot
and fails closed as `operator_proposal_result_evidence_capacity_unavailable` if
count/bytes/free-space cannot satisfy it. Unused reservation is released after
durable result commit. Automatic LRU eviction is allowed only for evidence with
known no-send or known restored/safe terminal state. Unknown-write,
rollback-unknown and manual-review evidence is pinned and never auto-evicted;
if pinned evidence consumes capacity, subsequent applies remain blocked.

If result persistence fails, the full result is retained, the global hazard
`operator_proposal_result_persistence_failed` blocks further apply, and an
unknown physical state latches emergency/manual review. Capacity pressure may
deny future proposals; it may not discard possible-write or rollback evidence.

### Queue list contract

The first-page request is exactly
`{"type":"get_decision_queue","contract_version":2,"page_size":50}`.
Later pages add both `cursor` and `expected_snapshot_revision`; new filters or
page-size are forbidden with a cursor. The success response uses
`type:"decision_queue"`, `contract_version:2` and
`wire_schema:"automixer.decision_queue.v2"`. It contains:

- `status` (`ok` or `degraded_overflow`) and `success:true`;
- `snapshot` with `id`, `revision`, `created_at`, `expires_at` and
  `fully_pageable`;
- `page` with `page_size`, `returned_count`, `remaining_count`, `next_cursor`
  and `has_more`;
- `counts` with exact `total_count`, `included_count`, `omitted_count`,
  `pinned_total_count`, `pinned_included_count`, `pinned_omitted_count`, plus
  fixed-map counts by section, priority and apply state;
- compact `items`, bounded `overflow`, and `wire` with `max_bytes` and the
  fixed-point exact `serialized_bytes` of the final payload.

Every item has exactly these native JSON fields: `id` (string, at most 256
UTF-8 bytes), `entity_kind` (`operator_proposal` or `mixing_agent_action`),
`queue_section` (`pending`, `history` or `analysis_only`), `title_preview`
(string, 512 bytes), `reason_preview` (string, 1,024 bytes), `display_target`
(string, 256 bytes), `kind` (string, 128 bytes), `severity`, `status`, `source`
(string, 128 bytes), ISO `created_at`/`updated_at`/`expires_at` or null,
`can_apply` (boolean), `requires_approval` (boolean), `apply_blocker_codes`
(at most 8 strings of 128 bytes), `apply_blocker_count` (integer),
`linked_change_count` (integer), `review_contract_version` (integer or null),
`review_contract_status`, `review_content_digest` (string or null),
`detail_available` (boolean), `hazard_scope`, `priority_class`,
`analysis_only` and `import_required` (booleans). UTF-8 preview truncation never
splits a code point. Rows never contain raw, result, plans, blobs or unbounded
ID arrays.

Priority is deterministic: explicit unresolved global hazards; other blocked
rows; critical/high rows; ordinary pending rows newest-first; terminal history
newest-first. Missing hazard classification is conservatively non-applyable
until regenerated/migrated. MixingAgent rows use stable source-qualified
identity and are `analysis_only/import_required` until converted into a real
proposal.

The exact oracle is
`len(serialize_message(payload, convert_numpy_types).encode("utf-8"))`. Every
list response is at most 131,072 bytes; packing targets
126,976 bytes to retain envelope reserve. Exact totals/included/remaining and
fixed-map aggregates account for omitted rows. If pinned metadata cannot fit,
return a small `degraded_overflow` envelope with counts/hazard digest and
`apply_ui_disabled:true`; never silently omit the safety condition.

`counts.total_count` is every row represented by the frozen snapshot.
`included_count` is the number actually materialized and therefore pageable;
`omitted_count = total_count - included_count`. `page.returned_count` is only
the current frame, and `page.remaining_count` is materialized rows after this
frame. `has_more` is true exactly when `next_cursor` exists.
`overflow.unpageable_count` equals `omitted_count`. The pinned counts use the
same definitions. Fixed maps have no dynamic keys:

- section: `pending`, `history`, `analysis_only`;
- priority: `global_hazard`, `blocked`, `high`, `ordinary`, `history`;
- apply state: `applyable`, `blocked`, `analysis_only`.

`overflow` has exactly `counts_only` (boolean), `reason` (null,
`snapshot_materialization_budget` or `pinned_metadata_budget`),
`unpageable_count`, `pinned_unpageable_count`, `apply_ui_disabled` and
`hazard_digest`. No omitted IDs/titles are returned.

If any pinned row is unpageable, every response for that snapshot has
`status:"degraded_overflow"`, `apply_ui_disabled:true`, and every item is
forced `can_apply:false`; no compact row may be treated as apply-ready.

### Global hazard envelope

Every response contains `hazards` with `fatal`, `apply_ui_disabled`, exact
`count`, up to 16 sorted fixed-enum `codes`, `overflow_count`, fixed source
counts (`runtime`, `manual_write`, `terminal_result`), and `digest`. The complete
fatal enum and source are:

- runtime: `server_shutdown_active`, `mixer_disconnected`,
  `wing_observation_only`, `operator_mode_blocks_apply`;
- manual write: `emergency_stop_active`, `emergency_stop_in_progress`,
  `emergency_stop_teardown_incomplete`,
  `manual_write_wal_recovery_required`, `console_writer_busy`,
  `console_writer_conflict_recorded`;
- terminal result: `terminal_write_state_unknown`,
  `terminal_rollback_state_unknown`, `terminal_manual_review_after_send`,
  `operator_proposal_result_persistence_failed`.

They are derived only from server lifecycle/connection/operator mode, bounded
manual supervision/lock state, and compact retained/tombstoned result truth.

Proposal-local master/headroom, feedback, stale-readback and transaction
blockers remain pinned blocked rows, not inferred global hazards. The hazard
digest is SHA-256 over sorted objects containing fixed code, source, bounded
entity-reference digest, review version/digest, current status/applyability,
hazard scope and priority. No free text or raw evidence enters this digest.

### Cursor and retention

Pagination uses frozen compact snapshots, random 128-bit tokens bound to the
WebSocket connection, snapshot and next offset. Snapshot `revision` is a
SHA-256 over its contract version, normalized request/filter, ordered compact
row identity/version/digest/status/applyability/expiry disposition, fixed
counts, MixingAgent revision and global-hazard digest. Subsequent requests must
echo the revision and recompute current safety/applyability revision before
serving. Any queue, expiry, MixingAgent or hazard change returns
`decision_queue_cursor_stale`; the client discards the snapshot and may perform
one explicit first-page refresh. Frozen ordering prevents skips only while the
safety revision remains current. Detail and apply always re-read current state.

TTL is 30 seconds; limits are
4 snapshots per connection, 32 global, 8 MiB global compact cache and 2 MiB per
snapshot. LRU/expiry has typed errors and never deletes proposals. Pending
proposals are never auto-evicted. Existing terminal history retention remains
200 plus the byte limit above; known-but-evicted detail returns
`operator_proposal_detail_not_retained`.

Cursor errors use response type `decision_queue_error`, `contract_version:2`,
`success:false`, and one of `decision_queue_cursor_invalid`,
`decision_queue_cursor_owner_mismatch`, `decision_queue_snapshot_expired`,
`decision_queue_snapshot_evicted`, `decision_queue_cursor_request_conflict`,
`decision_queue_snapshot_revision_required`, or
`decision_queue_snapshot_revision_mismatch`, or
`decision_queue_cursor_stale`. They perform no queue mutation or backend
contact and include `no_console_send_proven:true`,
`proposal_status_mutation:false` and `idempotency_reserved:false`.

### Detail contract

The exact detail request is
`{"type":"get_operator_proposal_detail","contract_version":2,` followed by
`"proposal_id":"...","expected_review_contract_version":2,`
`"expected_review_content_digest":"sha256:...","chunk_index":0}`. It is
authenticated. The success response uses `type:"operator_proposal_detail"`,
`contract_version:2`, the proposal identity/version/digest, bounded current
state, and immutable canonical content. The top level has exactly
`proposal_id`, `review_contract_version`, `review_content_digest`,
`current_state`, `content` and `wire` after the common type/version/status
fields. `current_state` has `status`, `updated_at`, `expired`, `can_apply`,
`apply_blocker_codes` and `apply_blocker_count`. Its `content` object has exactly
`encoding:"base64"`, media type
`application/vnd.automixer.operator-proposal-review+json;version=2`,
`chunk_index`, `chunk_count`, `raw_chunk_bytes`, `total_raw_bytes`,
`chunk_sha256`, `content_sha256` and `data`. Raw chunks are at most 65,536
bytes. The top-level `wire` contains `max_bytes:131072` and fixed-point actual
`serialized_bytes`. Every chunk response independently passes the same exact
wire oracle. The client concatenates decoded bytes, verifies every chunk and
the final content SHA-256, and only then parses the review document.

Detail failures use `operator_proposal_detail_error` and one of
`proposal_id_required`, `proposal_review_contract_version_required`,
`proposal_review_contract_version_invalid`,
`proposal_review_contract_version_mismatch`,
`operator_proposal_detail_digest_required`,
`operator_proposal_detail_digest_mismatch`, `operator_proposal_not_found`,
`operator_proposal_detail_not_retained`,
`operator_proposal_detail_chunk_index_invalid`,
`proposal_review_content_integrity_error`, or
`operator_proposal_detail_response_budget_invariant_breached`. Every detail
error includes `no_console_send_proven:true`,
`proposal_status_mutation:false` and `idempotency_reserved:false`.

Terminal result evidence uses the parallel authenticated request
`get_operator_proposal_result_detail` with proposal ID, expected result digest
and chunk index. Its `operator_proposal_result_detail` response uses the same
chunk fields, budget and verification rules with media type
`application/vnd.automixer.operator-proposal-result+json;version=2`. It reads
only the digest-bound retained bytes/artifact; callers never supply a file path.

### Apply contract

For a server-side v2 proposal, canonical apply requires both
`expected_review_contract_version:2` and
`expected_review_content_digest:"sha256:..."`. Before expiry mutation, idempotency
reservation, claim, backend contact, pre-read or send, reject missing,
malformed, mismatched or internally inconsistent version/digest with distinct
bounded errors and:

- `no_console_send_proven:true`
- `idempotency_reserved:false`
- `proposal_status_mutation:false`

All linked, batch, sequence, dry-run, gate and executor paths carry the same
binding. Request-side version cannot downgrade enforcement. Pre-v2 rows are
read-only and fail with `proposal_review_contract_upgrade_required`; they must
be regenerated.

Version failures are `proposal_review_contract_version_required`,
`proposal_review_contract_version_invalid` and
`proposal_review_contract_version_mismatch`. Digest failures are
`proposal_review_digest_required`, `proposal_review_digest_invalid`,
`proposal_review_digest_mismatch` and
`proposal_review_content_integrity_error`. Every failure response is
`operator_proposal_apply_blocked`; no error path mutates expiry/status merely to
report that the review binding is invalid.

Trust boundary: the server guarantees immutable content identity and trusts an
authenticated caller's explicit `approved`/approval ID. It does not claim to
prove human attention. The active frontend additionally requires verified full
detail before enabling Apply; list digest alone never enables its button.

### Frontend truth state

The active `main.tsx`/`AutomixerPrototype` path requests v2, preserves server
order, accounts for counts-only overflow, fetches and verifies detail before
enabling Apply, and sends the verified detail digest. Real empty means empty;
transport failure means unavailable/stale. Mock tasks remain only in explicit
demo mode. Close 1009 preserves code/reason, stops automatic reconnect and
requires manual retry.

### Other responses

Missing `contract_version` or `contract_version:1` on `get_decision_queue`
returns a required bounded read-only shim: `type:"decision_queue"`,
`contract_version:1`, `deprecated:true`, `migration_required:true`, scalar
summary counts, compact `pending_actions`/`history`, `apply_authority:false` and
the same 131,072-byte guard. Unknown versions return
`decision_queue_contract_version_unsupported`. The shim never restores full
raw/result or no-digest apply. First-party runtime runners migrate to v2.

Create, accept, dismiss, import, preflight, batch, plan, dry-run, gate,
executor and apply wire responses all pass through one compact-result projector
and exact 131,072-byte guard. Their `proposal` field is a bounded reference with
ID, stored review version/digest, status, expiry, `can_apply`, bounded blocker
codes/count, result digest/detail availability and compact safety truth. Arrays
are packed/counts-only; full proposal/result/normalized plans are accessible
only through authenticated digest-bound chunks. Internal full-object APIs
remain available to the soundcheck lifecycle and importers and are never sent
directly.

Before a possible write, an oversized response degrades to a fixed small error
that truthfully states `no_console_send_proven:true`. After any possible console
send, the fallback is instead `type:"operator_proposal_apply_result"`,
`contract_version:2`, actual `status`/`success`, proposal ID/version/digest,
`proposal_status`, `apply_status`, exact nullable `no_console_send_proven`,
`write_state_unknown`, nullable `physical_state_restored`,
`manual_review_required`, bounded WAL/rollback status, result digest/detail
reference, `emergency_stop_activated`, and `response_degraded:true`. It never
defaults no-send truth to true and is independently exact-oracle tested under
131,072 bytes. Unknown physical state latches emergency/manual review before
this receipt is sent.

## 4. Why this won

It fixes both failure classes without weakening live safety: payload size is a
measured invariant, complete authoritative content remains reviewable, and the
write gate proves that the executed proposal matches that content. It also
removes false operator truth caused by reconnect/mock fallback.

## 5. Rejected alternatives

- Larger frames merely postpone unbounded growth.
- Field truncation is not a byte proof and can hide an executable plan.
- List-only compaction leaves other responses oversized and apply unbound.
- Legacy no-digest apply is a downgrade bypass.
- One-frame detail reproduces the same availability failure.

## 6. Implementation plan

1. Add isolated pure canonicalization, compact-row, exact-packing and detail
   chunk functions plus private proposal review metadata and bounded result
   evidence persistence.
2. Integrate queue/detail handlers, connection-owned snapshots, bounded
   mutation responses and canonical apply validation across all entrypoints.
3. Integrate the active frontend reducer, detail review, digest apply, terminal
   close-1009 state and remove live mock fallback.
4. Run focused adversarial suites, standard backend suite, frontend tests/build,
   independent patch review and truth-rail update.

## 7. Test plan

- Deterministic canonicalization; Unicode; NaN/non-JSON failure; immutable
  mutation changes digest while status/result changes do not; current, stored
  and requested digest/version must all agree immediately before apply.
- Admission rejects the 129th pending proposal, a proposal over 2 MiB, and
  pending review content over 16 MiB without insertion or ID reservation;
  terminal byte eviction preserves a typed tombstone.
- 1 MiB raw/result and MixingAgent parameters: list and every detail chunk at
  most 131,072 bytes, exact reassembly and no marker leakage in list.
- Priority, pinned overflow, exact counts, stable identity and snapshot
  ownership/TTL/LRU/byte eviction; any expiry/status/agent/hazard mutation makes
  the next cursor request stale before rows are returned.
- Oversized and unknown-write terminal results persist before compaction;
  unsafe filenames/symlinks, disk reserve, artifact/count/byte exhaustion and
  persistence failure block further apply while preserving full evidence.
- Normal and degraded post-write receipts preserve actual no-send/rollback/WAL
  truth and remain under the exact wire budget.
- Missing/malformed/mismatch/integrity/downgrade apply failures on every route,
  with spies proving no reservation, claim, mutation, backend contact or send.
- Mutation between initial validation and an awaited safety check cannot alter
  `VerifiedProposalExecution`; concurrent apply claims exactly once and every
  write consumes the stored-byte snapshot rather than public proposal fields.
- Active frontend detail-before-apply, stale cursor once, counts-only state,
  close 1009 terminal behavior and no mock fallback.
- Mandatory backend command, frontend unit/build and diff review.

## 8. Risks and rollback

Risks are digest-field omission, memory pressure in snapshot/detail assembly,
incomplete internal-route migration and operator UI regressions. Mitigations
are a complete raw-inclusive projection, hard snapshot/LRU limits, central
pre-mutation validation and adversarial spies.

Rollback is source rollback of v2 handlers/frontend only. Do not restore
no-digest apply. If v2 cannot be safely served, leave proposals viewable but
non-applyable and report `soundcheck_incomplete`; no WING write is attempted.

## Council record

The Codex and independent proposals agreed on strict write binding. Independent
critique corrected the initial detail scope to include full authoritative raw
content through chunks. Kimi CLI was unavailable with HTTP 402, so an isolated
read-only architecture agent supplied the opposing proposal and critique.
