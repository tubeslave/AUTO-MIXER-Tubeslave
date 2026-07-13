# ADR: M005 runtime identity v2 and legacy trace quarantine

## Context

The final readiness audit found four related trust defects:

- generic historical synthetic traces without a `dry_run` source marker could
  still render as applied and verified;
- reading the 246 MB append-only decision-trace file scanned the whole file and
  blocked operator WebSocket handling for about two seconds;
- a transport timeout after `apply_operator_proposal` could be labelled
  fail-closed even though the console outcome was unknown;
- runtime identity v1 hashed only `backend/server.py`, so an old process or
  changed handler set could satisfy M005 provenance.

## Options considered

1. Delete or rewrite synthetic history.
2. Keep source-name deny lists and the existing full-file reader.
3. Quarantine unbound history at read time and bind M005 to a proof-owned
   backend process with an immutable runtime identity v2 manifest.
4. Trust an operator statement that an already-running backend is current.

## Decision

Choose option 3.

- Append-only trace bytes remain untouched. Terminal success is trusted only
  when the server stamps `automixer.automation_trace_provenance.v1`; legacy
  rows become `legacy_unbound_apply`/`legacy_unbound`, offline rows remain
  offline, and applied-but-unverified rows become `applied_unverified`.
- `read_recent` reads a bounded binary tail in reverse, skips malformed rows
  and returns the latest valid rows without scanning from byte zero.
- Any M005 transport failure after an apply request is
  `manual_review_required`, `fail_closed=false`; a lost rollback response is
  never treated as a completed rollback.
- Runtime identity v2 binds Git HEAD/tree/status, exact hashes for 15
  live-safety-critical files, backend artifact, PID, process nonce/start time,
  loopback endpoint, request challenge and a manifest hash.
- For M005 execute, provenance is captured first and the runner then launches
  its own exact `backend/server.py` with the provenance digest in a consumed
  pre-process environment binding. An existing listener, v1 identity, stale
  process, unbound process or changed critical file fails before mixer queries.
- M006 independently recomputes the current packet and requires the same v2
  source/process/event binding.

## Why this won

It preserves audit history while removing false physical trust, eliminates a
control-path latency spike, and makes the attended proof depend on the process
that actually loaded the safety code rather than on files observed later.

## Rejected alternatives

- Rewriting JSONL: violates append-only provenance and could hide prior test
  contamination.
- Source deny list only: cannot classify generic pytest/operator-queue rows.
- Already-running backend challenge only: proves response freshness, not that
  source and handlers were loaded after provenance capture.
- Automatic rollback after a lost apply response: the exact transaction id may
  be unknown, so a generic rollback can be blocked or target the wrong state.

## Implementation plan

1. Add provenance-aware trace normalization and bounded tail reads.
2. Stamp new server traces with immutable runtime provenance.
3. Add runtime identity v2 and proof-runner process launch binding.
4. Make M005 uncertain transport require physical/manual reconciliation.
5. Recompute all v2 checks in M006 and add mutation tests.
6. Rebuild and smoke the arm64 package, then run the post-lease full/no-WING
   gates.

## Test plan

- Legacy/offline/bound/unverified trace normalization and no-rewrite tests.
- Large-file tail, malformed-line and cross-chunk tests.
- M005 timeout-after-apply and timeout-after-rollback tests.
- Runtime identity v1, stale-time, changed-handler and unbound-launch mutation
  tests in both M005 and M006.
- Full backend/frontend/E2E/no-WING suites after writer-lease release.

## Risks and rollback

The M005 runner now owns backend startup and refuses an occupied canonical
port, so the operator must stop any earlier backend first. That operational
cost is intentional proof isolation. Rollback may restore development tooling,
but must not re-accept v1/unbound identities, relabel uncertain apply outcomes
as fail-closed, or rewrite retained trace/WAL history.

## Final implementation addendum

The proof contract was tightened after an independent execution-path review:

- An executed M005 run now requires `--wing-ip`, connects the proof-owned
  backend to that exact WING before baseline with `observation_only=false` and
  `record_telemetry=false`, and stops before baseline/apply if that initial
  connection is not verified.
- After the operator-gated cable loss, the recovery connection is separately
  required with `observation_only=true`. The M006 verifier requires exactly
  these two ordered connection events: identity, initial connection, baseline,
  stale/blocked apply, recovery connection and recovery readback.
- New executed evidence is restricted to the canonical Git-ignored
  `artifacts/wing_timeout_disconnect_proof/` root, preventing the runner's own
  files from changing its source-status digest. Existing historical evidence
  is preserved.
- Generic exceptions use the same transport-failure classifier as timeout and
  connection failures. Any exception after an apply or rollback request is
  `manual_review_required`; an unknown transport outcome is never labelled
  fail-closed or rolled back.
- Trace normalization recursively quarantines nested unbound physical or
  verified claims even when the top-level row is already blocked or pending.
  The latest 500 retained rows contain zero applied, zero verified and zero
  remaining unbound nested physical claims without changing the append-only
  store.

The post-fix backend regression set passed `282` tests. The historical
`20260711T131505Z` M005 run remains behavioral-only and fails current source,
runtime-identity-v2, initial-connection, owner-binding, owner-challenge,
event-hash and canonical-URL acceptance. This addendum grants no live authority.

The subsequent full local validation passed `6091` backend tests, `168`
frontend unit/contracts, `10` Chromium E2E tests and all `65` four-store
no-WING checks. The final source-content-bound package/smoke evidence is
`frontend/dist/local/20260712T072932Z_9e3d9f29-5ccf-41ca-a46e-f70933a45b23_local/`
and
`artifacts/packaged_runtime_smoke/20260712T073254Z_af976647-7612-419d-a5a5-9922930b203e_packaged_runtime_smoke/result.json`.
