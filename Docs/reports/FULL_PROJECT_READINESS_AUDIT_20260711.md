# Full Project Readiness Audit — 2026-07-11

## Verdict

Initial full-project readiness was `42/100`. After the verified repair,
packaged-runtime and real-console read-only validation slices, local
engineering readiness is `95/100`: the requested engineering target is met.
The stricter public-release score remains `88/100`, public release readiness is
false, and normal supervised transaction live write kinds are empty.

Post-audit addendum: the later soundcheck decomposition review found and closed
an additional P1 rollback-boundary defect. The corrected kernel is scored
separately in `Docs/reports/SOUNDCHECK_PLAN_DECOMPOSITION_20260712.md`; this
does not change the `88/100` public-release score or empty live allowlist.

These are deliberately separate scores. The local score covers code, safety,
tests, local packaging and read-only runtime behavior. The public-release score
also requires clean signed/notarized distribution, current-source physical
provenance and operator-present audible room/PA proof.

## Initial and strict public-release score

| Area | Initial | Current | Maximum |
|---|---:|---:|---:|
| Safety, authority and rollback | 10 | 24 | 25 |
| End-to-end mixing capability | 10 | 13 | 20 |
| Runtime reliability and recovery | 7 | 15 | 15 |
| Operator UX | 7 | 9 | 10 |
| Verification and reproducibility | 5 | 14 | 15 |
| Release evidence, docs and operations | 3 | 13 | 15 |
| **Total** | **42** | **88** | **100** |

## Local engineering score

| Area | Verified | Maximum |
|---|---:|---:|
| Safety, authority, emergency and rollback | 25 | 25 |
| End-to-end mixing capability | 17 | 20 |
| Runtime reliability and recovery | 15 | 15 |
| Operator UX | 9 | 10 |
| Verification and reproducibility | 15 | 15 |
| Local release evidence, docs and operations | 14 | 15 |
| **Total** | **95** | **100** |

## Confirmed critical findings

1. Static family metadata granted broad live authority without an accepted
   owner/readiness decision.
2. The supervised transaction entrypoint bypassed observation-only and the
   effective operator/deployment allowlist.
3. Partial multi-address writes, exceptions, cancellation and later-operation
   failures could leave earlier state changed without complete recovery.
4. A fresh coordinator per request did not serialize or deduplicate concurrent
   server transactions.
5. Per-address sends lacked a last-moment physical compare-and-swap and stale
   targets could overwrite operator changes or create large upward moves.
6. The family proof runner self-authorized direct WING writes and conflated a
   passed proof with acceptance.
7. Unit tests appended synthetic events to the mutable canonical manual-write
   evidence log.
8. Release truth was fail-open: `95/100` and public-ready were published while
   external owner/hardware gates and zero standing family authority remained.
9. The operator UI can render negative readback/rollback strings green, remain
   visually offline after reconnect, retain stale run evidence and lose family
   blockers when mapping the registry.
10. Manual fader, gain and rollback handlers coerced truthy values, so a string
    such as `"false"` could become an approved write.
11. Last-write rollback compared physical state before acquiring the shared
    writer lock and lacked a second physical CAS immediately before send.
12. Two already-running server processes could reserve the same proposal
    idempotency key before either wrote the WAL, allowing sequential replay.
13. A loopback WebSocket accepted arbitrary browser Origins, allowing an
   unrelated web page to reach local control messages.
14. Direct WING WebSocket fader/gain and other legacy control handlers could
    fall back to `LiveApplyService` when a deployment-boundary flag was false,
    bypassing the canonical operator-proposal transaction path.
15. Synchronous physical reads blocked the asyncio event loop; a stale numeric
    value could also pass without exact address, `received` status and two
    confirmed round trips, while cancellation could leave the worker running.
16. The file lock was global but recovery WAL scope still followed a
    configurable evidence root, and fresh nested WAL directories did not fsync
    each newly created parent entry.
17. Packaged Electron had no per-launch token bootstrap, used `file://`, could
    report connected before authentication ACK, and could package a stale
    backend binary.
18. Startup recovery accepted readback from a different OSC address and
    coerced fractional/string round-trip counts; cancellation could also
    release the writer scope while the blocking recovery query still ran.
19. Canonical single and linked `apply_operator_proposal` calls omitted the
    normalized preimage, bypassing the post-prepare physical CAS and allowing
    an operator move between validation and send to be overwritten.
20. Supervised family adapters accepted wrong-address/coerced readback and
    missing/non-finite preimages, so snapshot, verify or rollback could report
    physical proof for the wrong parameter or without a usable rollback value.
21. macOS packaging emitted x64 Electron with the same arm64-only PyInstaller
    backend copied into it, producing an Intel package that could not start its
    backend.
22. Normal soundcheck corrections shared one monolithic preflight/rollback
    boundary. In the archived 51-operation shape, 15 local residuals therefore
    suppressed all 36 locally admissible operations and yielded a
    counterfactual 0/51 under current policy. Completion state also failed to
    distinguish derivation success from a musically resolved mix.

## Repairs completed in this slice

### Post-audit soundcheck closure — 2026-07-12

- Parent manifests now decompose work into semantic children with stable
  revisions, canonical content digests and durable outcome journaling.
- Local residuals block only their child; disconnect, stale evidence,
  emergency, writer, rollback and Main-headroom hazards remain global.
- Bounded safe replan steps and a complete down-only fader-vector translation
  preserve progress without adding constructive authority.
- Selected-source fader plans bind fresh fader/mute/Main/routing/DCA evidence,
  exact analysis deadline identity and final audible-path reconciliation.
- `safe_core_ready`, `show_mix_ready` and `soundcheck_incomplete` are now
  evidence states. A lost source path also revokes the per-operation checkpoint
  and blocks dependent stages.
- Possible-send failures are terminal; no-send CAS aborts release retry
  reservations without rolling back over a human move.
- Normal live authority remains zero.

- All hardware-proven family rows now carry zero normal live write kinds and an
  explicit owner-promotion blocker.
- Server execution intersects observation, operator mode and deployment gates
  before physical query or send.
- Every operation is address/target-preflighted before the first snapshot, all
  snapshots are captured before the first mutation, and every send performs a
  fresh physical compare-and-swap.
- Conservative direction/delta policies block gain-up, fader/send boosts,
  relative EQ boosts, makeup boosts and large moves.
- Partial writes, exceptions, emergency-generation changes and cancellation
  enter reverse recovery; safe addresses continue restoring after another
  address conflicts or fails.
- Manual and multi-parameter mutations share one server-wide writer lock.
- The writer lock now spans processes, and both manual and supervised
  transactions write an fsync-backed prepare record before mutation and a
  durable commit/rollback terminal record afterwards.
- Startup scans unresolved/corrupt WAL rows, latches emergency stop and blocks
  reset until physical readback reconciles every address to either the exact
  target or exact preimage.
- Proposal apply uses an atomic pending-to-applying claim before the first
  await; history replay, queue-index apply, ID reuse, double-click races,
  post-claim exceptions and unrecovered linked writes are fail-closed.
- Channels, identifiers, booleans and numeric targets now reject boolean,
  fractional and non-finite coercions before query or send.
- Rollback performs a fresh physical compare-and-swap immediately before every
  address restore, and repeated task cancellation cannot release the
  coordinator while a blocking send is still in flight.
- Manual approval and live-confirmation fields now accept only literal JSON
  `true`; integer, string, mapping and list coercions fail closed.
- Last-write and linked rollback hold the shared process/file lock across CAS
  and apply, then repeat physical CAS after durable prepare and immediately
  before the first send.
- Durable proposal idempotency is rechecked from a WAL-derived map after the
  cross-process lock is acquired; stale reservations in a second process can
  no longer replay the same key.
- Browser WebSocket handshakes now use an exact Origin allowlist for the local
  Vite UI and explicit configured origins. Native no-Origin proof runners stay
  in the local OS trust boundary.
- Every direct WING WebSocket control surface is proposal-only regardless of
  the legacy boundary flag; the effective WING boundary now follows the actual
  WING connection/client and cannot be disabled by that flag.
- Physical readback runs blocking queries off-loop but drains the worker before
  cancellation exits, and accepts proof only for the exact requested address
  with `status=received` and at least two integer-confirmed round trips.
- Startup reconciliation uses that same cancellation-draining, exact-address,
  non-coercing contract and keeps the WAL/emergency latch unresolved on any
  mismatch.
- Canonical single and linked proposal apply bind the normalized physical/cache
  preimage into a second physical no-cache CAS after durable prepare and before
  send.
- Supervised family snapshot, apply, verify and rollback now require exact
  address, strict integer round trips and usable finite preimages; rollback
  preserves both initial and per-address CAS query evidence.
- The console writer lock and recovery WAL now share one canonical per-user
  runtime directory independent of evidence roots. WAL appends fsync the file,
  leaf directory and every newly created parent entry.
- Packaged Electron now generates a 32-byte memory-only token per launch,
  passes it only in the child backend environment, scrubs backend plaintext,
  restricts bootstrap IPC to the trusted main frame, serves the renderer from
  an ephemeral loopback HTTP origin and waits for a positive auth ACK before
  exposing a ready transport. Pre-auth transport failures retry; policy/auth
  close code `1008` does not. Packaging commands rebuild the backend first.
- macOS packaging is fail-closed to arm64 on this release path: the backend
  build rejects cross-architecture hosts, verifies its Mach-O architecture,
  and the completed package compares the Electron and embedded backend
  architecture sets. The superseded mixed x64/arm64 artifact is not release
  evidence.
- The current dirty-tree Electron 43 arm64 app is explicitly ad-hoc signed,
  passed deep/strict signature verification and completed the source-bound
  packaged-runtime smoke at
  `artifacts/packaged_runtime_smoke/20260712T060214Z_bb83311f-1043-4f35-9ff5-6933708e0abe_packaged_runtime_smoke/result.json`.
  The OS-enforced backend-only network boundary passed its 9-probe TCP/UDP,
  IPv4/IPv6 and descendant-process matrix; invalid authentication was rejected,
  a fresh memory-only token authenticated, hardware operations stayed hidden
  and blocked, and continuous socket observation found no external or WING
  connections. The packaged backend became ready in `4917 ms` against the
  exact `20000 ms` budget. The canonical manual-write log, decision traces,
  decision memory, console WAL, backend bundle, package files and app tree were
  unchanged. All `360` app Mach-O files and all `345` backend Mach-O files are
  arm64. Provenance remains explicitly
  `dirty_tree_non_clean_checkout`, so this is local evidence, not a public
  release artifact.
- Release dependency inputs now use exact hashes and an exact CPython/PyInstaller
  toolchain. The local clean dependency reconstruction verified all `131/131`
  distributions. `watchdog==6.0.0` remains a hash-bound sdist and therefore
  prevents a claim of byte-for-byte build reproducibility.
- The first 2026-07-12 full no-WING reruns exposed two gate defects instead of
  hiding them: the Electron auth E2E fixture omitted the required
  `packagedOfflineLock` boolean, and four standalone fake gain proofs wrote
  `32/32` committed synthetic rows to the per-user recovery WAL. The final
  reproducer changed
  the WAL from 8,497 bytes/SHA-256 `86a361a7411f3e9d20a7e2d60cb26b0a65977f2cfa45acd5fc8ef468f10cf566`
  to 11,329 bytes/SHA-256
  `58d548ecd28593d1791ff7784b72c394ae644fa7b101331528eec2454dfe7bca`.
  No WING/backend transport was contacted and every synthetic transaction was
  committed against `FakeProofMixer`; the rows are retained for provenance.
  The fake proof now uses a UUID-bearing exclusively-created run directory and
  run-local lock/WAL. The no-WING gate now fails if any of the four canonical
  stores changes during a run: manual-write events, decision traces, decision
  memory or the per-user console WAL. The E2E fixture now matches the complete
  secure bootstrap schema.
- EQ adjustments no longer implicitly enable the whole EQ block; proof probes
  no longer unmute channels or raise inactive sends/DCAs.
- The direct proof tool is plan-only until it is integrated with the backend
  exclusive writer and emergency generation.
- Pytest uses per-test evidence/runtime roots. The append-only stores remain at
  these preserved baselines: manual-write events `115380009` bytes / SHA-256
  `aaaba5d3b0dd7e9fb9f54b12affdf54ff444a16bba31c1d92297afda39941fc4`;
  decision traces `246109600` bytes / SHA-256
  `2392e248399f7c789b76f75af34bdb781df395a3924513c288567bc703d70556`;
  decision memory `20845559` bytes / SHA-256
  `47a38efe999e037ea3b8b3226498680135f9ea3e76447e6e98f6bf5d791eb724`;
  console WAL `11329` bytes / SHA-256
  `58d548ecd28593d1791ff7784b72c394ae644fa7b101331528eec2454dfe7bca`.
- Retained dry-run/replay/fake trace rows are quarantined at read time. Their
  original bytes and values stay auditable, but operator read models normalize
  them to offline/offline-simulated/unverified states and exclude them from
  verified/applied counts. The frontend independently refuses to render those
  sources as green physical proof.
- Automation trace reads now scan a bounded binary tail instead of parsing the
  full 246 MB append-only file. The current 500-row read measured `0.0032 s`
  and normalized to zero applied and zero verified rows, while the file hash
  remained unchanged.
- M005 execute mode now requires a clean checkout before artifact creation,
  the canonical source backend and loopback WebSocket URL, an operator gate,
  an operator-confirmed WING Rack firmware `3.0.5` identity challenge, a first
  immutable provenance event, and runtime identity v2. The proof runner starts
  its own backend only after provenance capture; identity binds Git state, 15
  safety-critical files, PID/nonce/time, request challenge and the pre-process
  digest. A lost apply or rollback response requires manual review. M006
  recomputes the current packet
  verifier rather than trusting a saved report. The historical 2026-07-11
  M005 behavioral artifact remains visible but is not accepted as current
  release provenance.
- Frontend negative/unknown readback and rollback states now fail red; stale
  payloads cannot re-enable apply after reconnect, and the supervised modal
  requires a connected, complete, fresh backend snapshot.
- The master audio recheck requires a SHA-256-bound route artifact, both
  channels 47/48 and fresh finite signal; an environment flag alone grants no
  authority.
- Verified results: `327` focused server/WAL safety tests, `157`
  frontend unit/contract tests, `10` Chromium E2E tests, clean TypeScript and
  production build, and `6039 passed, 1 skipped, 1 warning` for the complete
  repository suite including all WING and writer-lease rails.
- The earlier no-WING release package at
  `artifacts/no_wing_release_check/20260712T033415Z_no_wing_release_check/`
  passed its then-current `63` checks, but is superseded because it guarded only
  the manual-write log and console WAL. A fresh post-lease four-store no-WING
  run is required before final handoff.
- Audit repros and historical fake runners left `25,730` dry-run rows in the
  canonical decision-trace store plus related unbound decision-memory rows.
  They are retained for provenance, normalized to offline at read time and
  explicitly excluded from physical/readiness evidence.

## Final P1 closure addendum — 2026-07-12

The final independent pass found and closed three remaining local P1 gaps
without widening authority:

- The packaged backend no longer relies on unavailable Git metadata. The
  superseding local build at
  `frontend/dist/local/20260712T072932Z_9e3d9f29-5ccf-41ca-a46e-f70933a45b23_local/`
  embeds a self-hashed 15-file source manifest and fails closed on manifest
  tampering. Its challenged runtime identity binds the exact backend binary,
  launcher-token digest and observed backend PID.
- M005 now verifies an initial proof-owned non-observation WING connection
  before baseline and uses a separate observation-only recovery connection.
  Executed evidence is confined to the canonical ignored artifact root, and
  any generic exception after apply/rollback requires manual review. M006
  requires the exact two-connection event sequence.
- Legacy trace normalization recursively quarantines nested unbound physical
  claims, including claims hidden beneath blocked or pending top-level rows.

The superseding packaged smoke at
`artifacts/packaged_runtime_smoke/20260712T073254Z_af976647-7612-419d-a5a5-9922930b203e_packaged_runtime_smoke/result.json`
passed all 18 identity checks with backend ready in `5038/20000 ms`, zero
external/WING sockets and all four canonical stores unchanged. Focused
post-fix verification passed `282` backend and `34` macOS package-contract
tests. This was an intermediate checkpoint; the final local engineering score
and the stricter public-release score are recorded separately below.

Final breadth-first validation then passed `6091` backend tests, `168`
frontend unit/contracts, production TypeScript/build, `10` Chromium E2E tests
and all `65` four-store no-WING checks. Detached manifest verification passed;
manual-write events, decision traces, decision memory and console WAL retained
their exact baseline sizes and SHA-256 values. This closes the remaining local
validation gate but does not replace attended WING/owner/public-release proof.

## Final local engineering closure — 2026-07-12

The breadth-first repair pass is complete at `95/100` for the local engineering
scope. The final evidence set is:

- `6197 passed, 1 skipped, 1 warning` in the complete backend suite;
- `189` frontend unit/contracts, production build and `13` Chromium E2E tests;
- local arm64 package
  `frontend/dist/local/20260712T124936Z_f1894655-feea-48d3-b184-a41c5cb6d739_local/`;
- packaged runtime smoke
  `artifacts/packaged_runtime_smoke/20260712T125342Z_145618f8-ef94-4eda-9b90-43eb995a3289_packaged_runtime_smoke/result.json`,
  with all 18 identity checks, zero external/WING sockets and four unchanged
  canonical stores;
- `65/65` final no-WING checks plus detached manifest verification at
  `artifacts/no_wing_release_check/20260712T135456Z_no_wing_release_check/`;
- real WING Rack read-only signal observation at
  `artifacts/soundcheck_runtime_observation/20260712T093447Z_soundcheck_runtime_observation/summary.json`:
  30.003 seconds, 285 real-input/signal/master-signal frames, 48 channels, up
  to 47 active channels, Main/analyzer USB 47/48 and zero write/live events;
- final post-validation silence attempt at
  `artifacts/soundcheck_runtime_observation/20260712T131015Z_soundcheck_runtime_observation/summary.json`,
  which correctly blocked rather than accepting a silent capture. Direct
  readback confirmed USB 47/48 still map to MAIN L/R and Main remains unmuted at
  0 dB; all four canonical hashes stayed unchanged.

M005 behavior is closed by the passed physical artifact
`artifacts/wing_timeout_disconnect_proof/20260711T131505Z_wing_m005_disconnect_timeout_stale_proof/summary.json`
and the owner's confirmation. The owner also recorded the narrow M006 decision
for `fader_supervised` and gain-down-only `gain_supervised`. The technical M006
verifier still blocks promotion because the historical M005 artifact is not
bound to the current source/event-provenance contract. This distinction does
not reopen M005 behavior and does not grant live authority.

## Required before public-release 95

1. Execute the canonical backend transaction flow on the exact WING Rack and
   firmware with the operator present using only `apply_operator_proposal`, and
   capture physical snapshot/apply/readback/audio/rollback/emergency evidence.
2. From a clean checkout, rebind or migrate the closed M005 physical
   cable-loss, timeout, stale-readback, blocked-write and reconnect-recovery
   evidence to the current source/runtime-identity-v2 contract. Do not repeat
   the cable pull unless the owner explicitly requests it.
3. Produce an immutable bundle bound to source commit/tree, runner/backend/
   package digests, console serial, firmware, capture-route hash, master
   channels 47/48 and timestamps.
4. Re-run the M006 technical verifier for the already recorded narrow owner
   decision after current-source M005 provenance is accepted; broad families
   retain zero standing authority.
5. Reproduce the real current-tree arm64 packaged-app launch from a clean
   checkout, prove operator-accessible remote-token provisioning and
   invalid/expired-token recovery without persistence or disclosure, and
   complete Developer ID signing, notarization, stapling and Gatekeeper
   acceptance.
6. Perform operator-present room/PA listening, audible rollback, stereo master
   47/48 route/true-peak and emergency-stop validation.
7. Finish a clean-checkout full suite plus reproduction of the final no-WING,
   hashed-dependency and arm64 package evidence after releasing the writer
   lease, preserving the canonical log hash and excluding the documented
   synthetic audit rows from evidence intake.

## Safety boundary

No live WING write was performed by the final audit validation. The audit did
perform read-only console observation and route/state queries. Hardware proof
does not grant standing authority, a green test does not replace audible room/
PA proof, and public-release readiness must remain below its `95/100` threshold
while any item above is open.
