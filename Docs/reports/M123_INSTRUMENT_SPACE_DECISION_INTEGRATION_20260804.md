# M123 Instrument Space decision integration

Date: 2026-08-04

Status: software implementation complete; final post-truth validation scheduled

## Outcome

Instrument Space is now a real read-only decision-evidence path rather than a
visual-only projection. Measured pan, spectral centroid and channel level form
deterministic pair relations. Relations rank existing analyzers and annotate
already-created soundcheck proposals without choosing a console value.

## Runtime path

1. The server-owned audio capture publishes bounded, asynchronous per-channel
   spectral metrics.
2. `build_channel_inventory()` combines cached console pan/source identity with
   audio level and spectral evidence.
3. `InstrumentSpaceHistory` retains at most 12 in-memory snapshots and resets
   when mixer/audio source identity changes.
4. Decision support computes pair competition and snapshot drift.
5. Soundcheck proposal review receives only fresh, read-only annotations with
   measurement provenance.
6. The dashboard renders objects, relations, drift, provenance and next
   analyzers without drag or apply controls.

## Safety boundary

- Instrument Space has no mixer client, proposal queue, executor or transport
  dependency.
- Every scene, history, relation, drift row and annotation denies queue
  mutation, console apply, live authority and readiness closure.
- Stale or undated decision evidence is rejected at proposal annotation.
- Centroid proximity is triage, not an EQ diagnosis.
- Existing source identity, frequency separation, mono/phase, true-peak,
  feedback, Safety Kernel, operator approval, exact mapping, readback, rollback,
  WAL/audit and per-console hardware gates remain authoritative.
- WING and dLive hardware evidence is not shared or inferred from this software
  path.
- WING pan may use fresh cached OSC readback. dLive pan remains null until its
  exact read mapping and physical readback are verified; frequency/level-only
  relations carry `pan_evidence_missing` rather than inventing a coordinate.

## Evidence

- Focused backend/runtime tests: `43 passed`.
- Broad Product Layer/M123/soundcheck regression: `337 passed`.
- Pre-final full frontend contract suite: `278 passed`.
- TypeScript `--noEmit`: passed after the no-WING gate exposed and the patch
  closed an incomplete safe fallback DTO for history/decision support.
- Production Vite build: passed.
- 64-channel bounded diagnostic: 2,016 evaluated pairs, at most 128 emitted;
  200 runs measured 16.677 ms mean, 18.590 ms p95 and 19.882 ms max locally.
- The 250 ms test ceiling is a regression guard, not a production SLA.

## Remaining validation

After this final truth/ledger synchronization, repeat the broader
M123/soundcheck regression, frontend suite/typecheck/build, no-console release
check and standard backend suite. A dirty-tree CP05 failure must remain visible
and cannot be bypassed.
Attended WING/dLive proofs remain external and cannot be replaced by green
software checks.
