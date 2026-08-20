# Kimi proposal — full Codex Security remediation

## 1. Short thesis

Freeze a machine-readable 143-row disposition ledger before production edits,
prove current-checkout reachability for every occurrence, then fix in
file-clustered slices behind a uniform quarantine guard. No occurrence moves to
`fixed` without an executable pre/post probe; prose-only dispositions are not
accepted.

## 2. What I understood

- The sealed scan contains 143 reportable findings, while 104 original
  occurrences have no explicit disposition in the current progress report.
- The checkout is heavily dirty, so the conflict unit is the file and its
  existing hunks. Severity ordering alone would repeatedly touch central files
  such as `backend/server.py` against a moving baseline.
- A fresh connector-backed scan is unavailable, so local results remain
  `locally verified, unsealed`.
- Full validation may itself be blocked by a pipe-discipline hang and the
  expected writer-lease/CP05 dirty-tree rails. These must be distinguished from
  remediation regressions.
- Disk is nearly full; broad repository copies and artifact trees are unsafe.

## 3. Proposed solution

Phase 0 creates a 143-row disposition ledger with occurrence, severity,
locations, root-cause class, current reachability, guard/removal evidence, and
the exact regression test. Rows are `reachable`, `guarded`, `path_gone`, or
`unknown`; `unknown` remains blocked.

Phase 1 makes the standard verification gate runnable if the reported
`tests/test_supervised_proof_capability.py` pipe hang reproduces. Writer-lease
and CP05 dirty-tree failures remain named blockers, not silently waived tests.

Phase 2 adds a small shared fail-closed quarantine guard for legacy mutation
surfaces that are not integrated with the canonical supervised transaction.
Such tools retain dry-run/plan behavior and exit unavailable for live mutation.
This is a guard, not a new capability broker.

Phase 3 remediates high, then medium, then low findings, batching within each
severity tier by touched file. Each slice snapshots the pre-existing scoped
diff, adds a failing probe, applies the minimal fix, runs the post-fix and
positive-control probes, the owning suite, diff hygiene, and no-WING quick.
Rollback removes only task-owned hunks.

Phase 4 adds a read-only disposition checker that verifies 143 rows, reruns
the named regression nodes and class-level sink scans, and rejects blocked or
unknown rows. A fresh full Codex Security scan remains the only workbench
closure authority.

## 4. Likely files to touch

- New ledger/checker, focused security tests, and a small shared legacy
  live-write guard.
- Central handlers/transports, replay modules, ML/archive loaders, session and
  configuration stores, legacy WING tools, MCP/Telegram bridges, and native
  CoreAudio code cited by the sealed findings.
- ADR, final report, and autopilot truth surfaces only after source and tests
  are final.

## 5. Alternatives considered

- Pure severity-first interleaving: keeps risk ordering but creates repeated
  conflicts in already dirty central files. Retain severity tiers, then cluster
  by file inside them.
- Central capability broker: too broad and itself security-sensitive.
- Clean-room branch: risks losing or misbinding the current dirty state.
- Waiting for the connector: unnecessary for local proof, but fresh scan
  remains required for final closure.

## 6. Risks

- A shared guard is a common failure point; keep it small and independently
  tested, then rerun owning suites for every adoption slice.
- Ledger tooling can become a second test framework; restrict it to invoking
  existing pytest node IDs and deterministic sink scans.
- Reachability evidence can age as other tasks edit files; bind every guarded
  or removed path to a regression test.
- Disk exhaustion can invalidate evidence; use bounded temp roots and store
  commands/counts rather than copied payloads.
- Local green tests can create false confidence; retain `unsealed` language and
  keep physical-console proof separate.

## 7. Test plan

- Ledger parses to exactly 143 rows and the 77/59/7 severity histogram.
- Every occurrence has a current reachability proof and named regression.
- Each slice passes pre/post/positive probes, owning suite, compilation, scoped
  diff hygiene, and no-WING quick with authority denied.
- Each severity milestone runs the standard full pytest command after any
  verification-harness blocker is corrected.
- Final disposition checker passes 143/143, dependency/native checks pass where
  relevant, and a fresh full current-checkout Security scan is requested.

## 8. Where the other agent may disagree

Codex may prefer sink-local guards and a matrix that evolves during patching.
Kimi recommends freezing the ledger first to prevent omissions and using one
small shared quarantine guard for unsupported legacy live-write surfaces. Kimi
also treats a reproducible full-suite hang as an in-scope evidence blocker that
must be fixed before production remediation claims can be finalized.
