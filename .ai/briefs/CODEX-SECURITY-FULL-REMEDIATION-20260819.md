# CODEX-SECURITY-FULL-REMEDIATION-20260819

## Goal

Remediate every reportable first-party finding from the sealed Codex Security
scan and its later checkpoint scans without widening live-console authority or
overwriting unrelated work in the collaborative checkout.

## Authoritative inputs

- Sealed scan `2c00402a-b02e-4b80-a68f-c9c079eb70c4` at revision
  `f8a33c44f0144bbc62d6cadc135493ed10cd496e`: 143 reportable findings,
  77 high, 59 medium, 7 low, partial coverage.
- Checkpoint scan `59cb954a-42ab-4eae-b6d0-eba34941710c`: 12 findings.
- Post-fix checkpoint `e01f678c-e332-4688-b22c-2a9a9ab1d8e9`: zero findings
  only across its explicitly bounded surfaces; it is not a replacement full
  scan.
- `Docs/reports/CODEX_SECURITY_REMEDIATION_PROGRESS_20260807.md` records local
  verification for 39 original occurrences plus later checkpoint siblings.
  It does not establish closure for the remaining original occurrences.

## Current checkout constraints

- Branch: `codex/automixer-autopilot-control-plane` at `f8a33c44f`.
- The tree contains extensive staged and unstaged work owned by earlier tasks.
  Preserve it and edit only exact remediation-owned hunks.
- The previous writer lease was stale and has been replaced under the current
  task before production edits.
- Codex Security connector access is currently unavailable in this task, so
  authoritative local artifacts drive remediation. A fresh current-checkout
  scan remains required for workbench closure.
- Filesystem free space is low; do not copy the repository or generate broad
  artifact trees.

## Safety boundary

- Offline software work only. Do not start the backend, enumerate audio
  devices, discover LAN endpoints, open WING/dLive/Ableton transports, or send
  OSC/MIDI.
- Preserve `ANALYZE_ONLY` / proposal-only behavior and deny legacy direct-write
  utilities unless they enter the canonical supervised transaction path.
- Authentication alone is not live-write authority. Each operation family
  still requires current identity, explicit operator approval, bounded
  pre-state, exact readback, rollback, and teardown evidence.
- Static green tests do not establish console motion, acoustic safety, or
  concert readiness.

## Remediation classes

1. Live-write authorization and rollback: handlers, legacy CLIs, replay and
   proof tools must fail closed before transport unless using an existing
   operation-specific supervised transaction.
2. Transport authentication and destination binding: WebSocket, WING, dLive,
   Mixing Station, Ableton, MCP, Telegram, and HTTP clients must bind peers,
   ports, origins, owners, and credentials.
3. Filesystem containment and artifact integrity: all caller-controlled names,
   roots, run IDs, session IDs, rollback files, and model outputs remain under
   server-owned roots with no-follow and digest/schema checks.
4. Resource bounds and numeric safety: reject booleans masquerading as
   integers, non-finite values, archive/decompression bombs, unbounded XML/JSON
   or audio dimensions, and uncontrolled training work.
5. Replay/evidence integrity: recompute required hashes/state transitions,
   preserve supplied initial state, require explicit booleans, and never infer
   readiness from missing evidence.
6. Credential handling and subprocess boundaries: no secrets in argv or
   plaintext world-readable files; executable vectors and remote content are
   allowlisted and instruction-isolated.

## Required evidence

- A disposition row for every sealed occurrence: fixed by exact patch, closed
  by a verified overlapping root-cause fix, not applicable with concrete
  reachability proof, or still blocked. No silent omission.
- Focused pre-fix/post-fix tests for each root-cause class.
- Security regression aggregate, `tools/no_wing_release_check.py --quick`, the
  standard full pytest command, dependency audits where relevant, syntax and
  `git diff --check`.
- Final ADR and remediation report that separate software evidence from
  external WING/dLive/acoustic proof.

## Rollback

Every remediation slice must be an isolated, reviewable diff. If a change
conflicts with existing work or breaks an intended bounded path, revert only
the task-owned hunk and keep the affected occurrence open.
