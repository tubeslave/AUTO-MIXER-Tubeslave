# Kimi critique of Codex proposal

## 1. Strongest part of the other proposal

The trust-boundary grouping and sink-local fail-closed fixes are the right core
for a dirty live-sound checkout. Reusing existing archive and supervised
transaction controls avoids a second authorization framework. High-severity
live mutation, transport replacement, and false-success evidence also belong
first.

## 2. Possible mistakes or blind spots

- The Codex matrix has no mechanical 143-row completeness invariant. The
  existing partial progress report is evidence that prose alone permits silent
  omission.
- The named final full-suite gate may be unrunnable if the previously reported
  pipe hang still reproduces; the plan must investigate it rather than merely
  name the command.
- Per-sink authorization guards are hard to audit across many unsupported
  legacy tools. A shared quarantine guard can make omissions detectable, while
  supported transports still need sink-local assertions.
- File clustering is needed inside severity tiers to avoid repeatedly editing
  central dirty files.

## 3. Where the solution is too complex, fragile, or costly

Repeated occurrence-by-occurrence edits to `backend/server.py` and replay
modules multiply conflict risk. A matrix that evolves during patching can drift
and leaves only prose until the connector returns. Reusable versus sink-local
validation also needs an explicit boundary rule to prevent divergent guards.

Kimi accepts Codex's refinements: Phase 0 may retain `unknown` only with an
assigned executable proof task; the alleged pipe hang must be reproduced before
any test-harness change; no-WING quick belongs at cohort boundaries; and the
checker must never execute commands stored in the ledger.

## 4. What should be kept from the other proposal

- Six root-cause classes and high-to-low risk ordering.
- Dry-run retention plus fail-closed live mutation for unsupported legacy tools.
- Sink-local transport/readback/CAS/rollback assertions for supported paths.
- Explicit dirty-tree CP05/lease blocker handling.
- `locally_verified_unsealed` until a fresh full scan agrees.

## 5. My revised recommendation

Freeze and schema-check all 143 rows first, with a disposition or assigned
proof task for every row. Reproduce any validation-harness blocker. Remediate
in severity tiers clustered by file, using a shared guard only to quarantine
legacy mutators and sink-local validation elsewhere. Run targeted checks per
slice, no-WING per cohort, and full pytest at severity milestones. Finish with
a data-only 143/143 checker and a fresh full Codex Security scan.
