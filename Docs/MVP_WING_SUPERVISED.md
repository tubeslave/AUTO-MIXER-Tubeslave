# MVP: Supervised WING Rack Proof

Status: current execution contract

## Purpose

This is the current MVP rail for Automixer. The long-term goal is a capable AI
mixing engineer, but the current project milestone is narrower: prove one safe
supervised hardware loop on WING Rack before broad autonomy work continues.

## In Scope

The MVP includes only:

1. WING read-only connection proof.
2. Physical console readback proof.
3. Supervised fader write proof with explicit operator approval.
4. Supervised gain-down / trim proof with explicit operator approval.
5. Rollback proof.
6. Emergency-stop proof.
7. Connection-loss / timeout fail-closed proof.
8. Operator-visible evidence report.
9. Readiness matrix update based only on real proof.

## Out Of Scope

These are not part of the current MVP:

- Full autonomous live mixing.
- Live EQ, compressor, gate, routing, mute, FX, scene, matrix, or DCA writes.
- New broad proposal-only surfaces.
- New shadow-only methods that do not close a blocker below.
- New memory-trust, queue-import, result-consumer, report-only, or read-model
  chains that do not directly close a blocker below.
- Measurement-microphone / room-tuning workflows. They are useful later, but
  not required for the current console proof gate.

Correction-method development for Gain/Trim, Pan, EQ, Comp, FX, Balance, and
related families continues through `Docs/CORRECTION_METHODS_ROADMAP.md`, but
before this MVP is proven those methods must remain offline, dry-run,
proposal-only, read-only, and tested. They must not widen live writes.

## Source Of Truth

Use these files in this order:

1. `AGENTS.md`
2. `CLAUDE.md`
3. `Docs/MVP_WING_SUPERVISED.md`
4. `.ai/project-layer/wing_mvp_blockers.md`
5. `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
6. `Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md`
7. `Docs/reports/MVP_PROGRESS_LATEST.md`
8. `Docs/CORRECTION_METHODS_ROADMAP.md`
9. `.ai/project-layer/correction_methods_backlog.md`

## Definition Of Done

The MVP is done only when every row in
`.ai/project-layer/wing_mvp_blockers.md` is `closed` with real proof artifacts.

Offline tests, dry-run artifacts, shadow-only evidence, valid report shape, or
readiness candidates are not enough to close a blocker by themselves.

## Correct Work Order

1. Run read-only preflight against the intended WING backend.
2. Verify physical readback from the artifact and event log.
3. Review the clean preflight candidate manually.
4. Run supervised fader proof only with explicit operator approval.
5. Run supervised gain-down / trim proof only with explicit operator approval.
6. Prove rollback.
7. Prove emergency stop.
8. Prove connection-loss / timeout fail-closed behavior.
9. Update the readiness matrix from real evidence.

## Stop-The-Line Rules

Stop the current Codex work direction and return to the first open blocker if:

- `tools/codex_autoloop/codex_direction_guard.py` reports `drifting`.
- The work adds memory/queue/result-consumer/report-only layers without closing
  a blocker.
- The work treats candidate, shadow, offline, or dry-run evidence as live
  readiness.
- A supervised proof is planned before clean read-only preflight evidence is
  available for manual review.

This is not a stop to development. Codex must continue uninterrupted Goal-mode
work, but only on the first non-closed MVP blocker.

## Current Next Action

Run the read-only preflight runbook when WING Rack and backend are ready:

`Docs/runbooks/WING_READONLY_PREFLIGHT.md`
