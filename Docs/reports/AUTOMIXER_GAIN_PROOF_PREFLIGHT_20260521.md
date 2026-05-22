# Automixer Gain Proof Preflight - 2026-05-21

## Status

Status: **prepared, physical WING proof not run**.

This document is the live-validation checklist for the next narrow hardware
step. It does not authorize autonomous apply or any EQ/routing/snapshot write.

## Non-Live Dry Run Evidence

The fake/in-process gain proof runner passed:

- Run id: `20260521T161344Z_gain_proof_dry_run_ch3`
- Summary:
  `artifacts/wing_telemetry/gain_proof_dry_run/20260521T161344Z_gain_proof_dry_run_ch3/summary.json`
- Physical WING: `false`
- Write policy: `fake_mixer_apply_operator_proposal_only`
- Apply proof: `verified`
- Apply readback: `verified_from_physical_readback`
- Rollback proof: `verified`
- Rollback readback: `verified_from_physical_readback`
- Emergency blocked proof: `emergency_stop_active`

Latest refreshed fake proof:

- Run id: `20260521T233529Z_gain_proof_dry_run_ch3`
- Local date: 2026-05-22 Europe/Moscow
- Summary:
  `artifacts/wing_telemetry/gain_proof_dry_run/20260521T233529Z_gain_proof_dry_run_ch3/summary.json`
- Physical WING: `false`
- Focused safety/runner tests: `73 passed`
- Cooldown proof: blocked
- Emergency blocked proof: `emergency_stop_active`

Previous refreshed fake proof:

- Run id: `20260521T175356Z_gain_proof_dry_run_ch3`
- Summary:
  `artifacts/wing_telemetry/gain_proof_dry_run/20260521T175356Z_gain_proof_dry_run_ch3/summary.json`
- Physical WING: `false`
- Focused safety/runner tests: `73 passed`
- Cooldown proof: blocked
- Emergency blocked proof: `emergency_stop_active`

## Required Live Preconditions

Before physical gain proof:

1. WING Rack is connected and read-only preflight is green.
2. Operator mode is `Supervised`.
3. Emergency stop latch is clear. If it is active, reset requires explicit
   operator approval and audit event.
4. Target channel is known and low risk.
5. Current/old gain value is physically/readably captured.
6. Proposed move is a small gain trim-down only, default `-0.25 dB`.
7. Rollback target is the captured old value.

## Allowed Live Flow

Only this flow is allowed:

1. Read current gain value.
2. Create or select one Product Layer gain proposal.
3. Apply only through `apply_operator_proposal`.
4. Verify apply readback.
5. Roll back through the supervised rollback path.
6. Verify rollback readback.
7. Trigger emergency-stop blocked proof.
8. Save telemetry, backend logs, frontend state, and summary artifacts.

## Forbidden During This Proof

- No direct `set_gain` or `set_fader` websocket messages.
- No EQ, compressor, routing, FX, phase/delay, snapshot, scene, training, or
  rule writes.
- No Auto mode.
- No autonomous agent apply.
- No multi-channel proof.

## Success Wording

If all live proof points pass, update project status to:

**Supervised hardware-proven for fader and one gain trim-down path.**

If readback or rollback proof is missing, status remains `partial/blocked`.
