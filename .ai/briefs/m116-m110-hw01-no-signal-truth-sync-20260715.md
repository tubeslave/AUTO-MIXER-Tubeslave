# M116 M110 HW01 No-Signal Truth Sync

## Goal

Replace the stale canonical claim that no physical WING exists with the exact
operator-attended HW01 result, without promoting an incomplete route attempt or
granting any live authority.

## Verified current facts

- Source checkout was the clean detached M115 tree at
  `cc1a2ca503d424b2e3dd307be3cbe66f296dcc6c` with a passing dynamic CP05 gate.
- WING Rack `wing-rack` at `192.168.1.102`, serial `0100Q1V0604AAE`, reported
  firmware `3.0.5-4-ga4ea833d:release`.
- CoreAudio device `WING` was opened as 48 inputs, 48 outputs and 48 kHz.
- The proof-owned backend contacted WING and opened a 48-channel capture, then
  failed closed before route issuance with `master_channel_signal_missing`.
- The runner summary is the ephemeral local diagnostic
  `/Users/dmitrijvolkov/.automixer/runtime/m110_attended/plan-summary.json`,
  SHA-256 `3a300561f3a6b31099cf2586f0047d674ff1342abb767f761b856f384ef91273`.
- Its backend log is
  `/Users/dmitrijvolkov/.automixer/runtime/m110_attended/proof-owned-backend-2c7407d5ba86e1fd4b17084323df08d5.log`,
  SHA-256 `19d0d41ae6e52321f4ebf55e2dfe822bdbd500b7ad37ecaa68542c87c929e490`.
- Operator-observed short and 120-second read-only captures showed exact
  digital zero on all 48 channels. Operator-observed read-only console checks
  found Main 1 unmuted, USB 47/48 assigned to Main L/R and channel/input meters
  silent. These observations are not independently machine-verified and are
  not accepted route proof.
- The attempt performed zero reset, zero correction and zero live write; the
  backend, stream, OSC path and writer lease were torn down cleanly.

## Required truth

- `M110-HW01` is attempted and blocked at its master-signal gate, not blocked
  before start and not complete.
- The ephemeral diagnostic is not an issued route artifact and cannot satisfy
  HW01, CP05 physical evidence, readiness-matrix promotion or live authority.
- HW02-HW05 remain pending.
- Strict real-live readiness remains `88/100`; normal allowlist remains `0`;
  `safe_core_ready` and `show_mix_ready` remain unproven.
- The next attended action is to start a confirmed non-silent source into WING,
  physically contain outputs, keep playback running and repeat HW01 from a
  clean CP05-bound checkout.

## Constraints

- No unattended WING, OSC, CoreAudio or backend contact during M116.
- Preserve append-only ledger history and unrelated generated artifacts.
- Update tests that currently pin the stale missing-device claim.
- Run the full validation contract and rebind CP05 marker-only after the
  non-marker M116 commit.
