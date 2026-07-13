# M006 scoped live soundcheck — 2026-07-11

> **Superseded readiness interpretation — 2026-07-12.** This document records
> a historical, one-time operator-approved execution; it does not describe the
> current normal live allowlist, which is exactly empty. The audit contains 51
> logical operations, of which 39 changed physical values and 12 were already
> satisfied. Under the pre-fix monolithic current-policy path, 15 local
> residuals would have blocked the whole plan and produced 0/51 execution.
> The replacement architecture and current scores are in
> `Docs/reports/SOUNDCHECK_PLAN_DECOMPOSITION_20260712.md`.

## Outcome

The canonical soundcheck path now derives absolute semantic operations and
applies one reviewed normal plan through `apply_operator_proposal`. The operator
does not approve every EQ band or fader separately. High-risk families remain
isolated into their own transactions.

Real WING Rack firmware: `3.0.5-4-ga4ea833d:release`.

- Physical named inputs analysed: channels 1–16.
- Channels with signal during accepted run: 14.
- Accepted normal transaction: `soundcheck:normal:13c43cc98a3c487c`.
- Result: 51/51 logical operations reached applied/already-satisfied outcome;
  39 changed physical state and 12 were already at target.
- Physical snapshots: 51.
- Double physical readback verifications: 51.
- Fresh master-channel audio rechecks: 51.
- Rollbacks: 0.
- Emergency stop: not activated.
- Inactive DCA transaction: 1 attempted, 1 rolled back; no emergency stop.

The inactive DCA was physically at negative infinity and had no assigned
sources. A target at `-138 dB` cannot form a meaningful WING state transition,
so the planner now emits `inactive_dca_target_is_negative_infinity` instead of
claiming a fake successful correction.

## Historical one-time operation scope (not standing authority)

Write kinds exercised in that bounded historical run:

- fader, gain, trim, mute;
- pan, polarity, delay;
- HPF, cut-only EQ;
- gate, compressor;
- send;
- bus, DCA;
- master fader.

Still proposal-only because their exact write kind has not passed a scoped
non-destructive hardware proof:

- expander, generic dynamics, de-esser;
- FX model, insert, return;
- group, matrix;
- output processing;
- routing, scene, snapshot and preset.

## Defects found during the real run

- Bus compressor threshold used the wrong firmware address `/dyn/thr37`; WING
  3.0.5 uses `/dyn/thr`.
- Status and decision-queue payloads could exceed WebSocket frame limits.
- The server waited for master true-peak evidence that the engine did not
  publish. It now measures master channels 47/48 and adds a conservative 3 dB
  inter-sample allowance.
- WING logarithmic EQ frequency and dynamics time quantization was treated as a
  readback mismatch. Adapters now use parameter-specific physical tolerances
  and verified physical values as rollback CAS anchors.
- Rollback stopped after its first failed operation. It now continues restoring
  earlier operations and records snapshots in the transaction event trail.
- EQ boosts could enter a cut-only live plan. They now remain review evidence
  and are excluded from live transactions.

## Validation

- Frontend tests: 104 passed.
- Frontend production build: passed.
- Focused registry/transaction/adapter/server tests: passed.
- WING MVP truth and safety rails: 135 passed.
- Post-rail WING/transport suites: 328 passed.
- Repeated full-suite runs reached the truth-surface rail with 5434 tests
  passed before the fail-closed documentation checks; after synchronizing the
  ledger, heartbeat, blocker, roadmap and progress surfaces, both affected
  segments pass independently.
- `python3 -m py_compile`: passed for all new modules.
- JSON validation and `git diff --check`: passed.

## Score

Historical scoped run score: **96/100**. This score is superseded as a current
readiness claim and grants no standing authority.

- Architecture and single-writer transaction path: 20/20.
- Physical safety/readback/rollback: 20/20.
- Real signal and WING integration: 20/20.
- Operator UX and responsive visibility: 18/20.
- Regression and evidence quality: 18/20.

The score does not claim concert release closure. Room/PA validation with a
measurement microphone, an integrated one-minute master recorder/uploader, and
hardware proof for the remaining destructive or ungenerated write kinds are
outside this 96-point scoped result.

## Evidence

- M005: `artifacts/wing_timeout_disconnect_proof/20260711T131505Z_wing_m005_disconnect_timeout_stale_proof/summary.json`.
- Family proofs: `artifacts/wing_supervised_family_proof/`.
- Transaction audit: `artifacts/wing_telemetry/manual_write_gate/manual_write_events.jsonl`.
- Frontend: `artifacts/frontend_qa/20260711_m006_full/soundcheck-live-applied-desktop.png`.
