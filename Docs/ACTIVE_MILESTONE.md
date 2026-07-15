# Active Milestone

ID: M001

Title: WING read-only preflight evidence gate

Status: WAITING_FOR_EXTERNAL_VALIDATION

## M118 M111 sealed replay zero-I/O rail — 2026-07-15

M118 closes a proof-harness regression without widening live scope. The sealed
replay now preloads urllib3 under already-active socket guards with only its
import-time IPv6 probe suppressed, restores the real capability before project
imports, and asserts all writer/network/audio counters are zero. Pre-CP05
validation passed 9 migration tests (2 skipped), 137 WING rails (1 dynamic gate
deselected) and 7215 repository tests (3 skipped, 1 deselected, 1 warning).
The tracked change invalidates the previous M111 challenge; commit, marker-only
CP05 rebind and a fresh challenge remain required. No backend, audio or WING
contact occurred, and strict readiness remains `88/100`.

## M117 diagnostic provenance rail — 2026-07-15

M117 corrects current truth only. The M116 route-gate failure is
machine-recorded; its additional all-channel captures are operator-observed and
not independently machine-verified. The append-only M116 contact row remains
intact after the new offline/no-contact M117 row. No score, hardware checkpoint,
normal allowlist or live authority changes.

## M116 real WING reached; HW01 blocked on digital silence — 2026-07-15

The global M001 shell remains unchanged. Operator-directed active work is
M110: CP00-CP05 are complete under the canonical source-bound marker contract.
The clean M115 checkout at `cc1a2ca503d424b2e3dd307be3cbe66f296dcc6c`
reached WING Rack serial `0100Q1V0604AAE`, firmware
`3.0.5-4-ga4ea833d:release`, and opened the CoreAudio `WING` stream at 48
inputs, 48 outputs and 48 kHz. The machine-recorded route gate then blocked
fail-closed before artifact issuance on `master_channel_signal_missing`.
Operator-observed short and 120-second diagnostics showed exact digital zero on
all 48 channels; those extra observations are not independently
machine-verified. HW01 is incomplete; HW02-HW05 remain pending. Reset,
correction and write counts are zero, and serial-bound USB identity was not
reached.

The committed M115 product/runtime baseline is
`cc1a2ca503d424b2e3dd307be3cbe66f296dcc6c`: `7214 passed, 3 skipped in
428.23s`, 136 WING/static rails, and the 66/66 no-WING capsule at
`artifacts/no_wing_release_check/20260714T225611Z_no_wing_release_check`.
This does not replace the historical `5c054eb9e` launch-only provenance and
does not start HW01 or grant authority.

M113 requires exact integer 48 input, 48 output and 48000 Hz CoreAudio
metadata in both production collection and independent route-artifact
verification. Output channels are now identity-hash-bound; missing, wrong or
coerced metadata blocks, and output-mode refresh failure revokes authority.

M005 behavior is closed. The operator's M006 permission is recorded only for
one CP05-bound attended session and is not standing authority. Reset means 64
processor bypass addresses on channels 1-16 followed by exact removal of the
known M006 corrections: stages 23/14/14, scopes 33/63/14, changed 19/52/6 and
77 unique changed addresses; state outside the archived M006 transaction is
preserved. The full route -> reset -> M006 restore -> analysis -> staged lanes
-> two HW05 audits with SHA-bound listening path is implemented.

M111 adds a fail-closed, offline-only `verified_migration` route for the exact
immutable M005 bytes. It does not relabel them as native current-release
provenance and grants no authority. Activation requires a fresh run/source-
bound challenge, exact same-console WING Rack serial attestation, clean
committed replay, recomputed packet verification and a new explicit M006
owner decision.

Until HW01-HW05 succeed, strict real-live readiness remains **88/100**,
not 95: normal allowlist `0`, standing authority `false`, and
`show_mix_ready` is not physically proved on this source. CP05 is checked
dynamically against the current non-marker Git tree; any tracked non-marker
change requires one canonical marker-only rebind before hardware work.
Canonical current details are in
`ops/autopilot/STATE.json.active_work`; marker files remain the hardware gate.

## M109-CP01 validation restored — 2026-07-13

The committed CP01 source checkpoint
`b3f7d05ac1f0a2ddc9812e4723a2b3f130fefa6a` passes the mandatory command:
`6368 passed,
1 skipped in 382.42s`. M109-CP01 made four positive synthetic/contract tests
artifact-history-hermetic, retained deterministic negative checks, and aligned
two current-evidence truth rails with the preserved invalid WING attempts.
Production validation and canonical WING evidence are unchanged.

Validated local engineering is restored to **95/100**. Strict real-live and
public-release readiness remain **88/100**, below the 95 target: normal
allowlist `0`, `soundcheck_incomplete`, no DSP reset, no applied correction,
and no live authority. Active hardening plan:
`Docs/plans/M109-local-baseline-wire-truth-hardening.md`. The older ledger-backed
checkpoint below remains the previous full handoff; this section is the newer
CP01 source-commit validation binding.

## Superseding soundcheck execution status — 2026-07-12

The local P1 monolithic-transaction defect is closed. Normal soundcheck work is
now decomposed into semantic child transactions under a content-bound,
journaled parent. Local residuals stay local; global disconnect, stale
readback, emergency, writer, rollback and Main-headroom hazards still stop the
run. Bounded replans and a down-only complete fader-vector translation preserve
safe progress. Final source-path loss revokes both run readiness and the
affected operation checkpoint.

Stage-local reanalysis is now isolated and atomic: only the exact
server-accepted bundle becomes the committed revision, and failed/cancelled
candidates cannot leak readiness, observation shadow or journal state.

Current scores are intentionally separate: full-project local engineering
`95/100`, staged execution kernel `96/100`, strict real-live soundcheck
`88/100`, strict public release `88/100`. The normal live allowlist remains
exactly `0`; real-WING `safe_core_ready` and `show_mix_ready` have not been
proved on this current tree. See
`Docs/reports/SOUNDCHECK_PLAN_DECOMPOSITION_20260712.md`.

M005 behavior remains closed and the M006 owner intent remains recorded. The
next physical step is not another M005 cable pull: it is a confirmed non-silent
source routed from Ableton or the physical inputs through WING to USB Main
47/48, followed by an operator-present repeat of HW01. Candidate disposition
and current-source M005/M006 promotion remain separate fail-closed requirements.

Current blocker from `STATE.json`: M110 now reaches the real WING Rack and its 48x48/48 kHz CoreAudio capture, so the missing-device prerequisite is closed. The operator-attended route-only HW01 attempt opened the proof-owned backend, OSC/UDP and 48-channel audio stream, but the machine-recorded route gate failed closed before artifact issuance with reason master_channel_signal_missing; operator-observed 3-second and 120-second captures showed exact digital zero on all 48 channels; these extra observations are not independently machine-verified. Therefore HW01 remains incomplete and HW02-HW05 have not started; reset, corrections and live writes remain 0. The local runtime summary is ephemeral diagnostic evidence only, not accepted physical route proof. M111 still requires exact same-console attestation, a clean committed compatibility replay and an explicit migration-bound M006 decision. Strict real-live readiness stays 88/100: normal allowlist 0, standing authority false, safe_core_ready/show_mix_ready unproven.

Next operator action from `STATE.json`: Do not widen live authority. Physically contain PA and monitor outputs; in Ableton select WING as the Audio Output Device, route the source to WING USB 1-16, start playback, verify a non-silent Main 47/48 feed and leave playback running. Then, with the operator present and a clean dynamic-CP05-bound checkout, repeat M110-HW01 and proceed to HW02-HW05 only after an issued route artifact. Separately provide the exact WING Rack serial/same-console attestation for the 2026-07-11 M005 run, complete the clean committed migration replay and record an explicit verified_migration M006 owner decision. Complete signing/notarization/Gatekeeper and production remote-auth recovery separately.

Latest ledger handoff row: `m119_private_artifact_root_verifier_cli_source_checkpoint_ready` at `2026-07-15T02:52:38Z`.

## Earlier superseding status — 2026-07-11

The full-project safety rebaseline supersedes earlier scoped `51/51 live`
wording. Local engineering readiness has reached `95/100`; strict public-release
readiness remains `88/100` and false. Ten console families are hardware-proven
candidates, not standing live authority, and normal supervised transaction live
write kinds remain empty.

The runtime now fails closed across registry, operator mode, deployment and
observation gates; keeps every direct WING WebSocket surface proposal-only;
uses a per-user global writer and recovery WAL with parent-directory fsync,
startup recovery latch, strict address-bound physical readback, atomic proposal
claims and per-address apply/rollback compare-and-swap. Frontend authority also
requires a complete fresh backend snapshot and an authenticated ACK after the
memory-only Electron bootstrap. Direct hardware-proof execution remains
plan-only until an operator runs it through the backend on the exact console.
The full findings and remaining gates are recorded in
`Docs/reports/FULL_PROJECT_READINESS_AUDIT_20260711.md`.

Full-release requirements before `95/100`:

- operator-present canonical backend transaction proof on the exact WING/firmware;
- current-contract M005 cable-loss, timeout, stale/blocked-write and reconnect proof;
- immutable source/runner/backend/package/console/firmware/route evidence bundle;
- M006 owner promotion for each exact family, write kind and safe direction;
- a clean-checkout replay of the passing current-tree arm64 packaged Electron
  launch (the existing proof is explicitly dirty-tree-bound);
- Developer ID signing, app/DMG notarization, stapling and Gatekeeper acceptance;
- trusted same-origin HTTPS/WSS proxy, secure token lifecycle and physical iPhone recovery proof;
- operator-present room/PA listening, master 47/48 true-peak, rollback and emergency validation;
- a clean-checkout full/no-WING/package run with synthetic audit rows excluded.

The shorter M005/M006 list later in this file is the active M001 external
handoff only, not the complete full-release checklist.

Latest local verification is the dirty-tree-bound, ad-hoc-signed arm64 build at
`frontend/dist/local/20260712T124936Z_f1894655-feea-48d3-b184-a41c5cb6d739_local/build-profile.json`
and packaged smoke at
`artifacts/packaged_runtime_smoke/20260712T125342Z_145618f8-ef94-4eda-9b90-43eb995a3289_packaged_runtime_smoke/result.json`.
The backend started in `5207/20000 ms`; all `18/18` challenged runtime identity
v2 checks bound
the self-hashed embedded 15-file source manifest, exact backend binary, fresh
launcher-token digest and observed backend PID. Every app/backend Mach-O was
arm64; no external or WING socket opened; the manual-write log, decision
traces, decision memory and console WAL were unchanged. This remains
`releaseEligible=false`.

The final local five-store no-WING gate at
`artifacts/no_wing_release_check/20260712T224514Z_no_wing_release_check/summary.json`
passed all `66` checks and detached manifest verification with no failed or
environment-blocked checks, no WING/backend contact or live authority, and all
five canonical stores unchanged.

The passed M005 artifact closes disconnect/timeout/stale-state behavior, but it
is not current-release provenance. Public-release work must rebind or migrate
that evidence from a clean checkout; it must not repeat the physical cable pull
unless the owner explicitly requests it. The recorded M006 decision path
recomputes current provenance rather than trusting a saved packet report.

## Desired end state

The project can resume autonomous work from a clean, finite plan that keeps
the supervised WING Rack MVP externally gated. The historical exact 2026-07-02
read-only source is accepted-ready, while the current aggregate ledger remains
`operator_review_required` until two valid 2026-07-13 candidates receive
explicit decisions. CH3 fader/gain-down proof artifacts are available, M005
behavior is closed and the M006 owner decision is recorded. M110 hardware
proof, technical promotion and public release remain gated; no live authority
is inferred.

The historical exact-source decision is accepted-ready; the current aggregate
decision is not. CH3 fader state restoration/verification is complete.

Real hardware proof now requires WING Rack only for current-source public-release provenance or a new owner-requested run; M005 behavior itself is closed.

Status marker: local engineering `95/100`; M005 behavior closed; M006 owner intent recorded; technical promotion and public release remain open.

## Why this advances the project goal

The first Project Goal dependency is a safe WING read-only proof. Without this,
supervised fader/gain proof, rollback, emergency stop, and later autonomy cannot
be trusted.

## Dependencies

- `Docs/MVP_WING_SUPERVISED.md`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `tools/no_wing_release_check.py`
- `tools/wing_readonly_preflight_runner.py`
- `tools/review_wing_readonly_preflight_artifacts.py`
- `tools/review_wing_readonly_preflight_decisions.py`

## Included scope

- Validate no-WING prestart behavior.
- Validate read-only preflight input contract.
- Confirm that all generated evidence denies live authority.
- Prepare exact operator/hardware handoff steps.
- Record evidence in `ops/autopilot` and project docs.

## Excluded scope

- No live OSC writes.
- No real WING contact during unattended automation.
- No EQ, dynamics, mute, routing, FX, scene, matrix, DCA or autonomous writes.
- No broad proposal-only feature work.

## Acceptance criteria

- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick` passes or
  its failure is recorded as a blocker with log path.
- `tools/wing_readonly_preflight_runner.py` input validation path is checked
  without live WING mutation.
- Evidence says `live_authority_granted: false` and
  `readiness_closure_allowed_by_packet: false` unless real hardware review
  explicitly proves otherwise.
- `ops/autopilot/STATE.json`, `RUN_LEDGER.jsonl`, `HEARTBEAT.md` and
  `LAST_REPORT.md` are updated.
- Required tests for changed files pass.

## Historical M001 evidence snapshot — superseded

Current M110, decision-ledger and operator-action truth is the handoff at the
top of this file. The entries below preserve the earlier M001 audit snapshot.

- Baseline pytest: `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
  passed with `4027 passed, 1 skipped`.
- No-WING quick gate: `PYTHONPATH=backend python tools/no_wing_release_check.py --quick`
  passed.
- New no-WING artifact bundle:
  `artifacts/no_wing_release_check/20260618T204503Z_no_wing_release_check`.
- CP04 handoff report:
  `Docs/reports/AUTOMIXER_AUTOPILOT_M001_CP04_WING_READONLY_HANDOFF_20260618.md`.
- Validate-only runner status for the fresh expected plan: `ready_for_websocket`
  with `websocket_opened: false` and `write_request_sent: false`.
- CP05 milestone review:
  `Docs/reports/AUTOMIXER_AUTOPILOT_M001_CP05_MILESTONE_REVIEW_20260618.md`.
- M001 autonomous pre-hardware work is complete, the canonical read-only
  decision is `accepted_decision_ready`, and CH3 fader state
  restoration/verification plus the fader/gain-down proof rerun have passed
  with physical readback, rollback and emergency-stop/block evidence. The
  earlier `rollback_not_applied` artifact remains closed audit history; the
  M005 disconnect/timeout/stale-state behavior is closed by the passed physical
  artifact, and the narrow M006 owner decision is recorded. Only the
  current-source provenance rebind and technical promotion remain open.
- Historical next operator action snapshot: Do not widen live authority. Treat M005 disconnect/timeout/stale-state behavior as closed and the narrow M006 readiness review owner decision as recorded; do not repeat the physical cable pull unless the owner explicitly requests it. For public release, rebind or migrate the closed M005 evidence to a clean current-source provenance packet, rerun the M006 technical verifier, reproduce the package from a clean checkout, complete Developer ID signing/notarization/Gatekeeper checks, and perform operator-present room/PA audible rollback validation.
- Historical ledger checkpoint:
  `local_engineering_95_final_tree_full_suite_no_wing_validation` at
  `2026-07-12T13:56:06Z`. The final Electron 43 arm64 packaged runtime passed
  its backend-only sandbox/auth/reload/socket smoke at
  `artifacts/packaged_runtime_smoke/20260712T125342Z_145618f8-ef94-4eda-9b90-43eb995a3289_packaged_runtime_smoke/result.json`.
  It contacted no WING and remains `releaseEligible=false`. No live authority
  changed; release-ready live
  correction families remain `0`, live-allowed safety rows remain `0`, and
  current supervised live-write kinds remain `0`.

Latest M005 readback target binding guard: `d3eb93a62`; M005 baseline, stale-readback and reconnect-readback payloads, including `events.jsonl` stale/reconnect payloads, now must match the requested `channel`, `value_type` and WING `address` before M005 completion or M006 owner-review input can be ready. This grants no backend/WING contact, proposal creation, live apply, readiness-matrix update or live authority.

Latest M005 physical reconnect target guard: `bbf7833e2`; executed `physical-cable-loss` M005 proofs now fail closed before backend/WebSocket contact unless `--reconnect-wing-ip` is supplied. This keeps the operator-present cable-loss proof bound to reconnect-recovery evidence and grants no backend/WING contact, proposal creation, live apply, readiness-matrix update or live authority.
- Latest M006 packet verifier hardening: `3ea749d70`; it requires a ready
  packet-verifier report with the current M005 stale-readback event-evidence
  checks and matching reconnect-recovery physical readback event checks from
  `events.jsonl`.
- Latest M006 owner decision verifier: `5eda78126`; it requires a ready
  packet-verifier report with the current M005 stale-readback event-evidence
  checks, matching reconnect-recovery physical readback event checks and the
  accepted M005 summary sibling `events.jsonl` path. The owner decision is now
  recorded, but technical promotion blocks with
  `reason=m006_owner_decision_failed_closed` because current-source M005
  provenance is not accepted; it does not grant a readiness-matrix update.

## Current external validation requirements

- Confirmed non-silent source reaches the physical WING and USB Main 47/48 during an operator-attended 48x48/48 kHz capture; digital-zero input cannot satisfy HW01
- Operator-present CP05-bound M110-HW01 through M110-HW05 run with route, reset, staged soundcheck, rollback, SHA-bound listening and clean teardown evidence
- Operator accept or reject the two valid 2026-07-13 read-only preflight candidates; this decision-ledger requirement grants no live authority and does not reopen closed M005 behavior
- Exact WING Rack serial and same-console owner attestation for the immutable 2026-07-11 M005 evidence; this is identity-only and grants no live authority
- Clean committed M005 current-release compatibility replay plus an explicit verified_migration M006 owner decision bound to the recomputed migration and packet-verifier paths

## Exact evidence

See `Docs/plans/M001-wing-readonly-preflight-evidence-gate.md`.

## Real WING interaction plan

The full operator/backend route for real-console work is
`Docs/plans/M001-real-wing-interaction-plan.md`. It now starts from the
accepted read-only decision gate and passed CH3 fader/gain-down proof
artifacts, while preserving the earlier `rollback_not_applied` artifact as
closed audit history. M005 behavior and M006 owner intent are closed; only
current-source promotion provenance and the broader public-release gates remain.

## Rollback

Rollback is a docs/state-only rollback: revert the M001 plan/state updates and
leave product runtime untouched. If a checkpoint changed code in a later run,
that checkpoint must have its own commit and rollback note.

## Stopping condition

Stop if `ops/autopilot/STOP` exists, if another writer owns a non-stale lease,
or if no-WING evidence unexpectedly grants live authority.

## External blockers

M005 physical behavior and the M006 owner decision are complete. The remaining
external work is current-source provenance/technical promotion plus clean,
signed/notarized public distribution and operator-present audible room/PA
validation. None of those gates grants standing live authority by implication.
