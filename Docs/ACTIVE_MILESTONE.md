# Active Milestone

ID: M001

Title: WING read-only preflight evidence gate

Status: WAITING_FOR_EXTERNAL_VALIDATION

## M110 CP05 verified; HW01 blocked on physical WING USB — 2026-07-14

The global M001 shell remains unchanged. Operator-directed active work is
M110: CP00-CP05 are complete under the canonical source-bound marker contract.
Commit `91907b8fc620857172a96176baba6e52ced081e1` passed a clean detached
launch-only probe with loopback backend contact only; no WING, OSC/UDP, audio
stream or live-write surface opened. Passive CoreAudio and macOS USB enumeration
then found zero physical WING devices. HW01 is blocked before start on
`m110_wing_usb_coreaudio_device_missing`; HW02-HW05 remain pending. Reset and
applied correction counts remain zero.

M005 behavior is closed. The operator's M006 permission is recorded only for
one CP05-bound attended session and is not standing authority. Reset means 64
processor bypass addresses on channels 1-16 followed by exact removal of the
known M006 corrections: stages 23/14/14, scopes 33/63/14, changed 19/52/6 and
77 unique changed addresses; state outside the archived M006 transaction is
preserved. The full route -> reset -> M006 restore -> analysis -> staged lanes
-> two HW05 audits with SHA-bound listening path is implemented.

Until real HW01-HW05 succeed, strict real-live readiness remains **88/100**,
not 95: normal allowlist `0`, standing authority `false`, and
`show_mix_ready` is not physically proved on this source. The truth-sync
writer lease is released and its directory is absent; final marker rebind is
the remaining source-integrity step before any hardware attempt. Canonical current details are in
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
next physical step is not another M005 cable pull: it is connection of the
physical WING USB/CoreAudio interface followed by the operator-present M110
staged-flow proof, timing the two-pass source-path sweep and demonstrating
terminal readiness. Candidate disposition and current-source M005/M006
promotion remain separate fail-closed requirements.

Current blocker from `STATE.json`: M110 CP05 and the clean detached launch-only probe are verified on the committed control plane, but passive CoreAudio and macOS USB enumeration found no physical WING device. Therefore M110-HW01 has not started, no capture stream or OSC/UDP path was opened, and reset/corrections remain 0. Strict real-live readiness stays 88/100: normal allowlist 0, standing authority false, safe_core_ready/show_mix_ready unproven. M005 behavior is closed and narrow M006 owner intent is recorded, while current-source promotion and the remaining public-release gates stay fail-closed.

Next operator action from `STATE.json`: Do not widen live authority. Connect the physical WING USB audio interface to this Mac and require one CoreAudio device named WING with 48 input and 48 output channels at 48 kHz; Dante Virtual Soundcard is not an accepted substitute. Then rerun the CP05-bound attended M110-HW01-HW05 process for route, reset, staged soundcheck, rollback, SHA-bound listening and teardown. Operator uses the Operator Decision Queue in Docs/reports/WING_READONLY_PREFLIGHT_DECISION_REVIEW_LATEST.md to accept or reject the two valid 2026-07-13 read-only preflight candidates without granting authority. Separately rebind the closed M005 disconnect/timeout/stale-state evidence and rerun the M006 readiness review; complete clean-checkout signing/notarization/Gatekeeper and production remote-auth recovery separately.

Latest ledger handoff row: `m110_independent_review_full_suite_green` at `2026-07-14T17:24:24Z`.

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

- Physical WING USB/CoreAudio device named WING with 48 input and 48 output channels at 48 kHz; Dante Virtual Soundcard is not an accepted substitute
- Operator-present CP05-bound M110-HW01 through M110-HW05 run with route, reset, staged soundcheck, rollback, SHA-bound listening and clean teardown evidence
- Operator accept or reject the two valid 2026-07-13 read-only preflight candidates; this decision-ledger requirement grants no live authority and does not reopen closed M005 behavior
- Current-release provenance rebind for the closed M005 disconnect/timeout/stale-state proof without repeating the physical cable pull unless the owner explicitly requests it
- Technical M006 readiness review promotion after the current verifier accepts that provenance; the narrow owner decision is already recorded and grants no live authority

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
