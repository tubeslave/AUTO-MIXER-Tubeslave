# Plan M001: WING read-only preflight evidence gate

> Soundcheck execution update (2026-07-12): the local staged kernel is
> `96/100`, strict real-live soundcheck is `88/100`, and normal live authority
> remains empty. This legacy M001 plan now gates provenance/promotion and the
> operator-present staged proof; it does not require another M005 cable pull by
> default.

> The current tree also has atomic stage-local reanalysis with exact accepted
> bundle identity and cancellation-safe candidate cleanup. This remains
> offline/proposal authority until the external staged proof is completed.

Current blocker from `STATE.json`: M110 now reaches the real WING Rack and its 48x48/48 kHz CoreAudio capture, so the missing-device prerequisite is closed. The operator-attended route-only HW01 attempt opened the proof-owned backend, OSC/UDP and 48-channel audio stream, but the machine-recorded route gate failed closed before artifact issuance with reason master_channel_signal_missing; operator-observed 3-second and 120-second captures showed exact digital zero on all 48 channels; these extra observations are not independently machine-verified. Therefore HW01 remains incomplete and HW02-HW05 have not started; reset, corrections and live writes remain 0. The local runtime summary is ephemeral diagnostic evidence only, not accepted physical route proof. M111 still requires exact same-console attestation, a clean committed compatibility replay and an explicit migration-bound M006 decision. Strict real-live readiness stays 88/100: normal allowlist 0, standing authority false, safe_core_ready/show_mix_ready unproven.

Next operator action from `STATE.json`: Do not widen live authority. Physically contain PA and monitor outputs; in Ableton select WING as the Audio Output Device, route the source to WING USB 1-16, start playback, verify a non-silent Main 47/48 feed and leave playback running. Then, with the operator present and a clean dynamic-CP05-bound checkout, repeat M110-HW01 and proceed to HW02-HW05 only after an issued route artifact. Separately provide the exact WING Rack serial/same-console attestation for the 2026-07-11 M005 run, complete the clean committed migration replay and record an explicit verified_migration M006 owner decision. Complete signing/notarization/Gatekeeper and production remote-auth recovery separately.

Latest ledger handoff row: `m118_m111_sealed_replay_zero_io_source_checkpoint_ready` at `2026-07-15T01:48:55Z`.

## 2026-07-12 current-contract addendum

The historical M005 artifact remains behavioral history only. Current M005
execute is allowed only from a clean checkout. After first-event immutable
provenance, the runner must launch the canonical source backend itself at
`ws://127.0.0.1:8765` and bind runtime identity v2, explicit operator gate and a pre-existing
operator-confirmed WING Rack serial/firmware `3.0.5` challenge under
`artifacts/wing_console_identity_owner_challenge/`. The editable example is
`Docs/templates/WING_CONSOLE_IDENTITY_OWNER_CHALLENGE.example.json`; its
placeholder and `operator_confirmed=false` values are deliberately non-runnable.
M006 recomputes the current packet verifier and cannot rely on a saved ready
report. None of this grants live authority.

## Checkpoints

| Checkpoint | Objective | Files | Expected behavior | Boundary | Tests | Evidence | Rollback point | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M001-CP01 | Establish no-WING baseline | `tools/no_wing_release_check.py`, artifacts under `artifacts/no_wing_release_check/` | Quick check runs without WING contact and denies live authority | No hardware, no OSC writes | `PYTHONPATH=backend python tools/no_wing_release_check.py --quick` | `artifacts/no_wing_release_check/20260618T204503Z_no_wing_release_check`; command passed | revert only state/report updates | completed |
| M001-CP02 | Validate read-only preflight inputs | `tools/wing_readonly_preflight_runner.py`, `Docs/runbooks/WING_READONLY_PREFLIGHT.md` | Input validation can run before backend/WING session | No live mutation | no-WING summary validate-run-inputs fields | same artifact; `validate_run_inputs_only_*_matches_expected: true` | revert state/report updates | completed |
| M001-CP03 | Confirm authority denial surfaces | WING handoff packet/reports | Evidence has `live_authority_granted: false` and `readiness_closure_allowed_by_packet: false` before hardware proof | Report-only | no-WING artifact/verifier surfaces | same artifact; live/readiness authority denied | revert state/report updates | completed |
| M001-CP04 | Prepare real hardware handoff | `Docs/runbooks/WING_READONLY_PREFLIGHT.md`, `ops/autopilot/LAST_REPORT.md` | Operator sees exact next WING steps and blockers | No unattended hardware contact | plan-only plus validate-run-inputs-only | `Docs/reports/AUTOMIXER_AUTOPILOT_M001_CP04_WING_READONLY_HANDOFF_20260618.md`; `ops/autopilot/M001_CP04_wing_readonly_preflight_plan.json`; validate-only status `ready_for_websocket` | revert handoff text | completed |
| M001-CP05 | Milestone review | `Docs/ACTIVE_MILESTONE.md`, matrices, ledger | Acceptance criteria mapped to evidence | No product runtime changes | standard test command and structured artifact validation | `Docs/reports/AUTOMIXER_AUTOPILOT_M001_CP05_MILESTONE_REVIEW_20260618.md`; status `WAITING_FOR_EXTERNAL_VALIDATION` | revert M001 state commit | completed |

## Required validation at milestone close

- Required checkpoint commands pass or are recorded as concrete blockers.
- `git diff` shows no unintended product runtime edits from the milestone.
- `ops/autopilot/STATE.json` is valid JSON.
- `ops/autopilot/RUN_LEDGER.jsonl` is valid JSONL.
- STOP, lock, stale-run and recovery behavior remain documented.

## First finite Goal

M001-CP05 completed. M001 remains `WAITING_FOR_EXTERNAL_VALIDATION`. The
historical exact 2026-07-02 source retains `accepted_decision_ready`, while the
current aggregate machine status is `operator_review_required` because two
valid 2026-07-13 candidates remain unreviewed. The earlier failed CH3 fader
rollback proof with reason `rollback_not_applied` remains audit history, and
CH3 fader state restoration/verification plus the narrow fader/gain-down proof
rerun passed with apply, physical readback, rollback and emergency-stop/block
evidence. M005 behavior is closed and the narrow M006 owner decision is
recorded. The remaining external handoff includes candidate disposition,
physical WING USB plus attended M110 proof, current-source provenance and
technical promotion; it does not require a repeated M005 cable pull.

Audit marker: failed CH3 fader rollback proof is closed by the passed rerun;
it does not grant release readiness.

## Current external validation requirements

- Confirmed non-silent source reaches the physical WING and USB Main 47/48 during an operator-attended 48x48/48 kHz capture; digital-zero input cannot satisfy HW01
- Operator-present CP05-bound M110-HW01 through M110-HW05 run with route, reset, staged soundcheck, rollback, SHA-bound listening and clean teardown evidence
- Operator accept or reject the two valid 2026-07-13 read-only preflight candidates; this decision-ledger requirement grants no live authority and does not reopen closed M005 behavior
- Exact WING Rack serial and same-console owner attestation for the immutable 2026-07-11 M005 evidence; this is identity-only and grants no live authority
- Clean committed M005 current-release compatibility replay plus an explicit verified_migration M006 owner decision bound to the recomputed migration and packet-verifier paths

No live authority changed: release-ready live correction families remain `0`,
live-allowed safety rows remain `0`, and current supervised live-write kinds
remain `0`.

Latest M005 readback target binding guard: `d3eb93a62`; M005 baseline, stale-readback and reconnect-readback payloads, including `events.jsonl` stale/reconnect payloads, now must match the requested `channel`, `value_type` and WING `address` before M005 completion or M006 owner-review input can be ready. This grants no backend/WING contact, proposal creation, live apply, readiness-matrix update or live authority.

Latest M005 physical reconnect target guard: `bbf7833e2`; executed `physical-cable-loss` M005 proofs now fail closed before backend/WebSocket contact unless `--reconnect-wing-ip` is supplied. This keeps the operator-present cable-loss proof bound to reconnect-recovery evidence and grants no backend/WING contact, proposal creation, live apply, readiness-matrix update or live authority.

Latest M006 packet verifier hardening: `3ea749d70`; it requires a ready
packet-verifier report with the current M005 stale-readback event-evidence
checks and matching reconnect-recovery physical readback event checks from
`events.jsonl`.

Latest M006 owner decision verifier: `5eda78126`; it requires a ready
packet-verifier report with the current M005 stale-readback event-evidence
checks, matching reconnect-recovery physical readback event checks and the
accepted M005 summary sibling `events.jsonl` path. The owner decision is
recorded, but technical promotion blocks with
`reason=m006_owner_decision_failed_closed` because current-source M005
provenance is not accepted; it does not grant a readiness-matrix update.
