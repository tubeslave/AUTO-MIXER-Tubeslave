# Plan M001: WING read-only preflight evidence gate

> Soundcheck execution update (2026-07-12): the local staged kernel is
> `96/100`, strict real-live soundcheck is `88/100`, and normal live authority
> remains empty. This legacy M001 plan now gates provenance/promotion and the
> operator-present staged proof; it does not require another M005 cable pull by
> default.

> The current tree also has atomic stage-local reanalysis with exact accepted
> bundle identity and cancellation-safe candidate cleanup. This remains
> offline/proposal authority until the external staged proof is completed.

Current blocker from `STATE.json`: M110-HW01 is no longer blocked by digital silence. On clean CP05-bound source 45e2a2ae66853dc79a8db693f8ccfe15cbc67cc8, accepted route artifact /Users/dmitrijvolkov/.automixer/runtime/m110_master_route_proofs/4937d06623084a2d98e7bc3f9b8f8a17.json verified the exact WING serial and firmware, serial-bound USB/CoreAudio 48x48/48 kHz identity, two complete Main 47/48 MAIN L/R route passes, fresh signal at conservative -12.318 dBTP, and 0 console mutation writes. The runner remained blocked only by proof_owned_master_route_teardown_failed:wing_disconnect_failed:RuntimeError because the expected post-disconnect readiness_invalidated broadcast was treated as fatal; the backend log shows capture stop, route revocation, WING/OSC disconnect and clean backend shutdown. M120 adds a teardown-only exact-envelope receiver; independent rereview is ACCEPT and 142 runner tests passed. The pre-fix attempt cannot close HW01: a fresh clean committed M120/CP05-bound route-only repetition with verified teardown is required. HW02-HW05 remain pending; reset, corrections and live writes remain 0. M111 still requires exact same-console attestation, a clean committed compatibility replay and an explicit migration-bound M006 decision. Strict real-live readiness stays 88/100: normal allowlist 0, standing authority false, safe_core_ready/show_mix_ready unproven.

Next operator action from `STATE.json`: Do not widen live authority. Finish M120 validation, commit the source/truth change, release the writer lease, create exactly one marker-only CP05 rebind and verify the dynamic gate. Keep the proven non-silent Ableton-to-WING Main 47/48 feed running and repeat route-only M110-HW01 from a clean committed checkout. Proceed to HW02-HW05 only if the fresh artifact is accepted and teardown verifies capture stopped, mixer disconnected, route authority revoked and backend terminated cleanly. Before any HW02-HW05 write, physically contain all PA and monitor outputs and confirm Main 1 muted. Separately provide the exact WING Rack serial/same-console attestation for the 2026-07-11 M005 run, complete the clean committed compatibility replay and record an explicit verified_migration M006 owner decision. Complete signing/notarization/Gatekeeper and production remote-auth recovery separately.

Latest ledger handoff row: `offline_balance_probe_overhead_identity_fix` at `2026-09-23T20:28:42Z`.

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

- Repeat M110-HW01 from the clean committed M120 source after one marker-only CP05 rebind; keep the already proven non-silent Main 47/48 feed active and require a fresh accepted route artifact plus verified capture stop, mixer disconnect, route revocation and clean backend termination
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
