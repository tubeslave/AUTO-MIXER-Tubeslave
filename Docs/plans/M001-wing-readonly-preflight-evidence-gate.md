# Plan M001: WING read-only preflight evidence gate

> Soundcheck execution update (2026-07-12): the local staged kernel is
> `96/100`, strict real-live soundcheck is `88/100`, and normal live authority
> remains empty. This legacy M001 plan now gates provenance/promotion and the
> operator-present staged proof; it does not require another M005 cable pull by
> default.

> The current tree also has atomic stage-local reanalysis with exact accepted
> bundle identity and cancellation-safe candidate cleanup. This remains
> offline/proposal authority until the external staged proof is completed.

Current blocker from `STATE.json`: The monolithic soundcheck defect is closed locally: staged execution kernel readiness is 96/100 and full-project local engineering remains 95/100. Strict real-live soundcheck readiness is 88/100 because the normal live allowlist is 0, the current-tree staged child flow has no physical WING completion proof, real safe_core_ready/show_mix_ready are unproven, source-path query latency is unproven, and EQ boost/compressor makeup/gain-up/send-up substitutions remain operator-review-only. M005 behavior is closed and narrow M006 owner intent is recorded, but current-source provenance and technical promotion remain fail-closed. Strict public-release readiness stays 88/100 and standing live authority remains denied.

Next operator action from `STATE.json`: Do not widen live authority. Rebind the closed M005 disconnect/timeout/stale-state behavior to current-source provenance without another cable pull unless explicitly requested, rerun the M006 technical readiness review verifier against the recorded narrow owner decision, then perform an operator-present current-tree staged-child proof on the exact WING with source-path latency, rollback, safe_core_ready/show_mix_ready and room/PA evidence. Complete clean-checkout reproduction, Developer ID signing/notarization/Gatekeeper and production remote-auth recovery separately.

Latest ledger handoff row: `soundcheck_atomic_reanalysis_final_validation` at `2026-07-12T22:48:19Z`.

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

M001-CP05 completed. M001 remains `WAITING_FOR_EXTERNAL_VALIDATION` because the
accepted read-only decision does not close release readiness. Its current
machine status is `accepted_decision_ready`. The earlier failed CH3 fader
rollback proof with reason `rollback_not_applied` remains audit history, and
CH3 fader state restoration/verification plus the narrow fader/gain-down proof
rerun have passed with apply, physical readback, rollback and
emergency-stop/block evidence. M005 disconnect/timeout/stale-state behavior is
closed, and the narrow M006 owner decision is recorded. The remaining external
handoff is current-source provenance plus technical promotion, not a repeated
cable-pull run.

Audit marker: failed CH3 fader rollback proof is closed by the passed rerun;
it does not grant release readiness.

## Current external validation requirements

- Current-release provenance rebind for the closed M005 disconnect/timeout/stale-state proof without repeating the physical cable pull unless the owner explicitly requests it
- Technical M006 readiness review promotion after the current verifier accepts that provenance; the narrow owner decision is already recorded and grants no live authority

Next operator action from `STATE.json`: Do not widen live authority. Treat M005 disconnect/timeout/stale-state behavior as closed and the narrow M006 readiness review owner decision as recorded; do not repeat the physical cable pull unless the owner explicitly requests it. For public release, rebind or migrate the closed M005 evidence to a clean current-source provenance packet, rerun the M006 technical verifier, reproduce the package from a clean checkout, complete Developer ID signing/notarization/Gatekeeper checks, and perform operator-present room/PA audible rollback validation.

Latest ledger handoff row:
`local_engineering_95_final_tree_full_suite_no_wing_validation` at
`2026-07-12T13:56:06Z`. No live authority changed; release-ready live
correction families remain `0`, live-allowed safety rows remain `0`, and
current supervised live-write kinds remain `0`.

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
