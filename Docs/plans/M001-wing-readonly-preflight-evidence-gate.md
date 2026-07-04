# Plan M001: WING read-only preflight evidence gate

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
emergency-stop/block evidence. The current external blockers are M005
disconnect/timeout/stale-state protection and M006 owner readiness review.

Audit marker: failed CH3 fader rollback proof is closed by the passed rerun;
it does not grant release readiness.

## Current external validation requirements

- Supervised WING disconnect/timeout/stale-state protection proof (M005) with operator present
- Owner readiness review that updates the MVP readiness matrix from the accepted CH3 fader and gain-down proof artifacts (M006)

Next operator action from `STATE.json`: Do not widen live authority beyond the proven narrow supervised fader/gain-down scope. Next: run the M005 supervised disconnect/timeout/stale-state protection proof with the operator present, then complete the M006 owner readiness review that updates the MVP readiness matrix from the accepted CH3 fader and gain-down proof artifacts; only after that reassess wider release scope.

Latest ledger handoff row:
`m005_event_sequence_owner_verifier_bound` at
`2026-07-04T22:24:01Z`. No live authority changed; release-ready live
correction families remain `0`, live-allowed safety rows remain `0`, and
current supervised live-write kinds remain `0`.
