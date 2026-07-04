# Autopilot Blockers

## Current blockers

- M001 remains waiting for external validation, but the real WING Rack read-only
  artifact operator decision gate is no longer the active blocker. The current
  decision report is `accepted_decision_ready` with 1 accepted, 2 rejected, 0
  invalid and 0 unreviewed records.
- Supervised WING fader and gain-down proofs have now passed on the real
  console. The earlier failed rollback artifact
  `artifacts/wing_operator_proposal_validation/20260702T210956Z_wing_operator_proposal_ch3_fader/summary.json`
  (status `failed`, reason `rollback_not_applied`) is closed by the passed
  rerun `20260704T045741Z_wing_operator_proposal_ch3_fader` and the passed
  gain-down proof `20260704T045820Z_wing_operator_proposal_ch3_gain`; both
  carry apply, physical readback, rollback and emergency-stop/block evidence.
  The release readiness gate still requires the M005 disconnect/timeout/stale-state
  proof and the M006 owner readiness review.
- Release-ready live correction families remain `0` in
  `Docs/reports/WING_CORRECTION_APPLY_READINESS_20260703.md`; fader and gain
  are supervised policy candidates only, not release-ready live families.
- `backend/safety_coverage_matrix.py` has no live-allowed safety rows for this
  state. The fader and gain safety rows are now `hardware_proven_narrow` with
  passed apply/rollback/emergency-stop evidence, but `live_allowed` stays
  `false` for both until the M006 owner readiness review accepts the proof
  artifacts, so downstream autonomy promotion and mode-transition gates keep
  supervised live apply blocked.
- Operator product-state autonomy overlays also fail closed for this state:
  `supervised_live_allowed_static` is normalized to `false`, and the no-WING
  release gate rejects any overlay row that tries to claim it before proof
  passes.
- no-WING autonomy gap-rollup validation also fails closed for this state:
  rollup current live-write kinds/surfaces must match the mode-transition gate,
  so policy-level fader/gain cannot be accepted as current live authority while
  the gate exposes empty current live-write lists. The helper API now exposes
  this distinction as `policy_supervised_live_write_*` versus
  `mode_gate_supervised_live_write_*`, and the validator rejects the legacy
  gap-rollup claim phrase that called fader/gain the current supervised
  live-write boundary.
- Read-only preflight evidence also fails closed for readiness updates:
  `readiness_matrix_update_candidate_available` may be true for a clean
  candidate, but evidence-only `readiness_matrix_update_allowed` remains false.
  The same `readiness_matrix_update_allowed=false` denial is included in the
  read-only evidence `source_authority_flags`, and no-WING summary helpers
  re-check nested flags plus require an explicit source-authority-denied boolean
  before preserving readiness candidate availability; a separate manual
  review/update action is still required.
- Supporting real WING read-only evidence exists in
  `Docs/reports/WING_REAL_HARDWARE_READONLY_20260702.md`, but it is a
  temporary-root capture only. It cannot close readiness or authorize
  supervised writes.
- The next operator step is the M005 supervised disconnect/timeout/stale-state
  protection proof with the operator present, followed by the M006 owner
  readiness review that updates the MVP readiness matrix from the accepted
  CH3 fader and gain-down proof artifacts.
- A report-only M006 owner readiness review packet now exists at
  `artifacts/wing_owner_readiness_review/20260704T191101Z_m006_owner_readiness_review_packet/summary.json`
  and `Docs/reports/WING_M006_OWNER_READINESS_REVIEW_PACKET_20260704.md`. It
  prepares the owner review from the accepted CH3 fader/gain-down artifacts but
  is not an owner decision: M005 is still missing, readiness matrix updates are
  still disallowed, and live authority remains denied.
- The M005/M006 verifier contract is now stricter: `ad7f0da18` makes
  `tools/wing_timeout_disconnect_proof_runner.py` require physical reconnect
  readback before M005 reconnect recovery counts complete, and makes
  `tools/verify_m006_owner_readiness_packet.py` independently check the M005
  physical cable-loss token, unavailable stale-readback payload, safe
  blocked-write reason, connected reconnect status and physical reconnect
  readback before owner-review input can be ready. This is offline
  verification only; it does not close M005 or M006.
- Local runtime/tooling verification on 2026-07-04 is green but does not close
  the blocker: the current backend listener completed the 600.005 second
  read-only observation artifact
  `artifacts/soundcheck_runtime_observation/20260704T180231Z_soundcheck_runtime_observation/summary.json`
  with 2491 read-only meter frames, 119 dashboard snapshots, 0 live-write
  meter frames, 0 unexpected live events, no forbidden sent request types and
  no live apply request sent. Frontend test/build passed, targeted backend
  meter/runtime-observation/M005-runner/proposal tests passed, and WING MVP
  rails passed. Full backend validation also passed:
  `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`: 5453
  passed, 1 skipped, 1 warning. The no-WING quick gate failed closed on
  `backend_port_free` because the existing listener still needs operator
  confirmation. M005 still requires the operator-present physical cable-loss,
  stale-readback, blocked-write and reconnect proof, and M006 still requires
  owner readiness review.
- Offline long-run drift audit is now available, and it rejects the old daemon
  evidence instead of treating it as readiness proof. Audit artifact
  `artifacts/soundcheck_long_run_drift_audit/20260704T183743Z_soundcheck_long_run_drift_audit/summary.json`
  over `reports/soundcheck_daemon_log_20260703T221457.jsonl` failed because
  the source covered only 188.9 seconds, contained 287 live executor actions,
  73 positive gain/trim boosts, 422 relative-only fader/trim/gain targets and
  28 cumulative relative-drift surfaces. This is an offline negative check
  only: `live_authority_granted=false`, `readiness_closure_allowed=false` and
  `readiness_matrix_update_allowed=false`.
- Decision queue proposals now expose a read-only operator review coverage
  summary for `reason`, `risk`, `requested_change`, `old_value`, `new_value`
  and `rollback`. This makes missing proposal context visible to the operator
  without granting apply authority: it does not mutate the queue, does not run
  preflight, does not call `apply_operator_proposal`, and keeps
  `live_authority_granted=false`.
- Dirty worktree split was recorded on `2026-07-04T18:27:09Z`: HEAD is
  `22ece58e09`, `ops/autopilot/WRITER_LEASE/owner.json` is absent, and
  `ops/autopilot/STOP` is absent. Keep WING proof/readiness WIP
  (`tools/wing_timeout_disconnect_proof_runner.py`, its tests/ADR and the
  active plan), soundcheck runtime WIP (meter stream backend, runtime
  observation runner/tests/artifact and frontend meter subscription), and
  frontend UI restyle WIP (`AutomixerDashboard.tsx`,
  `AutomixerComponents.tsx`, `DirectorPanel.tsx` and
  `Docs/reports/screenshots_frontend_restyle_20260704.png`) as separate commit
  groups. The UI restyle/screenshot must not be mixed into a WING readiness
  proof commit. The meter stream subscription remains read-only and now
  unsubscribes during frontend hook cleanup when the WebSocket is still open;
  frontend test/build passed after that cleanup. The reproducible split
  handoff is now recorded in
  `artifacts/worktree_split/20260704T185842Z_worktree_split/summary.json` and
  `Docs/reports/WORKTREE_SPLIT_20260704.md`. Generated review patches now live
  under `artifacts/worktree_split/20260704T185842Z_worktree_split/patches/`
  for WING proof/readiness WIP, soundcheck backend/runtime WIP, frontend
  runtime binding and frontend UI restyle. The patch bundle is not fully
  self-contained: the 60,032,994-byte runtime observation `events.jsonl`
  remains local evidence tracked by SHA-256 only, and the earlier
  `20260704T183538Z`/`20260704T183627Z` drift-audit summaries are superseded
  by the accepted `20260704T183743Z` audit summary. This split evidence grants
  no live authority and does not close M005 or M006. Pre-landing review patches
  01-04 and 06 passed `git apply --check` on a temporary clean detached
  worktree at `22ece58e0`; post-landing truth patch 05 is validated against
  the current landed split head. JSON checks, `git diff --check` and WING MVP
  rails also passed.
- The split has now been landed as separate local commits: `7a792457e` for
  read-only soundcheck runtime evidence, `3cf9b53a1` for frontend read-only
  meter binding, `e745438ab` for frontend UI restyle, `3f5d5c0eb` for the M005
  disconnect proof runner, and `9ce9e29ba` for the report-only M006 owner
  readiness packet. The large runtime observation `events.jsonl` and two
  superseded drift-audit attempts are ignored as local evidence only.
- Next operator action from `STATE.json`: Do not widen live authority beyond the proven narrow supervised fader/gain-down scope. Next: run the M005 supervised disconnect/timeout/stale-state protection proof with the operator present, then complete the M006 owner readiness review that updates the MVP readiness matrix from the accepted CH3 fader and gain-down proof artifacts; only after that reassess wider release scope.
- Latest ledger handoff row:
  `m005_reconnect_evidence_contract_hardened` at
  `2026-07-04T21:16:22Z`. That row preserves the same `current_blocker`,
  `next_operator_action` and `external_validation_requirements` as
  `STATE.json`, so the safe handoff can be audited after mutable state moves.
- Live authority remains denied: release-ready live correction families remain
  `0`, live-allowed safety rows remain `0`, and current supervised live-write
  kinds remain `0`.
- `Docs/plans/M001-real-wing-interaction-plan.md` is synchronized to this
  blocker state: it shows `accepted_decision_ready`, 1 accepted, 2 rejected,
  0 invalid and 0 unreviewed decision records; the earlier
  `rollback_not_applied` supervised-proof blocker is closed by the passed
  CH3 fader and gain-down proofs, and the open items are the M005
  disconnect/timeout/stale-state proof and the M006 readiness review.
- `Docs/reports/WING_READONLY_PREFLIGHT_HANDOFF_PACKET_LATEST.md` is scoped as
  read-only preflight evidence handoff only. It no longer stands alone as the
  current next-action source after `accepted_decision_ready`; current
  operator action must come from `Docs/CURRENT_STATE.md`, the active
  real-WING plan and `ops/autopilot/STATE.json`.
- The current worktree contains local WING evidence/report updates and SmartDSP
  scoring work. Future autopilot runs must avoid overwriting unrelated local
  changes if they appear.
- Future heartbeat wake-ups must verify native Goal transition tools before
  replacing a thread Goal.

## Current external validation requirements

- Supervised WING disconnect/timeout/stale-state protection proof (M005) with operator present
- Owner readiness review that updates the MVP readiness matrix from the accepted CH3 fader and gain-down proof artifacts (M006)

## Not blockers

- `ops/autopilot/STOP` does not exist after setup.
- No active writer lease is present; `ops/autopilot/WRITER_LEASE/owner.json`
  is absent and the runtime lease directory is ignored.
- Control-plane docs/state can proceed without live hardware contact.
