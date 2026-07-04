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
- Next operator action from `STATE.json`: Do not widen live authority beyond the proven narrow supervised fader/gain-down scope. Next: run the M005 supervised disconnect/timeout/stale-state protection proof with the operator present, then complete the M006 owner readiness review that updates the MVP readiness matrix from the accepted CH3 fader and gain-down proof artifacts; only after that reassess wider release scope.
- Latest ledger handoff row:
  `supervised_m005_software_disconnect_partial_evidence_recorded` at
  `2026-07-04T08:04:59Z`. That row preserves the same `current_blocker`,
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
