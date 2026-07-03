# Autopilot Blockers

## Current blockers

- M001 remains waiting for external validation, but the real WING Rack read-only
  artifact operator decision gate is no longer the active blocker. The current
  decision report is `accepted_decision_ready` with 1 accepted, 2 rejected, 0
  invalid and 0 unreviewed records.
- Supervised WING proof is still blocked from release readiness. The available
  CH3 fader proof artifact
  `artifacts/wing_operator_proposal_validation/20260702T210956Z_wing_operator_proposal_ch3_fader/summary.json`
  has status `failed` with reason `rollback_not_applied`.
- Release-ready live correction families remain `0` in
  `Docs/reports/WING_CORRECTION_APPLY_READINESS_20260703.md`; fader and gain
  are supervised policy candidates only, not release-ready live families.
- `backend/safety_coverage_matrix.py` has no live-allowed safety rows for this
  state. The fader safety row is a `supervised_candidate` with failed rollback
  proof, so downstream autonomy promotion and mode-transition gates must keep
  supervised live apply blocked until the proof is rerun cleanly.
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
- The next operator step is to restore or verify CH3 fader state after the
  failed rollback, then rerun the narrow fader proof until apply, physical
  readback, rollback and emergency-stop/block evidence all pass. Gain proof
  must be separate and headroom-safe.
- Next operator action from `STATE.json`: Do not widen live authority. Operator
  must restore or verify CH3 fader state after the failed rollback with reason
  rollback_not_applied, then rerun the narrow fader proof until apply, physical
  readback, rollback and emergency-stop/block evidence all pass. After that,
  run a separate headroom-safe gain-down proof with the same evidence
  requirements; only then reassess release readiness.
- Latest ledger handoff row:
  `full_suite_after_structured_pre_release_score_surface_sync_passed` at
  `2026-07-03T12:31:14Z`. That row preserves the same `current_blocker`,
  `next_operator_action` and `external_validation_requirements` as
  `STATE.json`, so the safe handoff can be audited after mutable state moves.
- Live authority remains denied: release-ready live correction families remain
  `0`, live-allowed safety rows remain `0`, and current supervised live-write
  kinds remain `0`.
- `Docs/plans/M001-real-wing-interaction-plan.md` is synchronized to this
  blocker state: it now shows `accepted_decision_ready`, 1 accepted, 2
  rejected, 0 invalid and 0 unreviewed decision records, and
  `rollback_not_applied` as the active supervised-proof blocker.
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

- Operator restores or verifies CH3 fader state after rollback_not_applied
- Clean supervised CH3 fader proof rerun with apply, physical readback, rollback and emergency-stop/block evidence
- Separate headroom-safe supervised gain-down proof with the same evidence requirements

## Not blockers

- `ops/autopilot/STOP` does not exist after setup.
- No active writer lease is present; `ops/autopilot/WRITER_LEASE/owner.json`
  is absent and the runtime lease directory is ignored.
- Control-plane docs/state can proceed without live hardware contact.
