# Active Milestone

ID: M001

Title: WING read-only preflight evidence gate

Status: WAITING_FOR_EXTERNAL_VALIDATION

## Desired end state

The project can resume autonomous work from a clean, finite plan that verifies
the no-WING prestart and read-only preflight input gate before any real WING
contact. The canonical read-only decision is now accepted-ready, but the
remaining supervised proof/rollback run remains external validation and must
not be simulated.

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

## Current evidence

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
- M001 autonomous pre-hardware work is complete, and the canonical read-only
  decision is `accepted_decision_ready`. The current required external proof is
  the failed CH3 fader rollback recovery and rerun, followed by separate
  headroom-safe gain-down proof evidence.
- Next operator action from `STATE.json`: Do not widen live authority. Operator
  must restore or verify CH3 fader state after the failed rollback with reason
  rollback_not_applied, then rerun the narrow fader proof until apply, physical
  readback, rollback and emergency-stop/block evidence all pass. After that,
  run a separate headroom-safe gain-down proof with the same evidence
  requirements; only then reassess release readiness.
- Latest ledger handoff row:
  `full_suite_after_structured_pre_release_score_surface_sync_passed` at
  `2026-07-03T12:31:14Z`. No live authority changed; release-ready live
  correction families remain `0`, live-allowed safety rows remain `0`, and
  current supervised live-write kinds remain `0`.

## Current external validation requirements

- Operator restores or verifies CH3 fader state after rollback_not_applied
- Clean supervised CH3 fader proof rerun with apply, physical readback, rollback and emergency-stop/block evidence
- Separate headroom-safe supervised gain-down proof with the same evidence requirements

## Exact evidence

See `Docs/plans/M001-wing-readonly-preflight-evidence-gate.md`.

## Real WING interaction plan

The full operator/backend route for real-console work is
`Docs/plans/M001-real-wing-interaction-plan.md`. It now starts from the
accepted read-only decision gate and keeps supervised writes blocked until the
operator restores or verifies CH3 fader state after `rollback_not_applied`.
Only the explicit supervised `fader`/`gain-down` proof path with physical
readback, rollback and emergency-stop evidence can move the milestone forward.

## Rollback

Rollback is a docs/state-only rollback: revert the M001 plan/state updates and
leave product runtime untouched. If a checkpoint changed code in a later run,
that checkpoint must have its own commit and rollback note.

## Stopping condition

Stop if `ops/autopilot/STOP` exists, if another writer owns a non-stale lease,
or if no-WING evidence unexpectedly grants live authority.

## External blockers

Real hardware proof now requires WING Rack, intended backend session, operator
presence, CH3 fader state restoration/verification after `rollback_not_applied`,
and a clean supervised proof rerun. The accepted read-only decision is
necessary evidence, but it does not close readiness or grant live authority, so
the milestone remains `WAITING_FOR_EXTERNAL_VALIDATION`.
