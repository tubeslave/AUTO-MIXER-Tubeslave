# Active Milestone

ID: M001

Title: WING read-only preflight evidence gate

Status: WAITING_FOR_EXTERNAL_VALIDATION

## Desired end state

The project can resume autonomous work from a clean, finite plan that verifies
the no-WING prestart and read-only preflight input gate before any real WING
contact. A real hardware run remains external validation and must not be
simulated.

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
- M001 autonomous work is complete. The next required proof is external
  hardware/operator validation.

## Exact evidence

See `Docs/plans/M001-wing-readonly-preflight-evidence-gate.md`.

## Real WING interaction plan

The full operator/backend route for real-console work is
`Docs/plans/M001-real-wing-interaction-plan.md`. It keeps the current milestone
blocked before writes until read-only artifacts have accepted operator decision
records, then allows only the supervised `fader`/`gain-down` proof with
physical readback, rollback and emergency-stop evidence.

## Rollback

Rollback is a docs/state-only rollback: revert the M001 plan/state updates and
leave product runtime untouched. If a checkpoint changed code in a later run,
that checkpoint must have its own commit and rollback note.

## Stopping condition

Stop if `ops/autopilot/STOP` exists, if another writer owns a non-stale lease,
or if no-WING evidence unexpectedly grants live authority.

## External blockers

Real hardware proof requires WING Rack, intended backend session, operator
presence and accepted review records. Until that exists, the milestone remains
`WAITING_FOR_EXTERNAL_VALIDATION`.
