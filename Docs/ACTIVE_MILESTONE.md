# Active Milestone

ID: M001

Title: WING read-only preflight evidence gate

Status: WAITING_FOR_EXTERNAL_VALIDATION

## Desired end state

The project can resume autonomous work from a clean, finite plan that keeps
the supervised WING Rack MVP externally gated. The canonical read-only decision
is now accepted-ready, and CH3 fader/gain-down proof artifacts are available,
but M005 physical disconnect/timeout/stale-state proof and M006 owner readiness
review remain external validation and must not be simulated.

Status marker: canonical read-only decision is now accepted-ready; CH3 fader state restoration/verification is complete, but M005/M006 remain open.

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
- M001 autonomous pre-hardware work is complete, the canonical read-only
  decision is `accepted_decision_ready`, and CH3 fader state
  restoration/verification plus the fader/gain-down proof rerun have passed
  with physical readback, rollback and emergency-stop/block evidence. The
  earlier `rollback_not_applied` artifact remains closed audit history; the
  current required external proof is M005 disconnect/timeout/stale-state
  protection, followed by M006 owner readiness review.
- Next operator action from `STATE.json`: Do not widen live authority beyond the proven narrow supervised fader/gain-down scope. Next: run the M005 supervised disconnect/timeout/stale-state protection proof with the operator present, then complete the M006 owner readiness review that updates the MVP readiness matrix from the accepted CH3 fader and gain-down proof artifacts; only after that reassess wider release scope.
- Latest ledger handoff row:
  `m006_owner_decision_requires_stale_event_packet_checks` at
  `2026-07-05T01:07:32Z`. No live authority changed; release-ready live
  correction families remain `0`, live-allowed safety rows remain `0`, and
  current supervised live-write kinds remain `0`.
- Latest M006 owner decision verifier: `0936030c5`; it requires a ready
  packet-verifier report with the current M005 stale-readback event-evidence
  checks. The current pending template blocks with
  `reason=m006_owner_decision_pending` and does not grant a readiness-matrix
  update.

## Current external validation requirements

- Supervised WING disconnect/timeout/stale-state protection proof (M005) with operator present
- Owner readiness review that updates the MVP readiness matrix from the accepted CH3 fader and gain-down proof artifacts (M006)

## Exact evidence

See `Docs/plans/M001-wing-readonly-preflight-evidence-gate.md`.

## Real WING interaction plan

The full operator/backend route for real-console work is
`Docs/plans/M001-real-wing-interaction-plan.md`. It now starts from the
accepted read-only decision gate and passed CH3 fader/gain-down proof
artifacts, while preserving the earlier `rollback_not_applied` artifact as
closed audit history. Only the M005 operator-present disconnect proof and the
M006 owner readiness review can move the milestone forward.

## Rollback

Rollback is a docs/state-only rollback: revert the M001 plan/state updates and
leave product runtime untouched. If a checkpoint changed code in a later run,
that checkpoint must have its own commit and rollback note.

## Stopping condition

Stop if `ops/autopilot/STOP` exists, if another writer owns a non-stale lease,
or if no-WING evidence unexpectedly grants live authority.

## External blockers

Real hardware proof now requires WING Rack, intended backend session and
operator presence for M005 physical disconnect/timeout/stale-state protection.
The accepted read-only decision plus passed CH3 fader/gain-down proofs are
necessary evidence, and the earlier `rollback_not_applied` artifact is closed
by the rerun, but they do not close readiness or grant standing live authority;
the milestone remains `WAITING_FOR_EXTERNAL_VALIDATION` until M005 and M006 pass.
