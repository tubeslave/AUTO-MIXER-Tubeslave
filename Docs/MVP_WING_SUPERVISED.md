# MVP: Supervised WING Rack Proof

Status: current execution contract

## Purpose

This is the current MVP rail for Automixer. The long-term goal is a capable AI
mixing engineer, but the current project milestone is narrower: prove one safe
supervised hardware loop on WING Rack before broad autonomy work continues.

Goal-mode Codex automation must also follow
`.ai/project-layer/autonomous_mvp_and_model_routing.md` and
`.ai/agent-core/model_budget_policy.md`. Those files preserve the full project
goal while preventing daily/weekly/monthly model budget from being consumed by
broad, high-reasoning work before the supervised WING proof is ready.

## In Scope

The MVP includes only:

1. WING read-only connection proof.
2. Physical console readback proof.
3. Supervised fader write proof with explicit operator approval.
4. Supervised gain-down / trim proof with explicit operator approval.
5. Rollback proof.
6. Emergency-stop proof.
7. Connection-loss / timeout fail-closed proof.
8. Operator-visible evidence report.
9. Readiness matrix update based only on real proof.

## Out Of Scope

These are not part of the current MVP:

- Full autonomous live mixing.
- Live EQ, compressor, gate, routing, mute, FX, scene, matrix, or DCA writes.
- New broad proposal-only surfaces.
- New shadow-only methods that do not close a blocker below.
- New memory-trust, queue-import, result-consumer, report-only, or read-model
  chains that do not directly close a blocker below.
- Measurement-microphone / room-tuning workflows. They are useful later, but
  not required for the current console proof gate.

Correction-method development for Gain/Trim, Pan, EQ, Comp, FX, Balance, and
related families continues through `Docs/CORRECTION_METHODS_ROADMAP.md`, but
before this MVP is proven those methods must remain offline, dry-run,
proposal-only, read-only, and tested. They must not widen live writes.

## Source Of Truth

Use these files in this order:

1. `AGENTS.md`
2. `CLAUDE.md`
3. `Docs/MVP_WING_SUPERVISED.md`
4. `.ai/project-layer/wing_mvp_blockers.md`
5. `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
6. `tools/wing_readonly_preflight_runner.py`
7. `tools/review_wing_readonly_preflight_artifacts.py`
8. `backend/wing_readonly_preflight_review_decision.py`
9. `backend/supervised_wing_proof_plan.py`
10. `tools/no_wing_release_check.py`
11. `Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md`
12. `Docs/reports/MVP_PROGRESS_LATEST.md`
13. `Docs/CORRECTION_METHODS_ROADMAP.md`
14. `.ai/project-layer/correction_methods_backlog.md`

## Definition Of Done

The MVP is done only when every row in
`.ai/project-layer/wing_mvp_blockers.md` is `closed` with real proof artifacts.

Offline tests, dry-run artifacts, shadow-only evidence, valid report shape, or
readiness candidates are not enough to close a blocker by themselves.

## Correct Work Order

1. Run read-only preflight against the intended WING backend.
2. Verify physical readback from the artifact and event log.
3. Review the clean preflight candidate manually.
4. Record an accepted operator decision JSON for that candidate.
5. Run supervised fader proof only with explicit operator approval.
6. Run supervised gain-down / trim proof only with explicit operator approval.
7. Prove rollback.
8. Prove emergency stop.
9. Prove connection-loss / timeout fail-closed behavior.
10. Update the readiness matrix from real evidence.

## Stop-The-Line Rules

Stop the current Codex work direction and return to the first open blocker if:

- `tools/codex_autoloop/codex_direction_guard.py` reports `drifting`.
- The work adds memory/queue/result-consumer/report-only layers without closing
  a blocker.
- The work treats candidate, shadow, offline, or dry-run evidence as live
  readiness.
- A supervised proof is planned before clean read-only preflight evidence is
  available for manual review.
- A supervised proof is planned before the clean read-only preflight candidate
  has an accepted review decision record.
- A supervised proof is planned while accepted review source ids lack a valid
  accepted-candidate prestart trace, or while the proof plan still carries
  `accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof`.
- A supervised proof is planned before the no-WING handoff summary shows
  `backend_port_prestart_gate` exactly matching
  `expected_backend_port_prestart_gate`.
- A supervised proof is planned while `backend_port_prestart_gate` claims
  `grants_live_authority: true`, or while
  `backend_port_prestart_gate.must_pass_before_backend_wing_session` is not
  `true`.
- A supervised proof is planned while `backend_port_prestart_gate` does not show
  `operator_sequence_step_index: 0` and
  `operator_sequence_step_must_be_first: true`.
- A supervised proof is planned while `backend_port_prestart_gate` does not show
  `runs_command: prestart_backend_port_check`.
- A supervised proof is planned while `backend_port_prestart_gate` does not
  exactly list `required_markers` for
  `Before starting the intended backend/WING session`,
  `prestart_backend_port_check`, and `unexpected listeners`.
- A supervised proof is planned while the no-WING handoff summary does not show
  `backend_port_prestart_check_command` exactly matching
  `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick`.
- A supervised proof is planned before the no-WING handoff summary shows
  `supervised_prestart_gate` exactly matching
  `expected_supervised_prestart_gate`.
- A supervised proof is planned while `supervised_prestart_gate` claims
  `grants_live_authority: true`.
- A supervised proof is planned while `supervised_prestart_gate` does not show
  `operator_sequence_step_index: 9` and
  `operator_sequence_step_must_be_final: true`.
- A supervised proof is planned while the no-WING handoff summary does not show
  `operator_sequence_mvp_blocker_anchor_digest_review_step_index: 8` and
  `operator_sequence_supervised_prestart_gate_step_index: 9`.
- A supervised proof is planned while the no-WING handoff summary does not show
  `operator_sequence_count: 10`.
- A supervised proof is planned while
  `backend_port_prestart_check_operator_sequence_step` does not include
  `prestart_backend_port_check` and `unexpected listeners`.

This is not a stop to development. Codex must continue uninterrupted Goal-mode
work, but only on the first non-closed MVP blocker.

## Current Next Action

Run the read-only preflight runbook when WING Rack and backend are ready:

`Docs/runbooks/WING_READONLY_PREFLIGHT.md`

Before any supervised proof, accepted review source ids must have a valid
accepted-candidate prestart trace from read-only preflight review. If that trace
is missing or invalid, the proof plan must keep
`accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof`
as an operator blocker.

The no-WING handoff summary must also show `backend_port_prestart_gate` exactly
matching `expected_backend_port_prestart_gate`, with
`backend_port_prestart_gate.grants_live_authority: false` and
`backend_port_prestart_gate.must_pass_before_backend_wing_session: true`, before
any supervised fader/gain proof runbook is used.

The same `backend_port_prestart_gate` must show
`backend_port_prestart_gate.operator_sequence_step_index: 0` and
`backend_port_prestart_gate.operator_sequence_step_must_be_first: true`.

The same `backend_port_prestart_gate` must also show
`backend_port_prestart_gate.runs_command: prestart_backend_port_check`.

The same `backend_port_prestart_gate` must also show
`backend_port_prestart_gate.required_markers` exactly listing
`Before starting the intended backend/WING session`,
`prestart_backend_port_check`, and `unexpected listeners`.

The no-WING handoff summary must also show
`backend_port_prestart_check_command` exactly matching
`PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick`.

It must also show `supervised_prestart_gate` exactly matching
`expected_supervised_prestart_gate`, with
`supervised_prestart_gate.grants_live_authority: false`.

The same `supervised_prestart_gate` must show
`supervised_prestart_gate.operator_sequence_step_index: 9` and
`supervised_prestart_gate.operator_sequence_step_must_be_final: true`.

It must also show
`supervised_prestart_gate.requires_operator_decision_status: accepted_decision_ready`
and
`supervised_prestart_gate.unlocks_runbook: Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md`.

The handoff summary must also show `required_supervised_prestart_gate_markers`
exactly listing `accepted_decision_ready`,
`valid accepted-candidate prestart trace`, and
`WING_SUPERVISED_FADER_GAIN_PROOF`.

The operator sequence indexes must also remain explicit:
`operator_sequence_mvp_blocker_anchor_digest_review_step_index: 8` and
`operator_sequence_supervised_prestart_gate_step_index: 9`.

The handoff summary must also show `operator_sequence_count: 10` so reviewers
can reject packets with missing or extra operator steps.

The first operator step text must also be explicit:
`backend_port_prestart_check_operator_sequence_step` must include
`prestart_backend_port_check` and `unexpected listeners`.

## Autonomy Budget Rule

Automated cycles may continue without asking "continue?", but they must be
bounded by `tools/codex_autoloop/model_budget_profile.json`. If the expensive
profile is exhausted, Codex must keep work on the MVP rail with a cheaper model
profile or skip the cycle cleanly instead of spending GPT-5.5 high reasoning on
routine reports.
