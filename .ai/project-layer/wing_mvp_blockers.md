# WING MVP Blockers

Purpose: this is the live task table for the supervised WING Rack MVP. Codex
must choose the next task from the first row that is not `closed`.

Status values:

- `open`: no accepted real proof yet.
- `tooling_ready_waits_real_wing`: code/report tooling exists, but real WING
  artifact is still required.
- `blocked_by_readonly_preflight`: cannot start until a clean read-only
  preflight candidate is reviewed.
- `blocked_by_real_artifact_review`: cannot close until a real artifact is
  manually reviewed.
- `closed`: accepted real proof exists.

| Blocker | Status | Evidence | Next action |
| --- | --- | --- | --- |
| WING read-only connection proof | tooling_ready_waits_real_wing | `tools/wing_readonly_preflight_runner.py`; `tools/review_wing_readonly_preflight_artifacts.py`; `tools/review_wing_readonly_preflight_decisions.py`; `tools/build_wing_readonly_preflight_handoff_packet.py`; `backend/wing_readonly_preflight_review_decision.py`; `backend/supervised_wing_proof_plan.py`; `backend/wing_readonly_preflight_contract.py`; `tools/no_wing_release_check.py`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_RUNNER_20260606.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_EVIDENCE_20260606.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_EVENT_LOG_CROSSCHECK_20260606.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_REVIEW_RELEASE_GATE_20260607.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_OPERATOR_CONFIRMATION_20260607.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_SESSION_CONFIRMATION_GATE_20260607.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_SESSION_READY_REVIEW_CHECKLIST_20260607.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_DECISION_ARTIFACT_HASH_BINDING_20260607.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_MANUAL_REVIEW_CHECKLIST_20260607.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_REVIEW_DECISION_LEDGER_20260607.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_20260607.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_HANDOFF_PACKET_20260607.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_DECISION_GATED_PROOF_PLAN_20260607.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_HANDOFF_FORBIDDEN_REQUEST_COVERAGE_20260607.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_TRANSPORT_TIMEOUT_ARTIFACT_20260607.md` | Run `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick` before starting the intended backend/WING session, then run `Docs/runbooks/WING_READONLY_PREFLIGHT.md`. Before any supervised proof, the no-WING handoff summary must show `operator_sequence_count: 10`, `backend_port_prestart_gate` exactly matching `expected_backend_port_prestart_gate`, with `backend_port_prestart_gate.grants_live_authority: false`, `backend_port_prestart_gate.must_pass_before_backend_wing_session: true`, `backend_port_prestart_gate.operator_sequence_step_index: 0`, `backend_port_prestart_gate.operator_sequence_step_must_be_first: true`, `backend_port_prestart_gate.runs_command: prestart_backend_port_check`, `backend_port_prestart_gate.required_markers` exactly listing `Before starting the intended backend/WING session`, `prestart_backend_port_check`, and `unexpected listeners`, `backend_port_prestart_check_operator_sequence_step_index: 0`, `backend_port_prestart_check_operator_sequence_step` including `prestart_backend_port_check` and `unexpected listeners`, `backend_port_prestart_check_must_be_first_operator_step: true`, `backend_port_prestart_check_operator_sequence_step_is_first: true`, `mvp_blocker_anchor` matching the current first open blocker row, `mvp_blocker_anchor_matches_current_first_open: true`, `mvp_blocker_anchor_sha256_matches_current_first_open: true`, `mvp_blocker_anchor.packet_does_not_close_blocker: true`, `operator_sequence_has_mvp_blocker_anchor_digest_review: true`, `operator_sequence_mvp_blocker_anchor_digest_review_step_index: 8`, `operator_sequence_mvp_blocker_anchor_digest_review_step` including `mvp_blocker_anchor`, `mvp_blocker_anchor_sha256`, and `current first open blocker`, `operator_sequence_has_supervised_prestart_gate: true`, `operator_sequence_supervised_prestart_gate_is_final: true`, `operator_sequence_supervised_prestart_gate_step_index: 9`, `operator_sequence_supervised_prestart_gate_step` including `accepted_decision_ready`, `valid accepted-candidate prestart trace`, and `WING_SUPERVISED_FADER_GAIN_PROOF`, `required_supervised_prestart_gate_markers` exactly listing `accepted_decision_ready`, `valid accepted-candidate prestart trace`, and `WING_SUPERVISED_FADER_GAIN_PROOF`, `supervised_prestart_gate` exactly matching `expected_supervised_prestart_gate`, `supervised_prestart_gate.grants_live_authority: false`, `supervised_prestart_gate.operator_sequence_step_index: 9`, `supervised_prestart_gate.operator_sequence_step_must_be_final: true`, `supervised_prestart_gate.requires_operator_decision_status: accepted_decision_ready`, `supervised_prestart_gate.unlocks_runbook: Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md`, and `supervised_prestart_gate.requires_accepted_candidate_prestart_trace: true`; accepted review source ids must also have a valid accepted-candidate prestart trace, otherwise keep `accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof` in the proof plan blockers. |
| Physical console readback proof | tooling_ready_waits_real_wing | `wing_readonly_preflight.v1` schema requires physical readback; event-log crosscheck and `tools/review_wing_readonly_preflight_artifacts.py` review exist | Run read-only preflight with `--require-physical-readback` and review summary plus `events.jsonl`. |
| Supervised fader write proof | blocked_by_readonly_preflight | `Docs/reports/AUTOMIXER_SUPERVISED_WING_PROOF_READONLY_PREFLIGHT_GATE_20260606.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_DECISION_GATED_PROOF_PLAN_20260607.md` | After clean read-only preflight review, an accepted decision JSON, `backend_port_prestart_gate` exactly matching `expected_backend_port_prestart_gate` without live authority, the final handoff operator-sequence step with valid accepted-candidate prestart trace, and `supervised_prestart_gate` exactly matching `expected_supervised_prestart_gate`, run `Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md` for `--value-type fader`. |
| Supervised gain-down / trim proof | blocked_by_readonly_preflight | `Docs/reports/AUTOMIXER_SUPERVISED_WING_PROOF_READONLY_PREFLIGHT_GATE_20260606.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_DECISION_GATED_PROOF_PLAN_20260607.md` | After clean read-only preflight review, an accepted decision JSON, `backend_port_prestart_gate` exactly matching `expected_backend_port_prestart_gate` without live authority, the final handoff operator-sequence step with valid accepted-candidate prestart trace, and `supervised_prestart_gate` exactly matching `expected_supervised_prestart_gate`, run `Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md` for `--value-type gain` with a negative delta only. |
| Rollback proof | blocked_by_readonly_preflight | `tools/wing_operator_proposal_validation_runner.py` supports rollback through `rollback_last_manual_write` | Preserve rollback request, rollback readback, and artifact review from the supervised fader/gain run. |
| Emergency-stop proof | blocked_by_readonly_preflight | runner supports `--prove-emergency-stop`; server blocks later apply with `emergency_stop_active` | Use `--prove-emergency-stop` only inside the explicit supervised proof run. |
| Connection-loss / timeout fail-closed proof | tooling_ready_waits_real_wing | `Docs/reports/AUTOMIXER_WING_TIMEOUT_FAIL_CLOSED_PROOF_20260606.md` | Capture a controlled real-run timeout/connection-loss artifact and verify fail-closed semantics. |
| Operator-visible evidence report | tooling_ready_waits_real_wing | `.paperclip/reports/`; `Docs/reports/`; supervised proof report embeds autonomy readiness | Save preflight/proof artifacts and a clear operator-visible report for manual review. |
| Readiness matrix update based on real proof | blocked_by_real_artifact_review | `backend/autonomy_promotion_readiness.py`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_READINESS_BRIDGE_20260606.md` | Update readiness only after manual review of real WING artifacts; do not close rows from offline evidence. |

## Current First Open Blocker

WING read-only connection proof.

## Current First Command

Use `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick` before
starting the intended backend/WING session, then use
`Docs/runbooks/WING_READONLY_PREFLIGHT.md`. Before any supervised proof,
the no-WING handoff summary must show `operator_sequence_count: 10`;
it must also show `backend_port_prestart_gate` exactly
matching `expected_backend_port_prestart_gate`, with
`backend_port_prestart_gate.grants_live_authority: false` and
`backend_port_prestart_gate.must_pass_before_backend_wing_session: true`;
it must also show `backend_port_prestart_gate.operator_sequence_step_index: 0`;
it must also show
`backend_port_prestart_gate.operator_sequence_step_must_be_first: true`;
it must also show
`backend_port_prestart_gate.runs_command: prestart_backend_port_check`;
it must also show `backend_port_prestart_gate.required_markers` exactly listing
`Before starting the intended backend/WING session`,
`prestart_backend_port_check`, and `unexpected listeners`;
it must also show `backend_port_prestart_check_operator_sequence_step_index: 0`;
it must also show `backend_port_prestart_check_operator_sequence_step`
including `prestart_backend_port_check` and `unexpected listeners`;
it must also show
`backend_port_prestart_check_must_be_first_operator_step: true`;
it must also show
`backend_port_prestart_check_operator_sequence_step_is_first: true`;
it must also show `mvp_blocker_anchor` matching the current first open blocker
row and `mvp_blocker_anchor_matches_current_first_open: true`;
the no-WING handoff summary must also show
`mvp_blocker_anchor_sha256_matches_current_first_open: true`;
the no-WING handoff summary must also show
`mvp_blocker_anchor.packet_does_not_close_blocker: true`;
the no-WING handoff summary must also show
`operator_sequence_has_mvp_blocker_anchor_digest_review: true`;
the no-WING handoff summary must also show
`operator_sequence_mvp_blocker_anchor_digest_review_step_index: 8`;
the digest review `operator_sequence_mvp_blocker_anchor_digest_review_step`
must include `mvp_blocker_anchor`, `mvp_blocker_anchor_sha256`, and
`current first open blocker`;
the no-WING handoff summary must also show
`operator_sequence_has_supervised_prestart_gate: true`;
the no-WING handoff summary must also show
`operator_sequence_supervised_prestart_gate_is_final: true`;
the no-WING handoff summary must also show
`operator_sequence_supervised_prestart_gate_step_index: 9`;
the final `operator_sequence_supervised_prestart_gate_step` must include
`accepted_decision_ready`, `valid accepted-candidate prestart trace`, and
`WING_SUPERVISED_FADER_GAIN_PROOF`;
the no-WING handoff summary must also show
`required_supervised_prestart_gate_markers` exactly listing
`accepted_decision_ready`, `valid accepted-candidate prestart trace`, and
`WING_SUPERVISED_FADER_GAIN_PROOF`;
the no-WING handoff summary must show `supervised_prestart_gate` exactly
matching `expected_supervised_prestart_gate`;
the no-WING handoff summary must also show
`supervised_prestart_gate.grants_live_authority: false`;
the no-WING handoff summary must also show
`supervised_prestart_gate.operator_sequence_step_index: 9`;
the no-WING handoff summary must also show
`supervised_prestart_gate.operator_sequence_step_must_be_final: true`;
the no-WING handoff summary must also show
`supervised_prestart_gate.requires_operator_decision_status: accepted_decision_ready`;
the no-WING handoff summary must also show
`supervised_prestart_gate.unlocks_runbook: Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md`;
the no-WING handoff summary must also show
`supervised_prestart_gate.requires_accepted_candidate_prestart_trace: true`;
accepted review source ids must also have a valid accepted-candidate prestart
trace, otherwise keep
`accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof`
in the proof plan blockers.

## Do Not Work On Instead

- New memory-trust chains.
- New queue-import chains.
- New result consumers.
- New broad report/read-model/frontend surfaces.
- Full autonomous mode.

These are blocked unless the change directly closes one row in the table above.
