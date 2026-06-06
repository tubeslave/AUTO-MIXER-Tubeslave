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
| WING read-only connection proof | tooling_ready_waits_real_wing | `tools/wing_readonly_preflight_runner.py`; `tools/review_wing_readonly_preflight_artifacts.py`; `tools/review_wing_readonly_preflight_decisions.py`; `backend/wing_readonly_preflight_review_decision.py`; `backend/supervised_wing_proof_plan.py`; `tools/no_wing_release_check.py`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_RUNNER_20260606.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_EVIDENCE_20260606.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_EVENT_LOG_CROSSCHECK_20260606.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_REVIEW_RELEASE_GATE_20260607.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_OPERATOR_CONFIRMATION_20260607.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_MANUAL_REVIEW_CHECKLIST_20260607.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_REVIEW_DECISION_LEDGER_20260607.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_20260607.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_DECISION_GATED_PROOF_PLAN_20260607.md` | Run `Docs/runbooks/WING_READONLY_PREFLIGHT.md` against the intended backend/WING session. |
| Physical console readback proof | tooling_ready_waits_real_wing | `wing_readonly_preflight.v1` schema requires physical readback; event-log crosscheck and `tools/review_wing_readonly_preflight_artifacts.py` review exist | Run read-only preflight with `--require-physical-readback` and review summary plus `events.jsonl`. |
| Supervised fader write proof | blocked_by_readonly_preflight | `Docs/reports/AUTOMIXER_SUPERVISED_WING_PROOF_READONLY_PREFLIGHT_GATE_20260606.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_DECISION_GATED_PROOF_PLAN_20260607.md` | After clean read-only preflight review and an accepted decision JSON, run `Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md` for `--value-type fader`. |
| Supervised gain-down / trim proof | blocked_by_readonly_preflight | `Docs/reports/AUTOMIXER_SUPERVISED_WING_PROOF_READONLY_PREFLIGHT_GATE_20260606.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_DECISION_GATED_PROOF_PLAN_20260607.md` | After clean read-only preflight review and an accepted decision JSON, run `Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md` for `--value-type gain` with a negative delta only. |
| Rollback proof | blocked_by_readonly_preflight | `tools/wing_operator_proposal_validation_runner.py` supports rollback through `rollback_last_manual_write` | Preserve rollback request, rollback readback, and artifact review from the supervised fader/gain run. |
| Emergency-stop proof | blocked_by_readonly_preflight | runner supports `--prove-emergency-stop`; server blocks later apply with `emergency_stop_active` | Use `--prove-emergency-stop` only inside the explicit supervised proof run. |
| Connection-loss / timeout fail-closed proof | tooling_ready_waits_real_wing | `Docs/reports/AUTOMIXER_WING_TIMEOUT_FAIL_CLOSED_PROOF_20260606.md` | Capture a controlled real-run timeout/connection-loss artifact and verify fail-closed semantics. |
| Operator-visible evidence report | tooling_ready_waits_real_wing | `.paperclip/reports/`; `Docs/reports/`; supervised proof report embeds autonomy readiness | Save preflight/proof artifacts and a clear operator-visible report for manual review. |
| Readiness matrix update based on real proof | blocked_by_real_artifact_review | `backend/autonomy_promotion_readiness.py`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_READINESS_BRIDGE_20260606.md` | Update readiness only after manual review of real WING artifacts; do not close rows from offline evidence. |

## Current First Open Blocker

WING read-only connection proof.

## Current First Command

Use `Docs/runbooks/WING_READONLY_PREFLIGHT.md`.

## Do Not Work On Instead

- New memory-trust chains.
- New queue-import chains.
- New result consumers.
- New broad report/read-model/frontend surfaces.
- Full autonomous mode.

These are blocked unless the change directly closes one row in the table above.
