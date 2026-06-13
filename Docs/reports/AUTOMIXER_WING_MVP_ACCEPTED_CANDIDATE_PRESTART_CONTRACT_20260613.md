# Automixer WING MVP Accepted-Candidate Prestart Contract

Date: 2026-06-13

Status: Validated

## Summary

This change keeps the supervised WING proof blocked unless the accepted
read-only preflight candidate has a valid no-WING prestart trace. The trace
predicate now lives in `backend/wing_readonly_preflight_contract.py` and is
shared by:

- `backend/wing_readonly_preflight_review_decision.py`
- `backend/supervised_wing_proof_plan.py`

Runner and evidence intake also use the shared no-WING prestart summary type,
connection policy, and backend check name constants.

## Safety boundary

- No hardware contact.
- No live writes.
- No readiness closure.
- No supervised proof execution.
- No full-autonomous readiness claim.

## Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile
  backend/wing_readonly_preflight_contract.py
  backend/wing_readonly_preflight_review_decision.py
  backend/supervised_wing_proof_plan.py
  backend/wing_readonly_preflight_evidence.py
  tools/wing_readonly_preflight_runner.py`: passed.
- Focused tests:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_wing_readonly_preflight_review_decision.py
  tests/test_supervised_wing_proof_plan.py
  tests/test_wing_readonly_preflight_evidence.py
  tests/test_wing_readonly_preflight_runner.py
  tests/test_no_wing_release_check.py --tb=short`: `333 passed`.
- no-WING quick on unused backend port:
  `PYTHONPATH=backend /opt/homebrew/bin/python3
  tools/no_wing_release_check.py --quick --backend-port 65535
  --artifact-root
  /tmp/automixer_goal_candidate_prestart_contract_nowing_20260613`: passed.
  Summary:
  `/tmp/automixer_goal_candidate_prestart_contract_nowing_20260613/20260613T164805Z_no_wing_release_check/summary.json`.
- no-WING summary fields:
  `status: passed`, `failed_checks: []`, `live_write: false`,
  `connection_policy: no_hardware_no_network_no_live_writes`,
  `wing_readonly_preflight_plan_artifact.selected_mvp_blocker:
  WING read-only connection proof`,
  `wing_readonly_preflight_plan_artifact.selected_mvp_blocker_status:
  tooling_ready_waits_real_wing`,
  `wing_readonly_preflight_handoff_packet_artifact.operator_sequence_count:
  10`,
  `wing_readonly_preflight_handoff_packet_artifact.readiness_closure_allowed_by_packet:
  false`,
  `wing_readonly_preflight_handoff_packet_artifact.manual_review_required_before_readiness_update:
  true`,
  `wing_readonly_preflight_review_decision_ledger.accepted_candidate_prestart_contract_valid:
  false`,
  `next_wing_proof_plan.readonly_preflight_review_decision_candidate_prestart_contract_valid:
  false`.
- Direction guard:
  `status: on_target`, `current_focus: supervised_wing_rack_mvp`,
  `stop_the_line: false`, `score: 90`.
- Project Director:
  `current_mvp: supervised_wing_rack_proof`,
  `selected_blocker: WING read-only connection proof`,
  `selected_blocker_status: tooling_ready_waits_real_wing`,
  `guard_status: on_target`, `guard_stop_the_line: false`.
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py`: PASS.
- Full suite:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x
  --tb=short -q`: `3246 passed, 1 skipped, 1 warning`.

## Next real-world step

Run the no-WING quick prestart check before the intended backend/WING session,
then run `Docs/runbooks/WING_READONLY_PREFLIGHT.md` against real WING Rack and
manually review the captured `wing_readonly_preflight.v1` artifacts.
