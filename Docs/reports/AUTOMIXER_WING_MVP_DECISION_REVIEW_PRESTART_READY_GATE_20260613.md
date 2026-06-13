# Automixer WING MVP Decision Review Prestart Ready Gate

Date: 2026-06-13
Status: Validated

## Summary

The WING read-only preflight decision ledger and decision-review CLI now treat
`accepted_decision_ready` as a full next-step gate, not merely an accepted JSON
record. Accepted source ids must also have a valid accepted-candidate backend
prestart trace.

## Why

The first WING MVP blocker must advance only from real proof artifacts. An
accepted operator decision without the current backend prestart trace is still
not enough for supervised fader/gain proof planning. The CLI, release summary,
and proof plan now expose the same prestart validity boundary.

## Changed surfaces

- `backend/wing_readonly_preflight_contract.py` owns the shared prestart blocker
  id.
- `backend/wing_readonly_preflight_review_decision.py` exposes
  `accepted_candidate_prestart_contract_valid` and
  `accepted_candidate_prestart_source_ids`.
- `tools/review_wing_readonly_preflight_decisions.py` returns a blocked status
  instead of `accepted_decision_ready` when accepted source ids lack the valid
  trace.
- `tools/no_wing_release_check.py` validates and renders the same fields.
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md` and `Docs/MVP_WING_SUPERVISED.md`
  describe the stricter ready gate.
- Focused tests cover accepted decisions with and without the prestart trace.

## Validation

- `/opt/homebrew/bin/python3 -m py_compile
  backend/wing_readonly_preflight_contract.py
  backend/wing_readonly_preflight_review_decision.py
  backend/supervised_wing_proof_plan.py
  tools/review_wing_readonly_preflight_decisions.py
  tools/no_wing_release_check.py`: passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_wing_readonly_preflight_review_decision.py
  tests/test_wing_readonly_preflight_review_decision_tool.py
  tests/test_wing_readonly_preflight_evidence.py
  tests/test_wing_readonly_preflight_review_tool.py
  tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py
  tests/test_wing_mvp_rails.py tests/test_project_director_agent.py
  tests/test_codex_direction_guard.py`: 309 passed in 3.02s.
- Project Director JSON check selected `WING read-only connection proof`,
  reported `guard_status: on_target`, and `guard_stop_the_line: false`.
- Direction guard JSON check reported `status: on_target` and
  `stop_the_line: false`.
- `/opt/homebrew/bin/git diff --check`: passed.
- `python3 tools/memory/memory_lint.py`: PASS.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x
  --tb=short -q`: 3207 passed, 1 skipped, 1 warning in 235.30s (0:03:55).

## Boundary observation

`PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py
--quick` failed closed before this change because a local listener already
occupied `127.0.0.1:8765`. The gate reported
`backend_port_listener_requires_operator_confirmation`,
`safe_to_stop_listener_by_gate: false`, and did not stop or reconfigure the
listener.
