# WING MVP Decision Expected-Artifacts Binding

## Summary

The WING read-only decision rail now requires an accepted candidate source id
to keep the expected-artifact binding that was established during evidence
review. A candidate with a valid no-WING prestart trace is still blocked unless
it also carries:

- `expected_artifacts` with the summary/event globs.
- `expected_summary_artifact` and `expected_events_artifact`.
- `summary_path_matches_expected_artifact: true`.
- `events_path_matches_expected_artifact: true`.

This report is tooling evidence for the current first MVP blocker,
`WING read-only connection proof`. It does not close that blocker; a real WING
read-only artifact and manual operator review are still required.

## Safety Boundary

- No WING hardware contact.
- No backend WebSocket opened.
- No OSC writes.
- No fader, gain, trim, EQ, compressor, routing, or scene mutation.
- No live authority granted.
- Readiness closure remains blocked without real WING proof.

## Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile backend/wing_readonly_preflight_contract.py backend/wing_readonly_preflight_review_decision.py backend/supervised_wing_proof_plan.py tools/review_wing_readonly_preflight_decisions.py tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_supervised_wing_proof_plan.py`: passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py --tb=short`: `30 passed`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_supervised_wing_proof_plan.py tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py --tb=short`: `42 passed`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_runner.py tests/test_no_wing_release_check.py --tb=short`: `342 passed`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --backend-port 65535 --artifact-root /tmp/automixer_goal_decision_expected_artifacts_binding_20260614`: passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_project_director_agent.py tests/test_codex_direction_guard.py tests/test_wing_mvp_rails.py --tb=short`: `23 passed`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --print`: selected `WING read-only connection proof`, `guard_status: on_target`, `selected_evidence_file_count: 48`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/codex_direction_guard.py --format json`: `status: on_target`, `stop_the_line: false`, `score: 90`.
- `/opt/homebrew/bin/git diff --check`: passed.
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py`: PASS.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`: `3283 passed, 1 skipped, 1 warning`.
