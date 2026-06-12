# Automixer WING MVP Progress Prestart Trace Gate

## Summary

This change aligns the supervised WING MVP status rails with the current proof
plan gate. The first blocker remains `WING read-only connection proof` with
status `tooling_ready_waits_real_wing`, but the blocker table, Project Director
selected action, and generated MVP progress report now state that accepted
review source ids need a valid accepted-candidate prestart trace before any
supervised proof.

If that trace is missing or invalid, the next proof plan must keep
`accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof`
as an operator blocker.

## Safety Boundary

- No backend process was started.
- No WING transport was contacted.
- No OSC/WebSocket writes were added.
- No live authority was widened.

## Verification

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_mvp_rails.py tests/test_project_director_agent.py` - PASS, 9 passed.
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_mvp_rails.py tests/test_project_director_agent.py tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py` - PASS, 273 passed.
- `/opt/homebrew/bin/git diff --check` - PASS.
- `python3 tools/memory/memory_lint.py` - PASS.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q` - PASS, 3186 passed, 1 skipped, 1 warning.
