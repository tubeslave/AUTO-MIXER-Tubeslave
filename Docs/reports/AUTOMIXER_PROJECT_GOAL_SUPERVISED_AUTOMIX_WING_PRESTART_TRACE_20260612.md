# Automixer Project Goal Supervised AutoMix WING Prestart Trace

## Summary

This change aligns the high-level project goal capability matrix with the current WING proof gates. `mode_supervised_automix` now says physical proof before widening also requires accepted-candidate prestart trace from read-only WING preflight review.

## Safety Boundary

- Matrix/test/docs only.
- No backend startup.
- No WING transport usage.
- No OSC/WebSocket writes.
- No live-write widening.
- Current live-write kinds remain unchanged.

## Verification

- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_goal_capability_matrix.py -k "supervised_automix_goal_requires_wing_candidate_prestart_trace"`: 1 passed, 26 deselected.
- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_goal_capability_matrix.py tests/test_processing_registry.py tests/test_supervised_wing_proof_plan.py tests/test_supervised_proof_artifact_report.py tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_evidence.py tests/test_autonomy_promotion_readiness.py`: 350 passed.
- `/opt/homebrew/bin/git diff --check`: passed.
- `python3 tools/memory/memory_lint.py`: PASS.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`: 3186 passed, 1 skipped, 1 warning.
