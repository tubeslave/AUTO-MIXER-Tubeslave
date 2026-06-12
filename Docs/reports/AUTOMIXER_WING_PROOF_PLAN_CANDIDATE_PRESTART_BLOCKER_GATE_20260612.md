# Automixer WING Proof Plan Candidate Prestart Blocker Gate

## Summary

This change makes the no-WING release gate require the specific candidate-prestart blocker when a WING proof plan has accepted read-only preflight source ids but cannot bind them to prestart-valid candidate rows. The plan can still be valid in `manual_review_ready`, but it must tell the operator exactly what needs repair before physical proof steps are listed.

## Safety Boundary

- No backend startup.
- No WING transport usage.
- No OSC/WebSocket writes.
- No live authority granted.
- No automatic readiness closure.
- Physical proof steps remain absent while the candidate prestart trace is invalid.

## Verification

- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py -k "next_wing_plan_without_candidate_prestart_trace or next_wing_plan_blocked_by_candidate_prestart_trace or validates_next_wing_proof_plan_schema"`: 3 passed, 188 deselected.
- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_supervised_wing_proof_plan.py tests/test_supervised_proof_artifact_report.py tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_evidence.py tests/test_autonomy_promotion_readiness.py`: 287 passed.
- `/opt/homebrew/bin/git diff --check`: passed.
- `python3 tools/memory/memory_lint.py`: PASS.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`: 3184 passed, 1 skipped, 1 warning.
