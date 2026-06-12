# Automixer Offline no-WING Runbook Proof Plan Candidate Prestart Gate

## Summary

This change updates the offline no-WING validation runbook so the Next WING Proof Plan success criteria include the accepted candidate prestart trace fields and specific blocker. The runbook now matches the release gate: physical WING proof steps are not acceptable unless accepted read-only preflight source ids are backed by candidate review rows with the backend-port prestart contract.

## Safety Boundary

- Documentation and test only.
- No backend startup.
- No WING transport usage.
- No OSC/WebSocket writes.
- No live authority granted.

## Verification

- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py -k "offline_no_wing_runbook_lists_wing_proof_candidate_prestart_gate"`: 1 passed, 191 deselected.
- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_supervised_wing_proof_plan.py tests/test_supervised_proof_artifact_report.py tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_evidence.py tests/test_autonomy_promotion_readiness.py`: 288 passed.
- `/opt/homebrew/bin/git diff --check`: passed.
- `python3 tools/memory/memory_lint.py`: PASS.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`: 3185 passed, 1 skipped, 1 warning.
