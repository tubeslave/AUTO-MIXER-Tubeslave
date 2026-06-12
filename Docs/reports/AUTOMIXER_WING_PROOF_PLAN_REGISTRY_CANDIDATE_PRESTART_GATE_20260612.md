# Automixer WING Proof Plan Registry Candidate Prestart Gate

## Summary

This change updates the processing registry so `supervised_wing_proof_plan` exposes the accepted-candidate prestart trace gate. The registry now matches the proof-plan builder, no-WING release gate, and offline runbook: physical WING proof steps remain blocked until accepted read-only preflight source ids are backed by candidate review rows with the backend-port prestart contract.

## Safety Boundary

- Registry/test/docs only.
- No backend startup.
- No WING transport usage.
- No OSC/WebSocket writes.
- `supervised_wing_proof_plan` remains shadow-only and outside live executor profiles.

## Verification

- `PYTHONPATH=backend python3 -m pytest -q tests/test_processing_registry.py -k "supervised_wing_proof_plan_is_shadow_only_not_executor"`: 1 passed, 34 deselected.
- `PYTHONPATH=backend python3 -m pytest -q tests/test_processing_registry.py`: 35 passed.
- `PYTHONPATH=backend python3 -m pytest -q tests/test_processing_registry.py tests/test_supervised_wing_proof_plan.py tests/test_supervised_proof_artifact_report.py tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_evidence.py tests/test_autonomy_promotion_readiness.py`: 323 passed.
- `/opt/homebrew/bin/git diff --check`: passed.
- `python3 tools/memory/memory_lint.py`: PASS.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`: 3185 passed, 1 skipped, 1 warning.
