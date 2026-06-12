# Automixer WING Proof Plan Accepted Candidate Prestart Trace

## Summary

This change binds `next_wing_proof_plan` acceptance to accepted read-only preflight candidate rows that carry the shared backend-port prestart contract. A manual review decision with accepted source ids is no longer enough by itself to list physical WING proof steps; the accepted source ids must also be present in `candidate_reviews` with the required prestart command and failure detail fields.

## Safety Boundary

- No backend startup.
- No WING transport usage.
- No OSC/WebSocket writes.
- No live authority granted.
- No automatic readiness closure.
- Physical proof steps remain non-executing plan entries requiring operator approval.

## Verification

- `PYTHONPATH=backend python3 -m pytest -q tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py -k "supervised_wing_proof_plan or next_wing_proof_plan"`: 13 passed, 187 deselected.
- `PYTHONPATH=backend python3 -m pytest -q tests/test_supervised_wing_proof_plan.py tests/test_supervised_proof_artifact_report.py tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_evidence.py tests/test_autonomy_promotion_readiness.py`: 286 passed.
- `/opt/homebrew/bin/git diff --check`: passed.
- `python3 tools/memory/memory_lint.py`: PASS.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`: 3183 passed, 1 skipped, 1 warning.
