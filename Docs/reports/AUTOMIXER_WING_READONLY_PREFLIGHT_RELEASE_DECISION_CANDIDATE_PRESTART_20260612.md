# Automixer WING Read-Only Preflight Release Decision Candidate Prestart

## Summary

This change surfaces decision-ledger `candidate_reviews` in the no-WING release summary JSON and markdown. The release artifact can now show backend-port prestart contract fields for reviewed WING read-only preflight candidates without starting a backend, contacting a WING console, or widening live authority.

## Safety Boundary

- Read-only summary/reporting change only.
- No OSC or WING WebSocket writes.
- No backend startup requirement.
- No readiness closure permission.
- No live authority granted.
- Operator review remains required.

## Verification

- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py -k "summary_embeds_wing_preflight_review_decision"`: 1 passed, 188 deselected.
- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_evidence.py tests/test_autonomy_promotion_readiness.py`: 268 passed.
- `/opt/homebrew/bin/git diff --check`: passed.
- `python3 tools/memory/memory_lint.py`: PASS.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`: 3181 passed, 1 skipped, 1 warning.
