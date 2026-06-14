# WING MVP Accepted-Candidate Prestart Diagnostics

## Summary

The WING read-only review decision path now exposes exact per-source diagnostics
when an accepted candidate lacks a valid prestart trace. The same diagnostics
flow into the supervised proof plan and no-WING summary.

New report-only fields:

- `accepted_candidate_prestart_contract_violations_by_source`
- `readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source`

Both maps must be `{}` before supervised fader/gain proof steps are planned. A
non-empty map keeps
`accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof`
in force.

## Safety Boundary

This is offline/report-only. It does not contact WING, execute proof commands,
grant live authority, update readiness, or close the WING read-only connection
proof blocker.

## Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py --tb=short`
  passed: `303 passed in 2.95s`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short`
  passed: `314 passed in 3.17s`.
- `/opt/homebrew/bin/python3 -m py_compile backend/wing_readonly_preflight_contract.py backend/wing_readonly_preflight_review_decision.py backend/supervised_wing_proof_plan.py tools/review_wing_readonly_preflight_decisions.py tools/no_wing_release_check.py tools/codex_autoloop/project_director_agent.py`
  passed.
- `git diff --check` passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`
  passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --backend-port 58769 --artifact-root .paperclip/reports/no_wing_prestart_diagnostics_20260614`
  passed. The review-decision schema reported
  `accepted_candidate_prestart_contract_violations_by_source: {}`, and the
  handoff packet schema reported no violations.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --output .paperclip/reports/project_director_prestart_diagnostics_20260614.json --print`
  passed with `guard_status: on_target`, `guard_score: 90`, and selected
  blocker `WING read-only connection proof`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/codex_direction_guard.py --format json --output .paperclip/reports/codex_direction_guard_prestart_diagnostics_20260614.json`
  passed with `status: on_target`, `score: 90`, and `stop_the_line: false`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
  passed: `3299 passed, 1 skipped, 1 warning in 159.76s`.
