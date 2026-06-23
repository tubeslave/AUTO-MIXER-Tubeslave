# Automixer WING MVP Evidence Prestart Detail Contract

Date: 2026-06-14
Status: Validated

## Summary

Post-run WING read-only evidence intake now rejects captured artifacts when the
referenced no-WING prestart summary's backend-port detail contract is stale or
unsafe.

## Why

The first WING blocker remains `WING read-only connection proof`. Runner-side
pre-WebSocket validation is necessary, but post-run review must also enforce the
same no-WING prestart boundary before accepting a captured artifact as a
read-only preflight candidate.

## Changed surfaces

- `backend/wing_readonly_preflight_evidence.py` validates the referenced
  prestart summary's backend-port failure-detail list, required-false detail
  fields, and recorded-summary matches.
- `tests/test_wing_readonly_preflight_evidence.py` covers stale detail-list and
  unsafe required-false prestart detail cases.
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md` documents the post-run
  fail-closed condition.

## Safety Boundary

- No WING hardware contact.
- No backend/WING listener start, stop, or reconfiguration.
- No OSC/WING writes.
- No live authority granted.
- No readiness closure from offline evidence.

## Validation

- Focused evidence pytest:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_wing_readonly_preflight_evidence.py --tb=short -x`: 53 passed in
  0.19s.
- Focused review artifact pytest:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_wing_readonly_preflight_review_tool.py --tb=short -x`: 10 passed
  in 0.07s.
- Focused review decision pytest:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_wing_readonly_preflight_review_decision_tool.py --tb=short -x`:
  10 passed in 0.05s.
- Focused Project Director and WING rail pytest:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_project_director_agent.py tests/test_wing_mvp_rails.py
  --tb=short -x`: 11 passed in 0.13s.
- Project Director JSON selected `WING read-only connection proof`,
  `selected_blocker_status: tooling_ready_waits_real_wing`,
  `selected_action_source: Current First Command`,
  `current_first_command_baseline_coverage_passed: true`,
  `current_first_command_baseline_missing_markers: []`,
  `selected_evidence_file_count: 105`, `guard_status: on_target`,
  `guard_score: 90`, and `guard_stop_the_line: false`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile
  backend/wing_readonly_preflight_evidence.py
  tools/codex_autoloop/update_mvp_progress.py`: passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`:
  PASS.
- `/opt/homebrew/bin/git diff --check`: passed.
- Full standard suite:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x
  --tb=short -q`: 3310 passed, 9 skipped, 1 warning in 70.93s.
