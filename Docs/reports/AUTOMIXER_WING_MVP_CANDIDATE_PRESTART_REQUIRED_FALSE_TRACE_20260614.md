# Automixer WING MVP Candidate Prestart Required-False Trace

Date: 2026-06-14
Status: Validated

## Summary

Accepted-candidate WING read-only prestart traces now preserve required-false
backend-port safety detail fields through artifact evidence, artifact review,
decision review, and the shared prestart contract helper.

## Why

Evidence intake already rejects unsafe referenced no-WING prestart details.
Downstream proof planning also needs the same safety-denial fields visible in
candidate review rows before an accepted decision can become
`accepted_decision_ready`.

## Changed surfaces

- `backend/wing_readonly_preflight_contract.py` rejects candidate prestart rows
  when required-false fields are missing or truthy.
- `backend/wing_readonly_preflight_evidence.py` records the fields in artifact
  records.
- `backend/wing_readonly_preflight_review_decision.py` carries the fields into
  candidate review rows.
- `tools/review_wing_readonly_preflight_artifacts.py` and
  `tools/review_wing_readonly_preflight_decisions.py` render the fields.

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
  0.23s.
- Focused artifact review pytest:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_wing_readonly_preflight_review_tool.py --tb=short -x`: 10 passed
  in 0.08s.
- Focused decision ledger pytest:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_wing_readonly_preflight_review_decision.py --tb=short -x`: 21
  passed in 0.07s.
- Focused decision report pytest:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_wing_readonly_preflight_review_decision_tool.py --tb=short -x`:
  10 passed in 0.08s.
- Focused combined WING review pytest:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_wing_readonly_preflight_evidence.py
  tests/test_wing_readonly_preflight_review_tool.py
  tests/test_wing_readonly_preflight_review_decision.py
  tests/test_wing_readonly_preflight_review_decision_tool.py --tb=short -x`:
  94 passed in 0.23s.
- Focused supervised WING proof-plan pytest:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_supervised_wing_proof_plan.py --tb=short -x`: 13 passed in
  0.06s.
- Focused no-WING release-check pytest:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_no_wing_release_check.py --tb=short -x`: 267 passed, 8 skipped
  in 3.79s.
- Focused Project Director and WING rail pytest:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_project_director_agent.py tests/test_wing_mvp_rails.py
  --tb=short -x`: 11 passed in 0.14s.
- Focused WING/no-WING/readiness pytest after fixture alignment:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_autonomy_promotion_readiness.py
  tests/test_wing_readonly_preflight_evidence.py
  tests/test_wing_readonly_preflight_review_tool.py
  tests/test_wing_readonly_preflight_review_decision.py
  tests/test_wing_readonly_preflight_review_decision_tool.py
  tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py
  tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short
  -x`: 396 passed, 8 skipped in 3.32s.
- Project Director JSON selected `WING read-only connection proof`,
  `selected_blocker_status: tooling_ready_waits_real_wing`,
  `selected_action_source: Current First Command`,
  `current_first_command_baseline_coverage_passed: true`,
  `current_first_command_baseline_missing_markers: []`,
  `selected_evidence_file_count: 106`, `guard_status: on_target`,
  `guard_score: 90`, and `guard_stop_the_line: false`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile
  backend/wing_readonly_preflight_contract.py
  backend/wing_readonly_preflight_evidence.py
  backend/wing_readonly_preflight_review_decision.py
  tools/review_wing_readonly_preflight_artifacts.py
  tools/review_wing_readonly_preflight_decisions.py`: passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`:
  PASS.
- `/opt/homebrew/bin/git diff --check`: passed.
- Full standard suite:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x
  --tb=short -q`: 3313 passed, 9 skipped, 1 warning in 70.94s.
