# WING MVP Handoff Declared Operator Sequence Count

## Summary

The WING read-only handoff packet now declares its operator sequence count, and
the no-WING handoff gate validates that declaration against the actual operator
sequence and the shared contract.

Before supervised proof planning, the no-WING handoff summary must show:

- `declared_operator_sequence_count: 10`
- `declared_expected_operator_sequence_count: 10`
- `declared_operator_sequence_count_contract_flag: true`
- `declared_operator_sequence_count_matches_actual: true`
- `declared_operator_sequence_count_matches_contract: true`
- `declared_expected_operator_sequence_count_matches_contract: true`
- `declared_operator_sequence_count_contract_flag_matches_declared_count: true`

## Safety Boundary

This remains offline and read-only. It does not contact WING, execute packet
commands, grant live authority, or close the WING read-only connection proof
blocker.

## Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short`
  passed: `270 passed in 3.02s`.
- `/opt/homebrew/bin/python3 -m py_compile tools/no_wing_release_check.py tools/build_wing_readonly_preflight_handoff_packet.py tools/codex_autoloop/project_director_agent.py backend/wing_readonly_preflight_contract.py`
  passed.
- `git diff --check` passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --backend-port 58766 --artifact-root .paperclip/reports/no_wing_declared_operator_sequence_count_final_20260614`
  passed. The `wing_readonly_preflight_handoff_packet_artifact_schema`
  detail reported `operator_sequence_count: 10`,
  `expected_operator_sequence_count: 10`,
  `operator_sequence_count_matches_contract: true`,
  `declared_operator_sequence_count: 10`,
  `declared_expected_operator_sequence_count: 10`,
  `declared_operator_sequence_count_contract_flag: true`,
  `declared_operator_sequence_count_matches_actual: true`,
  `declared_operator_sequence_count_matches_contract: true`,
  `declared_expected_operator_sequence_count_matches_contract: true`,
  `declared_operator_sequence_count_contract_flag_matches_declared_count: true`,
  and no violations.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --output .paperclip/reports/project_director_declared_operator_sequence_count_20260614.json --print`
  passed: `guard_status: on_target`, `guard_score: 90`,
  `selected_blocker: WING read-only connection proof`, and
  `selected_evidence_file_count: 64`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/codex_direction_guard.py --format json --output .paperclip/reports/codex_direction_guard_declared_operator_sequence_count_20260614.json`
  passed: `status: on_target`, `score: 90`, `stop_the_line: false`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`
  passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
  passed: `3297 passed, 1 skipped, 1 warning in 161.04s`.
