# WING MVP Handoff Operator Sequence Contract

## Summary

The WING read-only handoff operator sequence now has a shared contract in
`backend/wing_readonly_preflight_contract.py`. The handoff builder emits
contract digest and match fields, and no-WING validation recomputes the expected
sequence before supervised proof planning.

Before supervised proof planning, the no-WING handoff summary must show:

- `expected_operator_sequence_sha256`
- `actual_expected_operator_sequence_sha256`
- `operator_sequence_matches_contract: true`
- `operator_sequence_sha256_matches_contract: true`
- `declared_expected_operator_sequence_sha256_matches_contract: true`
- `declared_operator_sequence_matches_contract_flag: true`
- `declared_operator_sequence_matches_contract_flag_matches_actual: true`
- `declared_operator_sequence_sha256_matches_contract_flag: true`
- `declared_operator_sequence_sha256_matches_contract_flag_matches_actual: true`

## Safety Boundary

This remains offline and read-only. It does not contact WING, execute packet
commands, grant live authority, or close the WING read-only connection proof
blocker.

## Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_handoff_packet.py --tb=short`
  passed as part of the focused implementation check: `279 passed in 3.84s`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short`
  passed: `290 passed in 3.03s`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --backend-port 58768 --artifact-root .paperclip/reports/no_wing_operator_sequence_contract_20260614`
  passed. The `wing_readonly_preflight_handoff_packet_artifact_schema`
  detail reported `operator_sequence_count: 10`,
  `expected_operator_sequence_count: 10`,
  `operator_sequence_count_matches_contract: true`,
  `operator_sequence_sha256: 99e574cddb02b198eb8eb09e511221292347504db2e4b89a7dd18b63c09c6e6b`,
  `actual_operator_sequence_sha256: 99e574cddb02b198eb8eb09e511221292347504db2e4b89a7dd18b63c09c6e6b`,
  `expected_operator_sequence_sha256: 99e574cddb02b198eb8eb09e511221292347504db2e4b89a7dd18b63c09c6e6b`,
  `actual_expected_operator_sequence_sha256: 99e574cddb02b198eb8eb09e511221292347504db2e4b89a7dd18b63c09c6e6b`,
  `operator_sequence_sha256_matches_actual: true`,
  `operator_sequence_matches_contract: true`,
  `operator_sequence_sha256_matches_contract: true`,
  all declared sequence contract flags true, and no violations.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --output .paperclip/reports/project_director_operator_sequence_contract_20260614.json --print`
  passed: `guard_status: on_target`, `guard_score: 90`,
  `selected_blocker: WING read-only connection proof`, and
  `selected_evidence_file_count: 66`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/codex_direction_guard.py --format json --output .paperclip/reports/codex_direction_guard_operator_sequence_contract_20260614.json`
  passed: `status: on_target`, `score: 90`, `stop_the_line: false`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`
  passed.
- `/opt/homebrew/bin/python3 -m py_compile backend/wing_readonly_preflight_contract.py tools/no_wing_release_check.py tools/build_wing_readonly_preflight_handoff_packet.py tools/codex_autoloop/project_director_agent.py`
  passed.
- `git diff --check` passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
  passed: `3299 passed, 1 skipped, 1 warning in 159.02s`.
