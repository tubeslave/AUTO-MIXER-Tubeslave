# WING MVP Handoff Operator Sequence SHA256

## Summary

The WING read-only handoff packet now declares `operator_sequence_sha256`, a
canonical SHA256 digest over the complete ordered operator sequence. The
no-WING handoff gate recomputes that digest from the packet sequence and fails
closed if the declared digest is missing or mismatched.

Before supervised proof planning, the no-WING handoff summary must show:

- `operator_sequence_sha256`
- `actual_operator_sequence_sha256`
- `operator_sequence_sha256_matches_actual: true`

## Safety Boundary

This remains offline and read-only. It does not contact WING, execute packet
commands, grant live authority, or close the WING read-only connection proof
blocker.

## Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_handoff_packet.py --tb=short`
  passed as part of the focused implementation check: `278 passed in 3.81s`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short`
  passed: `289 passed in 3.03s`.
- `/opt/homebrew/bin/python3 -m py_compile tools/no_wing_release_check.py tools/build_wing_readonly_preflight_handoff_packet.py tools/codex_autoloop/project_director_agent.py backend/wing_readonly_preflight_contract.py`
  passed.
- `git diff --check` passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --backend-port 58767 --artifact-root .paperclip/reports/no_wing_operator_sequence_sha256_20260614`
  passed. The `wing_readonly_preflight_handoff_packet_artifact_schema`
  detail reported `operator_sequence_count: 10`,
  `expected_operator_sequence_count: 10`,
  `operator_sequence_count_matches_contract: true`,
  `operator_sequence_sha256: 99e574cddb02b198eb8eb09e511221292347504db2e4b89a7dd18b63c09c6e6b`,
  `actual_operator_sequence_sha256: 99e574cddb02b198eb8eb09e511221292347504db2e4b89a7dd18b63c09c6e6b`,
  `operator_sequence_sha256_matches_actual: true`, all declared count
  fields true, and no violations.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --output .paperclip/reports/project_director_operator_sequence_sha256_20260614.json --print`
  passed: `guard_status: on_target`, `guard_score: 90`,
  `selected_blocker: WING read-only connection proof`, and
  `selected_evidence_file_count: 65`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/codex_direction_guard.py --format json --output .paperclip/reports/codex_direction_guard_operator_sequence_sha256_20260614.json`
  passed: `status: on_target`, `score: 90`, `stop_the_line: false`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`
  passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
  passed: `3298 passed, 1 skipped, 1 warning in 164.58s`.
