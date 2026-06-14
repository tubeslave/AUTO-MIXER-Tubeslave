# Automixer WING MVP Handoff Operator Sequence Cardinality

## Summary

The no-WING handoff validator now fails closed when the handoff operator
sequence does not exactly match the contract count or when the supervised
prestart gate moves away from index `9`.

## What changed

- `backend/wing_readonly_preflight_contract.py` defines
  `WING_READONLY_PREFLIGHT_HANDOFF_OPERATOR_SEQUENCE_COUNT`.
- `tools/no_wing_release_check.py` rejects shifted handoff sequences with:
  - `wing_readonly_preflight_handoff_packet_operator_sequence_count_mismatch`
  - `wing_readonly_preflight_handoff_packet_supervised_prestart_gate_step_index_mismatch`
- no-WING summary JSON/Markdown now show:
  - `expected_operator_sequence_count`
  - `operator_sequence_count_matches_contract`
  - `expected_operator_sequence_supervised_prestart_gate_step_index`
  - `operator_sequence_supervised_prestart_gate_step_index_matches_contract`

## Safety boundary

This remains offline and read-only. It does not contact WING, does not contact
the backend, does not mutate console state, and does not grant live authority.

## Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short`
  - `267 passed in 2.96s`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile backend/wing_readonly_preflight_contract.py tools/no_wing_release_check.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --backend-port 65535 --artifact-root /tmp/automixer_goal_handoff_sequence_cardinality_20260614`
  - `summary status: passed`
  - `handoff status: passed`
  - `operator sequence count: 10`
  - `expected operator sequence count: 10`
  - `count matches contract: True`
  - `supervised gate index: 9`
  - `expected supervised gate index: 9`
  - `supervised gate index matches contract: True`
  - `violations: []`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/codex_direction_guard.py --format json --output .paperclip/reports/codex_direction_guard_handoff_sequence_cardinality_20260614.md`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --output .paperclip/reports/project_director_handoff_sequence_cardinality_20260614.md --print`
  - `guard_status: on_target`
  - `selected_blocker: WING read-only connection proof`
  - `selected_blocker_status: tooling_ready_waits_real_wing`
  - `selected_evidence_file_count: 61`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`
  - `Status: PASS`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
  - `3294 passed, 1 skipped, 1 warning in 165.15s`

## Remaining blocker

The WING read-only connection proof still requires a real
`wing_readonly_preflight.v1` summary and `events.jsonl` from the intended WING
Rack session.
