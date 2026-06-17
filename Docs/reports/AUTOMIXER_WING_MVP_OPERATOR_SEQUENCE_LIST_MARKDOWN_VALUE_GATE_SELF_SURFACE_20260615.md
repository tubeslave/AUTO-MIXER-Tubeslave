# WING MVP Operator Sequence List Markdown Value Gate Self-Surface

## Summary

The no-WING operator-sequence-list Markdown-surface check now requires the
operator-readable `summary.md` to render the value-gate self-surface fields:

- `wing_readonly_operator_sequence_list_markdown_surface_required_field_lines_expected_value`
- `wing_readonly_operator_sequence_list_markdown_surface_required_field_lines_actual_value`
- `wing_readonly_operator_sequence_list_markdown_surface_required_field_lines_match_contract`

The check also records `values_match_contract` and `value_mismatches` in
`summary.json`, and fails closed if any value-gate line is missing or mismatched.

## Safety Boundary

This remains offline and read-only. It does not contact WING, start a backend,
execute handoff commands, grant live authority, or close the WING read-only
connection proof blocker.

## Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile tools/no_wing_release_check.py tools/codex_autoloop/update_mvp_progress.py tests/test_no_wing_release_check.py`
  passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py::test_wing_readonly_operator_sequence_list_markdown_surface_check_detects_required_field_lines_value_drift tests/test_no_wing_release_check.py::test_wing_readonly_operator_sequence_list_markdown_surface_check_detects_missing_required_field_lines_match_contract_line`
  passed: `2 passed in 0.09s`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_project_director_agent.py`
  passed: `406 passed in 4.36s`.
- Temporary-artifact no-WING quick passed with `status: passed`,
  artifact path:
  `/tmp/automixer_nowing_operator_sequence_value_gate_self_surface/20260617T084538Z_no_wing_release_check`,
  `wing_readonly_operator_sequence_list_markdown_surface_values_match_contract: true`,
  `wing_readonly_operator_sequence_list_markdown_surface_value_mismatches: []`,
  `wing_readonly_operator_sequence_list_markdown_surface_required_field_lines_match_contract: true`,
  `wing_readonly_operator_sequence_list_markdown_surface_missing_field_lines: []`,
  `wing_readonly_operator_sequence_list_markdown_surface_field_lines_rendered: true`,
  `wing_readonly_operator_sequence_list_markdown_surface_live_authority_granted: false`,
  `wing_readonly_operator_sequence_list_markdown_surface_readiness_closure_allowed: false`,
  `live_authority_granted: false`, and
  `readiness_closure_allowed_by_packet: false`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/verify_no_wing_artifact_manifest.py /tmp/automixer_nowing_operator_sequence_value_gate_self_surface/20260617T084538Z_no_wing_release_check`
  passed with `status: passed`, `failed_checks: []`,
  `read_only: true`, `contacts_wing: false`,
  `live_authority_granted: false`, and
  `readiness_closure_allowed: false`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --print`
  passed with `guard_status: drifting`, `guard_score: 30`,
  `selected_blocker: WING read-only connection proof`,
  `selected_evidence_file_count: 205`,
  `current_first_command_baseline_coverage_marker_count: 288`, and
  `current_first_command_baseline_missing_markers: []`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
  passed: `3758 passed, 1 skipped, 1 warning in 47.57s`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`
  passed.
- `/opt/homebrew/bin/git diff --check`
  passed.
