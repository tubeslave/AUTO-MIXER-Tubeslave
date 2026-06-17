# WING MVP Operator Sequence List Markdown Required Lines Value Gate

## Summary

The no-WING operator-sequence-list Markdown-surface check now compares the
rendered
`wing_readonly_operator_sequence_list_markdown_surface_required_field_lines`
value in `summary.md` against the runtime required-line contract.

The summary records expected, actual, and match verdict fields. A stale or empty
required-line value now fails closed even when every required line name is still
present.

## Safety Boundary

This remains offline and read-only. It does not contact WING, start a backend,
execute handoff commands, grant live authority, or close the WING read-only
connection proof blocker.

## Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile tools/no_wing_release_check.py tools/codex_autoloop/update_mvp_progress.py tests/test_no_wing_release_check.py`
  passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py::test_wing_readonly_operator_sequence_list_markdown_surface_check_detects_required_field_lines_value_drift`
  passed: `1 passed in 0.09s`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_project_director_agent.py`
  passed: `406 passed in 4.46s`.
- Temporary-artifact no-WING quick passed at
  `/tmp/automixer_nowing_operator_sequence_required_lines_value_gate/20260617T084048Z_no_wing_release_check/summary.json`
  with `status: passed`,
  `wing_readonly_operator_sequence_list_markdown_surface_required_field_lines_match_contract: true`,
  expected and actual required-field-lines values matching,
  `wing_readonly_operator_sequence_list_markdown_surface_missing_field_lines: []`,
  `wing_readonly_operator_sequence_list_markdown_surface_field_lines_rendered: true`,
  `wing_readonly_operator_sequence_list_markdown_surface_readiness_closure_allowed: false`,
  `wing_readonly_operator_sequence_list_markdown_surface_live_authority_granted: false`,
  `live_authority_granted: false`, and
  `readiness_closure_allowed_by_packet: false`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/verify_no_wing_artifact_manifest.py /tmp/automixer_nowing_operator_sequence_required_lines_value_gate/20260617T084048Z_no_wing_release_check`
  returned `status: passed`, `failed_checks: []`,
  `artifact_manifest_self_consistency_match: true`,
  `artifact_manifest_contract_cross_surface_match: true`,
  `readiness_closure_allowed: false`, and `live_authority_granted: false`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --print`
  passed with `guard_status: drifting`, `guard_score: 30`,
  `selected_blocker: WING read-only connection proof`,
  `selected_evidence_file_count: 205`,
  `current_first_command_baseline_coverage_marker_count: 288`,
  `current_first_command_baseline_missing_markers: []`, and the selected action
  requiring `wing_readonly_operator_sequence_list_markdown_surface_required_field_lines_match_contract: true`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
  passed: `3758 passed, 1 skipped, 1 warning in 48.09s`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`
  passed.
- `/opt/homebrew/bin/git diff --check`
  passed.
