# WING MVP Operator Sequence List Markdown Surface

## Summary

The no-WING summary now fail-closes when a WING read-only handoff artifact is
present but the operator-readable `summary.md` omits the promoted operator
sequence list review fields.

The conditional check records:

- `wing_readonly_operator_sequence_list_markdown_surface_required`
- `wing_readonly_operator_sequence_list_markdown_surface_required_field_lines`
- `wing_readonly_operator_sequence_list_markdown_surface_missing_field_lines`
- `wing_readonly_operator_sequence_list_markdown_surface_field_lines_rendered`
- `wing_readonly_operator_sequence_list_markdown_surface_readiness_closure_allowed`
- `wing_readonly_operator_sequence_list_markdown_surface_live_authority_granted`

## Safety Boundary

This remains offline and read-only. It does not contact WING, start a backend,
execute handoff commands, grant live authority, or close the WING read-only
connection proof blocker.

## Validation

- `PYTHONPATH=backend python -m py_compile tools/no_wing_release_check.py tools/codex_autoloop/update_mvp_progress.py tests/test_no_wing_release_check.py`
  passed.
- `PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_project_director_agent.py`
  passed: `357 passed, 8 skipped in 4.95s`.
- `PYTHONPATH=backend python tools/codex_autoloop/update_mvp_progress.py`
  regenerated the latest MVP progress report.
- `PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend`
  passed. The top-level summary showed
  `wing_readonly_operator_sequence_list_markdown_surface_required: true`,
  `wing_readonly_operator_sequence_list_markdown_surface_missing_field_lines: []`,
  `wing_readonly_operator_sequence_list_markdown_surface_field_lines_rendered: true`,
  `wing_readonly_operator_sequence_list_markdown_surface_readiness_closure_allowed: false`,
  `wing_readonly_operator_sequence_list_markdown_surface_live_authority_granted: false`,
  `live_authority_granted: false`,
  `readiness_closure_allowed_by_packet: false`, and
  `next_manual_artifact_required: real wing_readonly_preflight.v1 summary.json`.
- `PYTHONPATH=backend python tools/codex_autoloop/project_director_agent.py --format json --print`
  passed with `guard_status: on_target`, `selected_blocker: WING read-only connection proof`,
  `current_first_command_baseline_coverage_marker_count: 246`, and no missing baseline markers.
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
  passed: `3449 passed, 9 skipped in 51.90s`.
- `PYTHONPATH=backend python tools/memory/memory_lint.py`
  passed.
- `/opt/homebrew/bin/git diff --check`
  passed.
