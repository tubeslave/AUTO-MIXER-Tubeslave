# WING MVP Operator Sequence List Summary Surface Self-Check

## Summary

The no-WING handoff summary now includes a fail-closed self-check for the
promoted top-level operator sequence lists:

- `wing_readonly_operator_sequence_list_summary_surface_self_check_required`
- `wing_readonly_operator_sequence_list_summary_surface_required_fields`
- `wing_readonly_operator_sequence_list_summary_surface_mismatches`
- `wing_readonly_operator_sequence_list_summary_surface_match`
- `wing_readonly_operator_sequence_list_summary_surface_readiness_closure_allowed`
- `wing_readonly_operator_sequence_list_summary_surface_live_authority_granted`

The self-check compares the promoted `operator_sequence` and
`expected_operator_sequence` fields with the nested handoff artifact, the shared
expected checklist contract, and their count fields. A mismatch marks the
no-WING summary failed.

## Safety Boundary

This remains offline and read-only. It does not contact WING, start a backend,
execute handoff commands, grant live authority, or close the WING read-only
connection proof blocker.

## Validation

- `PYTHONPATH=backend python -m py_compile tools/no_wing_release_check.py tools/codex_autoloop/update_mvp_progress.py tests/test_no_wing_release_check.py`
  passed.
- `PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_project_director_agent.py`
  passed: `356 passed, 8 skipped in 4.93s`.
- `PYTHONPATH=backend python tools/codex_autoloop/update_mvp_progress.py`
  regenerated the latest MVP progress report.
- `PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend`
  passed. The top-level summary showed
  `wing_readonly_operator_sequence_list_summary_surface_self_check_required: true`,
  `wing_readonly_operator_sequence_list_summary_surface_mismatches: []`,
  `wing_readonly_operator_sequence_list_summary_surface_match: true`,
  `wing_readonly_operator_sequence_list_summary_surface_readiness_closure_allowed: false`,
  `wing_readonly_operator_sequence_list_summary_surface_live_authority_granted: false`,
  `live_authority_granted: false`,
  `readiness_closure_allowed_by_packet: false`, and
  `next_manual_artifact_required: real wing_readonly_preflight.v1 summary.json`.
- `PYTHONPATH=backend python tools/codex_autoloop/project_director_agent.py --format json --print`
  passed with `guard_status: on_target`, `selected_blocker: WING read-only connection proof`,
  `current_first_command_baseline_coverage_marker_count: 241`, and no missing baseline markers.
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
  passed: `3448 passed, 9 skipped in 51.71s`.
- `PYTHONPATH=backend python tools/memory/memory_lint.py`
  passed.
- `/opt/homebrew/bin/git diff --check`
  passed.
