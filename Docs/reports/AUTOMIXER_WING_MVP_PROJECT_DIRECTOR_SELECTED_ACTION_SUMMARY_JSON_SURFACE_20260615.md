# Automixer WING MVP Project Director Selected-Action Summary JSON Surface

Date: 2026-06-15

## Summary

The no-WING `summary.json` now exposes a top-level Project Director
selected-action surface. It includes the selected action source, SHA/match
verdicts, current-first-command baseline coverage, manifest-sidecar required
markers, manifest-sidecar missing markers, and sidecar denial marker verdicts.

## Safety

This is an offline no-WING reporting change. It does not contact WING, query a
console, mutate mixer state, close readiness, approve live writes, or grant live
authority.

## Validation

- `PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_embeds_wing_preflight_handoff_packet_artifact`
  passed: `1 passed in 0.11s`.
- `PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend`
  passed and wrote
  `/tmp/automixer_nowing_selected_action_surface/20260617T080832Z_no_wing_release_check/summary.json`.
  The summary exposes `project_director_selected_action_summary_surface_required:
  true`, `project_director_selected_action_source: Current First Command`,
  `project_director_selected_action_sha256_matches_current_directive: true`,
  `project_director_selected_action_manifest_sidecar_checklist_present: true`,
  `project_director_selected_action_manifest_sidecar_missing_markers: []`,
  `project_director_selected_action_manifest_sidecar_denial_present: true`,
  `project_director_selected_action_manifest_sidecar_denial_missing_markers: []`,
  `project_director_selected_action_value_gate_self_check_readiness_closure_allowed:
  false`, and
  `project_director_selected_action_value_gate_self_check_live_authority_granted:
  false`.
- `PYTHONPATH=backend python tools/verify_no_wing_artifact_manifest.py /tmp/automixer_nowing_selected_action_surface/20260617T080832Z_no_wing_release_check`
  reported `status: passed`, `failed_checks: []`,
  `artifact_manifest_self_consistency_match: true`,
  `artifact_manifest_contract_cross_surface_match: true`,
  `artifact_manifest_self_consistency_summary_surface_match: true`,
  `read_only: true`, `live_write: false`, `contacts_wing: false`,
  `contacts_backend: false`, `mutates_console_state: false`,
  `queries_console: false`, `executes_tests: false`,
  `readiness_closure_allowed: false`, and `live_authority_granted: false`.
- `PYTHONPATH=backend python tools/codex_autoloop/project_director_agent.py --format json --print`
  reported `guard_status: drifting`, `guard_score: 30`,
  `guard_stop_the_line: true`, selected blocker
  `WING read-only connection proof`, selected action source
  `Current First Command`, `selected_evidence_file_count: 205`,
  `current_first_command_baseline_coverage_passed: true`,
  `current_first_command_baseline_coverage_marker_count: 288`, and
  `current_first_command_baseline_missing_markers: []`.
- `PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py`
  passed: `393 passed in 4.25s`.
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
  passed: `3758 passed, 1 skipped, 1 warning in 47.58s`.
- `PYTHONPATH=backend python tools/memory/memory_lint.py` passed.
- `/opt/homebrew/bin/git diff --check` passed.
