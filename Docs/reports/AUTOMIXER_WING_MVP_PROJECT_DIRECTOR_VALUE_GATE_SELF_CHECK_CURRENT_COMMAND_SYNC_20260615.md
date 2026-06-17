# Automixer WING MVP Project Director Value Gate Self-Check Current Command Sync

Date: 2026-06-15

## Summary

Project Director now requires the accepted-candidate expected artifact trace
value-gate summary-surface self-check markers in `## Current First Command`.
Its selected action must include the self-check required flag, required fields,
empty mismatch list, match verdict, and readiness/live-authority denial fields.

This keeps Project Director aligned with the no-WING handoff contract. It does
not close readiness, contact WING, or grant live authority.

## Safety

This is an offline audit/planning change. It does not contact WING, query a
console, mutate mixer state, close readiness, approve live writes, or grant live
authority.

## Validation

- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_project_director_agent.py`
  returned `4 passed in 0.15s`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_nowing_current_command_sync`
  generated
  `/tmp/automixer_nowing_current_command_sync/20260617T082603Z_no_wing_release_check/summary.json`;
  the summary reported Project Director source `Current First Command`, selected
  action SHA `5b1acc23e25f71741ae72d48d77ceeefae3c5e91f9d3e156e1f95b7d07d5cf85`,
  current-first baseline marker count `288`, baseline coverage passed, no
  missing baseline markers, value-gate self-check missing markers `[]`,
  value-gate self-check markers present `true`, and
  `live_authority_granted: false`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/verify_no_wing_artifact_manifest.py /tmp/automixer_nowing_current_command_sync/20260617T082603Z_no_wing_release_check`
  returned `status: passed`, `failed_checks: []`,
  `artifact_manifest_self_consistency_match: true`,
  `artifact_manifest_contract_cross_surface_match: true`,
  `readiness_closure_allowed: false`, and `live_authority_granted: false`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --print`
  returned `guard_status: drifting`, `guard_score: 30`, selected
  `WING read-only connection proof`, reported
  `selected_evidence_file_count: 205`, `selected_action_source: Current First
  Command`, `current_first_command_baseline_coverage_marker_count: 288`, and
  `current_first_command_baseline_missing_markers: []`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py tests/test_supervised_wing_proof_plan.py`
  returned `415 passed in 4.24s`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
  returned `3758 passed, 1 skipped, 1 warning in 47.69s`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`.
- PASS: `/opt/homebrew/bin/git diff --check`.
