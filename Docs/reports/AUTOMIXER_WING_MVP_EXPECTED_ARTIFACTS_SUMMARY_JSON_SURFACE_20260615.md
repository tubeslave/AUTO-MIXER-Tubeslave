# Automixer WING MVP Expected Artifacts Summary JSON Surface

Date: 2026-06-15

## Summary

The no-WING `summary.json` now exposes the WING read-only handoff expected
artifact contract at the top level. Machine consumers can verify
`expected_artifacts`, `expected_artifact_globs`, `plan_expected_artifacts`,
`expected_artifacts_match_plan`, and summary/event glob match verdicts without
parsing the nested handoff packet artifact.

## Safety

This is an offline no-WING reporting change. It does not contact WING, query a
console, mutate mixer state, close readiness, approve live writes, or grant
live authority.

## Validation

- PASS: `PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_embeds_wing_preflight_handoff_packet_artifact`
  returned `1 passed in 1.25s`.
- PASS: `PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend`
  generated
  `artifacts/no_wing_release_check/20260615T015342Z_no_wing_release_check/summary.json`
  with `status: passed`, top-level
  `wing_readonly_expected_artifacts_summary_surface_required: true`,
  expected artifacts matching the nested handoff artifact,
  `expected_artifacts_match_plan: true`, `summary_glob_matches_plan: true`,
  `event_log_glob_matches_plan: true`, and readiness/live-authority denial
  flags set to `false`.
- PASS: `PYTHONPATH=backend python tools/codex_autoloop/project_director_agent.py --format json --print`
  kept the selected blocker at `WING read-only connection proof`, reported
  `selected_evidence_file_count: 197`, and kept expected-artifact markers in
  the selected action.
- PASS: `PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py`
  returned `321 passed, 8 skipped in 3.59s`.
- PASS: `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
  returned `3396 passed, 9 skipped in 51.41s`.
- PASS: `PYTHONPATH=backend python tools/memory/memory_lint.py`.
- PASS: `/opt/homebrew/bin/git diff --check`.
