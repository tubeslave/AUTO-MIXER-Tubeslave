# Automixer WING MVP Blocker Anchor Summary JSON Surface

Date: 2026-06-15

## Summary

The no-WING `summary.json` now exposes the WING MVP blocker anchor at the top
level. Machine consumers can verify `mvp_blocker_anchor`,
`mvp_blocker_anchor_sha256`, the current-first-open match verdicts, and
`mvp_blocker_anchor.packet_does_not_close_blocker` without parsing the nested
handoff packet artifact.

## Safety

This is an offline no-WING reporting change. It does not contact WING, query a
console, mutate mixer state, close readiness, approve live writes, or grant
live authority.

## Validation

- PASS: `PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_embeds_wing_preflight_handoff_packet_artifact`
  returned `1 passed in 1.24s`.
- PASS: `PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend`
  generated
  `artifacts/no_wing_release_check/20260615T014006Z_no_wing_release_check/summary.json`
  with `status: passed`, top-level
  `wing_readonly_mvp_blocker_anchor_summary_surface_required: true`,
  `mvp_blocker_anchor_matches_current_first_open: true`,
  `mvp_blocker_anchor_sha256_matches_current_first_open: true`,
  `mvp_blocker_anchor.packet_does_not_close_blocker: true`, and nested/top-level
  anchor fields matching.
- PASS: `PYTHONPATH=backend python tools/codex_autoloop/project_director_agent.py --format json --print`
  kept the selected blocker at `WING read-only connection proof`, reported
  `selected_evidence_file_count: 195`, and kept the blocker-anchor markers in
  the selected action.
- PASS: `PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py`
  returned `321 passed, 8 skipped in 3.59s`.
- PASS: `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
  returned `3396 passed, 9 skipped in 50.46s`.
- PASS: `PYTHONPATH=backend python tools/memory/memory_lint.py`.
- PASS: `/opt/homebrew/bin/git diff --check`.
