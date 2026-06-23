# Automixer WING MVP Next Proof Plan Candidate Contracts Summary JSON Surface

Date: 2026-06-15

## Summary

The no-WING `summary.json` now exposes `next_wing_proof_plan`
accepted-candidate prestart and artifact-root contract values at the top
level. Machine consumers can verify candidate source/root lists, expected
violation maps, invalid records, and contract validity without parsing the
nested proof-plan object.

## Safety

This is an offline no-WING reporting change. It does not contact WING, query a
console, mutate mixer state, close readiness, approve live writes, or grant
live authority.

## Validation

- PASS: `PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_embeds_next_wing_proof_plan`
  returned `1 passed in 1.30s`.
- PASS: `PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend`
  generated
  `artifacts/no_wing_release_check/20260615T010359Z_no_wing_release_check/summary.json`
  with `status: passed`,
  `next_wing_proof_plan_candidate_contracts_summary_surface_required: true`,
  and readiness/live-authority summary-surface flags set to `false`.
- PASS: `PYTHONPATH=backend python tools/codex_autoloop/project_director_agent.py --format json --print`
  kept the selected blocker at `WING read-only connection proof`, reported
  `selected_evidence_file_count: 190`, and surfaced the candidate prestart and
  artifact-root contract fields in the selected action.
- PASS: `PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py`
  returned `321 passed, 8 skipped in 3.58s`.
- PASS: `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
  returned `3396 passed, 9 skipped in 50.16s`.
- PASS: `PYTHONPATH=backend python tools/memory/memory_lint.py`.
- PASS: `/opt/homebrew/bin/git diff --check`.
