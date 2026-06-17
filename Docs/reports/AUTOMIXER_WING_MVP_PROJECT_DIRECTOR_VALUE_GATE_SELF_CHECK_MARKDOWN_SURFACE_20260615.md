# Automixer WING MVP Project Director Value Gate Self-Check Markdown Surface

Date: 2026-06-15

## Summary

no-WING validation now checks that the operator-readable `summary.md` renders
Project Director selected-action value-gate self-check fields. If any required
line is missing, no-WING appends
`project_director_selected_action_value_gate_self_check_markdown_surface` to
failed checks and sets the summary status to `failed`.

This is an offline operator-handoff guard. It does not contact WING, mutate
mixer state, close readiness, approve live writes, or grant live authority.

## Safety

The new Markdown-surface verdict keeps readiness closure and live authority
explicitly denied. It only proves operator visibility for already offline
Project Director selected-action self-check fields.

## Validation

- PASS: `PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_embeds_wing_preflight_handoff_packet_artifact tests/test_no_wing_release_check.py::test_no_wing_release_check_fails_summary_when_project_director_value_gate_markdown_surface_missing tests/test_project_director_agent.py`
  returned `6 passed in 0.23s`.
- PASS: `PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend`
  generated
  `/tmp/automixer_nowing_value_gate_markdown/20260617T081955Z_no_wing_release_check/summary.json`;
  the summary reported Project Director source `Current First Command`,
  Markdown-surface missing field lines `[]`, field lines rendered `true`,
  readiness closure allowed `false`, live authority granted `false`, and full
  autonomous ready `false`.
- PASS: `PYTHONPATH=backend python tools/codex_autoloop/project_director_agent.py --format json --print`
  returned `guard_status: drifting`, `guard_score: 30`,
  `guard_stop_the_line: true`, selected `WING read-only connection proof`,
  selected action source `Current First Command`,
  selected evidence file count `205`, baseline coverage passed,
  marker count `288`, and no missing markers.
- PASS: `PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py tests/test_supervised_wing_proof_plan.py`
  returned `415 passed in 4.40s`.
- PASS: `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
  returned `3758 passed, 1 skipped, 1 warning in 47.40s`.
- PASS: `PYTHONPATH=backend python tools/memory/memory_lint.py`.
- PASS: `/opt/homebrew/bin/git diff --check`.
