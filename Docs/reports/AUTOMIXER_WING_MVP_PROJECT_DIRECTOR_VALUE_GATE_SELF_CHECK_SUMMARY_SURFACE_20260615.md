# Automixer WING MVP Project Director Value Gate Self-Check Summary Surface

Date: 2026-06-15

## Summary

no-WING handoff summaries now expose a dedicated Project Director selected-action
value-gate self-check surface. The surface reports required markers, missing
markers, a `markers_present` verdict, and explicit readiness/live-authority
denial fields.

The selected action remains Project Director's audit/planning instruction for
the first open blocker. This change does not contact WING, mutate mixer state,
close readiness, approve live writes, or grant live authority.

## Safety

The new check fails closed if the selected action is missing any value-gate
self-check marker. Offline no-WING artifacts still cannot close the WING
read-only connection proof or enable supervised writes.

## Validation

- PASS: `PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_embeds_wing_preflight_handoff_packet_artifact tests/test_no_wing_release_check.py::test_no_wing_release_check_fails_summary_when_project_director_value_gate_markdown_surface_missing`
  returned `2 passed in 0.10s`.
- PASS: `PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_rejects_handoff_packet_project_director_value_gate_marker_missing tests/test_project_director_agent.py`
  returned `6 passed in 0.24s`.
- PASS: `PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend`
  generated
  `/tmp/automixer_nowing_value_gate_surface/20260617T081436Z_no_wing_release_check/summary.json`;
  the summary reported Project Director source `Current First Command`,
  value-gate self-check missing markers `[]`, markers present `true`,
  value-gate markdown surface missing field lines `[]`, markdown field lines
  rendered `true`, readiness closure allowed `false`, live authority granted
  `false`, and full autonomous ready `false`.
- PASS: `PYTHONPATH=backend python tools/codex_autoloop/project_director_agent.py --format json --print`
  returned `guard_status: drifting`, `guard_score: 30`,
  `guard_stop_the_line: true`, selected `WING read-only connection proof`,
  selected action source `Current First Command`, baseline coverage passed,
  marker count `288`, and no missing markers.
- PASS: `PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py tests/test_supervised_wing_proof_plan.py`
  returned `415 passed in 4.40s`.
- PASS: `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
  returned `3758 passed, 1 skipped, 1 warning in 47.27s`.
- PASS: `PYTHONPATH=backend python tools/memory/memory_lint.py`.
- PASS: `/opt/homebrew/bin/git diff --check`.
