# Automixer WING MVP Progress Director Evidence Summary

Date: 2026-06-14
Status: Validated

## Summary

`Docs/reports/MVP_PROGRESS_LATEST.md` now exposes the Current First Command
baseline coverage verdict and selected evidence file count that Project Director
uses for the active WING MVP blocker.

## Why

Goal-mode work is constrained to `WING read-only connection proof`. The visible
operator progress report should show whether the selected command still covers
the required no-WING handoff baseline and how much evidence Project Director is
using, instead of requiring a separate Project Director JSON inspection.

## Changed surfaces

- `tools/codex_autoloop/update_mvp_progress.py` now owns evidence-path
  extraction and renders coverage/evidence-count fields.
- `tools/codex_autoloop/project_director_agent.py` reuses the shared evidence
  extractor while preserving its compatibility wrapper.
- `tests/test_wing_mvp_rails.py` covers the new progress report fields.
- `Docs/reports/MVP_PROGRESS_LATEST.md` was regenerated with the new summary
  fields.

## Safety Boundary

- No WING hardware contact.
- No backend/WING listener start, stop, or reconfiguration.
- No OSC/WING writes.
- No live authority granted.
- No MVP blocker closed from documentation or offline evidence.

## Validation

- Focused no-WING/rails/Project Director pytest:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_no_wing_release_check.py tests/test_wing_mvp_rails.py
  tests/test_project_director_agent.py --tb=short -x`: 277 passed,
  8 skipped in 3.19s.
- Project Director JSON selected `WING read-only connection proof`,
  `selected_blocker_status: tooling_ready_waits_real_wing`,
  `selected_action_source: Current First Command`, `selected_evidence_file_count:
  102`, `current_first_command_baseline_coverage_passed: true`,
  `current_first_command_baseline_missing_markers: []`, `guard_status:
  on_target`, `guard_score: 90`, and `guard_stop_the_line: false`.
- Direction guard JSON reported `status: on_target`, `score: 90`, and
  `stop_the_line: false`.
- no-WING quick without frontend:
  `PYTHONPATH=backend /opt/homebrew/bin/python3
  tools/no_wing_release_check.py --quick --skip-frontend --artifact-root
  /tmp/automixer_goal_required_false_detail_nowing`: passed with
  `connection_policy: no_hardware_no_network_no_live_writes`,
  `live_write: false`, and `failed_checks: []`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile
  tools/no_wing_release_check.py
  tools/build_wing_readonly_preflight_handoff_packet.py
  tools/codex_autoloop/project_director_agent.py
  tools/codex_autoloop/update_mvp_progress.py`: passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`:
  PASS.
- `/opt/homebrew/bin/git diff --check`: passed.
- Standard full suite:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x
  --tb=short -q`: 3307 passed, 9 skipped, 1 warning in 71.28s.
