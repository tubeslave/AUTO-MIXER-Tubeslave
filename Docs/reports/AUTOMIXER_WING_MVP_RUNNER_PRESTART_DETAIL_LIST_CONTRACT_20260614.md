# Automixer WING MVP Runner Prestart Detail List Contract

Date: 2026-06-14
Status: Validated

## Summary

The real WING read-only preflight runner now rejects a no-WING prestart summary
when `backend_port_free.detail.backend_port_prestart_check_failure_detail_fields`
does not match the shared backend-port failure-detail contract.

## Why

The first WING blocker remains `WING read-only connection proof`. The real
runner must not open the backend/WING WebSocket from a stale no-WING prestart
summary. Required false values alone are not enough if the summary omitted the
current failure-detail field list, including `listener_confirmation_checklist`.

## Changed surfaces

- `tools/wing_readonly_preflight_runner.py` validates the prestart summary's
  backend-port failure-detail list before expected-plan/WebSocket work.
- `tests/test_wing_readonly_preflight_runner.py` adds a stale-list regression and
  makes the synthetic no-WING prestart fixture carry the real field list.
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md` documents the new fail-closed
  condition.

## Safety Boundary

- No WING hardware contact.
- No backend/WING listener start, stop, or reconfiguration.
- No OSC/WING writes.
- No live authority granted.
- No readiness closure from offline evidence.

## Validation

- Focused runner pytest:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_wing_readonly_preflight_runner.py --tb=short -x`: 36 passed in
  0.15s.
- Focused Project Director and WING rail pytest:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_project_director_agent.py tests/test_wing_mvp_rails.py
  --tb=short -x`: 11 passed in 0.14s.
- Project Director JSON selected `WING read-only connection proof`,
  `selected_blocker_status: tooling_ready_waits_real_wing`,
  `selected_action_source: Current First Command`,
  `current_first_command_baseline_coverage_passed: true`,
  `current_first_command_baseline_missing_markers: []`,
  `selected_evidence_file_count: 104`, `guard_status: on_target`,
  `guard_score: 90`, and `guard_stop_the_line: false`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile
  tools/wing_readonly_preflight_runner.py
  tools/codex_autoloop/update_mvp_progress.py`: passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`:
  PASS.
- `/opt/homebrew/bin/git diff --check`: passed.
- Standard full suite:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x
  --tb=short -q`: 3308 passed, 9 skipped, 1 warning in 71.12s.
