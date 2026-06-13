# Automixer WING MVP Backend Prestart Listener Probe Error Field

Date: 2026-06-13
Status: Validated

## Summary

The backend-port prestart failure-detail contract now requires
`listener_process_error` alongside `listener_processes`. If the no-WING gate
detects a busy backend port but cannot identify the listener owner through
`lsof` or `ps`, that owner-probe failure remains visible to the operator before
any real WING read-only preflight.

## Why

The first WING MVP blocker must not advance from ambiguous backend-port state.
The gate already failed closed when owner probing failed, but the exact-list
handoff contract did not require the diagnostic field. Requiring it makes the
busy-port evidence complete without granting permission to stop or reconfigure
anything automatically.

## Changed surfaces

- `backend/wing_readonly_preflight_contract.py` adds
  `listener_process_error` to
  `WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS`.
- `.ai/project-layer/wing_mvp_blockers.md`, `Docs/MVP_WING_SUPERVISED.md`,
  `Docs/runbooks/WING_READONLY_PREFLIGHT.md`, `tools/codex_autoloop/README.md`,
  and `tools/codex_autoloop/codex_cycle_prompt.md` now require the field in the
  backend-port prestart exact list.
- `tests/test_no_wing_release_check.py` covers the shared constant and the
  busy-port owner-probe failure detail.
- `tests/test_project_director_agent.py` and `tests/test_wing_mvp_rails.py`
  assert the new field remains visible in Project Director and MVP rail text.

## Safety Boundary

- No WING hardware contact.
- No backend/WING listener start, stop, or reconfiguration.
- No OSC/WING writes.
- No live authority granted.
- No MVP blocker closed from documentation or offline evidence.

## Validation

- Regenerated `Docs/reports/MVP_PROGRESS_LATEST.md` with
  `PYTHONPATH=backend /opt/homebrew/bin/python3
  tools/codex_autoloop/update_mvp_progress.py --blockers
  .ai/project-layer/wing_mvp_blockers.md --output
  Docs/reports/MVP_PROGRESS_LATEST.md`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_no_wing_release_check.py
  tests/test_wing_readonly_preflight_handoff_packet.py
  tests/test_wing_mvp_rails.py tests/test_project_director_agent.py
  tests/test_codex_direction_guard.py`: 238 passed in 3.65s.
- Project Director JSON check selected `WING read-only connection proof`,
  reported `guard_status: on_target`, `guard_stop_the_line: false`, and
  exposed the `listener_process_error` exact-list requirement in
  `selected_action`.
- Direction guard JSON check reported `status: on_target` and
  `stop_the_line: false`.
- `/opt/homebrew/bin/git diff --check`: passed.
- `python3 tools/memory/memory_lint.py`: PASS.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x
  --tb=short -q`: 3208 passed, 1 skipped, 1 warning in 332.75s (0:05:32).
