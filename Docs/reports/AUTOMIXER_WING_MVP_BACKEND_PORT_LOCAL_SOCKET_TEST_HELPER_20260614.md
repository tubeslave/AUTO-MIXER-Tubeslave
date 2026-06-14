# Automixer WING MVP Backend Port Local Socket Test Helper

Date: 2026-06-14
Status: Validated

## Summary

No-WING backend-port listener tests now use a shared local socket helper that
skips only when the environment disallows binding a local ephemeral test port.

## Why

The backend-port prestart gate must fail closed when an unexpected listener is
present before an intended backend/WING session. Tests should keep exercising
that busy-port path, but an environment-level `PermissionError` while binding a
local test socket should not be reported as a no-WING gate regression.

## Changed surfaces

- `tests/test_no_wing_release_check.py` adds
  `bind_local_test_server_or_skip()`.
- Backend-port listener tests use that helper instead of direct
  `socket.bind(("127.0.0.1", 0))`.
- `.ai/project-layer/wing_mvp_blockers.md` and `Docs/INDEX.md` include this
  report in the current WING blocker evidence trail.

## Safety Boundary

- No production no-WING checker logic changed.
- No WING hardware contact.
- No backend/WING listener start, stop, or reconfiguration.
- No OSC/WING writes.
- No live authority granted.
- No MVP blocker closed from documentation or offline evidence.

## Validation

- Backend-port no-WING subset:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_no_wing_release_check.py::test_backend_port_check_records_listener_owner_when_busy
  tests/test_no_wing_release_check.py::test_backend_port_check_accepts_operator_confirmed_intended_listener
  tests/test_no_wing_release_check.py::test_backend_listener_confirmation_template_stays_invalid_until_edited
  tests/test_no_wing_release_check.py::test_backend_listener_confirmation_rejects_partially_edited_template
  tests/test_no_wing_release_check.py::test_backend_listener_confirmation_accepts_fully_edited_template
  tests/test_no_wing_release_check.py::test_backend_port_check_rejects_mismatched_listener_confirmation
  tests/test_no_wing_release_check.py::test_backend_port_check_records_listener_owner_probe_error
  --tb=short`: 7 passed in 0.08s.
- Focused WING rails and Project Director tests:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_project_director_agent.py tests/test_wing_mvp_rails.py
  --tb=short -x`: 11 passed in 0.12s.
- Project Director JSON check selected `WING read-only connection proof`,
  reported `guard_status: on_target`, `guard_score: 90`,
  `guard_stop_the_line: false`, baseline coverage passed, exposed
  `selected_evidence_file_count: 98`, and found 47 indexed current reports with
  `missing_indexed_current_reports: []`.
- Direction guard JSON check reported `status: on_target`, `score: 90`, and
  `stop_the_line: false`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile
  tools/no_wing_release_check.py tools/codex_autoloop/project_director_agent.py
  tools/codex_autoloop/update_mvp_progress.py`: passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`:
  PASS.
- `/opt/homebrew/bin/git diff --check`: passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3
  tools/no_wing_release_check.py --quick --artifact-root
  /tmp/automixer_backend_port_socket_helper_nowing`: exited 1 fail-closed on
  existing listener `127.0.0.1:8765`, with
  `connection_policy: no_hardware_no_network_no_live_writes`,
  `live_write: false`, and no live authority.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x
  --tb=short -q`: 3310 passed, 1 skipped, 1 warning in 162.66s (0:02:42).
