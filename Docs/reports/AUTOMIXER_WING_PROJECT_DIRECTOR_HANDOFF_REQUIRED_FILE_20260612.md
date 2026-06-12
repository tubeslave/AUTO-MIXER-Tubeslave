# Automixer WING Project Director Handoff Required File

Date: 2026-06-12

## Summary

Project Director now lists
`tools/build_wing_readonly_preflight_handoff_packet.py` as a required file for
the supervised WING MVP path. Future autonomous cycles are directed to inspect
the tool that builds the structured `backend_port_prestart_gate` and
`supervised_prestart_gate` handoff contract before proof planning.

## Safety Boundary

- No WING hardware contact.
- No backend WING WebSocket session opened.
- No OSC/WING writes.
- No listener was stopped or reconfigured.
- No live authority granted.
- No MVP blocker was closed from documentation or offline evidence.

## Change

- Added the handoff packet builder to Project Director `required_files`.
- Added unit coverage so the required file remains present.

## Validation

- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py`
  - `10 passed`
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --format json --print`
  - required files include `tools/build_wing_readonly_preflight_handoff_packet.py`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
  - `status: on_target`
  - `stop_the_line: false`
- `/opt/homebrew/bin/git diff --check`
  - PASS
- `python3 tools/memory/memory_lint.py`
  - PASS
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
  - `3200 passed, 1 skipped, 1 warning`
