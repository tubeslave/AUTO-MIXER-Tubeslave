# Automixer WING Autoloop Backend Prestart Director Docs

Date: 2026-06-12

## Summary

The Codex autoloop README and Project Director tests now carry the structured
backend-port prestart gate. Automation-facing guidance requires the no-WING
handoff summary to show `backend_port_prestart_gate` exactly matching
`expected_backend_port_prestart_gate` before supervised fader/gain proof.
Direction guard drift scoring now ignores Project Director JSON policy fields
such as `forbidden_scope`, so negative policy boundaries do not trigger a false
`mixed_needs_review`.

## Safety Boundary

- No WING hardware contact.
- No backend WING WebSocket session opened.
- No OSC/WING writes.
- No listener was stopped or reconfigured.
- No live authority granted.
- No MVP blocker was closed from documentation or offline evidence.

## Change

- `tools/codex_autoloop/README.md` now documents the backend-port prestart gate
  alongside the final supervised prestart gate.
- `tests/test_project_director_agent.py` now asserts Project Director selected
  action, Codex prompt, and autoloop README all include the backend gate.
- `tools/codex_autoloop/codex_direction_guard.py` now filters JSON policy
  fields before drift scoring.
- `tests/test_codex_direction_guard.py` covers Project Director JSON
  `forbidden_scope` false positives.

## Validation

- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_director_agent.py tests/test_codex_direction_guard.py tests/test_wing_mvp_rails.py`
  - `21 passed`
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --format json --print`
  - selected blocker: `WING read-only connection proof`
  - selected action includes `backend_port_prestart_gate`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
  - `status: on_target`
  - `stop_the_line: false`
- `/opt/homebrew/bin/git diff --check`
  - PASS
- `python3 tools/memory/memory_lint.py`
  - PASS
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
  - `3200 passed, 1 skipped, 1 warning`
