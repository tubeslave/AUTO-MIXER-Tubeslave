# Automixer Codex Autoloop Structured WING Gate Report

Date: 2026-06-12

## Scope

This change updates Codex autoloop guidance and tests. It does not contact
hardware, open a backend WING WebSocket session, send OSC/WING requests, or
widen live write authority.

## Change

`tools/codex_autoloop/codex_cycle_prompt.md` and
`tools/codex_autoloop/README.md` now require, before any supervised fader/gain
proof:

- `supervised_prestart_gate` exactly matches
  `expected_supervised_prestart_gate`
- `supervised_prestart_gate.grants_live_authority: false`

## Verification

Passed validation:

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_mvp_rails.py -k codex_prompt`
  - `1 passed, 6 deselected`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_director_agent.py -k prompt_and_readme`
  - `1 passed, 2 deselected`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_mvp_rails.py tests/test_project_director_agent.py tests/test_project_goal_capability_matrix.py tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_handoff_packet.py`
  - `243 passed`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
  - `3191 passed, 1 skipped, 1 warning`
