# Automixer WING Autoloop Current Handoff Gates Docs

Date: 2026-06-12

## Summary

Codex autoloop prompt and README now document the current WING handoff gate list
from the first open MVP blocker, including:

- `mvp_blocker_anchor.packet_does_not_close_blocker: true`
- final `operator_sequence_supervised_prestart_gate_step` marker strings:
  `accepted_decision_ready`, `valid accepted-candidate prestart trace`, and
  `WING_SUPERVISED_FADER_GAIN_PROOF`

## Why

Automated Codex cycles read the autoloop prompt before task selection. This
keeps the top-level cycle contract aligned with the WING MVP blocker next action
and no-WING handoff summary before any supervised proof work.

## Changed

- Updated `tools/codex_autoloop/codex_cycle_prompt.md`.
- Updated `tools/codex_autoloop/README.md`.
- Extended Project Director tests so prompt and README retain the current WING
  handoff gate fields.

## Boundary

This report does not grant live authority. It does not contact WING, does not
start a backend/WING session, and does not close the read-only connection proof
blocker.

## Validation

- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py tests/test_codex_direction_guard.py`
  - `23 passed`
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --format json --print`
  - `selected_blocker: WING read-only connection proof`
  - `guard_status: on_target`
  - `guard_stop_the_line: false`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
  - `status: on_target`
  - `stop_the_line: false`
- `/opt/homebrew/bin/git diff --check`
  - PASS
- `python3 tools/memory/memory_lint.py`
  - PASS
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
  - `3205 passed, 1 skipped, 1 warning`
