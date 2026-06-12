# Automixer WING MVP Rows Structured Gate Report

Date: 2026-06-12

## Scope

This change updates WING MVP status/progress wording and tests. It does not
contact hardware, open a backend WING WebSocket session, send OSC/WING requests,
or widen live write authority.

## Change

`.ai/project-layer/wing_mvp_blockers.md` now keeps supervised fader and
gain-down/trim proof blocked until:

- clean read-only preflight review exists
- accepted decision JSON exists
- the final handoff operator-sequence step has valid accepted-candidate
  prestart trace
- `supervised_prestart_gate` exactly matches
  `expected_supervised_prestart_gate`

`Docs/reports/MVP_PROGRESS_LATEST.md` is regenerated from that blocker table.

## Verification

Passed validation:

- `PYTHONPATH=backend python3 tools/codex_autoloop/update_mvp_progress.py --blockers .ai/project-layer/wing_mvp_blockers.md --output Docs/reports/MVP_PROGRESS_LATEST.md`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_mvp_rails.py -k "blocker_table or progress_tool"`
  - `2 passed, 5 deselected`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_mvp_rails.py tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_project_director_agent.py`
  - `216 passed`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
  - `3191 passed, 1 skipped, 1 warning`
