# Automixer WING Project Director Decision Proof Required Files

Date: 2026-06-12

## Summary

Project Director now lists the read-only preflight decision ledger, decision
review tool, and supervised WING proof plan as required files. Future autonomous
cycles are directed to inspect the files that enforce accepted review source ids,
accepted-candidate prestart trace, and the proof-plan blocker before any
supervised fader/gain proof planning.

## Safety Boundary

- No WING hardware contact.
- No backend WING WebSocket session opened.
- No OSC/WING writes.
- No listener was stopped or reconfigured.
- No live authority granted.
- No MVP blocker was closed from documentation or offline evidence.

## Change

- Added `backend/wing_readonly_preflight_review_decision.py` to Project Director
  `required_files`.
- Added `tools/review_wing_readonly_preflight_decisions.py` to Project Director
  `required_files`.
- Added `backend/supervised_wing_proof_plan.py` to Project Director
  `required_files`.
- Added unit coverage so the required files remain present.

## Validation

- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py`
  - `10 passed`
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --format json --print`
  - required files include decision ledger, decision review tool, and supervised proof plan
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
  - `status: on_target`
  - `stop_the_line: false`
- `/opt/homebrew/bin/git diff --check`
  - PASS
- `python3 tools/memory/memory_lint.py`
  - PASS
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
  - `3200 passed, 1 skipped, 1 warning`
