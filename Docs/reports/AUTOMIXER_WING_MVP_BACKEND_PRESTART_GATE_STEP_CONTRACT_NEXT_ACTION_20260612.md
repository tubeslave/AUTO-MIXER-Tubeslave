# Automixer WING MVP Backend Prestart Gate Step Contract Next Action

Date: 2026-06-12

## Summary

The first open WING MVP blocker now explicitly requires the no-WING handoff
summary to show these structured `backend_port_prestart_gate` fields:

- `backend_port_prestart_gate.operator_sequence_step_index: 0`
- `backend_port_prestart_gate.operator_sequence_step_must_be_first: true`

## Why

The validator already expects `backend_port_prestart_gate` to match
`expected_backend_port_prestart_gate`. Naming the gate's first-step fields in the
selected action makes the operator-facing review contract explicit before any
backend/WING session or supervised fader/gain proof.

## Changed

- `.ai/project-layer/wing_mvp_blockers.md` names the structured gate first-step
  fields in the first blocker next action and current command.
- `Docs/MVP_WING_SUPERVISED.md` stops supervised proof planning when the
  structured gate lacks those first-step fields.
- `tools/codex_autoloop/codex_cycle_prompt.md` and
  `tools/codex_autoloop/README.md` require the same fields.
- `tests/test_project_director_agent.py` and `tests/test_wing_mvp_rails.py`
  assert the fields on selected-action and autoloop surfaces.

## Boundary

This report does not grant live authority. It does not contact WING, does not
start a backend/WING session, and does not close the read-only connection proof
blocker.

## Validation

- `PYTHONPATH=backend python3 tools/codex_autoloop/update_mvp_progress.py --blockers .ai/project-layer/wing_mvp_blockers.md --output Docs/reports/MVP_PROGRESS_LATEST.md`
  passed.
- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py tests/test_codex_direction_guard.py`
  passed: `23 passed in 0.14s`.
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --format json --print`
  reported `selected_blocker: WING read-only connection proof`,
  `guard_status: on_target`, and `guard_stop_the_line: false`.
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
  reported `status: on_target` and `stop_the_line: false`.
- `/opt/homebrew/bin/git diff --check` passed.
- `python3 tools/memory/memory_lint.py` passed.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q` passed:
  `3205 passed, 1 skipped, 1 warning in 90.92s`.
