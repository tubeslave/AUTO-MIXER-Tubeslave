# Automixer WING MVP Operator Sequence Step Indexes Next Action

Date: 2026-06-12

## Summary

The first open WING MVP blocker now explicitly requires the no-WING handoff
summary to show:

- `operator_sequence_mvp_blocker_anchor_digest_review_step_index: 8`
- `operator_sequence_supervised_prestart_gate_step_index: 9`

## Why

The handoff already requires a blocker-anchor digest review step and a final
supervised prestart gate. Naming both step indexes makes the operator-facing
review contract unambiguous before any supervised fader/gain proof.

## Changed

- `.ai/project-layer/wing_mvp_blockers.md` names both operator sequence step
  index fields in the first blocker next action and current command.
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md` and `Docs/MVP_WING_SUPERVISED.md`
  require the same fields before supervised proof planning.
- `tools/codex_autoloop/codex_cycle_prompt.md` and
  `tools/codex_autoloop/README.md` mirror the gate requirements for automated
  cycles.
- `tests/test_project_director_agent.py` and `tests/test_wing_mvp_rails.py`
  assert the fields on selected-action, runbook, MVP, progress, prompt, README,
  and docs-index surfaces.

## Boundary

This report does not grant live authority. It does not contact WING, does not
start a backend/WING session, and does not close the read-only connection proof
blocker.

## Validation

- `PYTHONPATH=backend python3 tools/codex_autoloop/update_mvp_progress.py --blockers .ai/project-layer/wing_mvp_blockers.md --output Docs/reports/MVP_PROGRESS_LATEST.md`
  passed.
- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py tests/test_codex_direction_guard.py`
  passed: `23 passed in 0.15s`.
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --format json --print`
  reported `selected_blocker: WING read-only connection proof`,
  `guard_status: on_target`, and `guard_stop_the_line: false`.
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
  reported `status: on_target` and `stop_the_line: false`.
- `/opt/homebrew/bin/git diff --check` passed.
- `python3 tools/memory/memory_lint.py` passed.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q` passed:
  `3205 passed, 1 skipped, 1 warning in 92.85s`.
