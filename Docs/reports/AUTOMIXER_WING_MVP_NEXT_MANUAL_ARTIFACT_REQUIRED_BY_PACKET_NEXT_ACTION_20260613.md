# Automixer WING MVP Next Manual Artifact Required By Packet Next Action

Date: 2026-06-13
Status: Validated

## Summary

The WING MVP rails now require the no-WING handoff summary to show:

- `next_manual_artifact_required: real wing_readonly_preflight.v1 summary.json`

before any supervised fader/gain proof planning.

## Why

The handoff packet is offline operator guidance. The first blocker remains open
until the project has a real WING read-only preflight summary artifact; the
packet cannot replace that evidence.

## Changed surfaces

- `tools/build_wing_readonly_preflight_handoff_packet.py` renders the explicit
  `next_manual_artifact_required` field in Markdown.
- `tools/no_wing_release_check.py` includes the field in release-check Markdown
  summaries.
- `.ai/project-layer/wing_mvp_blockers.md` requires the field in the first
  blocker next action and current command.
- `Docs/MVP_WING_SUPERVISED.md` lists missing
  `next_manual_artifact_required: real wing_readonly_preflight.v1 summary.json`
  as a stop-the-line condition.
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md` requires the field in handoff
  validation.
- `tools/codex_autoloop/README.md` and
  `tools/codex_autoloop/codex_cycle_prompt.md` require future cycles to verify
  the field.
- Regression tests cover packet rendering, no-WING release summary, Project
  Director output, and WING MVP rail surfaces.

## Validation

- Regenerated `Docs/reports/MVP_PROGRESS_LATEST.md` with
  `PYTHONPATH=backend /opt/homebrew/bin/python3
  tools/codex_autoloop/update_mvp_progress.py --blockers
  .ai/project-layer/wing_mvp_blockers.md --output
  Docs/reports/MVP_PROGRESS_LATEST.md`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_wing_readonly_preflight_handoff_packet.py
  tests/test_no_wing_release_check.py tests/test_project_director_agent.py
  tests/test_wing_mvp_rails.py tests/test_codex_direction_guard.py`: 237 passed
  in 3.57s.
- Project Director JSON check selected `WING read-only connection proof`, found
  `next_manual_artifact_required: real wing_readonly_preflight.v1 summary.json`
  in the selected action, and reported `guard_status: on_target` with
  `guard_stop_the_line: false`.
- Direction guard JSON check reported `status: on_target` and
  `stop_the_line: false`.
- `/opt/homebrew/bin/git diff --check`: passed.
- `python3 tools/memory/memory_lint.py`: PASS.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x
  --tb=short -q`: 3205 passed, 1 skipped, 1 warning in 220.78s (0:03:40).
