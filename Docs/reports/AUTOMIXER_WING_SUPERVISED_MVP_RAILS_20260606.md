# Automixer WING Supervised MVP Rails - 2026-06-06

## Summary

This slice creates the durable MVP rails for the current project direction:
supervised WING Rack proof before broader autonomy.

Added rails:

- canonical MVP contract: `Docs/MVP_WING_SUPERVISED.md`;
- live blocker table: `.ai/project-layer/wing_mvp_blockers.md`;
- correction methods roadmap: `Docs/CORRECTION_METHODS_ROADMAP.md`;
- correction methods backlog: `.ai/project-layer/correction_methods_backlog.md`;
- read-only preflight runbook;
- supervised fader/gain proof runbook;
- latest MVP progress report;
- Project Director Agent;
- progress updater.

## Project Director Agent

`tools/codex_autoloop/project_director_agent.py` is a repo-local planning
agent. It reads the MVP blocker table and direction guard, then emits the next
Codex directive. It selects the first non-closed MVP blocker as the allowed
task direction.

It does not:

- edit production code by itself;
- connect to WING;
- run live OSC/WebSocket actions;
- grant live readiness.

## Activation

Every Codex autoloop cycle now has this order:

1. update `Docs/reports/MVP_PROGRESS_LATEST.md`;
2. run `codex_direction_guard.py`;
3. run `project_director_agent.py`;
4. choose work from the director's selected blocker;
5. repeat progress/guard/director before the final report.

The existing 1200-second LaunchAgent cadence is unchanged when installed.
If the director detects drift, development still continues in Goal mode; only
broad side work is stopped and the next task is redirected to the first
non-closed MVP blocker.

## Current First Blocker

WING read-only connection proof.

## Safety Boundary

No live WING/OSC action was run for this change. Documentation and planning
automation do not close hardware proof blockers.

## Validation

- `python3 -m py_compile tools/codex_autoloop/codex_direction_guard.py tools/codex_autoloop/update_mvp_progress.py tools/codex_autoloop/project_director_agent.py` - PASS
- `python3 tools/codex_autoloop/update_mvp_progress.py` - PASS
- `python3 tools/codex_autoloop/project_director_agent.py --format markdown --output .paperclip/reports/project_director_latest.md` - PASS
- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py tests/test_wing_readonly_preflight_review_tool.py` - `14 passed`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_mvp_rails.py tests/test_project_director_agent.py tests/test_codex_direction_guard.py tests/test_codex_cycle_prompt.py tests/test_codex_autoloop_contract.py` - `29 passed`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_mvp_rails.py tests/test_project_director_agent.py tests/test_codex_direction_guard.py tests/test_codex_cycle_prompt.py tests/test_codex_autoloop_contract.py` - `30 passed` after adding the correction-method roadmap/backlog.
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_evidence.py tests/test_wing_mvp_rails.py tests/test_no_wing_release_check.py tests/test_write_surface_inventory_static_scan.py::test_write_surface_inventory_covers_files_with_direct_write_tokens` - `153 passed`
- `python3 tools/memory/memory_lint.py` - PASS
- `/opt/homebrew/bin/git diff --check` - PASS
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q` - `2888 passed, 1 skipped, 1 warning`

Current Project Director result:

- guard status: `mixed_needs_review`
- stop broad work: `true`
- continue Goal mode: `true`
- selected blocker: `WING read-only connection proof`
- selected action: run `Docs/runbooks/WING_READONLY_PREFLIGHT.md` against the
  intended backend/WING session.
- correction methods: continue only when they support the selected blocker or
  stay offline/proposal-only until the supervised WING MVP is proven.
