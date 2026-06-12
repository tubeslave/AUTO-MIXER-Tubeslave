# Automixer WING Project Director Selected Evidence Files

Date: 2026-06-12

## Summary

Project Director now exposes the selected WING MVP blocker evidence files as
structured directive data. Future autonomous cycles can inspect the exact
read-only preflight proof files named by the first open blocker instead of
depending only on a hand-maintained required-file list.

## Safety Boundary

- No WING hardware contact.
- No backend WING WebSocket session opened.
- No OSC/WING writes.
- No listener was stopped or reconfigured.
- No live authority granted.
- No MVP blocker was closed from documentation or offline evidence.

## Change

- Added `selected_evidence_files` to Project Director JSON.
- Added `selected_evidence_file_count` to Project Director JSON and Markdown.
- Rendered selected evidence files in Project Director Markdown.
- Derived extra required-file candidates from selected blocker evidence for
  `backend/`, `tools/`, and `Docs/runbooks/` paths only.
- Kept historical `Docs/reports/` entries as evidence, not required source
  files.

## Why It Matters

The supervised WING MVP path depends on the real read-only preflight proof
chain. This change makes the blocker table evidence visible to every future
Codex cycle and reduces the chance that a proof contract or review tool is
missed before supervised fader/gain proof planning.

## Validation

- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py tests/test_codex_direction_guard.py`
  - `23 passed`
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --format json --print`
  - `selected_blocker: WING read-only connection proof`
  - `selected_evidence_file_count: 23`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
  - `status: on_target`
  - `stop_the_line: false`
- `/opt/homebrew/bin/git diff --check`
  - PASS
- `python3 tools/memory/memory_lint.py`
  - PASS
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
  - `3202 passed, 1 skipped, 1 warning`
