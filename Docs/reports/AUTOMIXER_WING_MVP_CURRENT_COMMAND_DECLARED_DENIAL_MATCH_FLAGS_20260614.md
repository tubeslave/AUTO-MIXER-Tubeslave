# Automixer WING MVP Current Command Declared Denial Match Flags

Date: 2026-06-14
Status: Validated

## Summary

Project Director current-command baseline coverage now requires the no-WING
handoff packet's declared denial `*_matches_plan` self-check markers alongside
the recomputed denial match verdicts.

## Changed Surfaces

- `tools/codex_autoloop/update_mvp_progress.py` adds five
  `declared_<field>_matches_plan: true` markers to current-command baseline
  coverage.
- `tools/codex_autoloop/codex_cycle_prompt.md`, `tools/codex_autoloop/README.md`,
  and `.ai/project-layer/wing_mvp_blockers.md` require the same markers in the
  operator-visible WING read-only current command.
- `tests/test_project_director_agent.py` and `tests/test_wing_mvp_rails.py`
  assert the declared markers across Project Director, prompt, README, and
  progress rails.

## Safety Boundary

- No WING hardware contact.
- No OSC/WING writes.
- No backend or frontend server lifecycle changes.
- No readiness closure.
- No live authority granted.

## Validation

- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=backend python -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short -x`
  passed: 11 tests.
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short -x`
  passed: 301 tests, 8 skipped.
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
  passed: 3319 tests, 9 skipped.
- `PYTHONPATH=backend python -m py_compile tools/codex_autoloop/update_mvp_progress.py tools/codex_autoloop/project_director_agent.py tools/build_wing_readonly_preflight_handoff_packet.py tools/no_wing_release_check.py`
  passed.
- `PYTHONPATH=backend python tools/memory/memory_lint.py --root /Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main`
  passed.
- `/opt/homebrew/bin/git diff --check`
  passed.
