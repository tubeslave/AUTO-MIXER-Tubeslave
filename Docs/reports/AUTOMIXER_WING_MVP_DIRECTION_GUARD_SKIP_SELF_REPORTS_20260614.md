# WING MVP Direction Guard Skip Self Reports

## Summary

`codex_direction_guard.py` now skips generated direction guard reports when
scoring current-cycle evidence. These files are derived diagnostics, so they no
longer count as target, drift, or mixed evidence in later cycles.

The focused regression test covers both generated Markdown-style and JSON-style
guard reports, including truncated JSON snippets. Real WING read-only proof
evidence still scores normally.

This keeps Project Director routing focused on the first WING read-only
connection proof blocker. It does not contact WING, run hardware, write mixer
state, grant live authority, or close readiness.

## Validation

Passed before commit:

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_codex_direction_guard.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short`
  - Result: `34 passed`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile tools/codex_autoloop/codex_direction_guard.py tools/review_wing_readonly_preflight_decisions.py tests/test_codex_direction_guard.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py`
  - Result: passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --output .paperclip/reports/project_director_decision_review_prestart_bool_rendering_after_fix.md --print`
  - Result: selected blocker `WING read-only connection proof`, `guard_status=on_target`, `guard_score=90`, `guard_stop_the_line=false`, evidence file count `57`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --backend-port 65535 --artifact-root /tmp/automixer_goal_decision_review_bool_and_guard_20260614`
  - Result: passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --output .paperclip/reports/project_director_decision_review_bool_and_guard_final.md --print`
  - Result: selected blocker `WING read-only connection proof`, `guard_status=on_target`, `guard_score=90`, `guard_stop_the_line=false`, evidence file count `58`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
  - Result: `3290 passed, 1 skipped, 1 warning`.
- `/opt/homebrew/bin/git diff --check`
  - Result: passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`
  - Result: `Status: PASS`.
