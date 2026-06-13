# WING MVP Decision Review Prestart Summary Fields

## Summary

The WING read-only decision review Markdown report now renders the
accepted-candidate no-WING prestart summary binding that was already present in
the JSON ledger. Each candidate review row shows:

- `backend_port_prestart_summary_required`
- `backend_port_prestart_summary_path`
- `backend_port_prestart_summary_sha256`
- `backend_port_prestart_summary_type`
- `backend_port_prestart_summary_status`
- `backend_port_prestart_summary_connection_policy`
- `backend_port_prestart_summary_live_write`
- `backend_port_prestart_summary_contacts_backend`
- `backend_port_prestart_summary_queries_console`
- `backend_port_prestart_summary_executes_tests`
- `backend_port_prestart_check_status`

This keeps the accepted-candidate prestart trace reviewable in the
operator-facing report before any supervised WING proof planning. It does not
contact WING, run hardware, write mixer state, grant live authority, or close
the WING read-only connection proof blocker.

## Validation

Passed before commit:

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_wing_readonly_preflight_review_decision_tool.py --tb=short`
  - Result: `10 passed`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile tools/review_wing_readonly_preflight_decisions.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py`
  - Result: passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short`
  - Result: `21 passed`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --backend-port 65535 --artifact-root /tmp/automixer_goal_decision_review_prestart_summary_fields_20260614`
  - Result: passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --output .paperclip/reports/project_director_decision_review_prestart_summary_fields.md --print`
  - Result: selected blocker `WING read-only connection proof`, `guard_status=on_target`, `guard_score=90`, `guard_stop_the_line=false`, evidence file count `56`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/codex_direction_guard.py --format json --output .paperclip/reports/codex_direction_guard_decision_review_prestart_summary_fields.md`
  - Result: passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
  - Result: `3289 passed, 1 skipped, 1 warning`.
- `/opt/homebrew/bin/git diff --check`
  - Result: passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`
  - Result: `Status: PASS`.
