# Automixer WING MVP Listener Template Path Failure Field

## Summary

The backend-port prestart failure-detail contract now requires
`listener_confirmation_template_path`. The no-WING busy-port path already emits
this field when it creates a backend-listener confirmation template; the shared
contract now makes that artifact path mandatory for stale-plan and handoff
checks.

## Boundary

This is an offline artifact-contract change only. It does not start, stop, or
reconfigure the backend listener, connect to WING, query the console, mutate
console state, grant live authority, or close readiness.

## Validation

- PASS: focused no-WING/WING rail tests:
  `416 passed in 4.25s`.
- PASS: regenerated `Docs/reports/MVP_PROGRESS_LATEST.md`.
- PASS: Project Director stayed on `WING read-only connection proof` with
  guard status `on_target`, score `90`, `stop_broad_work: true`, and selected
  action containing `listener_confirmation_template_path`.
- PASS: Direction guard stayed `on_target`, score `90`, with no drift items.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick`
  failed closed on the existing backend listener and showed
  `listener_confirmation_template_path` in both the result detail and
  `backend_port_prestart_check_failure_detail_fields`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile` for the
  changed backend/tools Python modules.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`.
- PASS: `/opt/homebrew/bin/git diff --check`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
  returned `3299 passed, 1 skipped, 1 warning in 166.92s`.
