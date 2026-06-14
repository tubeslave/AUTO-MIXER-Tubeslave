# Automixer WING MVP Backend Prestart Command Text

## Summary

The WING read-only prestart artifact chain now carries
`backend_port_prestart_check_command_text` beside the existing
`backend_port_prestart_check_command` argv list. The text value is derived from
the shared contract and must equal
`PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick`.

## Boundary

This is an offline artifact-contract change only. It does not start the backend,
connect to WING, query the console, mutate console state, grant live authority,
or close readiness.

## Validation

- PASS: focused WING preflight/review/proof-plan/no-WING rail tests:
  `443 passed in 3.38s`.
- PASS: regenerated `Docs/reports/MVP_PROGRESS_LATEST.md`.
- PASS: Project Director stayed on `WING read-only connection proof` with
  guard status `on_target`, score `90`, and `stop_broad_work: true`.
- PASS: Direction guard stayed `on_target`, score `90`, with no drift items.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile` for the
  changed backend/tools Python modules.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`.
- PASS: `/opt/homebrew/bin/git diff --check`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
  returned `3299 passed, 1 skipped, 1 warning in 160.10s`.
