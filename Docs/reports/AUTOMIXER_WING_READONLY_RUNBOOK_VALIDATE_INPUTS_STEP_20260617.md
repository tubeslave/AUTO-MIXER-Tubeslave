# Automixer WING Read-Only Runbook Validate Inputs Step

## Summary

The WING read-only runbook now documents `validate_readonly_preflight_inputs` as the mandatory validation-only step before real capture.

## Safety boundary

- The added command uses `--validate-run-inputs-only`.
- It validates the real-run inputs against the no-WING prestart summary and expected plan.
- It must not open transport, contact backend, query console, or grant live authority.

## Validation

- Stale marker search over current WING rail surfaces: passed.
- `python3 -m py_compile tools/codex_autoloop/update_mvp_progress.py tests/test_wing_mvp_rails.py tests/test_project_director_agent.py`: passed.
- `PYTHONPATH=backend python3 -m pytest tests/test_project_director_agent.py tests/test_wing_mvp_rails.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py -q`: 414 passed.
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick`: passed (`artifacts/no_wing_release_check/20260617T111003Z_no_wing_release_check/`).
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`: 3761 passed, 1 skipped, 1 warning.
