# Automixer WING Read-Only Validate Inputs Handoff Step

## Summary

The WING read-only handoff packet now includes `validate_readonly_preflight_inputs` before `capture_readonly_preflight`.

## Safety boundary

- The step uses `--validate-run-inputs-only`.
- It validates prestart summary and expected plan bindings.
- It does not open transport, contact backend, query console, or grant live authority.

## Validation

- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick`: passed.
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format markdown`: current package appears as WING target evidence; repository still reports unrelated non-WING dirty worktree drift.
- `python3 tools/memory/memory_lint.py`: passed.
- `tests/test_wing_readonly_preflight_handoff_packet.py`: passed.
- `tests/test_no_wing_release_check.py` focused WING/no-WING subset: passed.
- `tests/test_wing_readonly_preflight_runner.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py`: passed.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`: 3761 passed, 1 skipped, 1 warning.
