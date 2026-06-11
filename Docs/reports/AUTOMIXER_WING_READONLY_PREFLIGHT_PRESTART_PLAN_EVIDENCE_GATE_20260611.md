# Automixer WING Read-Only Preflight Prestart Plan Evidence Gate - 2026-06-11

## Summary

This slice strengthens the first supervised WING Rack MVP blocker, `WING read-only connection proof`.

The file-backed WING read-only preflight evidence review now rejects a linked expected-plan file if it omits or changes the required prestart no-WING release check boundary.

## Boundary

- No WING Rack was contacted.
- No backend WebSocket WING session was opened.
- No live OSC/WING writes were executed.
- No MVP blocker is closed by this offline validation hardening.

## What Changed

- `backend/wing_readonly_preflight_evidence.py` now validates expected-plan prestart fields against `backend/wing_readonly_preflight_contract.py`.
- `tests/test_wing_readonly_preflight_evidence.py` now proves a tampered expected plan fails review when it changes the required prestart check flags, command, or diagnostic detail fields.

## Operator Impact

Before a real preflight artifact can be accepted for review, its linked expected plan must preserve the runbook boundary that `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick` is required before starting the intended backend/WING session.

## Verification

Verification:

```bash
python3 -m py_compile backend/wing_readonly_preflight_evidence.py
PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py
```

Result: `39 passed`.

Adjacent WING preflight/no-WING validation:

```bash
PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_runner.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_no_wing_release_check.py
```

Result: `267 passed`.

Full backend suite:

```bash
PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q
```

Result: `3178 passed, 1 skipped, 1 warning`.

Diagnostic no-WING quick run was executed with artifacts under `/tmp` to avoid
dirtying the repository:

```bash
PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --artifact-root /tmp/automixer_no_wing_release_check_prestart_plan_gate --command-timeout-sec 300
```

Result: failed on `backend_port_free` because `127.0.0.1:8765` already had a
`Python` listener (`pid 87210`). No process was stopped; the operator must
confirm whether that listener is the intended Automixer backend before any real
WING preflight run.
