# Automixer WING Read-Only Preflight Require Expected Plan - 2026-06-07

## Summary

This slice makes `--expected-plan` mandatory for real WING read-only preflight
runs. If the flag is omitted, the runner writes a blocked local summary and
exits before opening WebSocket.

## Safety Boundary

- No WING Rack session was opened.
- No OSC/WebSocket hardware transport was contacted.
- No fader, gain, trim, EQ, dynamics, routing, mute, FX, scene, matrix, or DCA
  write path was widened.
- No readiness matrix row or MVP blocker was closed.
- The next real action remains `Docs/runbooks/WING_READONLY_PREFLIGHT.md`.

## Files Changed

- `tools/wing_readonly_preflight_runner.py`
- `tests/test_wing_readonly_preflight_runner.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `.ai/briefs/wing-readonly-preflight-require-expected-plan-20260607.md`
- `.ai/proposals/codex-wing-readonly-preflight-require-expected-plan-20260607.md`
- `Docs/adr/wing-readonly-preflight-require-expected-plan-20260607.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_REQUIRE_EXPECTED_PLAN_20260607.md`
- `Docs/INDEX.md`

## Validation

Pre-commit validation run:

- `python3 -m py_compile tools/wing_readonly_preflight_runner.py` - PASS
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_runner.py` - `20 passed`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_no_wing_require_expected_plan_20260607` - PASS, top-level `status: passed`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q` - `3004 passed, 1 skipped, 1 warning`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py` - PASS, `on_target`, score `90`
- `python3 tools/memory/memory_lint.py` - PASS
- `/opt/homebrew/bin/git diff --check` - PASS

## Current Boundary

This improves the proof quality of the first WING blocker. It is not a real WING
connection proof by itself and does not authorize supervised writes.
