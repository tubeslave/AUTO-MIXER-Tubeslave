# Automixer WING Read-Only Preflight Plan Policy Flags Release Gate - 2026-06-07

## Summary

This slice makes the no-WING release checker reject offline WING read-only
preflight plans that do not include:

- `expected_plan_required_for_real_run: true`
- `physical_preflight_requires_existing_backend_connection: true`

The same fields are now exposed in release summary JSON and Markdown so the
operator handoff can be inspected before any real WING session opens.

## Safety Boundary

- No WING Rack session was opened.
- No OSC/WebSocket hardware transport was contacted.
- No fader, gain, trim, EQ, dynamics, routing, mute, FX, scene, matrix, or DCA
  write path was widened.
- No readiness matrix row or MVP blocker was closed.
- The next real action remains `Docs/runbooks/WING_READONLY_PREFLIGHT.md`.

## Files Changed

- `tools/no_wing_release_check.py`
- `tools/wing_readonly_preflight_runner.py`
- `tests/test_no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_runner.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `.ai/briefs/wing-readonly-preflight-plan-policy-flags-release-gate-20260607.md`
- `.ai/proposals/codex-wing-readonly-preflight-plan-policy-flags-release-gate-20260607.md`
- `Docs/adr/wing-readonly-preflight-plan-policy-flags-release-gate-20260607.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_PLAN_POLICY_FLAGS_RELEASE_GATE_20260607.md`
- `Docs/INDEX.md`

## Validation

Current-cycle validation:

- `python3 -m py_compile tools/no_wing_release_check.py tools/wing_readonly_preflight_runner.py` - PASS.
- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_runner.py` - `195 passed`.
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root artifacts/no_wing_release_check_current_cycle` - PASS, top-level `status: passed`.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q` - `3006 passed, 1 skipped, 1 warning`.
- `python3 tools/codex_autoloop/codex_direction_guard.py --format markdown --output .paperclip/reports/codex_direction_guard_final.md` - PASS, `on_target`, score `90`.
- `python3 tools/memory/memory_lint.py` - PASS.
- `/opt/homebrew/bin/git diff --check` - PASS.

Latest repo-local no-WING artifact retained:

- `artifacts/no_wing_release_check_current_cycle/20260607T035135Z_no_wing_release_check/`

## Current Boundary

This strengthens the operator handoff for the first WING blocker. It is not a
real WING connection proof and does not authorize supervised writes.
