# Automixer WING Read-Only Preflight Expected-Plan Gate - 2026-06-07

## Summary

This slice binds real WING read-only preflight capture to the offline plan
preview. The runner now accepts `--expected-plan`, validates runtime arguments
and request sequence against that JSON before opening WebSocket, records the
plan SHA256 in `summary.json`, and fails closed if the plan is missing, invalid,
or mismatched.

The evidence validator now rejects artifacts that do not prove:

- `expected_plan_path` is present;
- `expected_plan_sha256` is present;
- `expected_plan_match` is `true`;
- `expected_plan_violations` is empty.

## Safety Boundary

- No WING Rack session was opened.
- No OSC/WebSocket hardware transport was contacted.
- No fader, gain, trim, EQ, dynamics, routing, mute, FX, scene, matrix, or DCA
  write path was widened.
- No readiness matrix row or MVP blocker was closed.
- The next real action remains `Docs/runbooks/WING_READONLY_PREFLIGHT.md`.

## Files Changed

- `tools/wing_readonly_preflight_runner.py`
- `backend/wing_readonly_preflight_evidence.py`
- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- focused tests for runner, evidence, review, handoff, no-WING, and autonomy overlay behavior
- `.ai/briefs/wing-readonly-preflight-expected-plan-gate-20260607.md`
- `.ai/proposals/codex-wing-readonly-preflight-expected-plan-gate-20260607.md`
- `Docs/adr/wing-readonly-preflight-expected-plan-gate-20260607.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_EXPECTED_PLAN_GATE_20260607.md`
- `Docs/INDEX.md`

## Validation

Pre-commit validation run:

- `python3 -m py_compile tools/wing_readonly_preflight_runner.py backend/wing_readonly_preflight_evidence.py tools/no_wing_release_check.py tools/build_wing_readonly_preflight_handoff_packet.py` - PASS
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py` - `182 passed`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_no_wing_expected_plan_gate_20260607` - PASS, top-level `status: passed`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q` - `3003 passed, 1 skipped, 1 warning`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py` - PASS, `on_target`, score `90`
- `python3 tools/memory/memory_lint.py` - PASS
- `/opt/homebrew/bin/git diff --check` - PASS

## Current Boundary

This improves the proof quality of the first WING blocker. It is not a real WING
connection proof by itself and does not authorize supervised writes.
