# Automixer WING Read-Only Preflight Expected-Plan Required Evidence Gate - 2026-06-07

## Summary

This slice makes WING read-only evidence intake reject preflight summaries that
do not include `expected_plan_required_for_real_run: true`.

The runner already emits this field for real runs; the evidence layer now
requires it before a summary can become a clean read-only preflight candidate.

## Safety Boundary

- No WING Rack session was opened.
- No OSC/WebSocket hardware transport was contacted.
- No fader, gain, trim, EQ, dynamics, routing, mute, FX, scene, matrix, or DCA
  write path was widened.
- No readiness matrix row or MVP blocker was closed.
- The next real action remains `Docs/runbooks/WING_READONLY_PREFLIGHT.md`.

## Files Changed

- `backend/wing_readonly_preflight_evidence.py`
- focused WING evidence, review, decision, and autonomy readiness tests
- `.ai/briefs/wing-readonly-preflight-expected-plan-required-evidence-gate-20260607.md`
- `.ai/proposals/codex-wing-readonly-preflight-expected-plan-required-evidence-gate-20260607.md`
- `Docs/adr/wing-readonly-preflight-expected-plan-required-evidence-gate-20260607.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_EXPECTED_PLAN_REQUIRED_EVIDENCE_GATE_20260607.md`
- `Docs/INDEX.md`

## Validation

Pre-commit validation run:

- `python3 -m py_compile backend/wing_readonly_preflight_evidence.py` - PASS
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_autonomy_promotion_readiness.py` - `50 passed`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_no_wing_expected_plan_required_evidence_gate_20260607` - PASS, top-level `status: passed`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q` - `3005 passed, 1 skipped, 1 warning`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py` - PASS, `on_target`, score `90`
- `python3 tools/memory/memory_lint.py` - PASS
- `/opt/homebrew/bin/git diff --check` - PASS

## Current Boundary

This improves the proof quality of the first WING blocker. It is not a real WING
connection proof by itself and does not authorize supervised writes.
