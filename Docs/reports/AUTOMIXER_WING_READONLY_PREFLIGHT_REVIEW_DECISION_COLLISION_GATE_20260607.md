# Automixer WING Read-Only Preflight Review Decision Collision Gate - 2026-06-07

## Summary

This slice makes the WING read-only preflight review-decision ledger fail closed
when multiple valid decision records target the same clean preflight candidate.
Duplicate accepted decisions and accepted/rejected conflicts no longer leave an
accepted readiness candidate.

The check is offline only. It does not inspect hardware and does not authorize
readiness or live writes.

## Safety Boundary

- No WING Rack session was opened.
- No OSC/WebSocket hardware transport was contacted.
- No fader, gain, trim, EQ, dynamics, routing, mute, FX, scene, matrix, or DCA
  write path was widened.
- No readiness matrix row or MVP blocker was closed.
- The next real action remains `Docs/runbooks/WING_READONLY_PREFLIGHT.md`.

## Files Changed

- `backend/wing_readonly_preflight_review_decision.py`
- `tools/review_wing_readonly_preflight_decisions.py`
- `tools/no_wing_release_check.py`
- focused WING decision tests
- `.ai/briefs/wing-readonly-preflight-review-decision-collision-gate-20260607.md`
- `.ai/proposals/codex-wing-readonly-preflight-review-decision-collision-gate-20260607.md`
- `Docs/adr/wing-readonly-preflight-review-decision-collision-gate-20260607.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_REVIEW_DECISION_COLLISION_GATE_20260607.md`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `Docs/INDEX.md`

## Validation

Pre-commit validation run:

- `python3 -m py_compile backend/wing_readonly_preflight_review_decision.py tools/review_wing_readonly_preflight_decisions.py tools/no_wing_release_check.py` - PASS
- `PYTHONPATH=backend python3 -m pytest tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py -q --tb=short` - `18 passed`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_autonomy_promotion_readiness.py` - `41 passed`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_no_wing_review_decision_collision_gate_20260607` - PASS, top-level `status: passed`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q` - `3012 passed, 1 skipped, 1 warning`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py` - PASS, `on_target`, score `90`
- `python3 tools/memory/memory_lint.py` - PASS

## Current Boundary

This strengthens the operator-review gate after a real WING read-only preflight
artifact exists. It is not a real WING connection proof and does not authorize
supervised writes.
