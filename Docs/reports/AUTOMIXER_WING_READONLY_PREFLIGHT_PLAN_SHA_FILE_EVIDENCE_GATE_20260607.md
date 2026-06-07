# Automixer WING Read-Only Preflight Plan SHA File Evidence Gate - 2026-06-07

## Summary

This slice makes file-based WING read-only evidence review reject preflight
summaries whose `expected_plan_path` is missing or whose referenced offline plan
file does not hash to `expected_plan_sha256`.

The check runs only when evidence review loads a `summary.json` from disk.
Inline payload validation remains filesystem-free.

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
- `.ai/briefs/wing-readonly-preflight-plan-sha-file-evidence-gate-20260607.md`
- `.ai/proposals/codex-wing-readonly-preflight-plan-sha-file-evidence-gate-20260607.md`
- `Docs/adr/wing-readonly-preflight-plan-sha-file-evidence-gate-20260607.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_PLAN_SHA_FILE_EVIDENCE_GATE_20260607.md`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `Docs/INDEX.md`

## Validation

Pre-commit validation run:

- `python3 -m py_compile backend/wing_readonly_preflight_evidence.py` - PASS
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_autonomy_promotion_readiness.py` - `53 passed`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_no_wing_plan_sha_file_evidence_gate_20260607` - PASS, top-level `status: passed`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q` - `3008 passed, 1 skipped, 1 warning`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py` - PASS, `on_target`, score `90`
- `python3 tools/memory/memory_lint.py` - PASS
- `/opt/homebrew/bin/git diff --check` - PASS

## Current Boundary

This improves the proof quality of the first WING blocker. It is not a real WING
connection proof by itself and does not authorize supervised writes.
