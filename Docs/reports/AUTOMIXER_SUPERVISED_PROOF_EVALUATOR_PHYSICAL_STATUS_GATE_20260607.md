# Automixer Supervised Proof Evaluator Physical Status Gate - 2026-06-07

## Summary

This slice hardens the downstream supervised proof evaluator. Physical WING
proof artifacts now need `verified_from_physical_readback` status, not only a
physical-looking source label, before they can become supervised candidates.

## What changed

- `backend/supervised_proof_artifact_evaluator.py` rejects physical proof rows
  whose pre-read, apply, or rollback status is cache-backed.
- `tools/no_wing_release_check.py` includes a static evaluator policy gate.
- Focused evaluator, report, and no-WING tests cover the forged cache-status
  case.

## Safety boundary

No WING hardware was contacted. This does not grant live authority, does not
close readiness rows, and does not widen writes beyond the current supervised
fader/gain proof path.

## Validation

- `python3 -m py_compile backend/supervised_proof_artifact_evaluator.py tools/no_wing_release_check.py`
  - PASS
- `PYTHONPATH=backend python3 -m pytest -q tests/test_supervised_proof_artifact_evaluator.py tests/test_supervised_proof_artifact_report.py tests/test_no_wing_release_check.py`
  - `202 passed`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_no_wing_evaluator_physical_status_gate_20260607`
  - PASS, top-level `status: passed`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py`
  - PASS, `on_target`, score `90`
- `python3 tools/memory/memory_lint.py`
  - PASS
- `/opt/homebrew/bin/git diff --check`
  - PASS
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
  - `3029 passed, 1 skipped, 1 warning`
