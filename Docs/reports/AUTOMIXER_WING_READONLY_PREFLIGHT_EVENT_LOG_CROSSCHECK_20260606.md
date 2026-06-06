# Automixer WING Read-Only Preflight Event-Log Crosscheck - 2026-06-06

## Summary

This slice hardens read-only WING preflight evidence intake. A filesystem
`wing_readonly_preflight.v1` summary is now accepted only when its `events.jsonl`
exists and agrees with the recorded read-only request/readback flow.

## Safety Boundary

- No live WING/OSC/WebSocket action was run for this change.
- The evaluator still reads local artifact files only.
- No readiness blocker is closed automatically.
- No fader, gain, trim, EQ, compressor, routing, mute, scene, insert, FX, or
  snapshot write authority was added.

## Behavior

- Valid summary plus matching event log: remains a manual review candidate.
- Missing event log: artifact replacement required.
- Corrupt event log: artifact replacement required.
- Event log with forbidden send request types: artifact replacement required.
- Event log with request order other than `get_operator_mode`, `get_state`,
  then `query_mixer_parameter` rows: artifact replacement required.
- Event log that is not strict `send`/`recv` pairs: artifact replacement
  required.
- Event log with missing, duplicate, skipped, or non-contiguous `seq` values:
  artifact replacement required.
- Event log readbacks that differ from summary readbacks: artifact replacement
  required.
- Event log readbacks whose channel/value-type/address do not match their
  query request: artifact replacement required.

## Files Changed

- `backend/wing_readonly_preflight_evidence.py`
- `backend/write_surface_inventory.py`
- `tools/review_wing_readonly_preflight_artifacts.py`
- `tests/test_wing_readonly_preflight_evidence.py`
- `tests/test_wing_readonly_preflight_review_tool.py`
- `.ai/briefs/wing-readonly-preflight-event-log-crosscheck-20260606.md`
- `.ai/proposals/codex-wing-readonly-preflight-event-log-crosscheck-20260606.md`
- `Docs/adr/wing-readonly-preflight-event-log-crosscheck-20260606.md`
- `Docs/reports/AUTOMIXER_WRITE_SURFACE_HARDENING_20260601.md`

## Validation

- `python3 -m py_compile backend/wing_readonly_preflight_evidence.py tools/wing_readonly_preflight_runner.py tools/no_wing_release_check.py tools/review_wing_readonly_preflight_artifacts.py` - PASS
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_autonomy_promotion_readiness.py tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_runner.py tests/test_write_surface_inventory_static_scan.py::test_write_surface_inventory_covers_files_with_direct_write_tokens` - `167 passed`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_evidence.py tests/test_wing_mvp_rails.py tests/test_no_wing_release_check.py tests/test_write_surface_inventory_static_scan.py::test_write_surface_inventory_covers_files_with_direct_write_tokens` - `153 passed`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_write_surface_inventory_static_scan.py::test_write_surface_inventory_covers_files_with_direct_write_tokens tests/test_wing_readonly_preflight_evidence.py tests/test_no_wing_release_check.py` - `139 passed`
- `python3 tools/memory/memory_lint.py` - PASS
- `/opt/homebrew/bin/git diff --check` - PASS
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q` - `2888 passed, 1 skipped, 1 warning`
- `PYTHONPATH=backend python3 -m pytest tests/test_wing_readonly_preflight_evidence.py -q` - `14 passed`

## Remaining WING MVP Blockers

This makes preflight evidence harder to trust accidentally when the artifact is
incomplete or tampered. Real hardware blockers remain open until an operator
captures a physical WING read-only preflight, reviews the artifact, approves a
bounded supervised fader/gain proof, verifies readback, rollback,
emergency-stop, timeout/fail-closed behavior, and records the readiness update
from real proof.
