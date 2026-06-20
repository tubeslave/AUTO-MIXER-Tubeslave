# Automixer Autopilot M099: WING Read-Only Run ID Path Segment Intake Guard

## Summary

M099 closes a WING read-only artifact-intake path traversal gap. Candidate
summaries now fail before manual review when `run_id` is not a single safe path
segment, including parent traversal such as `../stale_wing_readonly_preflight`.

The concrete violation is `run_id_not_single_safe_path_segment`.

## Boundary

This is read-only intake hardening only. It contacts no WING/backend transport,
opens no WebSocket/OSC write path, imports no proposal into a queue, applies no
proposal, grants no live authority, closes no readiness row, creates no
hardware evidence and does not replace real WING/operator validation.

## Validation

- Focused WING read-only evidence tests:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_readonly_preflight_evidence.py -q`:
  `71 passed`.
- M099 rail:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m099_wing_readonly_run_id_path_segment_intake_guard_stays_readonly -q`:
  `1 passed`.
- Full WING MVP rail:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`:
  `112 passed`.
- Full staged suite:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`:
  `4029 passed, 1 skipped, 1 warning`.
- Memory lint and JSON/JSONL validation: see `ops/autopilot/STATE.json`.

## Result

M001 remains `WAITING_FOR_EXTERNAL_VALIDATION`. The next real unblocker is still
an operator-supervised WING read-only preflight artifact and manual review.
