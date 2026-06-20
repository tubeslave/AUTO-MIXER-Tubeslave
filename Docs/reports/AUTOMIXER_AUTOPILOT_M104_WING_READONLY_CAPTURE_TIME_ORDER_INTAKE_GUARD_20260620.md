# Automixer Autopilot M104: WING Read-Only Capture Time Order Intake Guard

## Summary

M104 closes a WING read-only artifact-intake temporal provenance gap.
Candidate summaries now fail before manual review when `captured_at` is earlier
than the UTC capture timestamp encoded in `run_id`.

The concrete summary violation is `captured_at_before_run_id_timestamp`. For
file-backed artifacts, event logs can also report
`event_log_timestamp_before_run_id:<index>`.

## Boundary

This is read-only intake hardening only. It contacts no WING/backend transport,
opens no WebSocket/OSC write path, imports no proposal into a queue, applies no
proposal, grants no live authority, closes no readiness row, creates no
hardware evidence and does not replace real WING/operator validation.

## Validation

- Focused WING read-only evidence tests:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_readonly_preflight_evidence.py -q`.
- Review-template fixture validation:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_readonly_preflight_review_tool.py -q`.
- M104 rail:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m104_wing_readonly_capture_time_order_intake_guard_stays_readonly -q`.
- Full WING MVP rail:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`.
- Full staged suite:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`.
- Memory lint and JSON/JSONL validation: see `ops/autopilot/STATE.json`.

## Result

M001 remains `WAITING_FOR_EXTERNAL_VALIDATION`. The next real unblocker is still
an operator-supervised WING read-only preflight artifact and manual review.
