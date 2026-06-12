# WING Read-Only Review Operator Next Step Surface

## Summary

Review reports now preserve `operator_next_step` from blocked WING read-only
preflight summaries. This keeps expected-plan repair guidance visible during
manual artifact review while preserving the offline/read-only boundary.

## Safety Boundary

- No WING hardware contact.
- No backend WebSocket opened by this change.
- No OSC writes.
- No live authority granted.
- Invalid artifacts remain blocked.

## Validation

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_evidence.py`
