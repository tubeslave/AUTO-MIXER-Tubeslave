# WING Read-Only Review Next Action Promotion

## Summary

Blocked WING read-only review reports now promote artifact-specific
`operator_next_step` guidance into the primary `next_required_action`. This
makes expected-plan repair instructions visible before any backend/WING
WebSocket session is opened.

## Safety Boundary

- No WING hardware contact.
- No backend WebSocket opened.
- No OSC writes.
- No live authority granted.
- Clean-candidate acceptance gates are unchanged.

## Validation

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_evidence.py`
