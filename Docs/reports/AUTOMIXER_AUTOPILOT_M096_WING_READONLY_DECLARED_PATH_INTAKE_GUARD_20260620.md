# Automixer Autopilot M096 WING Read-Only Declared Path Intake Guard

## Summary

M096 adds a read-only artifact intake guard for the active M001 external WING
validation blocker. A candidate WING read-only summary is rejected before manual
review if `summary_path` or `events_path` does not match the declared
`artifact_root` and `run_id`.

## Evidence

- Runtime guard: `backend/wing_readonly_preflight_evidence.py`
- Focused regression: `tests/test_wing_readonly_preflight_evidence.py`
- Downstream review-decision fixture: `tests/test_wing_readonly_preflight_review_decision.py`
- M096 rail: `tests/test_wing_mvp_rails.py::test_m096_wing_readonly_declared_path_intake_guard_stays_readonly`

## Safety

This is read-only intake hardening. It creates no WING/backend contact, opens
no WebSocket/OSC transport, sends no live write, imports no proposal, grants no
live authority and closes no readiness row. M001 still waits for a real WING
Rack artifact and operator review.

## Result

Candidate summaries with `summary_path_declared_artifact_path_mismatch` or
`events_path_declared_artifact_path_mismatch` are invalid and cannot become
accepted real WING read-only candidates.
