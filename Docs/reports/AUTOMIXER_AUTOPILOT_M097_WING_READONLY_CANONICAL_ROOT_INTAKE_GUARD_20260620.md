# Automixer Autopilot M097 WING Read-Only Canonical Root Intake Guard

## Summary

M097 adds a read-only artifact intake guard for the active M001 external WING
validation blocker. A candidate WING read-only summary is rejected before manual
review if `artifact_root` is not the canonical WING read-only capture root:
`artifacts/wing_readonly_preflight`.

## Evidence

- Runtime guard: `backend/wing_readonly_preflight_evidence.py`
- Focused regression: `tests/test_wing_readonly_preflight_evidence.py`
- M097 rail: `tests/test_wing_mvp_rails.py::test_m097_wing_readonly_canonical_root_intake_guard_stays_readonly`

## Safety

This is read-only intake hardening. It creates no WING/backend contact, opens
no WebSocket/OSC transport, sends no live write, imports no proposal, grants no
live authority and closes no readiness row. M001 still waits for a real WING
Rack artifact and operator review.

## Result

Candidate summaries with `artifact_root_not_wing_readonly_capture_root` are
invalid and cannot become accepted real WING read-only candidates.
