# Automixer Autopilot M098 WING Read-Only Source Path Intake Guard

## Summary

M098 adds a read-only artifact intake guard for the active M001 external WING
validation blocker. A candidate WING read-only summary loaded from disk is
rejected before manual review when the source file path does not match the
payload's declared `summary_path`.

## Evidence

- Runtime guard: `backend/wing_readonly_preflight_evidence.py`
- Focused regression: `tests/test_wing_readonly_preflight_evidence.py`
- M098 rail: `tests/test_wing_mvp_rails.py::test_m098_wing_readonly_source_path_intake_guard_stays_readonly`

## Safety

This is read-only intake hardening. It creates no WING/backend contact, opens
no WebSocket/OSC transport, sends no live write, imports no proposal, grants no
live authority and closes no readiness row. M001 still waits for a real WING
Rack artifact and operator review.

## Result

Copied or stale-source summaries with
`source_path_declared_summary_path_mismatch` are invalid and cannot become
accepted real WING read-only candidates.
