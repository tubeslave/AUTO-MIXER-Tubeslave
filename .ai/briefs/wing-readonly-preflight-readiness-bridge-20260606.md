# Brief: WING Read-Only Preflight Readiness Bridge

Task id: `wing-readonly-preflight-readiness-bridge-20260606`
Date: 2026-06-06

## Goal

Make real `wing_readonly_preflight.v1` evidence visible in the autonomy
promotion readiness snapshot so the supervised WING Rack MVP can review
read-only connection, physical readback, operator-visible evidence, and
readiness-matrix update candidates from one canonical surface.

## Scope

- Extend `backend/autonomy_promotion_readiness.py` with a read-only overlay for
  `wing_readonly_preflight_evidence`.
- Keep all existing promotion rows, safety rows, and live-write policy
  unchanged.
- Allow valid preflight evidence to become a manual readiness-review candidate
  only.
- Keep readiness closure, live authority, and Full Autonomous authority false.

## Safety Boundary

This task does not run WING, OSC, WebSocket, or console commands. It reads the
existing transport-free evidence snapshot only. A valid preflight artifact does
not authorize supervised apply, rollback, emergency-stop proof, or live writes.

## Definition of Done

- Default readiness snapshot reports that read-only preflight evidence is still
  required when no artifact exists.
- A valid preflight snapshot appears as a manual review candidate without
  closing blocker rows.
- Targeted readiness, registry, no-WING, memory lint, and diff checks pass.
