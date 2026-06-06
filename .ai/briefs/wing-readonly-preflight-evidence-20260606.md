# Brief: WING Read-Only Preflight Evidence

Task id: `wing-readonly-preflight-evidence-20260606`
Date: 2026-06-06

## Goal

Reduce the supervised WING Rack MVP blockers for read-only connection proof,
physical console readback proof, operator-visible evidence, and readiness
matrix updates based on real proof.

## Scope

- Add a transport-free evaluator for `wing_readonly_preflight.v1` summaries.
- Accept only artifacts that passed in backend `wing` mode with required
  physical readback.
- Reject artifacts that sent write, proposal, operator-mode, rollback,
  emergency-stop, connect, or disconnect request types.
- Expose a fail-closed MVP blocker matrix without closing readiness rows or
  granting live authority.

## Safety Boundary

This task does not run WING, OSC, WebSocket, or console commands. It reads JSON
artifacts only. Valid evidence may justify manual operator review, but it does
not permit live writes or automatic readiness promotion.

## Definition of Done

- Offline tests cover missing, valid, invalid, and file-discovered artifacts.
- Processing registry and project-goal matrix expose the evaluator as
  `shadow_only`, not a live executor.
- `tools/no_wing_release_check.py` reports the evidence state without granting
  live authority or closing readiness automatically.
- Memory lint, focused tests, and diff checks pass.
