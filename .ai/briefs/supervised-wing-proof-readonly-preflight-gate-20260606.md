# Brief: Supervised WING Proof Read-Only Preflight Gate

Task id: `supervised-wing-proof-readonly-preflight-gate-20260606`
Date: 2026-06-06

## Goal

Enforce the supervised WING Rack MVP order: read-only preflight and physical
readback evidence must exist before the next supervised fader/gain proof plan
can expose physical proof-runner steps.

## Scope

- Gate `backend/supervised_wing_proof_plan.py` on the
  `autonomy_promotion_readiness` read-only preflight overlay.
- Add a blocked plan state for missing or invalid read-only preflight evidence.
- Keep the proof plan non-executing and operator-approval-only.
- Update no-WING release validation so plan steps cannot appear without a clean
  read-only preflight candidate.

## Safety Boundary

This task does not run WING, OSC, WebSocket, or console commands. It removes
premature proof-runner commands from the plan when read-only preflight evidence
is missing. It does not grant live writes, readiness closure, or Full
Autonomous authority.

## Definition of Done

- Missing read-only preflight evidence produces `blocked_by_readonly_preflight`
  with zero proof steps.
- Valid read-only preflight evidence is required before physical proof steps are
  shown.
- no-WING release validation rejects proof steps without preflight evidence.
- Focused tests, memory lint, diff check, and full pytest pass.
