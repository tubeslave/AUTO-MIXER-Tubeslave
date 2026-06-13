# WING MVP Plan Output Contract

Task id: `wing-mvp-plan-output-contract-20260613`

## Context

The first open MVP blocker remains `WING read-only connection proof`. The
read-only preflight flow depends on an offline expected plan before the operator
opens the backend/WING session.

The handoff packet output path is now contract-owned, but the expected plan
output path was still duplicated across handoff and no-WING tooling.

## Constraints

- Do not contact WING hardware.
- Do not start, stop, or reconfigure the backend listener.
- Do not grant OSC/WING write authority.
- Preserve the existing operator-facing plan output path.
- Keep readiness closure blocked until real WING proof and manual review exist.

## Required Outcome

Move the read-only preflight expected plan output path into the shared WING
preflight contract. Make handoff/no-WING tooling consume it, and make no-WING
fail closed when plan, capture, or handoff commands drift to a stale plan path.
