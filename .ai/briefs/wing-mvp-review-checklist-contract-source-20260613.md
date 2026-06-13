# WING MVP Review Checklist Contract Source

Task id: `wing-mvp-review-checklist-contract-source-20260613`

## Context

The first open MVP blocker remains `WING read-only connection proof`.
The WING read-only preflight manual-review checklist ids are part of the
operator safety contract used before any readiness update or supervised proof.

Today the ids are duplicated across review decision code and the no-WING
release checker, while the handoff packet builder reads them from the release
checker. That preserves a tool-to-tool coupling in the first hardware handoff
path.

## Constraints

- Do not contact WING hardware.
- Do not start, stop, or reconfigure the backend listener.
- Do not grant live authority from offline handoff or no-WING artifacts.
- Preserve existing public aliases used by tests and tools.
- Keep blocker status unchanged until real WING artifacts are manually reviewed.

## Required Outcome

Move the WING read-only manual-review checklist id tuple to
`backend/wing_readonly_preflight_contract.py`, then have review decision,
no-WING validation, review tooling, and the handoff packet consume that shared
contract.
