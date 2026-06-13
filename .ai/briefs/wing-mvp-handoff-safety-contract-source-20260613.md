# WING MVP Handoff Safety Contract Source

Task id: `wing-mvp-handoff-safety-contract-source-20260613`

## Context

The first open MVP blocker remains `WING read-only connection proof`.
The WING read-only handoff packet carries two safety lists that must not drift
between generator and validator:

- `forbidden_live_request_types`
- `blockers_not_closed_by_packet`

These lists define what the no-WING packet must not authorize and which MVP
blockers remain open until real WING evidence is manually reviewed.

## Constraints

- Do not contact WING hardware.
- Do not start, stop, or reconfigure the backend listener.
- Do not grant live authority from offline handoff or no-WING artifacts.
- Preserve existing public aliases in `tools/no_wing_release_check.py`.
- Keep blocker status unchanged until real WING artifacts are manually reviewed.

## Required Outcome

Move the handoff forbidden-request and not-closed-blocker lists to
`backend/wing_readonly_preflight_contract.py`, then have the no-WING validator
and handoff packet builder consume that shared contract.
