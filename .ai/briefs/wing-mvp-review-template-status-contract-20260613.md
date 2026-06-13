# WING MVP Review Template Status Contract

Task id: `wing-mvp-review-template-status-contract-20260613`

## Context

The first open MVP blocker remains `WING read-only connection proof`.
The review artifact tool can generate a pending operator decision template, but
that template is not an accepted decision and cannot close readiness.

The template required fields already come from the shared WING preflight
contract. The pending template status still lived in the review tool only, and
the no-WING release checker did not verify that status against a shared
contract.

## Constraints

- Do not contact WING hardware.
- Do not start, stop, or reconfigure the backend listener.
- Do not authorize OSC/WING writes.
- Preserve existing aliases used by tests and tools.
- Keep readiness closure blocked until real WING proof and manual review exist.

## Required Outcome

Move the pending review decision template status into
`backend/wing_readonly_preflight_contract.py`, have the review tool expose it
from that shared source, and make the no-WING release checker fail closed when
the tool status drifts.
