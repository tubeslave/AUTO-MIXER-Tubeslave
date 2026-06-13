# WING MVP Review Decision Template Contract Source

Task id: `wing-mvp-review-decision-template-contract-source-20260613`

## Context

The first open MVP blocker remains `WING read-only connection proof`.
Before any supervised proof or readiness update, a real read-only preflight
candidate must produce an operator-edited review decision JSON.

The required fields for that pending decision template are duplicated in the
review tool and the no-WING release checker. This keeps a drift point in the
operator decision artifact contract.

## Constraints

- Do not contact WING hardware.
- Do not start, stop, or reconfigure the backend listener.
- Do not grant live authority from offline handoff or no-WING artifacts.
- Preserve existing public aliases used by tests and tools.
- Keep blocker status unchanged until real WING artifacts are manually reviewed.

## Required Outcome

Move the WING read-only review decision template required-field tuple to
`backend/wing_readonly_preflight_contract.py`, then have the review tool and
no-WING validator consume that shared contract.
