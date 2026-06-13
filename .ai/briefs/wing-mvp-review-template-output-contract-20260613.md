# WING MVP Review Template Output Contract

Task id: `wing-mvp-review-template-output-contract-20260613`

## Context

The first open MVP blocker remains `WING read-only connection proof`. Before
any supervised proof, the operator must review a real read-only preflight
artifact and write an edited decision JSON.

The pending decision template path was duplicated across the review tool,
handoff packet builder, no-WING release checker, and read-only runner
post-run commands. If those paths drift, operators can be sent to review or
edit different JSON files.

## Constraints

- Do not contact WING hardware.
- Do not start, stop, or reconfigure the backend listener.
- Do not grant OSC/WING write authority.
- Preserve existing command shapes and public aliases.
- Keep readiness closure blocked until real WING proof and manual review exist.

## Required Outcome

Move the WING read-only review decision root and pending template output path
into `backend/wing_readonly_preflight_contract.py`, then make runner, handoff,
review, and no-WING tooling consume the shared path contract.
