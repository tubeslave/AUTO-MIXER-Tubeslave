# WING MVP Review Template Output Gate

Task id: `wing-mvp-review-template-output-gate-20260613`

## Context

The first open MVP blocker remains `WING read-only connection proof`. The
previous change centralized the pending review decision template output path in
the shared WING read-only preflight contract.

The no-WING validator already checks that the review command contains
`--decision-template-output`, but a stale command could still point the operator
to the wrong file while passing the flag-presence check.

## Constraints

- Do not contact WING hardware.
- Do not start, stop, or reconfigure the backend listener.
- Do not grant OSC/WING write authority.
- Preserve the existing public pending-template path.
- Keep readiness closure blocked until real WING proof and manual review exist.

## Required Outcome

Make no-WING validation extract the `--decision-template-output` value from the
review command and fail closed unless it exactly matches
`WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT`.
