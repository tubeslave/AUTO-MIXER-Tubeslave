# WING MVP Handoff Template Output Contract

Task id: `wing-mvp-handoff-template-output-contract-20260613`

## Context

The first open MVP blocker remains `WING read-only connection proof`. The
review decision template output path is now contract-owned and no-WING evidence
checks the review command value, but the operator-facing handoff packet summary
still only exposed whether the command had `--decision-template-output`.

That leaves a visibility gap: a packet can mention a pending decision template
without making the contract match explicit in the handoff artifact summary.

## Constraints

- Do not contact WING hardware.
- Do not start, stop, or reconfigure the backend listener.
- Do not grant OSC/WING write authority.
- Preserve the existing public pending-template path.
- Keep readiness closure blocked until real WING proof and manual review exist.

## Required Outcome

Make the handoff packet and no-WING handoff artifact summary expose whether the
pending review decision template output path matches
`WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT`, and fail closed when
the packet path or review command path drifts.
