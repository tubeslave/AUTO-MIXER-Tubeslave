# WING MVP Handoff Packet Output Contract

Task id: `wing-mvp-handoff-packet-output-contract-20260613`

## Context

The first open MVP blocker remains `WING read-only connection proof`. Recent
work moved the review template and review report outputs into the shared WING
preflight contract, but the handoff packet output path itself was still a
duplicated literal in handoff/no-WING tooling.

If that path drifts, the operator can be pointed to a stale or unexpected
handoff packet before the read-only WING proof run.

## Constraints

- Do not contact WING hardware.
- Do not start, stop, or reconfigure the backend listener.
- Do not grant OSC/WING write authority.
- Preserve the existing operator-facing handoff packet path.
- Keep readiness closure blocked until real WING proof and manual review exist.

## Required Outcome

Move the operator-facing handoff packet output path into the shared WING
read-only preflight contract, make handoff/no-WING tooling consume it, and make
no-WING validation fail closed when the handoff command `--output` path drifts.
