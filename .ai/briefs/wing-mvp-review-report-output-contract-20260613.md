# WING MVP Review Report Output Contract

Task id: `wing-mvp-review-report-output-contract-20260613`

## Context

The first open MVP blocker remains `WING read-only connection proof`. The
operator handoff packet now validates the pending review decision template path,
but the artifact-review and decision-review report output paths were still
duplicated as literals across handoff/no-WING tooling and tests.

If those report paths drift, operators can be pointed to stale review reports
even while the packet itself remains read-only and grants no live authority.

## Constraints

- Do not contact WING hardware.
- Do not start, stop, or reconfigure the backend listener.
- Do not grant OSC/WING write authority.
- Preserve the existing operator-facing report paths.
- Keep readiness closure blocked until real WING proof and manual review exist.

## Required Outcome

Move the operator-facing artifact-review and decision-review report output paths
into the shared WING read-only preflight contract, make handoff/no-WING tooling
consume them, and fail closed when packet fields or review command `--output`
values drift.
