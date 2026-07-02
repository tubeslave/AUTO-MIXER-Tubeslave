# Autopilot Blockers

## Current blockers

- M001 is waiting for external validation: real WING Rack read-only artifact
  and operator review are required before physical readback intake can begin.
- Supporting real WING read-only evidence exists in
  `Docs/reports/WING_REAL_HARDWARE_READONLY_20260702.md`, but it is a
  temporary-root capture only. It cannot close readiness, replace accepted
  operator decision records, or authorize supervised writes.
- Canonical readiness is still blocked until the operator records accepted or
  rejected decisions for the four unreviewed source ids listed in
  `Docs/reports/WING_READONLY_PREFLIGHT_DECISION_REVIEW_LATEST.md`.
- The current worktree contains local WING evidence/report updates and SmartDSP
  scoring work. Future autopilot runs must avoid overwriting unrelated local
  changes if they appear.
- Future heartbeat wake-ups must verify native Goal transition tools before
  replacing a thread Goal.

## Not blockers

- `ops/autopilot/STOP` does not exist after setup.
- Control-plane docs/state can proceed without live hardware contact.
