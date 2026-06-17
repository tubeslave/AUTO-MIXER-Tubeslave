# Brief: WING MVP Next Proof Plan Required Fields Summary JSON Surface

Date: 2026-06-15

## Task

Expose the next WING proof-plan required and missing supervised summary fields
as top-level no-WING `summary.json` fields.

## Constraints

- Keep `WING read-only connection proof` as the first open blocker.
- Do not contact WING, query a console, start a backend/WING session, or widen
  live authority.
- Mirror the existing `next_wing_proof_plan` required-field evidence instead of
  changing proof-plan readiness rules.
- Preserve fail-closed behavior when required summary fields are missing.

## Expected Evidence

- Top-level `summary.json` exposes
  `missing_required_supervised_proof_plan_summary_fields`.
- Top-level required and missing fields match the nested `next_wing_proof_plan`.
- The required-field contract match is explicit and readiness/live authority
  remain denied.
- Focused and standard tests pass.
