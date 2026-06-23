# Brief: WING MVP Backend Prestart Failure Detail Summary JSON Surface

Date: 2026-06-15

## Task

Expose the backend prestart failure-detail field contract as top-level no-WING
`summary.json` fields.

## Constraints

- Keep `WING read-only connection proof` as the first open blocker.
- Do not contact WING, query a console, start a backend/WING session, or widen
  live authority.
- Mirror the existing handoff-packet failure-detail contract.
- Preserve readiness closure and live authority denial.

## Expected Evidence

- Top-level `summary.json` exposes
  `backend_port_prestart_check_failure_detail_fields`,
  `expected_backend_port_prestart_check_failure_detail_fields`, and a match
  verdict.
- The field list matches the canonical backend prestart failure-detail contract.
- Focused and standard tests pass.
