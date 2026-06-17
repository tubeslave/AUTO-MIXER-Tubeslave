# Brief: WING MVP Accepted Candidate Expected Artifact Trace Value Gate Summary Surface

## Task

Expose the accepted-candidate expected artifact trace value gate as explicit
no-WING `summary.json` and operator `summary.md` fields.

## Context

The no-WING validator now fails closed when `next_wing_proof_plan.summary`
accepted-candidate expected artifact trace values drift from the canonical
review-decision ledger trace. The mismatch verdict should also be visible as a
machine-readable summary surface so operators and Project Director checks can
triage exact trace-value drift without reading raw check internals.

## Constraints

- No WING connection, OSC transport, backend WebSocket session, or live writes.
- Do not treat an offline no-WING packet as readiness authority.
- Preserve the current first blocker: `WING read-only connection proof`.
- Keep readiness closure and live authority denied.

## Done When

- `next_wing_proof_plan_schema` exposes trace value-gate violation ids,
  current violations, pass/fail verdict, and safety-denial fields.
- `write_summary()` mirrors those fields into `next_wing_proof_plan` and
  top-level `summary.json`.
- `summary.md` renders value-gate violations and pass/fail state.
- Focused and full validation pass.
