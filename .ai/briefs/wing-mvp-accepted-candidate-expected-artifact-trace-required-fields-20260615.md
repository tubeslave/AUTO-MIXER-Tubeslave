# Brief: WING MVP Accepted Candidate Expected Artifact Trace Required Fields

## Task

Make accepted-candidate expected artifact trace fields part of the required
supervised proof-plan summary contract.

## Context

The next WING supervised proof plan must not proceed unless accepted review
source ids are bound to concrete expected read-only preflight artifacts. The
no-WING summary already exposes the trace surface, but the proof-plan required
field contract should also fail closed if those fields disappear from
`next_wing_proof_plan.summary`.

## Constraints

- No WING connection, OSC transport, backend WebSocket session, or live writes.
- Do not treat an offline no-WING packet as readiness authority.
- Preserve the current first blocker: `WING read-only connection proof`.
- Keep readiness closure and live authority denied.

## Done When

- `WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_REQUIRED_PROOF_PLAN_SUMMARY_FIELDS`
  includes accepted-candidate expected artifact trace fields.
- `build_supervised_wing_proof_plan()` writes those fields in its summary.
- no-WING validation fails closed when any of those fields is missing.
- Focused and full validation pass.
