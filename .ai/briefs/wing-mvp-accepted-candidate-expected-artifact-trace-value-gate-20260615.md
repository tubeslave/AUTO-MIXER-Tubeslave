# Brief: WING MVP Accepted Candidate Expected Artifact Trace Value Gate

## Task

Make no-WING validation fail closed when accepted-candidate expected artifact
trace fields in `next_wing_proof_plan.summary` drift from the canonical trace
computed from the read-only preflight review decision ledger.

## Context

The proof-plan summary now requires accepted-candidate expected artifact trace
fields. Presence alone is not enough: a generated proof plan could carry all
required keys while pointing an accepted source id at stale summary/events
artifacts or false match flags.

## Constraints

- No WING connection, OSC transport, backend WebSocket session, or live writes.
- Do not treat an offline no-WING packet as readiness authority.
- Preserve the current first blocker: `WING read-only connection proof`.
- Keep readiness closure and live authority denied.

## Done When

- `validate_supervised_proof_report_next_wing_plan()` compares each accepted
  candidate expected artifact trace field against canonical values computed
  from `candidate_reviews`.
- no-WING validation fails closed on mismatched source ids, expected artifact
  maps, expected summary/events artifact maps, match flags, missing sources, or
  completion verdict.
- Focused and full validation pass.
