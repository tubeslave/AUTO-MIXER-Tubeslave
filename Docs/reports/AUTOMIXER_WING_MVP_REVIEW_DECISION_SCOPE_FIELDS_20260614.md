# WING MVP Review Decision Scope Fields

## Summary

WING read-only preflight decision records now preserve the reviewed proof scope:
channels, value types, operator channel/session confirmation, and readback
counts.

## Changed

- `backend/wing_readonly_preflight_contract.py` adds scope fields to the shared
  decision-template required fields.
- `tools/review_wing_readonly_preflight_artifacts.py` emits scope fields in the
  pending operator decision template.
- `backend/wing_readonly_preflight_review_decision.py` rejects decision records
  whose scope fields do not match the reviewed candidate.
- Focused tests prove stale channel/value-type/readback scope is rejected.
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md` documents the fields in the
  minimum accepted-decision JSON.

## Safety Boundary

This is an evidence-integrity gate. It does not contact WING, mutate console
state, grant live authority, close readiness, or authorize supervised fader/gain
proof.

## Remaining Blocker

The WING MVP still waits for a fresh real `wing_readonly_preflight.v1` capture
and an operator review decision that includes the scoped proof fields.
