# WING MVP Review Checklist Run-Id Binding

## Summary

WING read-only preflight operator decisions now require an explicit
`confirm_event_log_run_id_binding` checklist id.

## Changed

- `backend/wing_readonly_preflight_contract.py` adds the checklist id to the
  shared manual review checklist.
- `tools/review_wing_readonly_preflight_artifacts.py` renders the matching
  checklist label and decision templates inherit the expanded checklist.
- Focused tests prove the review tool, handoff packet, no-WING summary, and
  review-decision ledger all require the new id.
- Old accepted-decision JSON without the new id is rejected by the ledger.

## Safety Boundary

This is a manual-review integrity gate. It does not contact WING, mutate
console state, grant live authority, close readiness, or authorize supervised
fader/gain proof.

## Remaining Blocker

The WING MVP still waits for a fresh real `wing_readonly_preflight.v1` capture
and an operator review decision using the expanded checklist.
