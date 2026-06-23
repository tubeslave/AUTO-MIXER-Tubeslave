# WING MVP Review Decision Run-Id Verdict Field

## Summary

WING read-only preflight decision records now require
`event_log_run_id_matches_summary: true`.

## Changed

- `backend/wing_readonly_preflight_contract.py` adds the verdict field to the
  shared decision-template required fields.
- `tools/review_wing_readonly_preflight_artifacts.py` emits the field in the
  pending operator decision template.
- `backend/wing_readonly_preflight_review_decision.py` rejects decision records
  missing the true verdict.
- Focused tests prove the ledger, review tool, decision tool, and no-WING
  surfaces carry the new requirement.
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md` documents the field in the minimum
  accepted-decision JSON.

## Safety Boundary

This is an evidence-integrity gate. It does not contact WING, mutate console
state, grant live authority, close readiness, or authorize supervised fader/gain
proof.

## Remaining Blocker

The WING MVP still waits for a fresh real `wing_readonly_preflight.v1` capture
and an operator review decision that includes the run-id verdict field.
