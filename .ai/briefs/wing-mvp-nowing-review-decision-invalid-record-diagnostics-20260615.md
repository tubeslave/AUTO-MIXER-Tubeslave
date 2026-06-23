# Brief: WING MVP no-WING Review-Decision Invalid-Record Diagnostics

## Task

Expose invalid WING read-only review decision record diagnostics in the
aggregated no-WING release summary.

## Context

Standalone decision-review reports now render invalid operator records with
`template_status` and `violations`. The no-WING release summary still carried
only `invalid_record_count` and `invalid_record_ids`, so the quick operator gate
could say the ledger failed without showing the stale pending-template reason.

## Safety Boundary

This is an offline reporting change. It does not contact WING, mutate console
state, update readiness rows, or grant live authority.

## Expected Evidence

- no-WING schema detail includes `invalid_records`.
- no-WING `summary.json` mirrors invalid record diagnostics.
- no-WING `summary.md` renders the stale-template violation for operator triage.
