# WING MVP Read-Only Event Log Run-Id Binding

## Summary

Real WING read-only preflight `events.jsonl` rows now carry the same `run_id`
as `summary.json`, and the offline evidence validator fails closed when the
event log belongs to a different run.

## Changed

- `tools/wing_readonly_preflight_runner.py` writes `run_id` into each event log
  row.
- `backend/wing_readonly_preflight_evidence.py` validates event-row run ids
  against the summary run id.
- `tools/review_wing_readonly_preflight_artifacts.py` renders
  `event_log_run_id_matches_summary` in the operator review report.
- Focused tests cover both valid event logs and mismatched event-log run ids.

## Safety Boundary

This is an offline artifact integrity check for a future physical WING capture.
It does not contact WING during validation, mutate console state, grant live
authority, close readiness, or authorize fader/gain movement.

## Remaining Blocker

The WING MVP still needs a fresh real `wing_readonly_preflight.v1` capture and
operator review before any supervised fader/gain proof.
