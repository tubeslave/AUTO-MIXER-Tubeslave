# WING MVP Handoff Backend Connection Confirmation

## Summary

The WING read-only handoff now requires an operator-sequence step that confirms
the existing backend connection is the intended WING Rack and is ready for
read-only inspection before capture starts.

The no-WING handoff summary must show:

- `backend_connection_confirmation_step_index: 2`
- `operator_sequence_has_backend_connection_confirmation: true`
- `operator_sequence_backend_connection_confirmation_step_index: 2`
- `operator_sequence_backend_connection_confirmation_before_capture: true`
- `operator_sequence_backend_connection_confirmation_step` includes
  `existing backend connection`, `intended WING Rack`, and
  `read-only inspection`
- `operator_sequence_capture_readonly_preflight_step_index: 4`

This is a handoff/rail hardening change. It does not contact WING, start a
backend, write mixer state, grant live authority, or close the WING read-only
connection proof blocker.

## Validation

Completed before commit:

- Touched Python compile check: passed.
- Focused handoff/no-WING tests: `268 passed`.
- Focused Project Director/direction/WING rail tests: `291 passed`.
- no-WING quick release check with backend port `65535`: passed.
- Project Director stayed on `WING read-only connection proof` with
  `guard_status=on_target`, `guard_score=90`, and `guard_stop_the_line=False`.
- Direction guard stayed `on_target`, `score=90`, and
  `stop_the_line=False`.
- `git diff --check`: passed.
- `tools/memory/memory_lint.py`: PASS.
- Full pytest: `3287 passed, 1 skipped, 1 warning`.
