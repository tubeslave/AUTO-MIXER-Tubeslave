# Automixer WING MVP Current First Prestart Summary Match Flags

## Summary

This slice makes `Current First Command` require the no-WING handoff packet's
prestart summary match verdicts:

- `backend_port_prestart_summary_type_matches_plan: true`
- `backend_port_prestart_summary_status_matches_plan: true`
- `backend_port_prestart_summary_connection_policy_matches_plan: true`
- `backend_port_prestart_summary_live_write_matches_plan: true`
- `backend_port_prestart_summary_contacts_backend_matches_plan: true`
- `backend_port_prestart_summary_queries_console_matches_plan: true`
- `backend_port_prestart_summary_executes_tests_matches_plan: true`
- `backend_port_prestart_check_status_matches_plan: true`

The values were already emitted by the handoff packet validator. This change
makes them part of Project Director baseline coverage, so future Goal-mode work
does not proceed from a packet whose prestart summary contract drifted from the
plan artifact.

## Safety

This is offline rail hardening only. It does not contact WING, open the backend
WebSocket, grant live authority, close readiness, or change fader/gain behavior.

## Validation

- PASS: focused Project Director and WING MVP rail tests, `11 passed`.
- PASS: Project Director JSON selected `WING read-only connection proof`, used
  `Current First Command`, kept `guard_status: on_target`, and reported
  `current_first_command_baseline_coverage_passed: true` with marker count `99`
  and no missing markers.
- PASS: direction guard `on_target`, score `90`, stop-the-line false.
- PASS: py_compile for the changed autoloop tools.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`.
- PASS: `/opt/homebrew/bin/git diff --check`.
- PASS: full standard pytest, `3310 passed, 1 skipped, 1 warning`.
- PASS: no-WING quick failed closed on the existing `127.0.0.1:8765` listener
  with `backend_port_listener_requires_operator_confirmation`,
  `live_write: false`, and no execution/readiness/live-authority grant.
