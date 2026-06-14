# Automixer WING MVP Handoff Prestart Summary Contract

## Summary

This slice tightens the WING read-only handoff packet. The packet now exposes the
same backend-port prestart summary contract that the no-WING plan artifact
already requires:

- `backend_port_prestart_summary_type: no_wing_release_check`
- `backend_port_prestart_summary_status: passed`
- `backend_port_prestart_summary_connection_policy: no_hardware_no_network_no_live_writes`
- `backend_port_prestart_summary_live_write: false`
- `backend_port_prestart_summary_contacts_backend: false`
- `backend_port_prestart_summary_queries_console: false`
- `backend_port_prestart_summary_executes_tests: false`
- `backend_port_prestart_check_status: passed`

The no-WING handoff validator now compares those packet fields against the plan
artifact and the shared contract constants. A stale packet that changes type,
status, policy, no-contact flags, or backend check status fails closed.

## Safety

This is an offline contract and reporting change only. It does not contact WING,
does not open the backend WebSocket, does not grant live authority, and does not
close the `WING read-only connection proof` blocker.

## Validation

- PASS: focused handoff/no-WING pytest, `288 passed`.
- PASS: focused handoff/no-WING/director/rails pytest, `299 passed`.
- PASS: py_compile for the changed tools.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`.
- PASS: `/opt/homebrew/bin/git diff --check`.
- PASS: full standard pytest, `3310 passed, 1 skipped, 1 warning`.
- PASS: no-WING quick failed closed on the existing `127.0.0.1:8765` listener
  with `backend_port_listener_requires_operator_confirmation`,
  `live_write: false`, and no execution/readiness/live-authority grant.
