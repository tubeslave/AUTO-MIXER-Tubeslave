# ADR: WING Read-Only Preflight Listener Command-Line Diagnostics

## Context

The WING read-only preflight runbook requires a no-WING quick check before starting the intended backend/WING session. If the backend port is busy, the operator needs enough local evidence to decide whether the listener is the intended Automixer backend. PID and process name can be ambiguous.

## Options considered

1. Keep only `lsof` PID, command name, and socket name.
2. Add a local `ps` command-line probe for each discovered PID.
3. Probe the listener over WebSocket to identify Automixer.

## Decision

Use option 2. Busy-port diagnostics now include `command_line` when `ps` can read it, or `command_line_error` when it cannot.

## Why this won

It improves operator evidence without contacting the listener, stopping it, or changing any live-control path.

## Rejected alternatives

Option 1 leaves too little evidence for ambiguous `Python` listeners. Option 3 may blur the no-WING prestart boundary because it contacts a local control service.

## Implementation plan

- Add `_process_command_line`.
- Enrich `_listener_processes_for_port` records with `command_line` or `command_line_error`.
- Add tests for successful and failed command-line probes.

## Test plan

- `python3 -m py_compile tools/no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py -k "listener_processes or backend_port"`
- Broader no-WING/WING focused tests.

## Risks and rollback

Risk: platform-specific `ps` output may be unavailable. The fallback records an error and preserves the original listener record. Rollback: remove `_process_command_line` and the extra record fields. No live write path is changed.
