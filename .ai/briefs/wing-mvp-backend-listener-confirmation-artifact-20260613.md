# Brief: WING MVP Backend Listener Confirmation Artifact

## Task

Add a file-backed operator confirmation path for the no-WING backend-port
prestart gate when the detected listener is the intended Automixer backend.

## Context

The first WING MVP blocker still requires real read-only WING proof. The current
local no-WING quick gate can fail closed on `backend_port_free` when
`127.0.0.1:8765` already has a listener. The gate tells the operator to confirm
whether the listener is intended, but there was no machine-readable artifact the
operator could provide on rerun.

## Safety boundary

- Do not stop or reconfigure the listener.
- Do not contact WING hardware.
- Do not grant live authority.
- Do not close readiness from this local confirmation.

## Required outcome

- Add an optional confirmation JSON path to `tools/no_wing_release_check.py`.
- Require matching host, port, PID, and command line.
- Require `safe_to_stop_listener_by_gate: false` and
  `live_authority_granted: false`.
- Keep the default no-file path fail-closed.
- Document and test the operator path.
