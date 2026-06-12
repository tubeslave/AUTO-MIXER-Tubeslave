# ADR: WING Read-Only no-WING Current Backend-Port Blocker

## Context

The supervised WING MVP first open blocker is `WING read-only connection proof`.
The current runbook requires `PYTHONPATH=backend python3
tools/no_wing_release_check.py --quick` before starting the intended
backend/WING session.

The current local run was executed with `--continue-on-failure` and failed only
on `backend_port_free`. The check reported a Python listener on
`127.0.0.1:8765` and required operator confirmation before any local service is
stopped or reconfigured.

## Options considered

1. Automatically stop the listener.
2. Change the gate to allow the existing listener.
3. Record the current fail-closed result and leave the operator action explicit.

## Decision

Use option 3. Record the current fail-closed result as operator-visible
evidence, without changing code, stopping the listener, opening a backend WING
WebSocket, or contacting WING hardware.

## Why this won

The current code already exposes the required busy-port detail fields. The
safe next move is to preserve the exact state for handoff: the gate is working,
the first WING blocker remains open, and the operator must decide whether the
existing listener is the intended backend session.

## Rejected alternatives

- Automatic listener stop: unsafe because the process may be the intended
  backend.
- Gate relaxation: unsafe because a stale or unexpected listener could be
  mistaken for a ready WING session.

## Implementation plan

- Add a brief and proposal under `.ai/`.
- Add this ADR.
- Add a dated `Docs/reports/` note with the command result, listener detail,
  safety boundary, and next action.

## Test plan

- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --continue-on-failure`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`
- Focused WING/no-WING regression tests.

## Risks and rollback

The report could become stale after the operator confirms or clears the
listener. Roll back by replacing it with a newer dated report from the next
gate run; do not rewrite the safety boundary retroactively.
