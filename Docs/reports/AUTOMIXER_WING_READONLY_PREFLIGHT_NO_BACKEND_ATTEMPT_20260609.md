# Automixer WING Read-Only Preflight Attempt - 2026-06-09

## Summary

The read-only preflight attempt was executed against `ws://localhost:8765` with
an explicit offline plan and expected-plan binding.

- Offline plan command was generated successfully and saved to:
  - `/tmp/wing_preflight_goal_plan/latest_plan.json`
- Real run was executed with `--expected-plan` and returned:
  - status: `blocked`
  - blockers: `["transport_timeout_or_connection_loss"]`
  - reason: `read_only_preflight_transport_error`
  - run ID: `20260609T115334Z_wing_readonly_preflight`
  - connection fail details:
    - `127.0.0.1: ConnectionRefusedError [Errno 61]`
    - `::1: ConnectionRefusedError [Errno 61]`

No live request was executed and no real `summary.json`/`events.jsonl` candidate was
written.

## Next required action

Run `Docs/runbooks/WING_READONLY_PREFLIGHT.md` only when the intended WING backend
is available at `ws://localhost:8765`, then review a clean candidate and record a
valid accepted decision before any supervised fader/gain proof work.
