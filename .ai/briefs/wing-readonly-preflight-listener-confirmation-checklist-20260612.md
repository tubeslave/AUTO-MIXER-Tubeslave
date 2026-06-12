# WING Read-Only Preflight Listener Confirmation Checklist Brief

## Task

Strengthen the no-WING backend-port prestart failure artifact with a
machine-readable checklist for confirming whether an existing listener is the
intended Automixer backend.

## Constraints

- No hardware contact.
- No backend WING WebSocket session.
- No OSC/WING writes.
- Do not stop or reconfigure local listener processes.
- Keep the first WING MVP blocker open until real WING artifacts are reviewed.

## Expected Output

- Busy backend-port failures include `listener_confirmation_checklist`.
- The shared WING preflight contract lists that field as expected failure
  detail.
- The read-only preflight runbook tells operators to inspect the checklist.
- Tests verify JSON detail, Markdown summary, and contract/runbook coverage.
