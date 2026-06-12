# WING Read-Only Runner Stale Prestart Plan Regression Brief

## Task

Prove the read-only WING preflight runner blocks stale expected plans that do
not include the current backend-port prestart failure-detail contract.

## Constraints

- No hardware contact.
- No backend WING WebSocket session.
- No OSC/WING writes.
- Keep the first WING MVP blocker open until real WING artifacts are reviewed.

## Expected Output

- Runner tests prove a stale expected plan missing
  `listener_confirmation_checklist` fails before WebSocket connect.
- The failure reports
  `expected_plan_backend_port_prestart_check_failure_detail_fields_mismatch`.
- The summary still carries the current prestart contract.
