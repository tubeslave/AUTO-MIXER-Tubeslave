# WING Read-Only Prestart Detail Fail-Closed Condition Brief

## Task

Expose stale backend-port prestart failure-detail contracts as an explicit
offline plan fail-closed condition for the WING read-only preflight path.

## Constraints

- No hardware contact.
- No backend WING WebSocket session.
- No OSC/WING writes.
- Keep the first WING MVP blocker open until real WING artifacts are reviewed.

## Expected Output

- Offline expected plans include `stale_backend_port_prestart_detail_contract`.
- Evidence validation requires that fail-closed condition.
- The runbook tells operators stale prestart-detail contracts fail closed.
- Tests cover runner, evidence, and no-WING plan schema behavior.
