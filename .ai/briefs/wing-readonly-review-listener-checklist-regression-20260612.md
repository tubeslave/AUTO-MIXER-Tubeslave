# WING Read-Only Review Listener Checklist Regression Brief

## Task

Lock the new backend-port `listener_confirmation_checklist` into the manual
review and decision review surfaces so it cannot silently disappear from
operator-visible evidence.

## Constraints

- No hardware contact.
- No backend WING WebSocket session.
- No OSC/WING writes.
- No live authority widening.
- Keep the first WING MVP blocker open until real WING artifacts are reviewed.

## Expected Output

- Review report tests require `listener_confirmation_checklist`.
- Decision review report tests require `listener_confirmation_checklist`.
- no-WING release summary tests require the checklist inside decision ledger
  candidate reviews.
