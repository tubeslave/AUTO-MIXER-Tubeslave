# WING Read-Only Expected Plan Next Step Brief

## Task

Add explicit `operator_next_step` guidance to WING read-only preflight runner
summaries when the expected-plan gate fails before WebSocket contact.

## Constraints

- No hardware contact.
- No backend WING WebSocket session.
- No OSC/WING writes.
- Keep the first WING MVP blocker open until real WING artifacts are reviewed.

## Expected Output

- Missing expected plan summaries tell operators to generate a plan with
  `--plan-only` and rerun with `--expected-plan`.
- Invalid expected plan summaries tell operators to regenerate the plan with
  matching arguments.
- Mismatched expected plan summaries tell operators to match runtime arguments
  and the current prestart contract before opening WebSocket.
- Tests prove all branches send no requests.
