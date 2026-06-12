# WING Handoff Structured Gate Summary Brief

## Task

Expose the structured WING handoff `supervised_prestart_gate` contract in
no-WING release JSON and Markdown summaries.

## Constraints

- No hardware contact.
- No backend WING WebSocket session.
- No OSC/WING writes.
- No live authority widening.
- Preserve the existing validator behavior.

## Expected Output

- Release JSON summary includes `supervised_prestart_gate`.
- Release Markdown summary renders the same contract.
- Tests assert the summary keeps `grants_live_authority: false` and the
  accepted-candidate prestart trace requirement.
