# Brief: WING Proof Plan Accepted Review Source Trace

## Task ID

wing-proof-plan-accepted-review-source-trace-20260607

## Context

The next supervised WING fader/gain proof is gated by a clean read-only
preflight artifact and an accepted operator decision. The proof plan currently
knows that a decision was accepted, but it should also carry the accepted
artifact source id so the physical proof is traceable back to the reviewed
read-only evidence.

## Scope

- Keep the proof plan offline and non-executing.
- Do not contact WING Rack, WebSocket, OSC, MIDI, or any hardware transport.
- Do not grant readiness, live authority, or live write widening.
- Propagate accepted/rejected decision source ids from the review-decision
  ledger into the supervised proof plan and no-WING summaries.

## Deliverables

- `supervised_wing_proof_plan` summary includes accepted/rejected decision
  source ids.
- No-WING next-proof-plan validator compares those ids with the review-decision
  ledger.
- Summary/markdown output renders the source ids.
- Focused regression tests.

## Validation

- Focused supervised proof plan and no-WING tests.
- WING review-decision tests.
- No-WING quick release check.
