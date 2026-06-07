# Brief: WING Proof Plan Review Decision Collision Trace

## Task ID

wing-proof-plan-review-decision-collision-trace-20260607

## Context

The WING read-only preflight decision ledger now fails closed on duplicate or
conflicting operator decisions. The next supervised fader/gain proof plan should
carry the exact collision diagnostics forward so the operator can replace the
right files before any supervised proof is planned.

## Scope

- Keep the proof plan offline and non-executing.
- Do not contact WING Rack, WebSocket, OSC, MIDI, or any hardware transport.
- Do not grant readiness, live authority, or live write widening.
- Expose duplicate/conflicting decision source ids in the proof-plan summary
  and no-WING release summary.

## Deliverables

- `supervised_wing_proof_plan` summary includes invalid decision record ids,
  duplicate source ids, and conflicting source ids.
- No-WING next-proof-plan validator compares those fields with the review
  decision ledger.
- Summary/markdown output renders the collision diagnostics.
- Focused regression tests.

## Validation

- Focused supervised proof plan and no-WING tests.
- WING decision ledger/tool tests.
- No-WING quick release check.
