# Brief: WING Read-Only Preflight Review Decision Collision Gate

## Task ID

wing-readonly-preflight-review-decision-collision-gate-20260607

## Context

The WING read-only preflight path now requires a clean real artifact and an
operator review decision before the next supervised proof can be planned. The
decision ledger must not treat conflicting or duplicate decision JSON files for
the same preflight candidate as a valid accepted review.

## Scope

- Keep the change offline and read-only.
- Do not contact WING Rack, WebSocket, OSC, MIDI, or any hardware transport.
- Do not grant readiness or live authority from decision records.
- Fail closed when more than one valid decision targets the same clean
  preflight candidate.

## Deliverables

- Decision ledger detects duplicate artifact decision records.
- Accepted/rejected conflicts are reported as invalid records.
- Decision-review CLI exposes duplicate and conflicting source ids.
- Runbook, ADR, report, and documentation index update.

## Validation

- Focused decision ledger and decision-review tool tests.
- WING preflight focused tests.
- No-WING quick release check.
