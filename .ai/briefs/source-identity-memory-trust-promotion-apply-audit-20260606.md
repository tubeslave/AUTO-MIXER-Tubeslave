# Brief: source-identity-memory-trust-promotion-apply-audit-20260606

## Context

The source identity memory-trust promotion executor can now apply accepted
dry-run specs to an explicit caller-provided trust store. Before any applied
trust can inform downstream proposal priority, the project needs rollback audit
evidence that validates executor output and keeps priority/live authority
separate.

## Objective

Add a read-only apply audit that consumes promotion executor output, records the
rollback target trust state for applied results, and fails closed on forged
executor authority or summary drift.

## Safety Constraints

- Do not touch WING, dLive, OSC, MIDI, TCP, sockets, or mixer clients.
- Do not import audit records into the Product Layer task queue.
- Do not mutate runtime memory, classifier models, or downstream proposal
  priority.
- Product Layer preview must remain non-mutating and show zero audit records
  unless an executor was actually applied outside preview.
- Audit records are rollback evidence only; live authority and full autonomous
  readiness remain false.

## Expected Evidence

- A backend audit validates executor envelope, status, summary counts, safety
  flags, apply results, target paths, and duplicate IDs.
- Applied executor results produce rollback audit records with
  `rollback_target_trust_state`.
- Forged live, queue, model, or direct downstream-priority authority is rejected.
- Product Layer preview exposes the audit as not-ready with zero records.
- Registry and capability matrix register the audit as non-live
  memory-learning infrastructure.
