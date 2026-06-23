# Brief: WING MVP Review Decision Ledger Artifact Root Contract Gate

Date: 2026-06-14

## Objective

Keep accepted WING read-only review decisions from becoming manual readiness
candidates when their accepted candidate artifact-root contract is invalid.

## Context

The supervised proof plan and no-WING validator recompute the accepted-candidate
artifact-root contract, but the review-decision ledger still marked
`manual_readiness_update_candidate_available` from prestart validity alone.
That left a stale intermediate signal before later gates blocked proof steps.

## Safety Boundary

This is offline/read-only validation. It must not contact WING, mutate console
state, update readiness, or grant live authority.

## Acceptance

- The review-decision ledger exposes accepted-candidate artifact-root contract
  fields next to the existing prestart contract fields.
- `manual_readiness_update_candidate_available` is false when the accepted
  candidate artifact-root contract has violations.
- The artifact-root blocker is present in ledger `blocked_by`.
- Existing inline test evidence remains supported without weakening file-backed
  proof-plan/no-WING path checks.
