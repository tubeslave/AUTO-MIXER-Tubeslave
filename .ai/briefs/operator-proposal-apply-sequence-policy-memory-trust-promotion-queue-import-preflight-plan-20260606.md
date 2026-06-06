# Brief: Operator Proposal Apply Sequence Policy Memory Trust Promotion Queue Import Preflight Plan

## Context

Sequence-policy queue importer records can create pending operator proposals.
Before those proposals can enter apply-sequence planning, they must pass the
existing batch preflight flow with explicit operator approval and idempotency
evidence.

## Goal

Add a read-only bridge that scans pending sequence-policy imported proposals and
emits an ordered batch-preflight request template. The bridge must not run
preflight, reserve idempotency, build sequence plans, mutate queue state, or
touch live mixer transport.

## Constraints

- No live OSC/WING writes.
- No direct sequence execution.
- No queue mutation, proposal-state mutation, idempotency reservation, policy
  store mutation, runtime-memory mutation, or model mutation.
- Reject forged sequence-policy import metadata that claims live authority or
  removes the separate sequence-consumer requirement.
- Use the existing `preflight_operator_proposal_apply_batch` handler as the next
  step, with explicit operator approval.

## Validation

- Focused pytest for the preflight plan module, Product State handler, decision
  queue visibility, registry, and capability matrix.
- `python3 tools/memory/memory_lint.py`.
