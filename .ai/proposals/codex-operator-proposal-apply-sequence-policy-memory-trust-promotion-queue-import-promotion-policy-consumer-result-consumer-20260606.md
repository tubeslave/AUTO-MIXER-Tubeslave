# Codex Proposal: Promotion Policy-Consumer Result Consumer

## Short thesis

Add the next read-only consumer boundary after policy-consumer evidence, keeping
ordering preview and queue-order mutation as separate future steps.

## What I understood

Automixer is building a reviewable evidence chain for supervised
sequence-policy learning. A policy-consumer result should not silently become a
reordered proposal list, queue import, policy-store write, or live authority.

## Proposed solution

Add and register a queue-import promotion policy-consumer result consumer that
validates policy-consumer records against the prior sequence-policy gate result
consumer, then emits only an ordering-preview request template.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`

## Alternatives considered

- Let policy-consumer output call ordering preview directly.
- Treat policy-consumer records as already ordered evidence.

Both alternatives collapse a useful audit boundary and make later queue import
harder to review.

## Risks

The main risk is confusing an ordering-preview request template with proposal
order mutation. Tests and registry metadata must keep ordering invocation,
queue import, policy mutation, memory/model mutation, live writes, and readiness
false.

## Test plan

Run focused unit tests for the new consumer, Product Layer handler flow,
registry/matrix checks, memory lint, diff whitespace checks, and the standard
backend gate when practical.

## Where the other agent may disagree

Kimi may argue that the existing ordering preview already validates policy
consumer records. My counterpoint is that queue-import promotion needs its own
provenance verifier before handing evidence back to the generic ordering layer.
