# Codex Proposal: Promotion Sequence-Policy Gate Result Consumer

## Short thesis

Add the next read-only consumer boundary after sequence-policy gate evidence,
keeping policy-consumer invocation and proposal ordering as separate future
steps.

## What I understood

Automixer is building a long evidence chain for supervised sequence-policy
learning. Each stage must verify provenance and then stop at a request template
instead of silently mutating runtime state or granting live authority.

## Proposed solution

Register the existing sequence-policy gate result consumer as a `shadow_only`
decision-evidence method, expose a Product Layer verifier handler, and add
focused tests that prove it emits only a policy-consumer request template.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `backend/handlers/product_state_handlers.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`

## Alternatives considered

- Skip the consumer and let rollback-verification result consumption call the
  policy consumer directly.
- Treat sequence-policy gate records as already consumed.

Both alternatives blur a review boundary and make future queue/import behavior
harder to audit.

## Risks

The main risk is interpretive: a policy-consumer request template can be
mistaken for policy mutation. Tests and registry metadata must keep invocation,
proposal ordering, queue import, live writes, and readiness false.

## Test plan

Run focused unit tests for the new consumer, Product Layer handler flow,
registry/matrix checks, memory lint, diff whitespace checks, and the standard
backend gate when practical.

## Where the other agent may disagree

Kimi may prefer pausing the chain until current dirty work is committed. My
counterpoint is that this is a small completion slice for a file already present
in the worktree, and it reduces ambiguity without widening live authority.
