# Proposal: Queue Import Promotion Acceptance Ledger Result Consumer

## Short thesis

Add a read-only verifier between promotion acceptance recording and promotion
execution planning. It should validate provenance and surface only an
execution-plan request template.

## What I understood

The sequence-policy memory promotion chain advances only through explicit
operator-reviewed evidence. Acceptance records are still not executor authority;
they can only feed a separate read-only execution plan.

## Proposed solution

- Verify the prior promotion-contract result consumer envelope, summary,
  request, templates, and no-mutation flags.
- Verify the promotion acceptance ledger envelope, status, summary counts,
  accepted/rejected identities, record fields, and no-mutation flags.
- Emit a `plan_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution`
  request template only when at least one accepted promotion record exists.
- Register the method as `shadow_only` decision evidence and wire Product Layer
  dispatch for verification plus the separate execution-plan handler.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- Focused unit tests, Product Layer chain test, capability matrix test.
- `.ai/memory/project.md` and ADR.

## Alternatives considered

- Stop after promotion acceptance recording.
- Build the execution plan inside the verifier.
- Treat accepted promotion records as direct policy trust.

## Risks

The main risk is confusing accepted promotion evidence with policy execution.
The consumer avoids that by emitting only a request template and keeping all
execution, policy, queue, memory, model, live authority, and autonomous-ready
flags false.

## Test plan

- Compile touched modules.
- Run the focused consumer test.
- Run Product Layer and capability matrix tests.
- Run neighboring promotion-chain tests.
- Run memory lint and diff check.

## Where the other agent may disagree

Kimi may argue execution planning should wait until an executor contract is
ready. My view is that a non-executing request template is a useful boundary:
it makes the next missing executor step explicit without granting authority.
