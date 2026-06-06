# Proposal: Queue Import Promotion Contract Result Consumer

## Short thesis

Add a read-only verifier between promotion-contract creation and promotion
acceptance recording. It should validate provenance and surface only an empty
acceptance-ledger request template.

## What I understood

The sequence-policy memory promotion chain is intentionally split into narrow
evidence steps. Promotion contracts are still not authority: the operator must
explicitly record accept/reject decisions before any later executor path can
even consider promotion.

## Proposed solution

- Verify the prior promotion-plan result consumer envelope, summary, request,
  and no-mutation flags.
- Verify the promotion contract envelope, ready status, summary counts, record
  identities, rollback fields, and no-mutation flags.
- Emit a `record_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance`
  request template with `acceptance_records=[]`.
- Register the method as `shadow_only` decision evidence and wire Product Layer
  dispatch for verification plus the separate acceptance-ledger handler.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- Focused unit tests, Product Layer chain test, capability matrix test.
- `.ai/memory/project.md` and ADR.

## Alternatives considered

- Stop after promotion-contract creation.
- Build the acceptance ledger inside the verifier.
- Treat a ready contract as accepted policy trust.

## Risks

The main risk is confusing a ready contract with operator acceptance. The
consumer avoids that by keeping `acceptance_records` empty, setting all mutation
and authority flags false, and requiring a separate handler call.

## Test plan

- Compile touched modules.
- Run the focused consumer test.
- Run Product Layer and capability matrix tests.
- Run neighboring promotion-chain tests.
- Run memory lint, diff check, and full pytest.

## Where the other agent may disagree

Kimi may prefer stopping after the contract until an explicit acceptance-ledger
UX exists. My view is that a request template is still safe because it does not
record acceptance and makes the missing operator decision explicit.
