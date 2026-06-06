# ADR: source-identity-memory-trust-promotion-acceptance-ledger-20260606

## Context

Source identity memory-trust promotion contracts can produce operator-review
records after validating promotion candidates, fresh replay context, and
rollback evidence. The system needs to capture the operator decision on those
records without changing runtime trust or downstream proposal priority.

## Options Considered

1. Promote contract-ready records directly into downstream priority.
2. Keep contract-ready records without recording operator decisions.
3. Add a separate read-only promotion acceptance ledger.

## Decision

Use option 3. Add a transport-free ledger for operator accepted/rejected
decisions on promotion contract records.

## Why This Won

The ledger preserves the evidence chain and makes the next manual decision
auditable. It still leaves runtime trust, downstream proposal priority, queues,
models, memory, and live console state unchanged.

## Rejected Alternatives

- Direct priority promotion was rejected because a future executor and rollback
  apply path are still missing.
- Contract-only evidence was rejected because it loses the operator's explicit
  acceptance or rejection.
- Queueable acceptance was rejected because this is not a live mixer action.

## Implementation Plan

1. Add a backend promotion acceptance ledger module.
2. Validate contract envelope, status, safety flags, summary counts, and
   contract records.
3. Validate operator decision records and reject unsafe authority claims.
4. Wire ledger status and counts into `main_pipeline_preview`.
5. Register the ledger as `shadow_only` memory-learning evidence.
6. Add focused ledger, Product Layer, registry, and capability tests.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_trust_promotion_acceptance_ledger.py tests/test_source_identity_calibration_memory_trust_promotion_contract.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_memory_trust_promotion_acceptance_ledger.py backend/source_identity_calibration_memory_trust_promotion_contract.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks And Rollback

Risk: accepted promotion records could be mistaken for actual source identity
trust promotion. Mitigation: accepted records force
`execution_allowed_by_acceptance=false`,
`memory_trust_promotion_allowed_by_acceptance=false`,
`downstream_priority_change_allowed_by_acceptance=false`,
`queue_mutation_allowed_by_acceptance=false`,
`model_mutation_allowed_by_acceptance=false`,
`runtime_memory_mutation_allowed_by_acceptance=false`,
`live_authority_granted=false`, and `full_autonomous_ready=false`.

Rollback: remove the ledger call from `processing_pipeline_preview.py`, remove
registry/capability rows, and keep promotion contract records as review-ready
evidence only.
