# ADR: source-identity-calibration-memory-trust-promotion-contract-20260606

## Context

Source identity calibration memory now has read-only indexing, memory-trust
review, operator acceptance, and manual promotion planning. The missing boundary
is a contract that checks replay freshness and rollback evidence before any
future priority promotion can be reviewed.

## Options Considered

1. Promote accepted memory-trust candidates directly into downstream priority.
2. Keep manual promotion candidates without a readiness contract.
3. Add a separate read-only promotion contract.

## Decision

Use option 3. Add a transport-free promotion contract that validates manual
promotion candidates, fresh replay context, and rollback evidence, then exposes
operator-review-ready records only.

## Why This Won

The contract makes the next learning gate explicit while preserving current
safety boundaries. Accepted memory can become review-ready evidence, but models,
runtime memory, downstream proposal priority, queues, and live console state
remain unchanged.

## Rejected Alternatives

- Direct priority promotion was rejected because it lacks an audited executor,
  rollback path, and proposal-family semantics.
- Plan-only evidence was rejected because it does not prove freshness or
  rollback readiness.
- Queueable promotion was rejected because this is not a live mixer action.

## Implementation Plan

1. Add a backend promotion-contract module with strict promotion-plan validation.
2. Require fresh read-only replay context and safe rollback evidence.
3. Wire contract status and counts into `main_pipeline_preview`.
4. Register the contract as `shadow_only` memory-learning evidence.
5. Add focused contract, Product Layer, registry, and capability tests.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_trust_promotion_contract.py tests/test_source_identity_calibration_memory_trust_promotion_plan.py tests/test_source_identity_calibration_memory_trust_acceptance_ledger.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_memory_trust_promotion_contract.py backend/source_identity_calibration_memory_trust_promotion_plan.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks And Rollback

Risk: contract-ready records could be mistaken for actual downstream trust
authority. Mitigation: records force
`memory_trust_promotion_allowed_by_contract=false`,
`downstream_priority_change_allowed_by_contract=false`,
`queue_mutation_allowed_by_contract=false`, `model_mutation_allowed_by_contract=false`,
`runtime_memory_mutation_allowed_by_contract=false`,
`live_authority_granted=false`, and `full_autonomous_ready=false`.

Rollback: remove the contract call from `processing_pipeline_preview.py`, remove
registry/capability rows, and keep manual promotion candidates as plan evidence
only.
