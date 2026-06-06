# ADR: source-identity-calibration-memory-trust-promotion-plan-20260606

## Context

Source identity calibration memory now has an index, trust review, and operator
acceptance ledger. The next boundary is identifying which accepted records are
eligible for a future promotion contract without changing runtime behavior.

## Options Considered

1. Let accepted memory-trust records immediately update downstream source
   identity trust.
2. Keep accepted records only in the ledger.
3. Add a separate read-only manual promotion plan.

## Decision

Use option 3. Add a transport-free promotion plan that validates the acceptance
ledger and surfaces manual promotion candidates only.

## Why This Won

The plan keeps the learning chain explicit. Accepted memory-trust records become
visible as candidate inputs for future promotion work, while models, runtime
memory, downstream proposal priority, queues, and live console state remain
unchanged.

## Rejected Alternatives

- Direct trust promotion was rejected because accepted memory still lacks a
  promotion contract with replay freshness, rollback, and proposal-family
  semantics.
- Ledger-only evidence was rejected because future promotion needs a stable
  candidate payload.
- Queueable promotion planning was rejected because this is not an executable
  mixer action.

## Implementation Plan

1. Add a backend promotion-plan module with strict acceptance-ledger validation.
2. Validate ledger envelope, status, safety flags, summary counts, source-type
   lists, accepted-record structure, UTC timestamps, and false-authority flags.
3. Wire promotion-plan status and counts into `main_pipeline_preview`.
4. Register the plan as `shadow_only` memory-learning evidence.
5. Add focused plan, Product Layer, registry, and capability tests.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_trust_promotion_plan.py tests/test_source_identity_calibration_memory_trust_acceptance_ledger.py tests/test_source_identity_calibration_memory_trust_review.py tests/test_source_identity_calibration_memory_index.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_memory_trust_promotion_plan.py backend/source_identity_calibration_memory_trust_acceptance_ledger.py backend/source_identity_calibration_memory_trust_review.py backend/source_identity_calibration_memory_index.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks And Rollback

Risk: manual promotion candidates could be mistaken for actual downstream trust
authority. Mitigation: plan candidates force
`memory_trust_promotion_allowed_by_plan=false`,
`downstream_priority_change_allowed_by_plan=false`,
`queue_mutation_allowed_by_plan=false`, `model_mutation_allowed_by_plan=false`,
`runtime_memory_mutation_allowed_by_plan=false`, `live_authority_granted=false`,
and `full_autonomous_ready=false`.

Rollback: remove the promotion-plan call from `processing_pipeline_preview.py`,
remove registry/capability rows, and keep accepted memory-trust records as
ledger evidence only.
