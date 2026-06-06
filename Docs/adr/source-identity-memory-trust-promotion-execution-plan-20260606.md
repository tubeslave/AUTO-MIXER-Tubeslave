# ADR: source-identity-memory-trust-promotion-execution-plan-20260606

## Context

Source identity memory-trust promotion acceptance records can now capture an
operator decision on promotion contract evidence. The system still needs a
separate review artifact that describes what would be executed later without
actually mutating trust, proposal priority, queues, models, runtime memory, or
live console state.

## Options Considered

1. Execute accepted promotion records directly.
2. Keep accepted promotion records only in the acceptance ledger.
3. Add a separate read-only promotion execution plan.

## Decision

Use option 3. Add a transport-free execution plan that turns accepted promotion
records into non-executing candidates.

## Why This Won

The plan makes the next executor boundary auditable without granting authority.
It preserves the full evidence chain while keeping downstream priority changes
blocked until a future executor implementation and safety contract exist.

## Rejected Alternatives

- Direct execution was rejected because rollback application and executor safety
  semantics are not implemented.
- Acceptance-ledger-only evidence was rejected because it leaves the next
  promotion step implicit.
- Queueable execution candidates were rejected because memory trust promotion is
  not a live mixer action and must not enter the Product Layer task queue.

## Implementation Plan

1. Add a backend promotion execution plan module.
2. Validate promotion acceptance ledger envelope, status, summary counts, and
   accepted records.
3. Reject unsafe authority flags, forged summary counts, and ambiguous contract
   decisions.
4. Wire plan status and candidate counts into `main_pipeline_preview`.
5. Register the plan as `shadow_only` memory-learning evidence.
6. Add focused plan, Product Layer, registry, and capability tests.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_trust_promotion_execution_plan.py tests/test_source_identity_calibration_memory_trust_promotion_acceptance_ledger.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_memory_trust_promotion_execution_plan.py backend/source_identity_calibration_memory_trust_promotion_acceptance_ledger.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks And Rollback

Risk: accepted execution candidates could be mistaken for actual promotion
authority. Mitigation: candidates force `execution_allowed_by_plan=false`,
`memory_trust_promotion_allowed_by_plan=false`,
`downstream_priority_change_allowed_by_plan=false`,
`queue_mutation_allowed_by_plan=false`,
`model_mutation_allowed_by_plan=false`,
`runtime_memory_mutation_allowed_by_plan=false`,
`live_authority_granted=false`, and `full_autonomous_ready=false`.

Rollback: remove the plan call from `processing_pipeline_preview.py`, remove
registry/capability rows, and keep promotion acceptance records as evidence
only.
